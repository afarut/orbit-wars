"""Фасад над ``orbit_lite``: геометрия залпа (угол / упреждение / угол->планета).

ЕДИНСТВЕННОЕ место в репозитории, которое импортирует ``orbit_lite``. Прячет за собой
torch-тензоры, слоты (вместо planet-id), построение ``PlanetMovement`` и приватные
функции пакета. Снаружи — numpy/скалярный интерфейс по planet-id, как у самописных
тулов (:func:`intercept.intercept_angle`, :func:`utils.validate_launch`,
:func:`utils.planet_at_angle`), на замену которым он и сделан.

``orbit_lite`` — отдельный пакет без ``setup.py`` (см. ``producer-orbit-wars-utils/``),
поэтому его корень кладётся в ``sys.path``. В сабмишне пакет должен ехать рядом с
``model.py`` (он torch-only, numpy не тянет).

Прогноз орбит ``orbit_lite`` восстанавливает направление вращения из ``initial_planets``
-> текущей фазы (в отличие от самописного резолвера, где знак вращения был неизвестен),
поэтому ``obs``/``state`` обязан содержать ключ ``initial_planets`` (его сохраняет
``dataprep/convert.py`` и отдаёт движок Kaggle).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

# корень пакета orbit_lite в sys.path (нет setup.py)
_PKG = Path(__file__).resolve().parents[1] / "producer-orbit-wars-utils"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import torch

from orbit_lite.adapter import single_obs_to_tensor
from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import _analytic_first_contact as _ol_first_contact
from orbit_lite.intercept_aim import intercept_angle as _ol_intercept_angle
from orbit_lite.movement import PlanetMovement
from orbit_lite.movement_aiming import LAUNCH_SURFACE_OFFSET

# Горизонт прогноза: угол упреждения у orbit_lite зажат в [0, H], поэтому H должен
# покрывать самый долгий перелёт (диагональ поля ~141 при скорости 1). Сам прогноз —
# векторная тригонометрия, дёшев даже при таком H.
DEFAULT_HORIZON: int = 150


def _get(obs: Any, key: str, default: Any = None) -> Any:
    """Поле obs (kaggle Struct — наследник dict; на всякий случай и атрибут)."""
    v = obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)
    return default if v is None else v


def _obs_with_true_initial(obs: Any) -> dict:
    """Вернуть копию obs с восстановленным ``initial_planets`` (позиции игрового шага 0).

    ``orbit_lite`` реконструирует фазу орбиты из ``initial_planets`` по формуле движка
    ``angle = a0 + angvel*(step-1)`` (вращение вокруг центра поля). Но реплеи этого репо
    кладут в ``initial_planets`` ТЕКУЩИЕ планеты (a0 == текущий угол) — тогда прогноз
    orbit_lite уезжает на ``angvel*(step-1)`` (на доске 100 это до ~40 единиц). Поэтому
    восстанавливаем истинный a0, повернув текущие планеты НАЗАД на ``angvel*(step-1)``
    вокруг ``(50, 50)``. Операция идемпотентна: если obs уже содержит верный игровой
    ``initial_planets``, формула даёт то же a0. Проверено на парах кадров: ошибка
    1-шагового прогноза падает с ~40 до 0.
    """
    planets = _get(obs, "planets", []) or []
    av = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    step = int(_get(obs, "step", 0) or 0)
    back = -av * max(0, step - 1)
    c, s = math.cos(back), math.sin(back)
    initial = []
    for p in planets:
        row = list(p)
        dx, dy = float(row[2]) - 50.0, float(row[3]) - 50.0
        row[2] = 50.0 + dx * c - dy * s
        row[3] = 50.0 + dx * s + dy * c
        initial.append(row)
    out = dict(obs) if isinstance(obs, dict) else {
        k: _get(obs, k) for k in (
            "player", "planets", "fleets", "comets", "comet_planet_ids",
            "angular_velocity", "step", "episode_steps", "next_fleet_id",
            "remainingOverageTime",
        )
    }
    out["initial_planets"] = initial
    return out


@dataclass(frozen=True)
class LaunchPlan:
    """Совместимо с обращениями эвристик: ``.angle`` / ``.reaches`` / ``.safe``."""

    angle: float
    reaches: bool
    safe: bool


class GeoEngine:
    """Геометрический движок для одного ``obs`` (``PlanetMovement`` строится один раз).

    Методы принимают planet-id (как в ``obs``); slot<->id резолвится внутри. Создавать
    по одному на наблюдение/состояние и переиспользовать на всех источниках хода.
    """

    def __init__(self, obs: Any, *, player: Optional[int] = None,
                 horizon: int = DEFAULT_HORIZON, device: str = "cpu") -> None:
        pid = int(_get(obs, "player", 0)) if player is None else int(player)
        ot = single_obs_to_tensor(_obs_with_true_initial(obs), player_id=pid, device=device)
        self._mv = PlanetMovement.from_obs_tensors(ot, movement_horizon=int(horizon))
        self._id2slot = {int(p): s for s, p in enumerate(self._mv.planet_ids.tolist())}

    def _slot(self, planet_id: int) -> Optional[int]:
        return self._id2slot.get(int(planet_id))

    def intercept(self, src_id: int, dst_id: int, ships: float) -> Tuple[float, float, bool]:
        """``(angle_rad, eta, hit)`` для залпа ``src->dst`` флотом ``ships``.

        Угол считается всегда (даже если выстрел нежизнеспособен); ``hit`` == «флот
        первым касается цели», ``eta`` == inf для нежизнеспособного.
        """
        ss, ts = self._slot(src_id), self._slot(dst_id)
        if ss is None or ts is None:
            return 0.0, float("inf"), False
        mv = self._mv
        src = torch.tensor([ss], dtype=torch.long, device=mv.device)
        tgt = torch.tensor([ts], dtype=torch.long, device=mv.device)
        n = torch.tensor([float(ships)], dtype=mv.dtype, device=mv.device)
        r = _ol_intercept_angle(mv, src, tgt, n)
        return float(r["angle"][0]), float(r["eta"][0]), bool(r["viable"][0])

    def validate_launch(self, src_id: int, dst_id: int, ships: float) -> LaunchPlan:
        """Проверка прямого залпа: угол + достижимость/безопасность пути.

        У ``orbit_lite`` ``viable`` == «флот первым касается цели» == ``reaches`` И
        ``safe`` старого :func:`utils.validate_launch`, так что оба флага равны ему.
        """
        angle, _eta, viable = self.intercept(src_id, dst_id, ships)
        return LaunchPlan(angle=angle, reaches=viable, safe=viable)

    def planet_at_angle(self, src_id: int, angle: float, ships: float) -> Optional[int]:
        """Обратная операция: planet-id первой планеты на пути флота под этим углом.

        ``None`` — если первой попадает солнце/край поля или флот никого не задевает
        (как у самописного :func:`utils.planet_at_angle`).
        """
        ss = self._slot(src_id)
        if ss is None:
            return None
        mv = self._mv
        H = int(mv.movement_horizon)
        slot = torch.tensor([ss], dtype=torch.long, device=mv.device)
        sx, sy = mv.position_at_slots(slot, 0)
        ang = torch.tensor([float(angle)], dtype=mv.dtype, device=mv.device)
        ca, sa = torch.cos(ang), torch.sin(ang)
        speed = fleet_speed(
            torch.tensor([float(ships)], dtype=mv.dtype, device=mv.device)
        ).clamp(min=1e-6)
        # спавн чуть за поверхностью источника (как в движке) и летим по прямой
        launch_x = sx + ca * (mv.radii[ss] + LAUNCH_SURFACE_OFFSET)
        launch_y = sy + sa * (mv.radii[ss] + LAUNCH_SURFACE_OFFSET)
        # источник исключаем из мишеней: вылет нельзя «прибыть» на свою же планету
        # (орбитальный источник иначе может пересечь прямой путь флота позже по горизонту).
        alive = mv.alive_at(0).clone()
        alive[ss] = False
        contact, _eta = _ol_first_contact(
            launch_x=launch_x, launch_y=launch_y, cos_a=ca, sin_a=sa, speed=speed,
            px=mv.x[: H + 1, :], py=mv.y[: H + 1, :], p_alive0=alive,
            radii=mv.radii, H=H, seg_len=None,
        )
        slot_hit = int(contact[0])
        if slot_hit < 0:
            return None
        return int(mv.planet_ids[slot_hit])

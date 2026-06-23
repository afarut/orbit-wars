"""Observation -> тензоры для policy/value сети Orbit Wars.

Превращает сырой ``obs`` (dict) в фиксированные паддингованные тензоры фич по каждому типу
сущностей (планеты, кометы, флоты, фичи единственного токена-солнца и глобальные
«side»-фичи) плюс всё, что нужно для декода policy обратно в ходы: список ``places``
(метаданные планет/комет, включая готовый :class:`intercept.Target`) и индексы тех
планет/комет, которыми мы владеем.

Размерности фич по типам ниже — это контракт с ``model.py``: энкодеры строятся под
``*_FEAT_DIM``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import intercept
from .intercept import Target

# --- константы игры -----------------------------------------------------------
BOARD: float = 100.0
CENTER: Tuple[float, float] = (50.0, 50.0)
ROTATION_RADIUS_LIMIT: float = 50.0   # orbital_radius + planet_radius < этого -> вращается
SUN_RADIUS: float = 10.0
EPISODE_STEPS: int = 500
COMET_SPAWN_STEPS: Tuple[int, ...] = (50, 150, 250, 350, 450)

# --- размерности фич (контракт с model.py) ------------------------------------
PLANET_FEAT_DIM: int = 20
COMET_FEAT_DIM: int = 25          # база планеты (20) + 5 доп. фич кометы
FLEET_FEAT_DIM: int = 14          # база (10) + 4 фичи назначения (dest dx/dy, eta, флаг)
GLOBAL_FEAT_DIM: int = 11

# нормировка ETA флота (ходы); = geo_lite.DEFAULT_HORIZON, покрывает самый долгий перелёт
FLEET_ETA_NORM: float = 150.0


@dataclass
class FeatureConfig:
    max_planets: int = 40
    max_comets: int = 16
    max_fleets: int = 256
    horizons: Tuple[int, ...] = (4, 8, 16)   # горизонты предсказания будущей позиции


@dataclass
class PlaceInfo:
    """Метаданные для декода о «месте» (планета или комета, по которой можно целиться)."""

    id: int
    owner: int
    x: float
    y: float
    ships: float
    production: float
    radius: float
    kind: str                 # 'planet' | 'comet'
    is_mine: bool
    target: Target            # для intercept-тулы на этапе декода


@dataclass
class EncodedObs:
    planet_feats: torch.Tensor   # [1, max_planets, PLANET_FEAT_DIM]
    planet_mask: torch.Tensor    # [1, max_planets]  (True = реальный)
    comet_feats: torch.Tensor    # [1, max_comets, COMET_FEAT_DIM]
    comet_mask: torch.Tensor     # [1, max_comets]
    fleet_feats: torch.Tensor    # [1, max_fleets, FLEET_FEAT_DIM]
    fleet_mask: torch.Tensor     # [1, max_fleets]
    planet_owner_slot: torch.Tensor   # [1, max_planets]  long: относит. слот владельца (0..4)
    comet_owner_slot: torch.Tensor    # [1, max_comets]   long
    fleet_owner_slot: torch.Tensor    # [1, max_fleets]   long
    global_feats: torch.Tensor   # [1, GLOBAL_FEAT_DIM]
    places: List[Optional[PlaceInfo]]   # длина == max_planets + max_comets
    owned_idx: List[int]                # индексы мест, которыми мы владеем (ships > 0)
    player: int
    cfg: FeatureConfig = field(default_factory=FeatureConfig)


# --- мелкие хелперы парсинга --------------------------------------------------
def _get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _comet_path_map(obs: Any) -> Dict[int, Tuple[np.ndarray, int]]:
    """planet_id -> (path[L,2], path_index) для активных комет."""
    out: Dict[int, Tuple[np.ndarray, int]] = {}
    for group in _get(obs, "comets", []) or []:
        pids = _get(group, "planet_ids", []) or []
        paths = _get(group, "paths", []) or []
        pidx = int(_get(group, "path_index", 0) or 0)
        for k, pid in enumerate(pids):
            if k < len(paths) and paths[k] is not None and len(paths[k]):
                out[int(pid)] = (np.asarray(paths[k], dtype=np.float64), pidx)
    return out


def _owner_flags(owner: int, player: int) -> Tuple[float, float, float]:
    """(is_mine, is_enemy, is_neutral)."""
    if owner == player:
        return 1.0, 0.0, 0.0
    if owner == -1:
        return 0.0, 0.0, 1.0
    return 0.0, 1.0, 0.0


# owner id -> позиция по кругу (CCW, шаги 90°). Выведено из движка orbit_wars: owner j
# спавнится на копии группы, повёрнутой на j·90°; на всех сидах rel-позиция = {0:0,1:1,2:3,3:2}.
# Пары «напротив» (диагональ): id0↔id3, id1↔id2.
_OWNER_POS: Dict[int, int] = {0: 0, 1: 1, 2: 3, 3: 2}


def _n_players(planets: List[list], fleets: List[list]) -> int:
    """Число игроков: 4, если на доске виден owner id >= 2, иначе 2. Эвристика — в позднем
    4p с выбитыми игроками 2/3 может ошибиться на 2 (редко, слот всё равно «враг»)."""
    max_owner = -1
    for p in planets:
        max_owner = max(max_owner, int(p[1]))
    for f in fleets:
        max_owner = max(max_owner, int(f[1]))
    return 4 if max_owner >= 2 else 2


def _owner_slot(owner: int, player: int, n_players: int) -> int:
    """Относительный слот владельца для эмбеддинга: 0=мы, 1=CCW-сосед, 2=напротив,
    3=CW-сосед, 4=нейтрал. В 1v1 враг всегда напротив (диагональ)."""
    if owner == -1:
        return 4
    if owner == player:
        return 0
    if n_players == 2:
        return 2
    return (_OWNER_POS[owner] - _OWNER_POS[player]) % 4


def _build_target(
    pid: int, owner: int, x: float, y: float, radius: float,
    angular_velocity: float, comet_map: Dict[int, Tuple[np.ndarray, int]],
) -> Tuple[Target, str]:
    """Построить intercept Target и вернуть (target, строка_вида)."""
    if pid in comet_map:
        path, pidx = comet_map[pid]
        return Target(pos=(x, y), kind="comet", path=path, path_index=pidx,
                      radius=radius), "comet"
    orbital_radius = math.hypot(x - CENTER[0], y - CENTER[1])
    if orbital_radius + radius < ROTATION_RADIUS_LIMIT:
        # ЗАМЕЧАНИЕ: знак вращения из одного obs неизвестен; берём переданное значение.
        return Target(pos=(x, y), kind="orbit", center=CENTER,
                      angular_velocity=angular_velocity, radius=radius), "planet"
    return Target(pos=(x, y), kind="static", radius=radius), "planet"


def _place_base(
    owner: int, x: float, y: float, ships: float, production: float,
    radius: float, player: int, target: Target, is_orbiting: float,
    angular_velocity: float, horizons: Tuple[int, ...],
) -> List[float]:
    cx, cy = CENTER
    dist_sun = math.hypot(x - cx, y - cy)
    mine, enemy, neutral = _owner_flags(owner, player)
    feats = [
        x / BOARD, y / BOARD,
        (x - cx) / 50.0, (y - cy) / 50.0, dist_sun / 50.0,
        is_orbiting, angular_velocity * 20.0,
        mine, enemy, neutral,
        ships / 100.0, math.log1p(max(0.0, ships)) / 5.0,
        production / 5.0, radius / 5.0,
    ]
    # смещения будущих позиций (подсказка о движении), нормировка на 50
    for h in horizons:
        fx, fy = intercept.predict_position(target, float(h))
        feats.extend([(fx - x) / 50.0, (fy - y) / 50.0])
    return feats   # длина == 14 + 2*len(horizons) == 20 при 3 горизонтах


def _comet_extras(target: Target) -> List[float]:
    path = target.path
    plen = float(len(path)) if path is not None else 0.0
    remaining = plen - target.path_index if plen else 0.0
    vx, vy = intercept._target_velocity(target)
    return [1.0, target.path_index / 200.0, remaining / 200.0, vx / 6.0, vy / 6.0]


def _fleet_features(
    owner: int, x: float, y: float, angle: float, ships: float, player: int,
    my_planet_xyr: List[Tuple[float, float, float]],
    dest_xy: Optional[Tuple[float, float]], eta: float,
) -> List[float]:
    mine = 1.0 if owner == player else 0.0
    enemy = 1.0 if (owner != player and owner != -1) else 0.0
    hx, hy = math.cos(angle), math.sin(angle)
    incoming = 0.0
    if enemy:
        for px, py, pr in my_planet_xyr:
            rx, ry = px - x, py - y
            proj = rx * hx + ry * hy
            if proj <= 0:
                continue
            perp = abs(rx * hy - ry * hx)   # |rel x heading|
            if perp < pr + 3.0:
                incoming = 1.0
                break
    # фичи назначения: на какую планету летит (центр цели отн. флота) и когда долетит.
    # Нет цели (флот мимо/в край/в солнце) -> нули + has_target=0, чтобы не врать «(0,0), eta=0».
    if dest_xy is not None:
        dx = (dest_xy[0] - x) / 50.0
        dy = (dest_xy[1] - y) / 50.0
        eta_n = min(eta, FLEET_ETA_NORM) / FLEET_ETA_NORM
        has_target = 1.0
    else:
        dx = dy = eta_n = has_target = 0.0
    return [
        x / BOARD, y / BOARD, mine, enemy,
        ships / 100.0, math.log1p(max(0.0, ships)) / 5.0,
        hx, hy, intercept.fleet_speed(ships) / 6.0, incoming,
        dx, dy, eta_n, has_target,
    ]


def _global_features(obs: Any, planets: List[list], fleets: List[list],
                     player: int) -> Tuple[List[float], int]:
    step = int(_get(obs, "step", 0) or 0)
    remaining = EPISODE_STEPS - step

    n_players = _n_players(planets, fleets)

    totals: Dict[int, float] = {}
    for p in planets:
        o = int(p[1])
        if o >= 0:
            totals[o] = totals.get(o, 0.0) + float(p[5])
    for f in fleets:
        o = int(f[1])
        if o >= 0:
            totals[o] = totals.get(o, 0.0) + float(f[6])
    mine = totals.get(player, 0.0)
    opp = [v for o, v in totals.items() if o != player]
    max_opp = max(opp) if opp else 0.0
    adv = (mine - max_opp) / (mine + max_opp + 1.0)

    countdown = 50.0
    for s in COMET_SPAWN_STEPS:
        if s >= step:
            countdown = float(s - step)
            break

    player_oh = [1.0 if player == i else 0.0 for i in range(4)]
    feats = [
        step / EPISODE_STEPS, remaining / EPISODE_STEPS, n_players / 4.0,
        *player_oh,
        mine / 500.0, max_opp / 500.0, adv, countdown / 50.0,
    ]
    return feats, n_players


def encode(obs: Any, cfg: FeatureConfig = FeatureConfig(),
           device: Optional[torch.device] = None, geo: Any = None) -> EncodedObs:
    """Закодировать одно наблюдение в паддингованные тензоры + метаданные для декода.

    ``geo`` — опциональный :class:`core.geo_lite.GeoEngine` для этого ``obs`` (фичи
    назначения флота). Если не передан и флоты есть — строится лениво (на инференсе
    ``model.act`` передаёт уже готовый, чтобы не строить ``PlanetMovement`` дважды).
    """
    player = int(_get(obs, "player", 0) or 0)
    angular_velocity = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    planets = list(_get(obs, "planets", []) or [])
    fleets = list(_get(obs, "fleets", []) or [])
    comet_ids = set(int(i) for i in (_get(obs, "comet_planet_ids", []) or []))
    comet_map = _comet_path_map(obs)
    n_players = _n_players(planets, fleets)

    my_planet_xyr = [
        (float(p[2]), float(p[3]), float(p[4]))
        for p in planets if int(p[1]) == player
    ]

    # --- планеты / кометы -> токены «мест» ---
    planet_rows: List[List[float]] = []
    comet_rows: List[List[float]] = []
    planet_places: List[PlaceInfo] = []
    comet_places: List[PlaceInfo] = []
    planet_owner_slots: List[int] = []
    comet_owner_slots: List[int] = []

    for p in planets:
        pid, owner, x, y, radius, ships, production = (
            int(p[0]), int(p[1]), float(p[2]), float(p[3]),
            float(p[4]), float(p[5]), float(p[6]),
        )
        target, kind = _build_target(
            pid, owner, x, y, radius, angular_velocity, comet_map)
        is_comet = (pid in comet_ids) or kind == "comet"
        is_orbiting = 1.0 if target.kind == "orbit" else 0.0
        base = _place_base(owner, x, y, ships, production, radius, player,
                           target, is_orbiting, angular_velocity, cfg.horizons)
        info = PlaceInfo(id=pid, owner=owner, x=x, y=y, ships=ships,
                         production=production, radius=radius,
                         kind="comet" if is_comet else "planet",
                         is_mine=(owner == player), target=target)
        slot = _owner_slot(owner, player, n_players)
        if is_comet:
            comet_rows.append(base + _comet_extras(target))
            comet_places.append(info)
            comet_owner_slots.append(slot)
        else:
            planet_rows.append(base)
            planet_places.append(info)
            planet_owner_slots.append(slot)

    # назначение каждого флота (planet-id + ETA) через geo_lite first-contact
    fleet_dest_xy: List[Optional[Tuple[float, float]]] = [None] * len(fleets)
    fleet_eta: List[float] = [float("inf")] * len(fleets)
    if fleets:
        if geo is None:
            from .geo_lite import GeoEngine
            geo = GeoEngine(obs, player=player,
                            device=str(device) if device is not None else "cpu")
        pid_xy = {int(p[0]): (float(p[2]), float(p[3])) for p in planets}
        dest_ids, etas = geo.fleet_targets(
            [float(f[2]) for f in fleets], [float(f[3]) for f in fleets],
            [float(f[4]) for f in fleets], [float(f[6]) for f in fleets],
        )
        for i, (did, e) in enumerate(zip(dest_ids, etas)):
            if did is not None and did in pid_xy:
                fleet_dest_xy[i] = pid_xy[did]
                fleet_eta[i] = e

    fleet_rows = [
        _fleet_features(int(f[1]), float(f[2]), float(f[3]), float(f[4]),
                        float(f[6]), player, my_planet_xyr,
                        fleet_dest_xy[i], fleet_eta[i])
        for i, f in enumerate(fleets)
    ]
    fleet_owner_slots = [_owner_slot(int(f[1]), player, n_players) for f in fleets]

    global_feats, _ = _global_features(obs, planets, fleets, player)

    # --- паддинг / стек ---
    def _pad(rows: List[List[float]], n: int, dim: int):
        arr = np.zeros((n, dim), dtype=np.float32)
        mask = np.zeros((n,), dtype=bool)
        for i, r in enumerate(rows[:n]):
            arr[i] = r
            mask[i] = True
        return arr, mask

    def _pad_idx(vals: List[int], n: int):
        arr = np.zeros((n,), dtype=np.int64)   # pad-слот 0 безвреден: токен закрыт pad_mask
        for i, v in enumerate(vals[:n]):
            arr[i] = v
        return arr

    p_arr, p_mask = _pad(planet_rows, cfg.max_planets, PLANET_FEAT_DIM)
    c_arr, c_mask = _pad(comet_rows, cfg.max_comets, COMET_FEAT_DIM)
    f_arr, f_mask = _pad(fleet_rows, cfg.max_fleets, FLEET_FEAT_DIM)
    p_slot = _pad_idx(planet_owner_slots, cfg.max_planets)
    c_slot = _pad_idx(comet_owner_slots, cfg.max_comets)
    f_slot = _pad_idx(fleet_owner_slots, cfg.max_fleets)

    # вектор places выровнен с H_place = [планеты(max_planets), кометы(max_comets)]
    places: List[Optional[PlaceInfo]] = [None] * (cfg.max_planets + cfg.max_comets)
    for i, info in enumerate(planet_places[:cfg.max_planets]):
        places[i] = info
    for i, info in enumerate(comet_places[:cfg.max_comets]):
        places[cfg.max_planets + i] = info
    owned_idx = [i for i, pl in enumerate(places)
                 if pl is not None and pl.is_mine and pl.ships > 0]

    def _t(a):
        return torch.from_numpy(a).unsqueeze(0).to(device) if device else \
            torch.from_numpy(a).unsqueeze(0)

    return EncodedObs(
        planet_feats=_t(p_arr), planet_mask=_t(p_mask),
        comet_feats=_t(c_arr), comet_mask=_t(c_mask),
        fleet_feats=_t(f_arr), fleet_mask=_t(f_mask),
        planet_owner_slot=_t(p_slot), comet_owner_slot=_t(c_slot),
        fleet_owner_slot=_t(f_slot),
        global_feats=torch.from_numpy(np.asarray(global_feats, dtype=np.float32))
        .unsqueeze(0).to(device) if device else
        torch.from_numpy(np.asarray(global_feats, dtype=np.float32)).unsqueeze(0),
        places=places, owned_idx=owned_idx, player=player, cfg=cfg,
    )

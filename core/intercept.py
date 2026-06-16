"""Утилита упреждающего угла (lead-angle) / ETA для Orbit Wars.

Самостоятельная тула (только numpy + stdlib, без torch), отвечающая на один вопрос:
*чтобы попасть по (возможно движущейся) цели, в каком направлении и с каким ETA нужно
запустить флот из ``ships`` кораблей из точки ``src``?*

Почему это нетривиально:
  * орбитальные планеты вращаются вокруг солнца, а кометы летят по фиксированному пути,
    поэтому целиться нужно в **будущую** позицию цели, а не в текущую;
  * скорость флота зависит от числа отправленных кораблей (логарифмическая кривая),
    поэтому время полёта и угол упреждения связаны.

Цели описываются маленьким неизменяемым :class:`Target`. Три вида:
  * ``static``  — позиция постоянна;
  * ``orbit``   — вращается вокруг ``center`` со скоростью ``angular_velocity`` рад/ход;
  * ``comet``   — позиция читается из заранее посчитанного ``path`` по индексу.

ЗАМЕЧАНИЕ о точности движка: формула скорости флота и обновление вращения ниже следуют
опубликованным правилам. Сверить точную базу логарифма / знак вращения с
``kaggle_environments.envs.orbit_wars`` после установки пакета; сигнатуры вызовов при
этом не изменятся.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# --- константы игры (дефолты; сверить с движком) ------------------------------
CENTER: Tuple[float, float] = (50.0, 50.0)
MAX_SPEED: float = 6.0          # конфиг `shipSpeed`
SPEED_REF_SHIPS: float = 1000.0  # число кораблей, при котором достигается (почти) макс. скорость


def fleet_speed(ships: float, max_speed: float = MAX_SPEED) -> float:
    """Скорость флота как функция размера.

    ``speed = 1 + (max_speed - 1) * (log(ships) / log(1000)) ** 1.5``

    1 корабль -> 1.0 ед/ход, растёт к ``max_speed`` около 1000 кораблей.
    Логарифмическое отношение не зависит от базы; долю клампим в ``[0, 1]``, чтобы флоты
    больше референса никогда не превышали ``max_speed``.
    """
    s = max(1.0, float(ships))
    frac = math.log(s) / math.log(SPEED_REF_SHIPS)
    frac = min(1.0, max(0.0, frac)) ** 1.5
    return 1.0 + (max_speed - 1.0) * frac


@dataclass(frozen=True)
class Target:
    """То, по чему может целиться флот.

    pos:               текущие (x, y).
    kind:              'static' | 'orbit' | 'comet'.
    center:            центр вращения для 'orbit'.
    angular_velocity:  знаковая скорость рад/ход для 'orbit' (знак = направление вращения).
    path:              траектория [L, 2] для 'comet'.
    path_index:        текущий индекс в ``path`` для 'comet'.
    radius:            радиус цели (используется только для hit-теста на стороне вызова).
    """

    pos: Tuple[float, float]
    kind: str = "static"
    center: Tuple[float, float] = CENTER
    angular_velocity: float = 0.0
    path: Optional[np.ndarray] = None
    path_index: int = 0
    radius: float = 1.0


def predict_position(target: Target, t: float) -> np.ndarray:
    """Позиция цели через ``t`` ходов в будущем (t может быть дробным)."""
    p0 = np.asarray(target.pos, dtype=np.float64)
    if target.kind == "orbit":
        c = np.asarray(target.center, dtype=np.float64)
        theta = target.angular_velocity * t
        ct, st = math.cos(theta), math.sin(theta)
        r = p0 - c
        rot = np.array([ct * r[0] - st * r[1], st * r[0] + ct * r[1]])
        return c + rot
    if target.kind == "comet" and target.path is not None and len(target.path):
        # линейная интерполяция между точками пути, чтобы позиция была непрерывна по t
        # (движок продвигает цель на ~cometSpeed ед/ход вдоль пути).
        f = target.path_index + t
        lo = int(math.floor(f))
        n = len(target.path)
        if lo < 0:
            return np.asarray(target.path[0], dtype=np.float64)
        if lo >= n - 1:
            return np.asarray(target.path[n - 1], dtype=np.float64)
        frac = f - lo
        a = np.asarray(target.path[lo], dtype=np.float64)
        b = np.asarray(target.path[lo + 1], dtype=np.float64)
        return a + frac * (b - a)
    return p0  # static (или вырожденный случай)


def _target_velocity(target: Target) -> np.ndarray:
    """Мгновенная скорость цели при t=0 (для решения с постоянной скоростью)."""
    if target.kind == "orbit":
        c = np.asarray(target.center, dtype=np.float64)
        r = np.asarray(target.pos, dtype=np.float64) - c
        # d/dt [R(w t) r] |_{t=0} = w * (-r_y, r_x)
        return target.angular_velocity * np.array([-r[1], r[0]])
    if target.kind == "comet" and target.path is not None and len(target.path) > 1:
        i = min(max(target.path_index, 0), len(target.path) - 2)
        return np.asarray(target.path[i + 1], dtype=np.float64) - np.asarray(
            target.path[i], dtype=np.float64
        )
    return np.zeros(2, dtype=np.float64)


def _smallest_positive_root(a: float, b: float, c: float) -> Optional[float]:
    """Наименьший положительный корень ``a t^2 + b t + c = 0`` (или None)."""
    if abs(a) < 1e-12:  # линейный случай: b t + c = 0
        if abs(b) < 1e-12:
            return None
        t = -c / b
        return t if t > 1e-9 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    roots = sorted(((-b - sq) / (2 * a), (-b + sq) / (2 * a)))
    for t in roots:
        if t > 1e-9:
            return t
    return None


def intercept_angle(
    src_xy: Tuple[float, float],
    target: Target,
    ships: float,
    *,
    max_speed: float = MAX_SPEED,
    max_t: float = 500.0,
    refine_iters: int = 3,
) -> Tuple[float, float, bool]:
    """Угол запуска (радианы), ETA (ходы) и найден ли перехват.

    Стратегия:
      1. скорость флота ``v`` из ``ships``;
      2. closed-form перехват с постоянной скоростью (квадратное уравнение по t) по
         мгновенной скорости цели — точно для статики, хороший seed для кривых траекторий;
      3. несколько шагов фикс-точки с истинной (кривой) ``predict_position``, чтобы учесть
         кривизну орбиты/кометы.

    Соглашение об угле как в движке: ``0 = +x (вправо)``, ``+pi/2 = +y (вниз)``.
    При неудаче (цель недостижима на этой скорости) целимся в текущую позицию цели и
    возвращаем ``hit=False``.
    """
    src = np.asarray(src_xy, dtype=np.float64)
    v = fleet_speed(ships, max_speed)

    p0 = np.asarray(target.pos, dtype=np.float64)
    d = p0 - src
    vt = _target_velocity(target)

    # (|vt|^2 - v^2) t^2 + 2 (d . vt) t + |d|^2 = 0
    a = float(vt @ vt - v * v)
    b = float(2.0 * (d @ vt))
    c = float(d @ d)
    t = _smallest_positive_root(a, b, c)

    if t is None or t > max_t:
        # недостижимо: целимся в текущую позицию, сообщаем промах.
        angle = math.atan2(d[1], d[0])
        eta = float(np.linalg.norm(d) / v) if v > 0 else float("inf")
        return angle, eta, False

    # уточняем по истинной траектории (фикс-точка: t = dist(P(t), src) / v)
    for _ in range(max(0, refine_iters)):
        aim = predict_position(target, t)
        nt = float(np.linalg.norm(aim - src) / v)
        if not math.isfinite(nt) or nt > max_t:
            break
        if abs(nt - t) < 1e-4:
            t = nt
            break
        t = nt

    aim = predict_position(target, t)
    da = aim - src
    angle = math.atan2(da[1], da[0])
    return angle, float(t), True

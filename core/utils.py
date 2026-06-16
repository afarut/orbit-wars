from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Type

import numpy as np
import torch.nn as nn

from .features import ROTATION_RADIUS_LIMIT, _build_target, _comet_path_map
from .intercept import (
    MAX_SPEED,
    fleet_speed,
    intercept_angle,
)

# --- константы игры (зеркало конфига движка; сверить с kaggle_environments) -----
SUN_CENTER: Tuple[float, float] = (50.0, 50.0)
SUN_RADIUS: float = 10.0
BOARD: float = 100.0


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    *,
    activation: Type[nn.Module] = nn.GELU,
    norm: bool = True,
    dropout: float = 0.0,
    out_norm: bool = False,
) -> nn.Sequential:
    """Собрать feed-forward MLP ``in_dim -> *hidden_dims -> out_dim``.

    Скрытые слои: ``Linear -> [LayerNorm] -> активация -> [Dropout]``.
    Финальный слой — голый ``Linear`` (опционально с ``LayerNorm`` после, если задан
    ``out_norm`` — например, для токен-эмбеддингов, которые идут в attention).

    Один хелпер держит все блоки сети единообразными (одинаковые активация, нормировка
    и dropout) и позволяет каждому типу сущности проецироваться из своей сырой размерности
    фич в общий ``d_model``, который использует трансформер.
    """
    dims = [in_dim, *hidden_dims, out_dim]
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        last = i == len(dims) - 2
        layers.append(nn.Linear(a, b))
        if not last:
            if norm:
                layers.append(nn.LayerNorm(b))
            layers.append(activation())
            if dropout:
                layers.append(nn.Dropout(dropout))
        elif out_norm:
            layers.append(nn.LayerNorm(b))
    return nn.Sequential(*layers)


# === проверка траектории запуска флота ========================================
# Флот летит по прямой с постоянной скоростью fleet_speed(ships). Угол с упреждением
# на движущуюся цель даёт intercept.intercept_angle. Здесь проверяем, не стоят ли на
# прямом пути солнце, статичные/орбитальные планеты, кометы или граница поля, и при
# блокировке ищем безопасный запуск перебором числа кораблей.


@dataclass(frozen=True)
class LaunchPlan:
    """Результат проверки запуска флота из from_planet в to_planet."""

    angle: float                 # рад, направление запуска (0=+x, +pi/2=+y)
    eta: float                   # ходы до прибытия
    ships: int                   # число кораблей в залпе
    reaches: bool                # intercept нашёл перехват цели
    safe: bool                   # прямой путь свободен
    blocked_by: Optional[str]    # 'sun' | 'planet' | 'comet' | 'oob' | None
    blocker_idx: Optional[int]   # индекс мешающего тела в planets (если есть)
    block_t: Optional[float]     # ход первого столкновения


def _planet_xyr(row: Sequence[float]) -> Tuple[float, float, float]:
    """(x, y, radius) из строки planets [id, owner, x, y, radius, ships, production]."""
    return float(row[2]), float(row[3]), float(row[4])


def _oob_t(spawn: np.ndarray, direction: np.ndarray, v: float) -> float:
    """Ход выхода флота за границы поля [0, BOARD] (inf, если не выходит)."""
    best = float("inf")
    for axis in (0, 1):
        d = v * direction[axis]
        if abs(d) < 1e-12:
            continue
        for bound in (0.0, BOARD):
            t = (bound - spawn[axis]) / d
            if t > 1e-9:
                best = min(best, t)
    return best


def _in_bounds(p: np.ndarray) -> bool:
    return bool(0.0 <= p[0] <= BOARD and 0.0 <= p[1] <= BOARD)


# --- векторизованные версии тех же тестов (используются в validate_launch) -----
def _static_entry_t_batch(
    spawn: np.ndarray, direction: np.ndarray, v: float, horizon: float,
    centers: np.ndarray, radii: np.ndarray,
) -> np.ndarray:
    """Время входа луча ``spawn+v*t*dir`` в каждый круг ``(centers, radii)``; ``inf``, если нет.

    Квадратное уравнение луч-круг сразу по всем телам (берём меньший корень в ``(0, horizon]``;
    если спавн уже внутри круга — это время выхода, как у скалярного варианта).
    """
    n = centers.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    a = v * v
    if a < 1e-12:
        return np.full(n, np.inf)
    rel = spawn[None, :] - centers                      # [n, 2]
    b = 2.0 * v * (rel @ direction)                     # [n]
    c = np.einsum("ij,ij->i", rel, rel) - radii * radii  # [n]
    disc = b * b - 4.0 * a * c
    sq = np.sqrt(np.where(disc >= 0.0, disc, 0.0))
    tm = (-b - sq) / (2.0 * a)
    tp = (-b + sq) / (2.0 * a)
    # первый корень >= -1e-9 (как в скалярной версии: ко второму не переходим, если
    # первый уже за горизонтом)
    cand = np.where(tm >= -1e-9, tm, np.where(tp >= -1e-9, tp, np.inf))
    cand = np.where(cand <= horizon + 1e-9, np.maximum(cand, 0.0), np.inf)
    return np.where(disc >= 0.0, cand, np.inf)


def _interval_hit_t(
    P: np.ndarray, Q: np.ndarray, radius: float, t_a: float, t_b: float,
) -> float:
    """Наименьший ``t in [t_a, t_b]`` с ``‖P + t*Q‖ <= radius`` (``inf``, если нет).

    Для линейного относительного движения (корабль ↔ сегмент пути кометы).
    """
    a = float(Q @ Q)
    b = 2.0 * float(P @ Q)
    c = float(P @ P) - radius * radius
    if a < 1e-12:                       # относительно неподвижны
        return t_a if c <= 0.0 else np.inf
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return np.inf
    sq = math.sqrt(disc)
    r1 = (-b - sq) / (2.0 * a)
    r2 = (-b + sq) / (2.0 * a)          # ‖P+tQ‖<=radius на [r1, r2]
    lo = max(t_a, r1)
    hi = min(t_b, r2)
    return lo if lo <= hi else np.inf


def _comet_hit_t(
    spawn: np.ndarray, direction: np.ndarray, v: float, horizon: float,
    path: np.ndarray, path_index: float, radius: float,
) -> float:
    """Ход столкновения с кометой: точный минимум по сегментам пути (без сетки тиков).

    Позиция кометы линейна по ``t`` на каждом сегменте (движок продвигает её на сегмент за
    ход), корабль тоже линеен → ``‖ship-comet‖²`` квадратична, решаем точно по интервалам.
    """
    n = len(path)
    if n == 0:
        return np.inf
    if n == 1:                          # вырожденный путь -> точка
        return _interval_hit_t(spawn - path[0], v * direction, radius, 0.0, horizon)
    s0 = int(math.floor(path_index))
    for s in range(max(s0, 0), n - 1):
        t_a = max(0.0, s - path_index)
        t_b = min(horizon, (s + 1) - path_index)
        if t_a >= t_b:
            continue
        A = path[s]
        seg = path[s + 1] - A
        # comet(t) = A + (path_index + t - s)*seg ; rel = ship - comet = P + t*Q
        P = spawn - A - (path_index - s) * seg
        Q = v * direction - seg
        t_hit = _interval_hit_t(P, Q, radius, t_a, t_b)
        if np.isfinite(t_hit):
            return t_hit                # сегменты по возрастанию t -> первый и есть ранний
    # хвост: комета стоит в последней точке пути
    t_a = max(0.0, (n - 1) - path_index)
    if t_a < horizon:
        return _interval_hit_t(spawn - path[n - 1], v * direction, radius, t_a, horizon)
    return np.inf


def _orbit_hit_t_batch(
    spawn: np.ndarray, direction: np.ndarray, v: float, horizon: float,
    p0s: np.ndarray, radii: np.ndarray, center: np.ndarray, av: float,
) -> np.ndarray:
    """Время столкновения с каждым орбитальным телом — векторно по всем телам (same-time).

    Общая сетка по ``[0, horizon]`` локализует впадину зазора ``‖ship(t)-body(t)‖`` каждого
    тела (корабль и тело — в ОДИН момент ``t``). Затем векторный тернарный поиск уточняет
    минимум (есть ли столкновение), а векторная бисекция — самый ранний момент входа в диск
    тела. Точность даёт уточнение, поэтому шаг сетки можно брать грубым.
    """
    No = p0s.shape[0]
    if No == 0:
        return np.empty(0, dtype=np.float64)
    if v * v < 1e-12 or horizon <= 0.0:
        return np.full(No, np.inf)
    rvec = p0s - center                                  # [No, 2]
    cx, cy = float(center[0]), float(center[1])
    dx, dy = float(direction[0]), float(direction[1])
    rad = radii                                          # [No]

    def gap_at(tv: np.ndarray) -> np.ndarray:            # tv:[No] -> зазор[No] (свой t у тела)
        ct, st = np.cos(av * tv), np.sin(av * tv)
        bx = cx + ct * rvec[:, 0] - st * rvec[:, 1]
        by = cy + st * rvec[:, 0] + ct * rvec[:, 1]
        return np.hypot(spawn[0] + v * tv * dx - bx, spawn[1] + v * tv * dy - by)

    # общая сетка [No, T] (шаг ~0.1: только локализует впадины, точность ниже)
    T = max(3, int(math.ceil(horizon / 0.1)) + 1)
    ts = np.linspace(0.0, horizon, T)                    # [T]
    ct, st = np.cos(av * ts), np.sin(av * ts)            # [T]
    bx = cx + ct[None, :] * rvec[:, 0, None] - st[None, :] * rvec[:, 1, None]   # [No, T]
    by = cy + st[None, :] * rvec[:, 0, None] + ct[None, :] * rvec[:, 1, None]
    sx = spawn[0] + v * ts * dx                          # [T]
    sy = spawn[1] + v * ts * dy
    g = np.hypot(sx[None, :] - bx, sy[None, :] - by)     # [No, T]

    # --- существование: уточнить глобальный минимум тернарным поиском (векторно) ---
    imin = np.argmin(g, axis=1)                          # [No]
    a_t = ts[np.maximum(imin - 1, 0)]
    b_t = ts[np.minimum(imin + 1, T - 1)]
    for _ in range(34):
        m1 = a_t + (b_t - a_t) / 3.0
        m2 = b_t - (b_t - a_t) / 3.0
        keep_right = gap_at(m1) < gap_at(m2)
        b_t = np.where(keep_right, m2, b_t)
        a_t = np.where(keep_right, a_t, m1)
    t_min = 0.5 * (a_t + b_t)
    collide = gap_at(t_min) <= rad                       # [No]

    # --- самый ранний вход: первое пересечение radius на сетке -> бисекция (векторно) ---
    below = g <= rad[:, None]                            # [No, T]
    has = below.any(axis=1)
    first = np.argmax(below, axis=1)                     # [No] (0, если ниже нигде)
    lo = ts[np.maximum(first - 1, 0)]                    # gap(lo) > rad при has & first>0
    hi = ts[first]                                       # gap(hi) <= rad при has
    subgrid = collide & ~has                             # впадина уже шага сетки -> [0, t_min]
    lo = np.where(subgrid, 0.0, lo)
    hi = np.where(subgrid, t_min, hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cond = gap_at(mid) <= rad
        hi = np.where(cond, mid, hi)
        lo = np.where(cond, lo, mid)
    return np.where(collide, hi, np.inf)


def validate_launch(
    planets: Sequence[Sequence[float]],
    from_idx: int,
    to_idx: int,
    ships: int,
    *,
    angular_velocity: float = 0.0,
    comets: Optional[Sequence] = None,
    max_speed: float = MAX_SPEED,
    spawn_gap: float = 1e-3,
) -> LaunchPlan:
    """Проверить прямую траекторию запуска флота из ``from_idx`` в ``to_idx``.

    ``planets`` — сырой массив obs (строки ``[id, owner, x, y, radius, ships, production]``),
    ``from_idx``/``to_idx`` — индексы в нём. Для движущихся тел нужны ``angular_velocity``
    (орбиты) и ``comets`` (поле obs); без них тела считаются статичными. Возвращает угол,
    ETA и первую помеху на пути (солнце/планета/комета/край поля).

    Векторизованная версия: тела классифицируются пакетно (без цикла + ``_build_target``),
    статичные круги проверяются разом (``_static_entry_t_batch``), кометы — closed-form по
    сегментам пути (``_comet_hit_t``), орбиты — честной same-time проверкой векторно по всем
    телам (``_orbit_hit_t_batch``). Возвращает первую помеху по времени.
    """
    comet_map = _comet_path_map({"comets": comets or []})
    ships = int(ships)

    sx, sy, sr = _planet_xyr(planets[from_idx])
    tgt = planets[to_idx]
    tgt_target = _build_target(int(tgt[0]), int(tgt[1]), float(tgt[2]), float(tgt[3]),
                               float(tgt[4]), angular_velocity, comet_map)[0]

    v = fleet_speed(ships, max_speed)
    angle, eta, reaches = intercept_angle((sx, sy), tgt_target, ships, max_speed=max_speed)

    direction = np.array([math.cos(angle), math.sin(angle)])
    spawn = np.array([sx, sy]) + (sr + spawn_gap) * direction
    horizon = eta if reaches else min(_oob_t(spawn, direction, v), 500.0)
    sun_c = np.asarray(SUN_CENTER, dtype=np.float64)

    # --- классификация всех тел без питоновского цикла (зеркало _build_target) ---
    arr = np.asarray(planets, dtype=np.float64)
    n = arr.shape[0]
    idx_all = np.arange(n)
    keep = (idx_all != from_idx) & (idx_all != to_idx)
    ids = arr[:, 0].astype(np.int64)
    xs, ys, rs = arr[:, 2], arr[:, 3], arr[:, 4]
    is_comet = np.array([int(i) in comet_map for i in ids], dtype=bool)
    orbital_r = np.hypot(xs - SUN_CENTER[0], ys - SUN_CENTER[1])
    is_orbit = (~is_comet) & ((orbital_r + rs) < ROTATION_RADIUS_LIMIT) & (angular_velocity != 0.0)
    is_static = (~is_comet) & (~is_orbit)            # орбита при av==0 -> статична

    static_sel = keep & is_static
    orbit_idx = idx_all[keep & is_orbit].tolist()
    comet_idx = idx_all[keep & is_comet].tolist()

    # солнце + статичные планеты — пакетный тест луч-круг
    t_sun = _static_entry_t_batch(spawn, direction, v, horizon,
                                  sun_c[None, :], np.array([SUN_RADIUS]))[0]
    static_centers = np.stack([xs[static_sel], ys[static_sel]], axis=1) if static_sel.any() \
        else np.empty((0, 2))
    t_static = _static_entry_t_batch(spawn, direction, v, horizon, static_centers, rs[static_sel])

    # движущиеся тела
    if orbit_idx:
        p0s = np.stack([xs[orbit_idx], ys[orbit_idx]], axis=1)
        t_orbit = _orbit_hit_t_batch(spawn, direction, v, horizon, p0s,
                                     rs[orbit_idx], sun_c, angular_velocity)
    else:
        t_orbit = np.empty(0)
    t_comet = np.array(
        [_comet_hit_t(spawn, direction, v, horizon, comet_map[int(ids[i])][0],
                      float(comet_map[int(ids[i])][1]), float(rs[i])) for i in comet_idx],
        dtype=np.float64) if comet_idx else np.empty(0)

    all_t = np.concatenate([np.array([t_sun], dtype=np.float64), t_static, t_orbit, t_comet])
    all_kind = (["sun"] + ["planet"] * int(static_sel.sum())
                + ["planet"] * len(orbit_idx) + ["comet"] * len(comet_idx))
    all_idx = ([None] + idx_all[static_sel].tolist() + orbit_idx + comet_idx)

    j = int(np.argmin(all_t))
    if np.isfinite(all_t[j]):
        best_t: Optional[float] = float(all_t[j])
        blocked_by: Optional[str] = all_kind[j]
        blocker_idx: Optional[int] = all_idx[j]
    else:
        best_t, blocked_by, blocker_idx = None, None, None

    # вылет за поле: при неперехвате флот улетает; на всякий случай и для краевых целей
    end = spawn + v * horizon * direction
    if (not reaches) or (not _in_bounds(end)):
        t_oob = _oob_t(spawn, direction, v)
        if best_t is None or t_oob < best_t:
            best_t, blocked_by, blocker_idx = float(t_oob), "oob", None

    return LaunchPlan(
        angle=float(angle), eta=float(eta), ships=ships, reaches=bool(reaches),
        safe=best_t is None, blocked_by=blocked_by, blocker_idx=blocker_idx,
        block_t=best_t,
    )


def planet_at_angle(
    planets: Sequence[Sequence[float]],
    from_idx: int,
    angle: float,
    ships: int,
    *,
    angular_velocity: float = 0.0,
    comets: Optional[Sequence] = None,
    max_speed: float = MAX_SPEED,
    spawn_gap: float = 1e-3,
    max_horizon: float = 500.0,
) -> Optional[int]:
    """Индекс первой планеты на пути флота из ``from_idx`` под углом ``angle``.

    Инверсия :func:`validate_launch`: вместо ``to_idx`` задаётся ``angle`` (рад, 0=+x,
    +pi/2=+y), на выходе — индекс тела в ``planets``, в которое флот врежется первым.
    ``None``, если до выхода за поле помех нет либо первой помехой оказалось солнце
    (флот гибнет, ни одной планеты не достигает). Кометы/орбиты считаются движущимися
    телами, поэтому нужны ``angular_velocity`` и ``comets`` (без них тела статичны).
    """
    comet_map = _comet_path_map({"comets": comets or []})
    ships = int(ships)

    sx, sy, sr = _planet_xyr(planets[from_idx])
    v = fleet_speed(ships, max_speed)
    direction = np.array([math.cos(angle), math.sin(angle)])
    spawn = np.array([sx, sy]) + (sr + spawn_gap) * direction
    horizon = min(_oob_t(spawn, direction, v), max_horizon)
    sun_c = np.asarray(SUN_CENTER, dtype=np.float64)

    # --- классификация всех тел без питоновского цикла (зеркало validate_launch) ---
    arr = np.asarray(planets, dtype=np.float64)
    n = arr.shape[0]
    idx_all = np.arange(n)
    keep = idx_all != from_idx                       # цели нет — кандидаты все, кроме источника
    ids = arr[:, 0].astype(np.int64)
    xs, ys, rs = arr[:, 2], arr[:, 3], arr[:, 4]
    is_comet = np.array([int(i) in comet_map for i in ids], dtype=bool)
    orbital_r = np.hypot(xs - SUN_CENTER[0], ys - SUN_CENTER[1])
    is_orbit = (~is_comet) & ((orbital_r + rs) < ROTATION_RADIUS_LIMIT) & (angular_velocity != 0.0)
    is_static = (~is_comet) & (~is_orbit)            # орбита при av==0 -> статична

    static_sel = keep & is_static
    orbit_idx = idx_all[keep & is_orbit].tolist()
    comet_idx = idx_all[keep & is_comet].tolist()

    # солнце + статичные планеты — пакетный тест луч-круг
    t_sun = _static_entry_t_batch(spawn, direction, v, horizon,
                                  sun_c[None, :], np.array([SUN_RADIUS]))[0]
    static_centers = np.stack([xs[static_sel], ys[static_sel]], axis=1) if static_sel.any() \
        else np.empty((0, 2))
    t_static = _static_entry_t_batch(spawn, direction, v, horizon, static_centers, rs[static_sel])

    # движущиеся тела
    if orbit_idx:
        p0s = np.stack([xs[orbit_idx], ys[orbit_idx]], axis=1)
        t_orbit = _orbit_hit_t_batch(spawn, direction, v, horizon, p0s,
                                     rs[orbit_idx], sun_c, angular_velocity)
    else:
        t_orbit = np.empty(0)
    t_comet = np.array(
        [_comet_hit_t(spawn, direction, v, horizon, comet_map[int(ids[i])][0],
                      float(comet_map[int(ids[i])][1]), float(rs[i])) for i in comet_idx],
        dtype=np.float64) if comet_idx else np.empty(0)

    all_t = np.concatenate([np.array([t_sun], dtype=np.float64), t_static, t_orbit, t_comet])
    all_idx = ([None] + idx_all[static_sel].tolist() + orbit_idx + comet_idx)

    j = int(np.argmin(all_t))
    if not np.isfinite(all_t[j]):
        return None              # на пути до выхода за поле ничего нет
    return all_idx[j]            # None, если первой помехой оказалось солнце


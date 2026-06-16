"""Smoke-тесты архитектуры Orbit Wars (kaggle_environments не нужен).

Запуск: .venv/bin/python smoke_test.py
Покрывает раздел verification из плана:
  1. forward: формы, softmax по `to`, маски, value, set-инвариантность;
  2. intercept: статика closed-form, симулированное попадание по движущейся цели, кривая скорости;
  3. act(): валидные ходы [from_planet_id, angle, num_ships].
"""

import math
import random

import numpy as np
import torch

from core import (
    FeatureConfig, encode, fleet_speed, intercept_angle, predict_position,
)
from core.intercept import Target
from model import ModelConfig, PolicyValueNet


def radius_of(prod):
    return 1.0 + math.log(prod)


def make_obs(seed=0):
    """Небольшое, но репрезентативное наблюдение (вид игрока 0)."""
    rng = random.Random(seed)
    planets = [
        # id, owner, x, y, radius, ships, production
        [0, 0, 40.0, 50.0, radius_of(3), 10.0, 3],   # моя база, орбитальная (dist 10)
        [1, 1, 60.0, 50.0, radius_of(3), 10.0, 3],   # база врага, орбитальная
        [2, -1, 10.0, 10.0, radius_of(2), 30.0, 2],  # нейтрал, статичная (далеко)
        [3, -1, 90.0, 90.0, radius_of(2), 25.0, 2],  # нейтрал, статичная
        [4, -1, 50.0, 35.0, radius_of(4), 12.0, 4],  # нейтрал, орбитальная (dist 15)
        [5, -1, 20.0, 80.0, 1.0, 8.0, 1],            # комета (см. ниже)
    ]
    # путь кометы: короткая почти прямая траектория
    path = [[20.0 + 1.5 * k, 80.0 - 1.2 * k] for k in range(40)]
    comets = [{"planet_ids": [5], "paths": [path], "path_index": 0}]
    fleets = [
        # id, owner, x, y, angle, from_planet_id, ships  (вражеский флот к моей базе)
        [100, 1, 55.0, 50.0, math.atan2(0.0, -1.0), 1, 20.0],
    ]
    return {
        "player": 0,
        "planets": planets,
        "fleets": fleets,
        "angular_velocity": 0.04,
        "initial_planets": [p[:] for p in planets],
        "comets": comets,
        "comet_planet_ids": [5],
        "remainingOverageTime": 60.0,
        "step": 120,
    }


def finite_sorted(t):
    v = t.flatten()
    v = v[torch.isfinite(v)]
    return torch.sort(v).values


def test_forward():
    torch.manual_seed(0)
    cfg = FeatureConfig(max_planets=40, max_comets=16, max_fleets=256)
    net = PolicyValueNet(ModelConfig()).eval()

    enc = encode(make_obs(), cfg=cfg)
    out = net(enc)
    logits, pi, value = out["logits"], out["pi"], out["value"]

    M = cfg.max_planets + cfg.max_comets
    assert logits.shape == (1, M, M + 1), logits.shape
    assert pi.shape == (1, M, M + 1), pi.shape
    assert value.shape == (1,), value.shape

    # softmax по оси `to` суммируется в 1
    assert torch.allclose(pi.sum(-1), torch.ones(1, M), atol=1e-5), "pi не нормирован"

    # диагональ (self-target) замаскирована
    diag = pi[0, torch.arange(M), torch.arange(M)]
    assert torch.all(diag == 0), "диагональ не замаскирована"

    # паддинг-колонки целей получают нулевую вероятность
    place_mask = torch.cat([enc.planet_mask, enc.comet_mask], dim=1)[0]  # [M]
    pad_cols = ~place_mask
    assert torch.all(pi[0][:, :M][:, pad_cols] == 0), "паддинг-цели не замаскированы"

    assert torch.isfinite(value).all(), "value не конечный"
    print("  forward формы/маски/softmax: OK   value =", round(value.item(), 4))

    # --- set-инвариантность: перемешать порядок планет, мультимножество логитов не меняется ---
    obs2 = make_obs()
    perm = obs2["planets"][:]
    random.Random(1).shuffle(perm)
    obs2["planets"] = perm
    out2 = net(encode(obs2, cfg=cfg))
    a, b = finite_sorted(logits), finite_sorted(out2["logits"])
    assert a.shape == b.shape, (a.shape, b.shape)
    assert torch.allclose(a, b, atol=1e-3), (a - b).abs().max().item()
    print("  set-инвариантность при перестановке планет: OK")


def test_intercept():
    # кривая скорости
    assert abs(fleet_speed(1) - 1.0) < 1e-9
    speeds = [fleet_speed(n) for n in (1, 10, 100, 500, 1000)]
    assert all(x < y for x, y in zip(speeds, speeds[1:])), speeds
    assert speeds[-1] <= 6.0 + 1e-9 and speeds[-1] > 5.5, speeds
    print("  fleet_speed монотонна 1->6:", [round(s, 3) for s in speeds])

    # статичная цель: closed form
    src = (50.0, 50.0)
    tgt = Target(pos=(70.0, 50.0), kind="static", radius=1.0)
    angle, eta, hit = intercept_angle(src, tgt, ships=50)
    assert hit
    assert abs(angle - 0.0) < 1e-6, angle
    assert abs(eta - 20.0 / fleet_speed(50)) < 1e-4, eta
    print("  статичный перехват: OK   angle=%.4f eta=%.3f" % (angle, eta))

    # движущиеся цели (орбита + комета): симулируем полёт и проверяем промах
    def simulate_miss(src, tgt, ships):
        angle, eta, hit = intercept_angle(src, tgt, ships)
        assert hit, "перехват не найден"
        v = fleet_speed(ships)
        fx = src[0] + v * eta * math.cos(angle)
        fy = src[1] + v * eta * math.sin(angle)
        tx, ty = predict_position(tgt, eta)
        return math.hypot(fx - tx, fy - ty), eta

    orbit = Target(pos=(60.0, 50.0), kind="orbit", center=(50.0, 50.0),
                   angular_velocity=0.05, radius=1.5)
    miss_o, eta_o = simulate_miss((20.0, 20.0), orbit, ships=120)
    assert miss_o < 0.5, miss_o
    print("  орбитальный перехват: OK    miss=%.4f eta=%.2f" % (miss_o, eta_o))

    path = [[20.0 + 1.5 * k, 80.0 - 1.2 * k] for k in range(60)]
    comet = Target(pos=tuple(path[0]), kind="comet", path=np.array(path),
                   path_index=0, radius=1.0)
    miss_c, eta_c = simulate_miss((80.0, 20.0), comet, ships=80)
    assert miss_c < 0.5, miss_c   # интерполированный путь кометы -> точный перехват
    print("  перехват кометы: OK    miss=%.4f eta=%.2f" % (miss_c, eta_c))


def test_act():
    torch.manual_seed(0)
    net = PolicyValueNet(ModelConfig()).eval()
    obs = make_obs()
    moves = net.act(obs)

    my_ids = {int(p[0]) for p in obs["planets"] if int(p[1]) == obs["player"]}
    garrison = {int(p[0]): float(p[5]) for p in obs["planets"]}
    seen_sources = set()
    for from_id, angle, num_ships in moves:
        assert from_id in my_ids, f"запуск с не своей планеты {from_id}"
        assert isinstance(num_ships, int) and num_ships >= 1, num_ships
        assert num_ships <= garrison[from_id], "отправлено больше гарнизона"
        assert -math.pi - 1e-6 <= angle <= math.pi + 1e-6, angle
        assert from_id not in seen_sources, "больше одного хода на источник"
        seen_sources.add(from_id)
    print(f"  act(): OK   {len(moves)} валидных ход(ов) (необученная):", moves)

    # форсируем запуск (подавляем опцию hold), чтобы прогнать путь через intercept
    with torch.no_grad():
        net.mlp_hold[-1].bias.fill_(-1e9)
    forced = net.act(obs)
    n_owned = sum(1 for p in obs["planets"]
                  if int(p[1]) == obs["player"] and float(p[5]) > 0)
    assert len(forced) == n_owned, (len(forced), n_owned)
    for from_id, angle, num_ships in forced:
        assert from_id in my_ids and 1 <= num_ships <= garrison[from_id]
        assert math.isfinite(angle)
    print(f"  act(форс. запуск): OK   {len(forced)} ход(ов):", forced)


def test_validate_launch():
    """Векторизованный validate_launch: сверка с истинным same-time brute-force оракулом."""
    import time

    from core.utils import (
        SUN_CENTER, SUN_RADIUS, _in_bounds, _oob_t, validate_launch,
    )
    from core.features import _build_target, _comet_path_map
    from core import intercept

    AV = 0.03

    def random_board(rng, with_comet):
        n = rng.randint(4, 9)
        planets = []
        for i in range(n):
            prod = rng.randint(1, 5)
            planets.append([i, rng.choice([-1, 0, 1]), rng.uniform(5, 95),
                            rng.uniform(5, 95), radius_of(prod),
                            float(rng.randint(1, 50)), prod])
        comets = None
        if with_comet:
            cid = rng.randrange(n)
            px, py = planets[cid][2], planets[cid][3]
            dx, dy = rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)
            path = [[px + dx * k, py + dy * k] for k in range(30)]
            planets[cid][4] = 1.0
            comets = [{"planet_ids": [planets[cid][0]], "paths": [path],
                       "path_index": rng.randint(0, 3)}]
        a, b = rng.sample(range(n), 2)
        return planets, a, b, rng.randint(1, 800), comets

    def oracle(planets, from_idx, to_idx, ships, comets, dt=0.02):
        """Истина: мелкая сетка по t, точная проверка ‖ship(t)-body(t)‖<=radius."""
        comet_map = _comet_path_map({"comets": comets or []})
        sx, sy, sr = planets[from_idx][2], planets[from_idx][3], planets[from_idx][4]
        tgt = planets[to_idx]
        tgt_target = _build_target(int(tgt[0]), int(tgt[1]), float(tgt[2]), float(tgt[3]),
                                   float(tgt[4]), AV, comet_map)[0]
        v = intercept.fleet_speed(ships)
        angle, eta, reaches = intercept.intercept_angle((sx, sy), tgt_target, ships)
        direction = np.array([math.cos(angle), math.sin(angle)])
        spawn = np.array([sx, sy]) + (sr + 1e-3) * direction
        horizon = eta if reaches else min(_oob_t(spawn, direction, v), 500.0)
        bodies = []
        for i, row in enumerate(planets):
            if i in (from_idx, to_idx):
                continue
            tar, kind = _build_target(int(row[0]), int(row[1]), float(row[2]), float(row[3]),
                                      float(row[4]), AV, comet_map)
            bodies.append((i, "comet" if kind == "comet" else "planet", tar, float(row[4])))
        best_t, best_kind, best_idx = math.inf, None, None
        for s in range(1, int(horizon / dt) + 2):
            t = min(s * dt, horizon)
            ship = spawn + v * t * direction
            if np.linalg.norm(ship - np.asarray(SUN_CENTER)) <= SUN_RADIUS:
                best_t, best_kind, best_idx = t, "sun", None
            for (i, lbl, tar, rad) in bodies:
                if np.linalg.norm(ship - intercept.predict_position(tar, t)) <= rad and best_kind != "sun":
                    best_t, best_kind, best_idx = t, lbl, i
            if best_kind is not None:
                break
        end = spawn + v * horizon * direction
        if (not reaches) or (not _in_bounds(end)):
            t_oob = _oob_t(spawn, direction, v)
            if t_oob < best_t:
                best_t, best_kind, best_idx = t_oob, "oob", None
        return (best_kind is None), best_kind, best_idx, best_t, spawn, direction, v, horizon

    def far_board(rng):
        # все планеты далеко от солнца (orbital_radius+radius >= 50) -> только статичный путь;
        # так часть A целенаправленно проверяет статику/солнце/oob против истины.
        n = rng.randint(4, 8)
        planets = []
        while len(planets) < n:
            x, y = rng.uniform(0, 100), rng.uniform(0, 100)
            prod = rng.randint(1, 5)
            rad = radius_of(prod)
            if math.hypot(x - 50, y - 50) + rad < 50.5:   # гарантированно статичная (с запасом)
                continue
            planets.append([len(planets), rng.choice([-1, 0, 1]), x, y,
                            rad, float(rng.randint(1, 50)), prod])
        a, b = rng.sample(range(n), 2)
        return planets, a, b, rng.randint(1, 800)

    def compare(planets, a, b, ships, comets):
        """Сверить новую версию с оракулом; вернуть 1, если погранично разошёлся safe."""
        new = validate_launch(planets, a, b, ships, angular_velocity=AV, comets=comets)
        o_safe, o_kind, o_idx, o_t, spawn, _d, _v, _h = oracle(planets, a, b, ships, comets)
        if new.safe != o_safe:
            return 1                        # касательная грань на сетке оракула (мера ноль)
        if not new.safe:
            assert new.blocked_by == o_kind, (new.blocked_by, o_kind, planets, a, b, ships, comets)
            assert new.blocker_idx == o_idx, (new.blocker_idx, o_idx, planets, a, b, ships, comets)
            # «спавн внутри круга» (солнце/статик-планета): block_t = время выхода (соглашение
            # луч-круг), а оракул даёт ~0 -> сверять block_t там не нужно.
            inside = False
            if new.blocked_by == "sun":
                inside = float(np.linalg.norm(spawn - np.asarray(SUN_CENTER))) <= SUN_RADIUS
            elif new.blocked_by == "planet":
                r = planets[new.blocker_idx]
                inside = float(np.linalg.norm(spawn - np.array([r[2], r[3]]))) <= r[4]
            if not inside:
                assert abs(new.block_t - o_t) < 0.1, (new.blocked_by, new.block_t, o_t,
                                                      planets, a, b, ships, comets)
        return 0

    # --- A: статика/солнце/oob (дальние планеты) vs истинный оракул ---
    rng = random.Random(1)
    mism = sum(compare(*far_board(rng), None) for _ in range(200))
    assert mism <= 2, f"расхождений safe (A): {mism}/200"
    print(f"  A) статика/солнце/oob == оракул: OK (200 досок, погран.: {mism})")

    # --- B: орбиты + кометы vs истинный same-time оракул ---
    rng = random.Random(7)
    mism = 0
    for _ in range(200):
        planets, a, b, ships, comets = random_board(rng, with_comet=(rng.random() < 0.6))
        mism += compare(planets, a, b, ships, comets)
    assert mism <= 3, f"расхождений safe (B): {mism}/200"
    print(f"  B) орбиты+кометы == same-time оракул: OK (200 досок, погран.: {mism})")

    # --- C: пропускная способность на крупной доске (40 тел) ---
    rng = random.Random(3)
    planets = [[i, rng.choice([-1, 0, 1]), rng.uniform(5, 95), rng.uniform(5, 95),
                radius_of(rng.randint(1, 5)), float(rng.randint(1, 50)), rng.randint(1, 5)]
               for i in range(40)]
    cid = 7
    planets[cid][4] = 1.0
    px, py = planets[cid][2], planets[cid][3]
    comets = [{"planet_ids": [cid], "paths": [[[px + k, py + 0.5 * k] for k in range(60)]],
               "path_index": 0}]
    calls = [(rng.randrange(40), rng.randrange(40), rng.randint(1, 800)) for _ in range(300)]
    calls = [(a, b, s) for (a, b, s) in calls if a != b]

    def _bench(fn):
        t0 = time.perf_counter()
        for a, b, s in calls:
            fn(planets, a, b, s, angular_velocity=AV, comets=comets)
        return time.perf_counter() - t0

    t_new = _bench(validate_launch)
    print(f"  C) {len(calls)} вызовов на 40 телах: {t_new*1e3:.0f} мс "
          f"({t_new / len(calls) * 1e6:.0f} мкс/вызов)")


if __name__ == "__main__":
    print("[1] forward")
    test_forward()
    print("[2] intercept")
    test_intercept()
    print("[3] act")
    test_act()
    print("[4] validate_launch")
    test_validate_launch()
    print("\nВСЕ SMOKE-ТЕСТЫ ПРОЙДЕНЫ")

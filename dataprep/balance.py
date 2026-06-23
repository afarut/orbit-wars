r"""Баланс SFT-датасета по командам (JSONL -> JSONL).

НАЗНАЧЕНИЕ
----------
После ``dataprep/filter.py`` сэмплы сильно перекошены по командам: частые
победители (``meta.team``) дают непропорционально много ходов, и SFT имитирует в
основном их. Этот шаг выравнивает число сэмплов по командам. Инструмент ЭТАПА
РАЗРАБОТКИ, целиком ОФЛАЙН (между ``filter`` и ``preprocess``): лежит в ``dataprep/``
и в сабмит не входит. Балансируем именно ПОСЛЕ фильтра — тогда баланс в итоговом
обучающем файле точный (filter режет у разных команд разные доли ходов).

КЛЮЧ ГРУППИРОВКИ
---------------
``meta.team`` — строковое имя команды (источник ``info.TeamNames[player]``, см.
``dataprep/convert.py``); при ``--who winner`` это имя команды-победителя. Численного
id команды/сабмишна в реплеях нет, поэтому имя — единственный носитель «команды».
Разные сабмишены одного автора схлопываются в одну «команду» (для баланса это ОК).
Пустое/отсутствующее имя идёт под сентинелом ``<unknown>``.

КАК ВЫБИРАЕТСЯ ОБЪЁМ НА КОМАНДУ
------------------------------
ВАЖНО: распределение объёмов команд тяжелохвостое (медиана ~сотни ходов, у топ-команд
десятки-сотни ТЫСЯЧ). Поэтому строгое «у всех поровну» тут разрушительно: оно упирается
в самую мелкую команду и выкидывает почти всё (на реальных данных ~2.6% выживало).

**Рекомендуемый режим — ``--cap N`` (cap-only):** оставить ВСЕ команды, но обрезать
только верх до ``N`` ходов. Ни одна команда не доминирует (≤ ``N``), а средние/мелкие
сохраняются целиком — анти-доминирование без потери хвоста. Объём подбирается по
``N`` (см. прогноз, который печатает сам balance / ``dataprep/preprocess`` не нужен).

Альтернатива — **коридор** через ``--tol``/``--center`` (для РОВНЫХ распределений; на
тяжёлом хвосте даёт огромные потери): допуск отклонения ``--tol`` (доля, по умолч.
0.1 = 10%) от центра ``--center`` (``median`` по умолч. / ``mean``):

  * центр считается по числу сэмплов всех команд;
  * команды с числом сэмплов ``< center·(1-tol)`` ОТБРАСЫВАЮТСЯ целиком — выровнять
    их вверх нельзя (нельзя выдумать сэмплы), а тянуть ради них всех остальных вниз
    слишком расточительно (это и есть авто-решение «дроп vs выровнять»);
  * порог берётся ``n = floor = (минимум среди оставшихся)``, а ``cap = n·(1+tol)``;
    все оставшиеся команды режутся до ``min(их объём, cap)``. Итог: у всех оставшихся
    команд число сэмплов в коридоре ``[n, n·(1+tol)]`` — т.е. различаются не больше
    чем на ``tol`` (выровнено «до точности tol»), а крупные команды подрезаны.

``--tol 0`` -> строгий старый режим (ровно минимум на всех). ``--target-n N``
переопределяет авто-логику: ровно ``N`` на команду, команды с ``< N`` отбрасываются.
``--min-team K`` — дополнительный жёсткий префильтр (выкинуть команды ``< K`` ещё до
расчёта центра); по умолчанию 0.

ОТБОР СЭМПЛОВ
------------
Внутри команды нужные ``k`` сэмплов из её ``c`` выбираются последовательным отбором
без замены (Кнут, алгоритм S): на очередной сэмпл команды оставляем его с
вероятностью ``need/left``. Даёт РОВНО ``k`` равномерно выбранных сэмплов без хранения
индексов (память O(#команд)). Детерминизм — единый ``random.Random(seed)``.

ИСПОЛЬЗОВАНИЕ
-------------
  python -m dataprep.balance --in data/samples.full_send.jsonl \
      --out data/samples.full_send.balanced.jsonl --tol 0.1 --center median

  # как библиотека:
  from dataprep.balance import balance_by_team
  stats = balance_by_team("in.jsonl", "out.jsonl", tol=0.1, center="median", seed=0)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

try:                                             # прогресс-бар для офлайн-прогонов
    from tqdm import tqdm
except ImportError:                              # без tqdm тул всё равно работает (без бара)
    def tqdm(it: Iterable, **_kw: Any) -> Iterable:  # type: ignore[misc]
        return it

UNKNOWN = "<unknown>"                            # сентинел для пустого/отсутствующего имени


# --- хелперы -----------------------------------------------------------------
def _team_of(sample: dict, key: str) -> str:
    """Имя команды сэмпла из ``meta[key]``; пустое/None -> сентинел ``<unknown>``."""
    team = (sample.get("meta") or {}).get(key)
    if team is None or team == "":
        return UNKNOWN
    return str(team)


def _ensure_parent(path: str) -> None:
    """Создаёт родительский каталог для выходного файла, если его ещё нет."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _iter_lines(path: str, desc: str, total: Optional[int] = None):
    """Стримит непустые строки файла под tqdm-баром."""
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=desc, unit="rec", total=total):
            line = line.strip()
            if line:
                yield line


def _median(xs: List[int]) -> float:
    """Медиана списка (без numpy)."""
    s = sorted(xs)
    k = len(s)
    mid = k // 2
    return float(s[mid]) if k % 2 else (s[mid - 1] + s[mid]) / 2.0


def _center(counts: List[int], mode: str) -> float:
    """Центр распределения объёмов команд: ``median`` или ``mean``."""
    if not counts:
        return 0.0
    if mode == "mean":
        return sum(counts) / len(counts)
    if mode == "median":
        return _median(counts)
    raise ValueError(f"неизвестный --center: {mode} (ожидается median|mean)")


# --- ядро баланса ------------------------------------------------------------
def count_teams(in_path: str, key: str = "team") -> Counter:
    """Проход 1: число сэмплов на команду (``meta[key]``)."""
    counts: Counter = Counter()
    for line in _iter_lines(in_path, desc="balance:count"):
        try:
            sample = json.loads(line)
        except Exception:                        # noqa: BLE001 — битая строка не роняет прогон
            continue
        counts[_team_of(sample, key)] += 1
    return counts


def _plan_targets(counts: Dict[str, int], *, tol: float, center_mode: str,
                  target_n: Optional[int], cap: Optional[int],
                  min_team: int) -> Dict[str, Any]:
    """Решает, сколько сэмплов оставить у каждой команды (без чтения файла).

    Возвращает ``{"keep": {team: k}, "drop": {team: c}, "n", "cap", "center",
    "lower", "mode"}``. Логику см. в докстринге модуля.
    """
    # Жёсткий префильтр: команды меньше min_team вообще не рассматриваем.
    pre_drop = {t: c for t, c in counts.items() if c < min_team}
    pool = {t: c for t, c in counts.items() if c >= min_team}
    if not pool:
        raise ValueError(
            f"ни одна команда не прошла --min-team={min_team} "
            f"(макс. у команды: {max(counts.values()) if counts else 0})")

    if cap is not None:                          # cap-only: оставить ВСЕ, обрезать только верх
        cap_v = int(cap)
        keep = {t: min(c, cap_v) for t, c in pool.items()}
        return {"keep": keep, "drop": dict(pre_drop), "n": cap_v, "cap": cap_v,
                "center": 0.0, "lower": 0.0, "mode": f"cap-only={cap_v}"}

    if target_n is not None:                     # ручной режим: ровно N на команду
        n = cap = int(target_n)
        lower = float(target_n)
        center = float(target_n)
        mode = f"target-n={target_n}"
    else:
        center = _center(list(pool.values()), center_mode)
        lower = center * (1.0 - tol)
        # команды ниже коридора отбрасываем; floor = минимум среди оставшихся
        survivors = {t: c for t, c in pool.items() if c >= lower}
        if not survivors:                        # подстраховка (max всегда >= center >= lower)
            survivors = {max(pool, key=pool.get): max(pool.values())}
        n = min(survivors.values())
        cap = max(1, round(n * (1.0 + tol)))
        mode = f"center={center_mode}, tol={tol}"

    keep: Dict[str, int] = {}
    drop: Dict[str, int] = dict(pre_drop)
    for t, c in pool.items():
        if c < (lower if target_n is None else n):
            drop[t] = c                          # ниже коридора / меньше target_n -> дроп
        else:
            keep[t] = min(c, cap)                # подрезать до коридора
    return {"keep": keep, "drop": drop, "n": n, "cap": cap,
            "center": center, "lower": lower, "mode": mode}


def balance_by_team(in_path: str, out_path: str, *,
                    key: str = "team", tol: float = 0.1, center: str = "median",
                    target_n: Optional[int] = None, cap: Optional[int] = None,
                    min_team: int = 0, seed: int = 0,
                    overwrite: bool = False) -> Dict[str, Any]:
    """Стримит ``in_path`` -> ``out_path``, выравнивая число сэмплов по командам.

    Объём на команду выбирается авто-логикой (см. докстринг модуля): команды-хвосты
    ниже коридора ``center·(1±tol)`` отбрасываются, остальные режутся в коридор. Память
    плоская: два прохода по файлу + счётчики O(#команд). Возвращает сводку с полным
    раскладом ``before``/``after`` по командам — для печати (``_report``).
    """
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"{out_path} уже существует — укажи --overwrite или другой --out")
    _ensure_parent(out_path)

    counts = count_teams(in_path, key=key)
    if not counts:
        raise ValueError(f"во входе нет сэмплов: {in_path}")

    plan = _plan_targets(counts, tol=tol, center_mode=center,
                         target_n=target_n, cap=cap, min_team=min_team)
    keep = plan["keep"]                           # team -> сколько оставить

    # need/left для алгоритма S по каждой оставляемой команде.
    need = dict(keep)
    left = {t: counts[t] for t in keep}

    rng = random.Random(seed)
    total = sum(counts.values())
    stats: Dict[str, Any] = {
        "total": total, "kept": 0, "dropped": 0, "errors": 0,
        "teams_in": len(counts), "teams_kept": len(keep),
        "teams_dropped": len(plan["drop"]),
        "n": plan["n"], "cap": plan["cap"], "center": plan["center"],
        "lower": plan["lower"], "mode": plan["mode"],
        "tol": tol, "key": key, "seed": seed,
        "before": dict(counts), "after": dict(keep),
    }

    with open(out_path, "w", encoding="utf-8") as dst:
        for line in _iter_lines(in_path, desc="balance:write", total=total):
            try:
                sample = json.loads(line)
            except Exception:                    # noqa: BLE001 — битая строка не роняет прогон
                stats["errors"] += 1
                continue
            team = _team_of(sample, key)
            if team not in need:                 # команда-хвост -> дроп
                stats["dropped"] += 1
                continue
            take = need[team] > 0 and rng.random() < need[team] / left[team]
            left[team] -= 1
            if take:
                need[team] -= 1
                dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["kept"] += 1
            else:
                stats["dropped"] += 1
    return stats


# --- отчёт -------------------------------------------------------------------
def _report(stats: Dict[str, Any], out_path: str) -> None:
    """Печать сводки + ПОЛНЫЙ расклад по командам «до -> после» (сорт. по объёму до)."""
    before = stats["before"]
    after = stats["after"]
    total = stats["total"] or 1
    print(f"\n[balance] -> {out_path}")
    print(f"  режим: {stats['mode']}")
    if stats["center"]:
        print(f"  центр={stats['center']:,.0f}  коридор=[{stats['lower']:,.0f} .. "
              f"{stats['cap']:,}]  (n={stats['n']:,})")
    print(f"  команд: всего {stats['teams_in']:,}, оставлено {stats['teams_kept']:,}, "
          f"отброшено {stats['teams_dropped']:,}")
    print(f"  сэмплов: оставлено {stats['kept']:,}/{stats['total']:,} "
          f"({stats['kept']/total*100:.1f}%), отброшено {stats['dropped']:,}, "
          f"ошибок {stats['errors']:,}")
    print(f"  команды (до -> после, сорт. по объёму до):")
    print(f"    {'до':>9}  {'после':>9}   команда")
    for team, c in sorted(before.items(), key=lambda kv: kv[1], reverse=True):
        a = after.get(team)
        shown = f"{a:>9,}" if a is not None else f"{'(дроп)':>9}"
        print(f"    {c:>9,}  {shown}   {team}")


# --- CLI ---------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Баланс SFT-датасета Orbit Wars по командам "
                    "(JSONL {state, action, meta} -> JSONL, коридор по объёму команды)")
    ap.add_argument("--in", dest="input", required=True, help="входной JSONL-датасет")
    ap.add_argument("--out", required=True, help="выходной (сбалансированный) JSONL")
    ap.add_argument("--key", default="team",
                    help="поле meta для группировки команд (по умолч. %(default)s)")
    ap.add_argument("--tol", type=float, default=0.1,
                    help="допуск отклонения объёма от центра (доля, по умолч. 0.1=10%%; "
                         "0 -> строго минимум на всех)")
    ap.add_argument("--center", choices=["median", "mean"], default="median",
                    help="центр распределения для коридора (по умолч. median)")
    ap.add_argument("--cap", type=int, default=None,
                    help="cap-only: оставить ВСЕ команды, обрезать только верх до N "
                         "(анти-доминирование без потери хвоста; переопределяет --tol/--target-n)")
    ap.add_argument("--target-n", type=int, default=None,
                    help="ручной режим: ровно N сэмплов на команду (переопределяет --tol)")
    ap.add_argument("--min-team", type=int, default=0,
                    help="жёсткий префильтр: выкинуть команды с числом сэмплов < N")
    ap.add_argument("--seed", type=int, default=0, help="сид отбора (воспроизводимость)")
    ap.add_argument("--overwrite", action="store_true",
                    help="перезаписать --out, если он уже существует")
    args = ap.parse_args(argv)

    if not os.path.exists(args.input):
        ap.error(f"входной файл не найден: {args.input}")

    try:
        stats = balance_by_team(args.input, args.out, key=args.key, tol=args.tol,
                                center=args.center, target_n=args.target_n,
                                cap=args.cap, min_team=args.min_team, seed=args.seed,
                                overwrite=args.overwrite)
    except FileExistsError as e:                  # чистое сообщение вместо трейсбека
        ap.error(str(e))
    except ValueError as e:
        ap.error(str(e))

    _report(stats, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

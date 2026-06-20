r"""Препроцессинг SFT-датасета: угол -> планета-назначение (JSONL -> JSONL).

НАЗНАЧЕНИЕ
----------
В ``data/samples.full_send.jsonl`` таргет хранится как ``action`` =
``[[from_id, angle_rad, num_ships], ...]`` — то есть УГОЛ запуска. Сеть же
предсказывает не угол, а ИНДЕКС места-назначения (голова ``to``), а угол считает
на декоде через :class:`core.geo_lite.GeoEngine`. Поэтому здесь переводим угол в
планету-назначение готовой обратной операцией
:meth:`core.geo_lite.GeoEngine.planet_at_angle` (обёртка над ``orbit_lite``) и пишем
компактный таргет ``sends = [[from_id, dest_id, ships], ...]`` (угол выкидываем, число
кораблей ``ships`` оставляем — из него голова доли учит бакет {25,50,75,100}% гарнизона).

Это инструмент ЭТАПА РАЗРАБОТКИ, целиком ОФЛАЙН (последний шаг ETL, как convert/filter):
лежит в ``dataprep/`` и в боевой ``submission.tar.gz`` не попадает.

ФОРМАТ ВЫХОДА (одна строка = один ход)
--------------------------------------
    {
      "state":      {...},                 # obs as-is, готов под features.encode
      "sends":      [[from_id, dest_id, ships]],  # резолвнутые вылеты (hold -> [])
      "unresolved": [from_id, ...],         # источники, чей угол не восстановился (None)
      "meta":       {episode_id, player, step, is_winner, n_players}
    }

На обучении (см. sft/dataset.py): источник из ``sends`` -> таргет dest_id;
источник из ``unresolved`` -> ignore_index (не учим ложному hold); прочие свои
планеты -> hold.

ИСПОЛЬЗОВАНИЕ
-------------
  python -m dataprep.preprocess --in data/samples.full_send.jsonl --out data/sft.full_send.jsonl
  python -m dataprep.preprocess --in data/samples.full_send.jsonl --out /tmp/sft.smoke.jsonl --limit 2000
"""
from __future__ import annotations

# --- bootstrap: корень репо в sys.path (для запуска и как `-m`, и как файл) ---
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import math
from typing import Any, Dict, Iterable, Optional

try:
    from tqdm import tqdm
except ImportError:                              # без tqdm тул всё равно работает
    def tqdm(it: Iterable, **_kw: Any) -> Iterable:  # type: ignore[misc]
        return it

from core import geo_lite

OWNER_I, SHIPS_I = 1, 5   # индексы полей планеты: [id, owner, x, y, radius, ships, production]


def _ship_bucket(ships: float, garrison: float) -> int:
    """Доля ships/garrison -> бакет {0:25,1:50,2:75,3:100}% (для статистики прогона).

    Дублирует ``sft.dataset.ship_bucket`` намеренно — dataprep не тянет зависимости sft.
    """
    g = max(1.0, float(garrison))
    return min(3, max(0, math.ceil((float(ships) / g) / 0.25) - 1))


def resolve_sample(sample: dict, horizon: int = geo_lite.DEFAULT_HORIZON) -> dict:
    """Перевести углы хода в планеты-назначения для одного семпла.

    Возвращает запись ``{state, sends, unresolved, meta}`` + кладёт в ``_stats``
    счётчики для агрегированной статистики прогона. ``horizon`` — горизонт прогноза
    orbit_lite (должен покрывать самый долгий перелёт; см. ``geo_lite.DEFAULT_HORIZON``).
    """
    state = sample["state"]
    planets = state["planets"]
    player = int(state.get("player", 0) or 0)

    ids = {int(p[0]) for p in planets}   # известные planet-id в этом состоянии
    geo = None                           # geo_lite-движок строим лениво (один раз на state)

    sends = {}                # from_id -> (dest_id, ships) (dict гасит дубль источника)
    unresolved = []
    for move in sample.get("action") or []:
        from_id, angle, ships = int(move[0]), float(move[1]), int(move[2])
        if from_id not in ids:
            unresolved.append(from_id)   # источника нет в obs — аномалия, не учим
            continue
        if geo is None:
            geo = geo_lite.GeoEngine(state, player=player, horizon=horizon)
        dest_id = geo.planet_at_angle(from_id, angle, ships)
        if dest_id is None:
            unresolved.append(from_id)   # солнце/выход за поле/не восстановилось
            continue
        sends[from_id] = (int(dest_id), ships)

    # дёшево (без геометрии) считаем своих потенциальных источников для тюнинга w_hold
    n_owned = sum(1 for p in planets
                  if int(p[OWNER_I]) == player and float(p[SHIPS_I]) > 0)

    # гистограмма бакетов доли по резолвнутым вылетам (ориентир для frac_weights)
    garrison = {int(p[0]): float(p[SHIPS_I]) for p in planets}
    buckets = [0, 0, 0, 0]
    for fid, (_did, sh) in sends.items():
        g = garrison.get(fid, 0.0)
        if g > 0:
            buckets[_ship_bucket(sh, g)] += 1

    meta = sample.get("meta") or {}
    out = {
        "state": state,
        "sends": [[fid, did, sh] for fid, (did, sh) in sends.items()],
        "unresolved": unresolved,
        "meta": {
            "episode_id": meta.get("episode_id"),
            "player": player,
            "step": meta.get("step"),
            "is_winner": meta.get("is_winner"),
            "n_players": meta.get("n_players"),
        },
    }
    out["_stats"] = {
        "is_hold": not (sample.get("action")),
        "is_multi": len(sample.get("action") or []) > 1,
        "n_sends": len(sample.get("action") or []),
        "n_resolved": len(sends),
        "n_unresolved": len(unresolved),
        "n_owned": n_owned,
        "buckets": buckets,
    }
    return out


def run(in_path: str, out_path: str, limit: Optional[int] = None,
        insights_path: Optional[str] = None,
        horizon: int = geo_lite.DEFAULT_HORIZON) -> Dict[str, int]:
    """Стримингом прогнать препроцессинг, вернуть агрегированную статистику."""
    agg = {"states": 0, "hold_states": 0, "action_states": 0, "multi_states": 0,
           "sends_total": 0, "sends_resolved": 0, "sends_unresolved": 0,
           "owned_sources": 0, "send_sources": 0,
           "bucket_25": 0, "bucket_50": 0, "bucket_75": 0, "bucket_100": 0}

    with open(in_path, "r", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(tqdm(fin, desc="resolve angle->planet")):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = resolve_sample(json.loads(line), horizon=horizon)
            st = rec.pop("_stats")
            agg["states"] += 1
            agg["hold_states"] += int(st["is_hold"])
            agg["action_states"] += int(not st["is_hold"])
            agg["multi_states"] += int(st["is_multi"])
            agg["sends_total"] += st["n_sends"]
            agg["sends_resolved"] += st["n_resolved"]
            agg["sends_unresolved"] += st["n_unresolved"]
            agg["owned_sources"] += st["n_owned"]
            agg["send_sources"] += st["n_resolved"]
            b = st["buckets"]
            agg["bucket_25"] += b[0]; agg["bucket_50"] += b[1]
            agg["bucket_75"] += b[2]; agg["bucket_100"] += b[3]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    agg["hold_sources"] = agg["owned_sources"] - agg["send_sources"]
    _report(agg, out_path)
    if insights_path:
        _write_insights(agg, in_path, out_path, insights_path)
    return agg


def _report(agg: Dict[str, int], out_path: str) -> None:
    """Печать сводки + ориентир для w_hold (отношение hold/send источников)."""
    st = agg["states"] or 1
    sends = agg["sends_total"] or 1
    src = agg["owned_sources"] or 1
    print(f"\n[preprocess] -> {out_path}")
    print(f"  ходов:            {agg['states']:,}")
    print(f"  hold:             {agg['hold_states']:,} ({agg['hold_states']/st*100:.1f}%)")
    print(f"  с действием:      {agg['action_states']:,} ({agg['action_states']/st*100:.1f}%)")
    print(f"  мульти-send:      {agg['multi_states']:,}")
    print(f"  вылетов всего:    {agg['sends_total']:,}")
    print(f"  резолвнуто:       {agg['sends_resolved']:,} ({agg['sends_resolved']/sends*100:.2f}%)")
    print(f"  не резолвнуто:    {agg['sends_unresolved']:,} ({agg['sends_unresolved']/sends*100:.2f}%)")
    print(f"  своих источников: {agg['owned_sources']:,}")
    print(f"    из них send:    {agg['send_sources']:,} ({agg['send_sources']/src*100:.2f}%)")
    print(f"    из них hold:    {agg['hold_sources']:,} ({agg['hold_sources']/src*100:.2f}%)")
    if agg["send_sources"]:
        ratio = agg["send_sources"] / max(1, agg["hold_sources"])
        print(f"  ОРИЕНТИР w_hold ~ {ratio:.3f} (send/hold; даёт ~равный вклад классов)")
    _report_buckets(agg)


def _report_buckets(agg: Dict[str, int]) -> None:
    """Гистограмма бакетов доли + ориентир для train.frac_weights (обратная частота)."""
    counts = [agg["bucket_25"], agg["bucket_50"], agg["bucket_75"], agg["bucket_100"]]
    total = sum(counts) or 1
    labels = ["25%", "50%", "75%", "100%"]
    print(f"  бакеты доли (резолвнутые вылеты, всего {total:,}):")
    for lab, c in zip(labels, counts):
        print(f"    {lab:>4}: {c:,} ({c/total*100:.1f}%)")
    # обратная частота, нормированная к среднему 1.0 -> ориентир frac_weights
    inv = [total / (4 * c) if c else 0.0 for c in counts]
    print(f"  ОРИЕНТИР frac_weights ~ [{', '.join(f'{w:.2f}' for w in inv)}] (обратная частота)")


def _write_insights(agg: Dict[str, int], in_path: str, out_path: str,
                    path: str) -> None:
    """Сохранить сводку в insights/ (проектная конвенция: разборы датасета туда)."""
    st = agg["states"] or 1
    sends = agg["sends_total"] or 1
    src = agg["owned_sources"] or 1
    ratio = agg["send_sources"] / max(1, agg["hold_sources"]) if agg["send_sources"] else 0.0
    text = f"""# SFT препроцессинг: угол -> планета-назначение

Источник: `{in_path}` -> `{out_path}`
Скрипт: `dataprep/preprocess.py` (обратная операция `core.geo_lite.GeoEngine.planet_at_angle`).

| Метрика | Значение |
|---|---|
| Ходов | {agg['states']:,} |
| Hold | {agg['hold_states']:,} ({agg['hold_states']/st*100:.1f}%) |
| С действием | {agg['action_states']:,} ({agg['action_states']/st*100:.1f}%) |
| Мульти-send | {agg['multi_states']:,} |
| Вылетов всего | {agg['sends_total']:,} |
| Резолвнуто (угол->планета) | {agg['sends_resolved']:,} ({agg['sends_resolved']/sends*100:.2f}%) |
| Не резолвнуто (None) | {agg['sends_unresolved']:,} ({agg['sends_unresolved']/sends*100:.2f}%) |
| Своих источников | {agg['owned_sources']:,} |
| — send | {agg['send_sources']:,} ({agg['send_sources']/src*100:.2f}%) |
| — hold | {agg['hold_sources']:,} ({agg['hold_sources']/src*100:.2f}%) |

**Ориентир `w_hold` ≈ {ratio:.3f}** (send/hold — при таком весе вклад hold- и
send-источников в лосс примерно равный). Дефолт в конфиге 0.3; крути под датасет.

Не резолвнутые вылеты на обучении идут в `ignore_index` (не учим ложному hold).
"""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"  сводка: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT препроцессинг: угол -> планета")
    ap.add_argument("--in", dest="in_path", default="data/samples.full_send.jsonl")
    ap.add_argument("--out", dest="out_path", default="data/sft.full_send.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="ограничить число строк (smoke)")
    ap.add_argument("--horizon", type=int, default=geo_lite.DEFAULT_HORIZON,
                    help="горизонт прогноза orbit_lite (покрыть самый долгий перелёт)")
    ap.add_argument("--insights", default="insights/sft-angle-to-planet.md",
                    help="куда положить md-сводку (пусто -> не писать)")
    args = ap.parse_args()
    run(args.in_path, args.out_path, limit=args.limit,
        insights_path=(args.insights or None), horizon=args.horizon)


if __name__ == "__main__":
    main()

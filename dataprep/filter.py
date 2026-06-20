r"""Офлайн-фильтр SFT-датасета Orbit Wars (JSONL -> JSONL).

НАЗНАЧЕНИЕ
----------
Прореживает датасет, собранный ``dataprep/convert.py`` (одна строка —
``{state, action, meta}``), оставляя только семплы нужного класса действий.
Это инструмент **этапа разработки**, целиком ОФЛАЙН (как и конвертер): он лежит
в ``dataprep/`` и НЕ импортируется из ``core/`` (по §2.12 правил боевой сабмит
не тянет данные в рантайме).

ФИЛЬТРЫ
-------
Каждый фильтр — предикат ``sample -> bool`` (True == оставить). Семпл проходит,
если удовлетворяет ВСЕМ выбранным фильтрам (AND). Какие применять — задаёт флаг
``--keep`` (по умолчанию ``full_send``). Напоминание: ``action`` — это список
вылетов ``[[from_id, angle, ships], ...]``, а ``[]`` == бездействие (hold).

  * ``full_send``        — hold ЛИБО КАЖДЫЙ вылет отправляет весь гарнизон своей
                           планеты-источника (``ships == гарнизон``). Вылеты с
                           РАЗНЫХ планет разрешены (каждый — весь гарнизон),
                           частичные отправки отсекаются. [по умолчанию]
  * ``partial_send``     — hold ЛИБО любой вылет (включая ЧАСТИЧНЫЙ); отсекает только
                           вылеты с источником вне obs / пустым гарнизоном. Нужен как
                           обучающий сигнал для головы числа кораблей (доля гарнизона).
  * ``distinct_sources`` — hold ЛИБО ни одна планета не стреляет дважды за ход
                           (отсекает «одна планета -> несколько целей»).
  * ``single_launch``    — hold ЛИБО ровно один вылет за ход (отсекает ЛЮБОЙ
                           мультивылет; строгий одношаговый класс).

Доли в датасете (data/samples.jsonl, сэмпл): hold 52%, одиночный full 16%,
разные планеты «все-full» 16% -> ``full_send`` оставляет ~85%; добавив
``single_launch`` (старое строгое поведение) — ~68%. «Одна планета -> много
целей» в данных ~0.1%.

РАСШИРЕНИЕ
----------
Новые правила добавляются в реестр ``FILTERS`` (имя -> предикат).

ИСПОЛЬЗОВАНИЕ
-------------
  # по умолчанию (hold + все вылеты «весь гарнизон», вкл. разные планеты):
  python -m dataprep.filter --in data/samples.jsonl --out data/samples.full_send.jsonl

  # дополнительно запретить «одна планета -> много целей»:
  python -m dataprep.filter --in data/samples.jsonl --out data/out.jsonl \
      --keep full_send distinct_sources

  # строго: только hold или одиночный вылет всего гарнизона (старое поведение):
  python -m dataprep.filter --in data/samples.jsonl --out data/out.jsonl \
      --keep full_send single_launch

  # как библиотека:
  from dataprep.filter import FILTERS, keep_full_send
  kept = [s for s in samples if keep_full_send(s)]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional

try:                                             # прогресс-бар для офлайн-прогонов
    from tqdm import tqdm
except ImportError:                              # без tqdm тул всё равно работает (без бара)
    def tqdm(it: Iterable, **_kw: Any) -> Iterable:  # type: ignore[misc]
        return it

# Порядок полей планеты держим в одном месте — переиспользуем namedtuple конвертера.
try:
    from dataprep.convert import Planet          # запуск как `python -m dataprep.filter`
except ImportError:                              # запуск как `python dataprep/filter.py`
    from convert import Planet

# Индекс поля ships в строке планеты [id, owner, x, y, radius, ships, production].
# Берём из namedtuple конвертера — единый источник порядка полей.
SHIPS_IDX = Planet._fields.index("ships")

# Тип предиката фильтра: семпл -> оставить ли его (True == оставить).
Predicate = Callable[[dict], bool]


# --- хелперы -----------------------------------------------------------------
def _garrison_map(sample: dict) -> Dict[int, int]:
    """planet_id -> гарнизон (ships) из ``state.planets`` этого семпла.

    Гарнизон берётся из наблюдения момента решения (сдвиг obs<->action уже учтён
    конвертером), поэтому соответствует моменту хода. Для full-send гарнизон всегда
    целочисленный (insights/rounding-direction-by-bin.md) -> сравнение точное.
    """
    planets = (sample.get("state") or {}).get("planets") or []
    return {p[0]: int(p[SHIPS_IDX]) for p in planets if p}


# --- фильтры (предикаты; True == оставить семпл) -----------------------------
def keep_full_send(sample: dict) -> bool:
    """hold ЛИБО каждый вылет отправляет весь гарнизон своей планеты-источника.

    Разрешает вылеты с РАЗНЫХ планет (каждый — весь гарнизон); частичные отправки
    (в т.ч. дробление одной планеты по нескольким целям) отсекает.
    """
    action = sample.get("action") or []
    if not action:
        return True                              # бездействие (hold)
    g = _garrison_map(sample)
    for move in action:                          # move == [from_id, angle, ships]
        garrison = g.get(move[0])
        if garrison is None or int(move[2]) != garrison:
            return False                         # источник не найден или отправлено не всё
    return True


def keep_partial_send(sample: dict) -> bool:
    """hold ЛИБО любой вылет (включая частичный) — впускает дробные отправки.

    В отличие от ``full_send``, НЕ требует ``ships == гарнизон``: нужен как обучающий
    сигнал для головы числа кораблей. Отсекает только вылеты с источником вне obs или
    с нулевым гарнизоном (аномалия — нечего слать).
    """
    action = sample.get("action") or []
    if not action:
        return True                              # бездействие (hold)
    g = _garrison_map(sample)
    for move in action:                          # move == [from_id, angle, ships]
        garrison = g.get(move[0])
        if garrison is None or garrison <= 0:
            return False                         # источник не найден / пустой гарнизон
    return True


def keep_distinct_sources(sample: dict) -> bool:
    """hold ЛИБО ни одна планета не стреляет дважды за ход.

    Отсекает «одна планета -> несколько целей» (дробление гарнизона по целям);
    вылеты с разных планет пропускает.
    """
    sources = [move[0] for move in (sample.get("action") or [])]
    return len(sources) == len(set(sources))


def keep_single_launch(sample: dict) -> bool:
    """hold ЛИБО ровно один вылет за ход (отсекает любой мультивылет)."""
    return len(sample.get("action") or []) <= 1


# Реестр фильтров (имя -> предикат). Будущие правила дописывать сюда.
FILTERS: Dict[str, Predicate] = {
    "full_send": keep_full_send,
    "partial_send": keep_partial_send,
    "distinct_sources": keep_distinct_sources,
    "single_launch": keep_single_launch,
}

# Набор фильтров по умолчанию, если --keep не задан.
DEFAULT_KEEP: List[str] = ["full_send"]


# --- ядро фильтрации ----------------------------------------------------------
def _ensure_parent(path: str) -> None:
    """Создаёт родительский каталог для выходного файла, если его ещё нет."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def filter_dataset(in_path: str, out_path: str,
                   predicates: Optional[Dict[str, Predicate]] = None,
                   overwrite: bool = False) -> Dict[str, int]:
    """Стримит ``in_path`` -> ``out_path``, оставляя семплы, прошедшие ВСЕ предикаты.

    Память плоская: читаем и пишем построчно, весь датасет в RAM не держим.
    Предикаты прогоняются БЕЗ short-circuit — чтобы вести честную по-фильтрную
    статистику отбраковки (важно, когда фильтров станет несколько).

    Возвращает сводку: ``{"total", "kept", "dropped", "errors", "rejected:<имя>"...}``.
    Битая JSONL-строка не роняет прогон (учитывается в ``errors``).
    """
    preds = predicates if predicates is not None else dict(FILTERS)
    if not preds:
        raise ValueError("не выбрано ни одного фильтра (пустой набор предикатов)")

    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"{out_path} уже существует — укажи --overwrite или другой --out")
    _ensure_parent(out_path)

    stats: Dict[str, int] = {"total": 0, "kept": 0, "dropped": 0, "errors": 0}
    for name in preds:
        stats[f"rejected:{name}"] = 0

    with open(in_path, "r", encoding="utf-8") as src, \
            open(out_path, "w", encoding="utf-8") as dst:
        bar = tqdm(src, desc="filter", unit="rec")   # tqdm сам считает прочитанные строки
        for line in bar:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                sample = json.loads(line)
            except Exception:                    # noqa: BLE001 — битая строка не роняет прогон
                stats["errors"] += 1
                continue
            failed = [name for name, pred in preds.items() if not pred(sample)]
            for name in failed:
                stats[f"rejected:{name}"] += 1
            if failed:
                stats["dropped"] += 1
            else:
                dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["kept"] += 1
            if hasattr(bar, "set_postfix"):      # live-счётчики только при настоящем tqdm
                bar.set_postfix(kept=stats["kept"], dropped=stats["dropped"])
    return stats


# --- CLI ---------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Фильтр SFT-датасета Orbit Wars (JSONL {state, action, meta})")
    ap.add_argument("--in", dest="input", required=True, help="входной JSONL-датасет")
    ap.add_argument("--out", required=True, help="выходной (отфильтрованный) JSONL")
    ap.add_argument("--keep", nargs="+", choices=list(FILTERS), default=DEFAULT_KEEP,
                    help="какие фильтры применять, AND (по умолч. %(default)s; "
                         "варианты: " + ", ".join(FILTERS) + ")")
    ap.add_argument("--overwrite", action="store_true",
                    help="перезаписать --out, если он уже существует")
    args = ap.parse_args(argv)

    if not os.path.exists(args.input):
        ap.error(f"входной файл не найден: {args.input}")

    preds = {name: FILTERS[name] for name in args.keep}
    try:
        stats = filter_dataset(args.input, args.out, predicates=preds,
                               overwrite=args.overwrite)
    except FileExistsError as e:                 # чистое сообщение вместо трейсбека
        ap.error(str(e))

    total = stats["total"]
    kept, dropped = stats["kept"], stats["dropped"]
    pct = (100.0 * kept / total) if total else 0.0
    print(f"готово: оставлено {kept}/{total} ({pct:.1f}%), "
          f"отброшено {dropped}, ошибок {stats['errors']} -> {args.out}")
    for name in args.keep:                       # по-фильтрная отбраковка
        print(f"  отброшено фильтром {name}: {stats[f'rejected:{name}']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

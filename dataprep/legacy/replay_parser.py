r"""Офлайн-парсер реплеев Orbit Wars (kaggle_environments).

АРХИВ. Каноничный путь сбора датасета — ``dataprep/convert.py``; этот парсер
оставлен как альтернативная реализация и кодом пайплайна не используется.

НАЗНАЧЕНИЕ И ГРАНИЦЫ ИСПОЛЬЗОВАНИЯ
----------------------------------
Это инструмент **этапа разработки**: скачал реплеи -> распарсил -> посчитал
статистику / собрал датасет для обучения. И всё это ОФЛАЙН.

По правилам компетишена (§2.12 NO INGRESS OR EGRESS) боевой сабмит во время
эпизода не имеет права тянуть внешние данные или что-то отправлять наружу.
Поэтому:
  * этот файл НЕ импортируется из `core/` и НЕ кладётся в submission.tar.gz;
  * он лежит в `dataprep/legacy/` именно чтобы случайно не попасть в бандл сабмита;
  * результат его работы (статистика/датасет) запекается в веса/эвристики
    ЗАРАНЕЕ, а не читается агентом в рантайме.

Реплеи официально публичны и качаются (§2.11), как внешние данные допустимы
(§2.6). Поделиться парсером можно только публично на форуме компа, не приватно
вне команды (§3.6).

ОТКУДА БРАТЬ РЕПЛЕИ (вне этого скрипта, в шелле/ноутбуке)
--------------------------------------------------------
  # свои эпизоды через Kaggle CLI:
  kaggle competitions episodes <SUBMISSION_ID>
  kaggle competitions replay  <EPISODE_ID>      # -> <EPISODE_ID>.json

  # или сгенерить локально:
  from kaggle_environments import make
  env = make("orbit_wars", configuration={"seed": 42})
  env.run(["main.py", "random", "random", "random"])
  import json; json.dump(env.toJSON(), open("local.json", "w"))

ИСПОЛЬЗОВАНИЕ
-------------
  python -m dataprep.legacy.replay_parser path/to/episode.json            # сводка
  python -m dataprep.legacy.replay_parser "replays/*.json" --csv out.csv  # дамп пошагово

  # собрать SFT-датасет (стриминг, плоская память + прогресс-бар):
  python -m dataprep.legacy.replay_parser "replays/*.json" --sft data/sft.jsonl --who winner

  Рядом пишется data/sft.jsonl.manifest — список уже обработанных реплеев.
  Докачал новые реплеи -> запусти ту же команду: старое пропустится, новое
  допишется в конец. --overwrite — пересобрать датасет с нуля.

  # как библиотека:
  from dataprep.legacy.replay_parser import load_replay, parse_replay, iter_sft_records
  rep = parse_replay(load_replay("episode.json"))
  for st in rep.steps:
      print(st.step, st.scores, st.actions)
  for rec in iter_sft_records(rep, who="winner"):
      ...  # rec["obs"] -> features.encode, rec["moves"] -> таргет policy
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

try:                                             # прогресс-бар для офлайн-прогонов
    from tqdm import tqdm
except ImportError:                              # без tqdm тул всё равно работает (без бара)
    def tqdm(it: Iterable, **_kw: Any) -> Iterable:  # type: ignore[misc]
        return it

# Поля строго в порядке из Observation Reference правил.
Planet = namedtuple("Planet", "id owner x y radius ships production")
Fleet = namedtuple("Fleet", "id owner x y angle from_planet_id ships")

NEUTRAL = -1


# --- мелкие хелперы (как в orbit_wars/features.py: терпим dict и объект) -------
def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_move_list(action: Any) -> List[list]:
    """Нормализует action агента к списку ходов [[from_id, angle, ships], ...]."""
    if not action:
        return []
    if isinstance(action, dict):                 # на случай {"moves": [...]}
        action = action.get("moves", []) or []
    out: List[list] = []
    for m in action:
        if m is None:
            continue
        out.append(list(m))
    return out


# --- структуры результата -----------------------------------------------------
@dataclass
class StepState:
    """Состояние мира на одном ходу реплея."""

    step: int
    planets: List[Planet]
    fleets: List[Fleet]
    comet_ids: List[int]                         # planet_id, которые являются кометами
    actions: Dict[int, List[list]]               # player_id -> список ходов на этом ходу
    statuses: List[str]                          # статус каждого игрока ("ACTIVE"/"DONE"/...)
    scores: Dict[int, float]                     # player_id -> счёт (корабли планет + флотов)
    raw_obs: Dict[str, Any] = field(default_factory=dict)  # полное наблюдение шага (для features.encode)

    def owned_planets(self, player: int) -> List[Planet]:
        return [p for p in self.planets if p.owner == player]

    def owned_fleets(self, player: int) -> List[Fleet]:
        return [f for f in self.fleets if f.owner == player]


@dataclass
class Replay:
    """Разобранный реплей целиком."""

    episode_id: Optional[str]
    n_players: int
    config: Dict[str, Any]
    steps: List[StepState]
    rewards: List[Optional[float]]               # финальная награда от движка, по игрокам
    final_statuses: List[str]
    winner: Optional[int]                         # None == ничья/не определён
    meta: Dict[str, Any] = field(default_factory=dict)


# --- загрузка ----------------------------------------------------------------
def load_replay(path: str) -> dict:
    """Читает JSON реплея с диска. Поддерживает .json и .json.gz."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    # Иногда реплей завёрнут: {"replay": {...}} или это список steps без обёртки.
    if isinstance(data, dict) and "steps" not in data and "replay" in data:
        data = data["replay"]
    if isinstance(data, list):                   # совсем сырой список steps
        data = {"steps": data}
    return data


# --- разбор ------------------------------------------------------------------
def _shared_obs(step_agents: List[dict]) -> dict:
    """Глобальное наблюдение хранится в obs игрока 0; planets/fleets — общие.

    kaggle_environments часто кладёт полное состояние только в obs[0], а у
    остальных агентов obs содержит лишь их `player`/`action`. Берём первое
    наблюдение, где реально есть планеты.
    """
    for a in step_agents:
        obs = _get(a, "observation", {}) or {}
        if _get(obs, "planets"):
            return obs
    # фолбэк: хотя бы obs игрока 0
    return _get(step_agents[0], "observation", {}) or {}


def _score(planets: List[Planet], fleets: List[Fleet], player: int) -> float:
    """Финальный счёт по правилам (строка 161 rules): корабли на своих планетах
    + корабли в своих флотах."""
    s = sum(p.ships for p in planets if p.owner == player)
    s += sum(f.ships for f in fleets if f.owner == player)
    return float(s)


def parse_replay(data: dict) -> Replay:
    """dict реплея -> структурированный Replay (пошаговое состояние + итоги)."""
    raw_steps = _get(data, "steps", []) or []
    if not raw_steps:
        raise ValueError("В реплее нет 'steps' — это точно реплей Orbit Wars?")

    n_players = len(raw_steps[0])
    config = _get(data, "configuration", {}) or {}

    # Статические поля (initial_planets, angular_velocity) часто есть только в obs
    # нулевого шага. Запомним их, чтобы дописать в obs каждого шага -> каждый шаг
    # становится самодостаточным для features.encode.
    STATIC_KEYS = ("initial_planets", "angular_velocity")
    static_fields: Dict[str, Any] = {}

    steps: List[StepState] = []
    for t, step_agents in enumerate(raw_steps):
        obs = dict(_shared_obs(step_agents))     # копия полного наблюдения шага
        for k in STATIC_KEYS:
            if obs.get(k) not in (None, []):
                static_fields.setdefault(k, obs[k])
            elif k in static_fields:
                obs[k] = static_fields[k]        # форвард-филл недостающего

        planets = [Planet(*p) for p in (obs.get("planets") or [])]
        fleets = [Fleet(*f) for f in (obs.get("fleets") or [])]
        comet_ids = [int(c) for c in (obs.get("comet_planet_ids") or [])]

        actions: Dict[int, List[list]] = {}
        statuses: List[str] = []
        for p, agent in enumerate(step_agents):
            actions[p] = _as_move_list(_get(agent, "action"))
            statuses.append(str(_get(agent, "status", "")))

        scores = {p: _score(planets, fleets, p) for p in range(n_players)}
        steps.append(StepState(
            step=int(obs.get("step", t) or t),
            planets=planets, fleets=fleets, comet_ids=comet_ids,
            actions=actions, statuses=statuses, scores=scores,
            raw_obs=obs,
        ))

    # Награды/статусы: движок кладёт их в верхний уровень либо в последний шаг.
    rewards = _get(data, "rewards")
    if rewards is None:
        rewards = [_get(a, "reward") for a in raw_steps[-1]]
    final_statuses = _get(data, "statuses")
    if final_statuses is None:
        final_statuses = steps[-1].statuses

    winner = _determine_winner(rewards, steps[-1].scores)

    return Replay(
        episode_id=_get(data, "id"),
        n_players=n_players,
        config=config,
        steps=steps,
        rewards=list(rewards) if rewards is not None else [None] * n_players,
        final_statuses=list(final_statuses),
        winner=winner,
        meta={"name": _get(data, "name"), "version": _get(data, "version")},
    )


def _determine_winner(rewards: Optional[list], final_scores: Dict[int, float]) -> Optional[int]:
    """Победитель — по reward движка, при отсутствии — по финальному счёту.
    Возвращает None при ничьей."""
    table: Dict[int, float]
    if rewards is not None and any(r is not None for r in rewards):
        table = {i: (r if r is not None else float("-inf")) for i, r in enumerate(rewards)}
    else:
        table = dict(final_scores)
    if not table:
        return None
    best = max(table.values())
    leaders = [i for i, v in table.items() if v == best]
    return leaders[0] if len(leaders) == 1 else None


# --- CLI / отчёты ------------------------------------------------------------
def summarize(rep: Replay) -> str:
    """Короткая текстовая сводка по одному реплею."""
    last = rep.steps[-1]
    lines = [
        f"episode    : {rep.episode_id}",
        f"players    : {rep.n_players}",
        f"steps      : {len(rep.steps)}",
        f"rewards    : {rep.rewards}",
        f"statuses   : {rep.final_statuses}",
        f"end scores : {last.scores}",
        f"winner     : {rep.winner if rep.winner is not None else 'tie/unknown'}",
    ]
    # сколько ходов сделал каждый игрок за партию
    total_moves = {p: 0 for p in range(rep.n_players)}
    for st in rep.steps:
        for p, mv in st.actions.items():
            total_moves[p] += len(mv)
    lines.append(f"moves total: {total_moves}")
    return "\n".join(lines)


def dump_csv_stream(files: List[str], path: str) -> int:
    """Пишет пошаговый CSV: по строке на (эпизод, шаг, игрок). Удобно для анализа.

    Стриминг: реплеи парсятся по одному, строки сразу дозаписываются — память
    плоская независимо от числа файлов. Манифеста (resume) у CSV нет: он нужен
    только основному датасету (SFT)."""
    import csv

    _ensure_parent(path)
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "step", "player", "score",
                    "n_planets", "n_fleets", "n_moves", "status"])
        for fp in tqdm(files, desc="csv", unit="rep"):
            try:
                rep = parse_replay(load_replay(fp))
            except Exception:                     # noqa: BLE001 — битый файл не роняет прогон
                continue
            for st in rep.steps:
                for p in range(rep.n_players):
                    w.writerow([
                        rep.episode_id, st.step, p, st.scores[p],
                        len(st.owned_planets(p)), len(st.owned_fleets(p)),
                        len(st.actions.get(p, [])),
                        st.statuses[p] if p < len(st.statuses) else "",
                    ])
                    rows += 1
    return rows


def _select_sft_players(rep: Replay, who: str) -> List[int]:
    """Каких игроков берём в датасет с этого реплея.

    who:
        'winner' — только победитель эпизода (по умолчанию);
        'all'    — все игроки;
        'top:N'  — N лучших по финальному счёту.
    """
    if who == "winner":
        return [rep.winner] if rep.winner is not None else []
    if who == "all":
        return list(range(rep.n_players))
    if who.startswith("top:"):
        k = int(who.split(":", 1)[1])
        last_scores = rep.steps[-1].scores
        return [p for p, _ in sorted(
            last_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
    raise ValueError(f"неизвестный режим who={who!r}")


def iter_sft_records(rep: Replay, who: str = "winner",
                     skip_empty: bool = False) -> Iterator[dict]:
    """Генерит SFT-записи для одного реплея (без I/O).

    Одна запись = (наблюдение, по которому игрок `p` принимал решение; его ходы,
    принятые ПО ЭТОМУ наблюдению). Поле ``obs`` готово к подаче в
    ``core.features.encode`` (там уже есть ``player``/``step``), а ``moves``
    — экспертный таргет для policy-головы.

    ВАЖНО — сдвиг obs↔action на кадр. kaggle_environments сохраняет
    ``steps[t].observation`` уже ПОСЛЕ применения ``steps[t].action`` (корабли
    улетели, продакшен начислен). То есть ход ``actions[t]`` принят по состоянию
    ``obs[t-1]``, а не ``obs[t]``. Поэтому вход ``obs`` шага ``t`` спариваем с
    таргетом ``moves`` шага ``t+1``. Эмпирически это даёт 0% «отправлено больше
    гарнизона» против ~57% при наивном спаривании внутри одного шага. Последний
    шаг эпизода (за ним нет хода) при этом отбрасывается — это корректно, после
    финального наблюдения решение уже не принимается.

    skip_empty: пропускать шаги без ходов (по умолчанию НЕ пропускаем — «hold»
        это валидный экспертный таргет для policy-головы).
    """
    last_scores = rep.steps[-1].scores
    players = _select_sft_players(rep, who)
    # cur — наблюдение, по которому решали; nxt — шаг, в obs которого записан
    # результат, а в action — сам ход, принятый по `cur`. Берём (cur.obs, nxt.action).
    for cur, nxt in zip(rep.steps, rep.steps[1:]):
        for p in players:
            moves = nxt.actions.get(p, [])
            if skip_empty and not moves:
                continue
            obs = dict(cur.raw_obs)
            obs["player"] = p
            obs["step"] = cur.step
            yield {
                "episode": rep.episode_id, "step": cur.step, "player": p,
                "is_winner": (p == rep.winner),
                "final_score": last_scores.get(p),
                "obs": obs, "moves": moves,
            }


def _ensure_parent(path: str) -> None:
    """Создаёт родительский каталог для выходного файла, если его ещё нет."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_manifest(path: str) -> Set[str]:
    """Читает манифест обработанных реплеев в set basename'ов (нет файла -> пусто)."""
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def export_sft_stream(files: List[str], out_path: str, who: str = "winner",
                      skip_empty: bool = False, manifest_path: Optional[str] = None,
                      overwrite: bool = False) -> int:
    """Стримит SFT-сэмплы (JSONL) для имитационного обучения (behavioral cloning).

    Реплеи обрабатываются по одному с дозаписью — память плоская независимо от
    числа файлов (важно: датасет ~6 ГБ не влезет в RAM целиком).

    RESUME: рядом с ``out_path`` ведётся манифест ``<out_path>.manifest`` —
    по одному basename обработанного реплея на строку. Уже перечисленные файлы
    пропускаются ДО загрузки (не тратим память на чтение многомегабайтного JSON),
    новые — дописываются в конец датасета. Так можно докачать ещё реплеев и
    прогнать конвейер повторно без дублей и без переобработки старого.

    overwrite=True — игнорировать манифест и начать датасет (и манифест) с нуля.

    Crash-safety: на каждый файл сперва пишем и flush'им записи, и только ПОТОМ
    дописываем basename в манифест. При падении между ними один эпизод
    переобработается на resume (дубли возможны только для него).
    """
    manifest_path = manifest_path or (out_path + ".manifest")
    _ensure_parent(out_path)
    _ensure_parent(manifest_path)

    # Fresh-старт, если: явный overwrite; либо манифеста нет; либо манифест есть,
    # а датасет пропал (удалили) — иначе из-за skip'а в датасете будут дыры.
    fresh = overwrite or not os.path.exists(manifest_path) or not os.path.exists(out_path)
    done: Set[str] = set() if fresh else _load_manifest(manifest_path)
    file_mode = "w" if fresh else "a"

    written = skipped = errors = 0
    bar = tqdm(files, desc=f"sft:{who}", unit="rep")
    with open(out_path, file_mode, encoding="utf-8") as out, \
            open(manifest_path, file_mode, encoding="utf-8") as man:
        for fp in bar:
            key = os.path.basename(fp)
            if key in done:
                skipped += 1
                _set_postfix(bar, written, skipped, errors)
                continue
            try:
                rep = parse_replay(load_replay(fp))
            except Exception:                     # noqa: BLE001 — битый файл не роняет прогон
                errors += 1
                _set_postfix(bar, written, skipped, errors)
                continue
            for rec in iter_sft_records(rep, who=who, skip_empty=skip_empty):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            out.flush()
            man.write(key + "\n")                 # отмечаем файл обработанным ПОСЛЕ записи
            man.flush()
            done.add(key)
            _set_postfix(bar, written, skipped, errors)
    return written


def _set_postfix(bar: Any, written: int, skipped: int, errors: int) -> None:
    """Обновляет live-счётчики на tqdm-баре (без построчного логирования)."""
    if hasattr(bar, "set_postfix"):
        bar.set_postfix(written=written, skipped=skipped, errors=errors)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Офлайн-парсер реплеев Orbit Wars")
    ap.add_argument("paths", nargs="+", help="JSON-реплеи (можно с glob/wildcards)")
    ap.add_argument("--csv", help="сохранить пошаговый дамп в этот CSV (стриминг)")
    ap.add_argument("--sft", help="сохранить SFT-сэмплы (JSONL) для обучения (стриминг)")
    ap.add_argument("--who", default="winner",
                    help="чьи ходы в SFT: winner | all | top:N (по умолч. winner)")
    ap.add_argument("--skip-empty", action="store_true",
                    help="пропускать шаги без ходов при экспорте SFT")
    ap.add_argument("--manifest",
                    help="путь к манифесту обработанных реплеев "
                         "(по умолч. <sft>.manifest)")
    ap.add_argument("--overwrite", action="store_true",
                    help="игнорировать манифест и начать SFT-датасет с нуля")
    args = ap.parse_args(argv)

    files: List[str] = []
    for pat in args.paths:
        files.extend(sorted(glob.glob(pat)) or [pat])

    if args.sft:
        export_sft_stream(files, args.sft, who=args.who, skip_empty=args.skip_empty,
                          manifest_path=args.manifest, overwrite=args.overwrite)
    if args.csv:
        dump_csv_stream(files, args.csv)
    if not args.sft and not args.csv:             # режим инспекции: просто сводки
        for fp in files:
            try:
                rep = parse_replay(load_replay(fp))
            except Exception as e:                # noqa: BLE001 — не падаем на одном битом файле
                print(f"[skip] {fp}: {e}")
                continue
            print(f"=== {fp} ===")
            print(summarize(rep))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

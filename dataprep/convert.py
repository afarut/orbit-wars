r"""Офлайн-конвертер реплеев Orbit Wars -> SFT-семплы (JSONL).

НАЗНАЧЕНИЕ
----------
Берёт публичные реплеи Kaggle (kaggle_environments) и превращает их в датасет
для имитационного обучения (behavioral cloning). Один семпл — это ровно три поля:

    {
      "state":  {...},   # x — наблюдение, по которому игрок принимал решение
      "action": [...],   # y — экспертный ход(ы), принятый ПО ЭТОМУ наблюдению
      "meta":   {...},   # метаинформация (эпизод, команда, реворды, seed, ...)
    }

Это инструмент **этапа разработки**, целиком ОФЛАЙН: он лежит в ``dataprep/`` и НЕ
импортируется из ``core/`` (по §2.12 правил боевой сабмит не имеет права
тянуть/отправлять данные в рантайме). Результат запекается в веса заранее.

КЛЮЧЕВАЯ ТОНКОСТЬ — СДВИГ obs <-> action НА КАДР
------------------------------------------------
kaggle_environments сохраняет ``steps[t].observation`` уже ПОСЛЕ применения
``steps[t].action`` (корабли улетели, продакшен начислен). То есть ход
``actions[t]`` принят по состоянию ``obs[t-1]``, а не ``obs[t]``. Поэтому вход
``state`` шага ``t`` спариваем с таргетом ``action`` шага ``t+1``. Эмпирически
это даёт 0% «отправлено больше гарнизона» против ~57% при наивном спаривании
внутри одного шага. Последний шаг эпизода (за ним нет хода) отбрасывается — после
финального наблюдения решение уже не принимается.

ИСПОЛЬЗОВАНИЕ
-------------
  # собрать датасет победителей (по умолчанию), стриминг + resume:
  python -m dataprep.convert --in "replays/*.json" --out data/samples.jsonl

  # все игроки / N лучших по финальному счёту:
  python -m dataprep.convert --in replays/ --out data/all.jsonl --who all
  python -m dataprep.convert --in replays/ --out data/top2.jsonl --who top:2

  Рядом пишется <out>.manifest — список уже обработанных реплеев. Докачал новые
  реплеи -> запусти ту же команду: старое пропустится, новое допишется в конец.
  --overwrite — пересобрать датасет с нуля.

  # как библиотека:
  from dataprep.convert import load_replay, parse_replay, iter_samples
  rep = parse_replay(load_replay("episode.json"))
  for s in iter_samples(rep, who="winner"):
      ...  # s["state"] -> features.encode, s["action"] -> таргет policy
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

    episode_id: Optional[str]                     # uuid матча (data["id"])
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
    """Финальный счёт по правилам (§161 rules): корабли на своих планетах
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
    info = _get(data, "info", {}) or {}

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

    # seed обычно лежит в info.seed (configuration.seed в публичных реплеях пуст).
    seed = _get(info, "seed")
    if seed is None:
        seed = _get(config, "seed")

    return Replay(
        episode_id=_get(data, "id"),
        n_players=n_players,
        config=config,
        steps=steps,
        rewards=list(rewards) if rewards is not None else [None] * n_players,
        final_statuses=list(final_statuses),
        winner=winner,
        meta={
            "name": _get(data, "name"),
            "version": _get(data, "version"),
            "episode_id": _get(info, "EpisodeId"),     # числовой Kaggle-id
            "seed": seed,
            "teams": list(_get(info, "TeamNames") or []),  # имена команд по индексу игрока
        },
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


# --- выбор игроков -----------------------------------------------------------
def _select_players(rep: Replay, who: str) -> List[int]:
    """Каких игроков берём в датасет с этого реплея.

    who:
        'winner' — только победитель эпизода (по умолчанию; ничья -> никого);
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


# --- генерация семплов (ядро со сдвигом) -------------------------------------
def iter_samples(rep: Replay, who: str = "winner", skip_empty: bool = False,
                 source: Optional[str] = None) -> Iterator[dict]:
    """Генерит семплы {state, action, meta} для одного реплея (без I/O).

    Один семпл = (наблюдение `state`, по которому игрок `p` принимал решение;
    его ход(ы) `action`, принятые ПО ЭТОМУ наблюдению). ``state`` готов к подаче
    в ``core.features.encode`` (там уже проставлены ``player``/``step``),
    ``action`` — экспертный таргет для policy-головы.

    СДВИГ obs<->action: спариваем (state шага t, action шага t+1) — см. модульный
    докстринг. Из ``state`` удаляется ``initial_planets`` (в фичах не используется).

    skip_empty: пропускать шаги без ходов (по умолчанию НЕ пропускаем — «hold» это
        валидный экспертный таргет для policy-головы).
    """
    last_scores = rep.steps[-1].scores
    players = _select_players(rep, who)
    teams = rep.meta.get("teams") or []
    # cur — наблюдение, по которому решали; nxt — шаг, в obs которого записан
    # результат, а в action — сам ход, принятый по `cur`. Берём (cur.obs, nxt.action).
    for cur, nxt in zip(rep.steps, rep.steps[1:]):
        for p in players:
            action = nxt.actions.get(p, [])
            if skip_empty and not action:
                continue
            state = dict(cur.raw_obs)
            state.pop("initial_planets", None)   # в features.encode не читается
            state["player"] = p
            state["step"] = cur.step
            yield {
                "state": state,
                "action": action,
                "meta": {
                    "episode": rep.episode_id,                # uuid матча
                    "episode_id": rep.meta.get("episode_id"),  # числовой Kaggle-id
                    "seed": rep.meta.get("seed"),
                    "step": cur.step,
                    "player": p,
                    "team": teams[p] if p < len(teams) else None,  # команда текущего action
                    "teams": teams,
                    "is_winner": (p == rep.winner),
                    "winner": rep.winner,
                    "n_players": rep.n_players,
                    "final_score": last_scores.get(p),
                    "rewards": rep.rewards,                   # реворды команд по индексу
                    "source": source,
                },
            }


# --- экспорт (стриминг + resume) ---------------------------------------------
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


def _set_postfix(bar: Any, written: int, skipped: int, errors: int) -> None:
    """Обновляет live-счётчики на tqdm-баре (без построчного логирования)."""
    if hasattr(bar, "set_postfix"):
        bar.set_postfix(written=written, skipped=skipped, errors=errors)


def convert(files: List[str], out_path: str, who: str = "winner",
            skip_empty: bool = False, manifest_path: Optional[str] = None,
            overwrite: bool = False) -> int:
    """Стримит семплы {state, action, meta} (JSONL) для behavioral cloning.

    Реплеи обрабатываются по одному с дозаписью — память плоская независимо от
    числа файлов (датасет ~ГБ в RAM целиком не влезет).

    RESUME: рядом с ``out_path`` ведётся манифест ``<out_path>.manifest`` — по
    одному basename обработанного реплея на строку. Уже перечисленные файлы
    пропускаются ДО загрузки, новые — дописываются в конец. Можно докачать ещё
    реплеев и прогнать конвейер повторно без дублей и без переобработки старого.

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
    bar = tqdm(files, desc=f"convert:{who}", unit="rep")
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
            for sample in iter_samples(rep, who=who, skip_empty=skip_empty, source=key):
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
            out.flush()
            man.write(key + "\n")                 # отмечаем файл обработанным ПОСЛЕ записи
            man.flush()
            done.add(key)
            _set_postfix(bar, written, skipped, errors)
    return written


# --- CLI ---------------------------------------------------------------------
def _expand_inputs(patterns: List[str]) -> List[str]:
    """Разворачивает аргументы --in в список файлов реплеев.

    Каждый аргумент может быть: каталогом (берём *.json и *.json.gz внутри),
    glob-шаблоном или путём к конкретному файлу.
    """
    files: List[str] = []
    for pat in patterns:
        if os.path.isdir(pat):
            files.extend(sorted(glob.glob(os.path.join(pat, "*.json"))))
            files.extend(sorted(glob.glob(os.path.join(pat, "*.json.gz"))))
        else:
            files.extend(sorted(glob.glob(pat)) or [pat])
    return files


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Конвертер реплеев Orbit Wars -> SFT-семплы {state, action, meta} (JSONL)")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="реплеи: файл(ы), каталог или glob-шаблон(ы)")
    ap.add_argument("--out", required=True, help="выходной JSONL-датасет")
    ap.add_argument("--who", default="winner",
                    help="чьи ходы берём: winner | all | top:N (по умолч. winner — "
                         "фильтрация по победам включена)")
    ap.add_argument("--skip-empty", action="store_true",
                    help="пропускать шаги без ходов (по умолч. «hold» сохраняем)")
    ap.add_argument("--manifest",
                    help="путь к манифесту обработанных реплеев (по умолч. <out>.manifest)")
    ap.add_argument("--overwrite", action="store_true",
                    help="игнорировать манифест и начать датасет с нуля")
    args = ap.parse_args(argv)

    files = _expand_inputs(args.inputs)
    if not files:
        ap.error("по --in не найдено ни одного файла реплея")

    n = convert(files, args.out, who=args.who, skip_empty=args.skip_empty,
                manifest_path=args.manifest, overwrite=args.overwrite)
    print(f"готово: записано {n} семплов -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

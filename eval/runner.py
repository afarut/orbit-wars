"""Один эпизод Orbit Wars через настоящий движок ``make('orbit_wars')``.

Агенты передаются уже построенными и в порядке мест (seat 0..n-1); число агентов задаёт
режим (2 = 1v1, 4 = FFA). Итог сводится к:
  * ``rewards`` — сырой reward движка (1 победитель / -1 остальные);
  * ``scores``  — суммарные корабли по игрокам (планеты + флоты) в финале — это и есть
    официальный критерий победы, и он даёт ПОЛНЫЙ порядок мест (нужно для FFA-рейтинга);
  * ``ranks``   — место (0 = лучший, ничьи делят место) для TrueSkill.

Таймауты в конфиге подняты намеренно: локально сравниваем КАЧЕСТВО политики, а не скорость
инференса на CPU — иначе медленный чекпойнт штрафуется за время хода.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from kaggle_environments import make

from .agents import Agent

# щедрые таймауты: не штрафуем за время инференса на CPU
_LOOSE_TIMEOUTS = {"actTimeout": 60, "agentTimeout": 60, "runTimeout": 100000}


@dataclass
class EpisodeResult:
    labels: List[str]            # метки агентов в порядке мест (seat 0..n-1)
    rewards: List[Optional[float]]
    scores: List[float]          # суммарные корабли по игрокам
    ranks: List[int]             # 0 = лучший; ничьи делят место (для TrueSkill)
    statuses: List[str]          # DONE / ERROR / TIMEOUT ...
    seed: int
    n_steps: int


def _final_scores(env, n_players: int) -> List[float]:
    """Суммарные корабли каждого игрока в последнем шаге (планеты + флоты)."""
    obs = env.steps[-1][0]["observation"]      # полное состояние лежит в obs игрока 0
    planets = obs.get("planets") or []
    fleets = obs.get("fleets") or []
    tot = [0.0] * n_players
    for p in planets:
        o = int(p[1])
        if 0 <= o < n_players:
            tot[o] += float(p[5])
    for f in fleets:
        o = int(f[1])
        if 0 <= o < n_players:
            tot[o] += float(f[6])
    return tot


def _ranks_from_scores(scores: Sequence[float]) -> List[int]:
    """Места по убыванию счёта: 0 = лучший, ничьи делят место (standard competition rank)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    prev_score: Optional[float] = None
    prev_rank = 0
    for pos, i in enumerate(order):
        if prev_score is not None and scores[i] == prev_score:
            ranks[i] = prev_rank               # ничья -> то же место
        else:
            ranks[i] = pos
            prev_rank = pos
            prev_score = scores[i]
    return ranks


def run_episode(agents: Sequence[Agent], seed: int, *, episode_steps: int = 500,
                agent_seeds: Optional[Sequence[int]] = None,
                debug: bool = False) -> EpisodeResult:
    """Прогнать один эпизод; ``agents`` уже в порядке мест.

    ``seed`` — посев карты движка (одинаковый -> одинаковая карта). ``agent_seeds`` —
    посевы RNG агентов (для воспроизводимости стохастики); ``None`` -> ``reset()``.
    """
    if agent_seeds is None:
        for a in agents:
            a.reset()
    else:
        for a, s in zip(agents, agent_seeds):
            a.seed(s)
    n = len(agents)
    config = {"episodeSteps": episode_steps, "seed": seed, **_LOOSE_TIMEOUTS}
    env = make("orbit_wars", configuration=config, debug=debug)
    env.run(list(agents))

    last = env.steps[-1]
    rewards = [a.get("reward") for a in last]
    statuses = [a.get("status", "?") for a in last]
    scores = _final_scores(env, n)
    ranks = _ranks_from_scores(scores)
    return EpisodeResult(
        labels=[a.name for a in agents], rewards=rewards, scores=scores,
        ranks=ranks, statuses=statuses, seed=seed, n_steps=len(env.steps),
    )

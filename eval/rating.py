"""Рейтинг по результатам эпизодов: TrueSkill + матрица «финишировал выше».

TrueSkill — та же модель, что на ладдере Kaggle (μ — оценка силы, σ — неопределённость).
Шкала инициализирована под ладдер (μ₀=600), чтобы числа были сопоставимы. Нативно тянет
и 1v1, и многопользовательский FFA: каждый игрок — «команда» из одного, обновление по
полному порядку мест (``EpisodeResult.ranks``, ничьи поддержаны).

Матрица win-rate обобщена до «доля эпизодов, где A финишировал ВЫШЕ B» (по местам): в 1v1
это обычный win-rate, в FFA — попарное превосходство по итоговому месту.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import trueskill

from .runner import EpisodeResult


def make_env(mu: float = 600.0, draw_probability: float = 0.05) -> trueskill.TrueSkill:
    """TrueSkill-окружение в шкале ладдера (μ₀, σ=μ/3, β=μ/6, τ=μ/300)."""
    return trueskill.TrueSkill(
        mu=mu, sigma=mu / 3.0, beta=mu / 6.0, tau=mu / 300.0,
        draw_probability=draw_probability,
    )


@dataclass
class Standing:
    label: str
    mu: float
    sigma: float
    skill: float                 # консервативная оценка μ - 3σ (ранжируем по ней)
    games: int
    wins: int                    # место 0 (с учётом ничьих за 1-е)
    avg_rank: float              # среднее место (0 = лучший)
    placements: Dict[int, int] = field(default_factory=dict)   # место -> сколько раз


def compute_standings(results: Sequence[EpisodeResult], *, mu: float = 600.0,
                      draw_probability: float = 0.05) -> List[Standing]:
    """Прогнать TrueSkill по эпизодам (онлайн, в порядке поступления) -> отсортированные места.

    Сортировка по консервативному skill = μ - 3σ (по убыванию)."""
    env = make_env(mu, draw_probability)
    ratings: Dict[str, trueskill.Rating] = {}
    games: Dict[str, int] = {}
    wins: Dict[str, int] = {}
    rank_sum: Dict[str, int] = {}
    placements: Dict[str, Dict[int, int]] = {}

    for r in results:
        for lab in r.labels:
            ratings.setdefault(lab, env.create_rating())
        groups = [(ratings[lab],) for lab in r.labels]
        updated = env.rate(groups, ranks=list(r.ranks))
        for lab, grp, rank in zip(r.labels, updated, r.ranks):
            ratings[lab] = grp[0]
            games[lab] = games.get(lab, 0) + 1
            rank_sum[lab] = rank_sum.get(lab, 0) + rank
            if rank == 0:
                wins[lab] = wins.get(lab, 0) + 1
            placements.setdefault(lab, {})
            placements[lab][rank] = placements[lab].get(rank, 0) + 1

    out: List[Standing] = []
    for lab, rt in ratings.items():
        g = games.get(lab, 0)
        out.append(Standing(
            label=lab, mu=rt.mu, sigma=rt.sigma, skill=rt.mu - 3.0 * rt.sigma,
            games=g, wins=wins.get(lab, 0),
            avg_rank=(rank_sum.get(lab, 0) / g) if g else 0.0,
            placements=dict(sorted(placements.get(lab, {}).items())),
        ))
    out.sort(key=lambda s: s.skill, reverse=True)
    return out


def above_matrix(results: Sequence[EpisodeResult]) -> Tuple[List[str], List[List[float]]]:
    """(labels, M): M[i][j] — доля совместных эпизодов, где i финишировал ВЫШЕ j (по месту).

    Диагональ — NaN. Ничьи не считаются ни за, ни против (в знаменателе остаются)."""
    labels: List[str] = []
    for r in results:                            # порядок первого появления
        for lab in r.labels:
            if lab not in labels:
                labels.append(lab)
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    above = [[0] * n for _ in range(n)]
    total = [[0] * n for _ in range(n)]
    for r in results:
        for a in range(len(r.labels)):
            for b in range(len(r.labels)):
                if a == b:
                    continue
                ia, ib = idx[r.labels[a]], idx[r.labels[b]]
                total[ia][ib] += 1
                if r.ranks[a] < r.ranks[b]:      # меньше место = выше
                    above[ia][ib] += 1
    mat = [[(above[i][j] / total[i][j]) if total[i][j] else float("nan")
            for j in range(n)] for i in range(n)]
    for i in range(n):
        mat[i][i] = float("nan")
    return labels, mat

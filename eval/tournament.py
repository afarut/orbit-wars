"""Турнир: round-robin между чекпойнтами/эвристиками, 1v1 или 4p FFA.

Матчи:
  * ``1v1`` — все пары агентов; каждая пара играет ``episodes`` карт, на каждой — обе
    рассадки (циклические ротации мест), чтобы убрать позиционное преимущество;
  * ``4p``  — все четвёрки агентов (комбинации); каждая четвёрка на каждой карте играет
    4 циклические ротации мест. Полные перестановки (24) не нужны: карта 4-симметрична.

Посев карты (``seed``) одинаков для всех матчей -> одинаковые карты, честное сравнение.

Параллелизм: эпизоды независимы -> ``multiprocessing`` (fork). Модель чекпойнта грузится
ОДИН раз на воркер (кэш по спеке) и переиспользуется между эпизодами; стохастика
пере-сидируется на каждый эпизод (воспроизводимо при том же ``base_seed``).
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .agents import AgentSpec, build_agent
from .rating import Standing, above_matrix, compute_standings
from .runner import EpisodeResult, run_episode

# (индексы спек по местам, посев карты) — одна задача-эпизод
Task = Tuple[Tuple[int, ...], int]


@dataclass
class TournamentResult:
    mode: str
    results: List[EpisodeResult]
    standings: List[Standing]
    matrix_labels: List[str]
    matrix: List[List[float]]

    @property
    def n_games(self) -> int:
        return len(self.results)


def _cyclic_rotations(group: Sequence[int]) -> List[Tuple[int, ...]]:
    """Все циклические ротации (по числу элементов)."""
    g = list(group)
    return [tuple(g[i:] + g[:i]) for i in range(len(g))]


def generate_tasks(n_specs: int, mode: str, episodes: int, base_seed: int) -> List[Task]:
    """Список задач-эпизодов (индексы спек по местам + посев карты)."""
    if mode == "1v1":
        group_size = 2
    elif mode == "4p":
        group_size = 4
    else:
        raise ValueError(f"режим {mode!r}: ожидается '1v1' или '4p'")
    if n_specs < group_size:
        raise ValueError(f"для {mode} нужно ≥{group_size} агентов, дано {n_specs}")

    tasks: List[Task] = []
    for group in itertools.combinations(range(n_specs), group_size):
        for e in range(episodes):
            seed = base_seed + e
            for seats in _cyclic_rotations(group):
                tasks.append((seats, seed))
    return tasks


# --- воркер (модель грузится раз на процесс) ---------------------------------
_WORKER: Dict[str, object] = {}


def _init_worker(specs: List[AgentSpec], device: str, episode_steps: int) -> None:
    import torch
    torch.set_num_threads(1)                     # параллелим процессами -> без oversubscribe
    _WORKER["specs"] = specs
    _WORKER["device"] = device
    _WORKER["episode_steps"] = episode_steps
    _WORKER["cache"] = {}                        # индекс спеки -> построенный агент


def _get_agent(idx: int):
    cache = _WORKER["cache"]                      # type: ignore[index]
    if idx not in cache:
        spec = _WORKER["specs"][idx]              # type: ignore[index]
        cache[idx] = build_agent(spec, device=_WORKER["device"], seed=0)  # type: ignore[index]
    return cache[idx]


def _run_task(task: Task) -> EpisodeResult:
    seats, seed = task
    agents = [_get_agent(i) for i in seats]
    agent_seeds = [seed * 31 + pos for pos in range(len(seats))]
    return run_episode(agents, seed=seed, agent_seeds=agent_seeds,
                       episode_steps=_WORKER["episode_steps"])  # type: ignore[index]


def run_tournament(specs: Sequence[AgentSpec], *, mode: str = "1v1", episodes: int = 20,
                   base_seed: int = 0, episode_steps: int = 500, workers: int = 0,
                   device: str = "cpu",
                   progress=None) -> TournamentResult:
    """Прогнать турнир. ``workers``: 0 -> авто (cpu-2), 1 -> последовательно.

    ``progress`` — необязательный callback(done, total) для отображения прогресса."""
    specs = list(specs)
    if len(set(s.label for s in specs)) != len(specs):
        raise ValueError("метки агентов должны быть уникальны")
    tasks = generate_tasks(len(specs), mode, episodes, base_seed)
    total = len(tasks)

    if workers == 0:
        workers = max(1, (mp.cpu_count() or 2) - 2)

    results: List[EpisodeResult] = []
    if workers == 1:
        _init_worker(list(specs), device, episode_steps)
        for i, t in enumerate(tasks):
            results.append(_run_task(t))
            if progress:
                progress(i + 1, total)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers, initializer=_init_worker,
                      initargs=(list(specs), device, episode_steps)) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_task, tasks)):
                results.append(r)
                if progress:
                    progress(i + 1, total)

    standings = compute_standings(results)
    labels, matrix = above_matrix(results)
    return TournamentResult(mode=mode, results=results, standings=standings,
                            matrix_labels=labels, matrix=matrix)

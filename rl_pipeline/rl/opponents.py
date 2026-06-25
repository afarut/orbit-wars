"""Адаптивный пул оппонентов.

Для каждого оппонента отслеживает win_rate нашего агента против него.
Агенты с более высоким loss_rate (мы чаще проигрываем) выбираются чаще.

Поддерживает:
  - Rust built-in agents (GREEDY / PRODUCER / DEFENDER) через VecEnv.step_p0
  - Self-play: пул прошлых чекпойнтов (TODO — добавить позже)

Формат:
  agent_id = int  (совпадает с ow_rs.AGENT_* константами)
  special ids:
    -1  = самоигра против текущей модели (добавить позже)
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class OpponentPool:
    """Адаптивный выбор оппонентов по loss rate.

    Вероятность выбора P(i) ∝ exp(loss_rate(i) / τ).
    При τ→∞ — равномерно; при τ→0 — фокус на самом сложном.
    """

    def __init__(
        self,
        agent_ids: List[int],          # список доступных ow_rs.AGENT_* id
        temperature: float = 0.5,
        ema_alpha: float = 0.05,       # скорость обновления EMA win_rate
        min_games_to_weight: int = 20, # до этого — равномерно
        num_agents: int = 2,           # число игроков (2 или 4)
    ):
        self.agent_ids = list(agent_ids)
        self.temperature = temperature
        self.ema_alpha = ema_alpha
        self.min_games = min_games_to_weight
        self.num_agents = num_agents

        # 1v1: один слот оппонента
        # 4p:  три слота оппонентов
        self.opp_slots = num_agents - 1

        self._win_rate: Dict[int, float] = {i: 0.5 for i in agent_ids}
        self._games: Dict[int, int] = {i: 0 for i in agent_ids}

    @property
    def stats(self) -> Dict[int, Dict]:
        return {
            aid: {"win_rate": self._win_rate[aid], "games": self._games[aid]}
            for aid in self.agent_ids
        }

    def _probs(self) -> np.ndarray:
        """Вероятности выбора агентов (softmax по loss rate / τ)."""
        n = len(self.agent_ids)
        if all(self._games[i] < self.min_games for i in self.agent_ids):
            return np.ones(n) / n
        loss_rates = np.array([1.0 - self._win_rate[i] for i in self.agent_ids])
        logits = loss_rates / max(self.temperature, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        return probs / probs.sum()

    def sample_kinds(self, num_envs: int) -> List[List[int]]:
        """Для каждого env вернуть список agent_id оппонентов (длина opp_slots)."""
        # Если слотов >= агентов — берём всех (shuffle) и добираем с повтором до opp_slots.
        # (FFA с одним типом оппонента [PL] и opp_slots=3 → [PL, PL, PL], а не [PL].)
        if self.opp_slots >= len(self.agent_ids):
            out = []
            for _ in range(num_envs):
                base = random.sample(self.agent_ids, len(self.agent_ids))
                while len(base) < self.opp_slots:
                    base.append(random.choice(self.agent_ids))
                out.append(base[:self.opp_slots])
            return out
        probs = self._probs()
        kinds = []
        for _ in range(num_envs):
            env_kinds = [
                int(np.random.choice(self.agent_ids, p=probs))
                for _ in range(self.opp_slots)
            ]
            kinds.append(env_kinds)
        return kinds

    def flat_kinds(self, kinds: List[List[int]]) -> List[int]:
        """Развернуть [[k00,k01], [k10,k11], ...] в плоский список для set_opponents."""
        return [k for env_k in kinds for k in env_k]

    def update(
        self,
        kinds: List[List[int]],   # [N, opp_slots]
        rewards: np.ndarray,      # [N, A] int32 — финальные наградные сигналы
        dones: np.ndarray,        # [N] bool — завершившиеся эпизоды
    ) -> None:
        """Обновить win_rate по завершённым эпизодам."""
        for e in range(len(dones)):
            if not dones[e]:
                continue
            # Наша награда — rewards[e, 0]
            our_r = float(rewards[e, 0])
            win = 1.0 if our_r > 0 else 0.0 if our_r < 0 else 0.5
            # Обновляем EMA win_rate для каждого оппонента в этом env
            for aid in set(kinds[e]):
                self._games[aid] += 1
                self._win_rate[aid] = (
                    (1 - self.ema_alpha) * self._win_rate[aid]
                    + self.ema_alpha * win
                )

    def log_line(self) -> str:
        parts = []
        for aid in self.agent_ids:
            name = {0: "hold", 4: "producer_lite"}.get(aid, str(aid))
            wr = self._win_rate[aid]
            g = self._games[aid]
            parts.append(f"{name}={wr:.2f}({g}g)")
        return "opp_wr: " + "  ".join(parts)

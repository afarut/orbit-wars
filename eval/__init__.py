"""Локальный турнир Orbit Wars: стравливание чекпойнтов и эвристик между собой.

Offline-only (как ``sft/``/``dataprep/``) — тянет ``kaggle_environments`` и ``trueskill``,
в сабмишн НЕ попадает. Запуск через настоящий движок ``make('orbit_wars')``:

  * :mod:`eval.agents`     — единый интерфейс агента ``callable(obs, config) -> ходы``;
  * :mod:`eval.runner`     — один эпизод (1v1 / 4p FFA) -> результат с местами;
  * :mod:`eval.rating`     — TrueSkill + матрица win-rate;
  * :mod:`eval.tournament` — round-robin / challenger, параллельно по эпизодам.

CLI: ``.venv/bin/python -m eval ...`` (см. :mod:`eval.__main__`).
"""

import logging as _logging

# kaggle_environments при импорте грузит OpenSpiel, который сам ставит своему логгеру
# уровень INFO + свой handler на stdout и печатает ~30 строк. Уровень родителя его не
# гасит; помогает только disabled=True, выставленный ДО импорта kaggle_environments
# (logger — синглтон по имени; модуль не трогает .disabled).
_logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").disabled = True

from .agents import (Agent, AgentSpec, CheckpointAgent, HeuristicAgent,
                     build_agent, parse_spec, spec_from_config)

__all__ = ["Agent", "AgentSpec", "CheckpointAgent", "HeuristicAgent",
           "build_agent", "parse_spec", "spec_from_config"]

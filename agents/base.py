"""Общая база скриптовых ботов в ``agents/`` — интерфейс eval (``callable(obs, config)``).

Здесь же — настройка ``sys.path`` для пакета ``orbit_lite`` (живёт в
``producer-orbit-wars-utils/``, без ``setup.py``); зеркало ``core/geo_lite.py``. Импорт этого
модуля делает ``import orbit_lite`` рабочим, поэтому каждый бот импортирует базу ДО
``from orbit_lite ...``.

Модуль офлайновый (как и весь ``agents/``): в сабмишн не входит, нужен только для локального
турнира ``eval``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

# корень пакета orbit_lite в sys.path (нет setup.py) — как в core/geo_lite.py
_PKG = Path(__file__).resolve().parents[1] / "producer-orbit-wars-utils"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

Move = List[float]   # [from_planet_id, angle_rad, num_ships] — как в eval.agents


class ScriptedAgent:
    """База скриптового бота, совместимая с ``eval`` (дак-тайпинг под ``eval.agents.Agent``).

    Контракт: ``__call__(obs, config) -> список ходов``; раннер зовёт ``.name``,
    ``.seed(int)`` и ``.reset()`` по-эпизодно (``eval/runner.py``). Боты детерминированы,
    поэтому ``seed`` лишь хранит посев (паритет интерфейса), а ``reset`` сбрасывает
    накопленный за эпизод кэш forecast'а — переопределяется наследником.
    """

    name: str = "scripted"

    def __init__(self, name: str | None = None, *, seed: int = 0) -> None:
        if name:
            self.name = name
        self.seed_val = int(seed)

    def seed(self, s: int) -> None:
        self.seed_val = int(s)

    def reset(self) -> None:
        """Сброс к началу эпизода (наследник чистит рантайм)."""

    def __call__(self, obs: Any, config: Any = None) -> List[Move]:
        raise NotImplementedError

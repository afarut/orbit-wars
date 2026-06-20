"""Скриптовые боты Orbit Wars под интерфейс eval (офлайн, в сабмишн НЕ входят).

Перенесены из Kaggle-ноутбуков (планировщик «Producer / flow-diff» поверх ``orbit_lite``);
каждый — наследник :class:`agents.base.ScriptedAgent` (``callable(obs, config) -> ходы``).
``SCRIPTED_AGENTS`` — реестр по имени; его читает ``eval.agents.build_agent`` для спеки
``scripted:<name>`` (см. ``configs/pool/scripted.yaml``).
"""

from agents.apex_master import ApexMasterAgent
from agents.base import ScriptedAgent
from agents.producer_hybrid import ProducerHybridAgent

SCRIPTED_AGENTS = {
    "producer_hybrid": ProducerHybridAgent,
    "apex_master": ApexMasterAgent,
}

__all__ = ["ScriptedAgent", "ProducerHybridAgent", "ApexMasterAgent", "SCRIPTED_AGENTS"]

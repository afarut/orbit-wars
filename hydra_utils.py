"""Общие утилиты для всех Hydra-входов (офлайн; зависит только от omegaconf).

Лёгкий модуль без тяжёлых импортов — чтобы и тренировка (`sft`), и эвал (`eval`)
могли печатать конфиг, не таща пакет друг друга.
"""
from __future__ import annotations

from omegaconf import OmegaConf


def print_cfg(cfg, tag: str) -> None:
    """Печать всего конфига Hydra в начале запуска. Если есть незаданные
    интерполяции/обязательные поля (напр. eval-овский `ckpt_dir`), `resolve=True`
    падает — тогда печатаем нерезолвленный вид, а не роняем запуск."""
    try:
        text = OmegaConf.to_yaml(cfg, resolve=True)
    except Exception:
        text = OmegaConf.to_yaml(cfg)            # есть MISSING/${...} -> без резолва
    print(f"[{tag}] полный конфиг:\n{text}", flush=True)

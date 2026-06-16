r"""Точка входа SFT-обучения (Hydra).

Запуск:
  # одиночный процесс (Mac MPS / CPU / одна CUDA — выбор авто)
  python -m sft.train
  # несколько GPU Nvidia (DDP)
  torchrun --standalone --nproc_per_node=N sft/train.py
  # оверрайды Hydra
  python -m sft.train train.batch_size=512 train.lr=1e-4 data.w_hold=0.05
"""
from __future__ import annotations

# --- bootstrap: корень репо в sys.path (для запуска и как `-m`, и как файл torchrun) ---
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hydra

from sft import engine

_CONFIGS = str(_ROOT / "configs")


@hydra.main(version_base=None, config_path=_CONFIGS, config_name="sft")
def main(cfg) -> None:
    engine.run(cfg)


if __name__ == "__main__":
    main()

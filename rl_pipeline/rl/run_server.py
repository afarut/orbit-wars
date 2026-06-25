"""Запуск RL без hydra (на сервере, где hydra/omegaconf нет).

  python -m rl.run_server [key=value ...]

Грузит configs/rl_train.yaml + defaults (configs/model/...), мерджит, поддерживает
dot-path оверрайды (device=cuda, rl.lr=1e-4 и т.п.), зовёт rl.train.run(cfg).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


class Cfg(dict):
    """dict с доступом по атрибуту + .get() (минимальная замена OmegaConf DictConfig)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return Cfg(v) if isinstance(v, dict) else v

    def get(self, k, d=None):
        v = dict.get(self, k, d)
        return Cfg(v) if isinstance(v, dict) else v


def _load(p: Path):
    with open(p) as f:
        return yaml.safe_load(f)


def build_cfg(config_name: str = "rl_train") -> dict:
    cdir = Path(__file__).resolve().parent.parent / "configs"
    raw = _load(cdir / f"{config_name}.yaml")
    defaults = raw.pop("defaults", []) or []
    cfg: dict = {}
    for d in defaults:
        if d == "_self_":
            continue
        if isinstance(d, dict):
            for group, choice in d.items():
                cfg[group] = _load(cdir / group / f"{choice}.yaml")
    for k, v in raw.items():          # _self_ поверх defaults
        cfg[k] = v
    return cfg


def _apply_override(cfg: dict, dotted: str, val: str) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    try:
        d[keys[-1]] = yaml.safe_load(val)   # парсим тип (int/float/bool/str)
    except Exception:
        d[keys[-1]] = val


def main() -> None:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = build_cfg()
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            _apply_override(cfg, k, v)
    from rl.train import run
    run(Cfg(cfg))


if __name__ == "__main__":
    main()

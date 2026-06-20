r"""Точка входа локального турнира Orbit Wars (Hydra).

Запуск:
  # пул по умолчанию (бейзлайны-эвристики), 1v1
  python -m eval.run
  # 4p FFA, больше карт, фикс. посев для воспроизводимости
  python -m eval.run mode=4p episodes=25 seed=123
  # выбор агентов «флажками» — список имён из каталога configs/agent/
  python -m eval.run roster=[best, bestT, sniper] ckpt_dir=outputs/<ts>/checkpoints
  # готовый пресет ростера из configs/pool/<name>.yaml
  python -m eval.run pool=scripted
  # ad-hoc чекпойнты из разных прогонов: имя в ростере -> путь через `ckpts`
  python -m eval.run roster=[runA,runB,sniper] ckpts='{runA: outputs/A/checkpoints/best.pt, runB: outputs/B/checkpoints/best.pt}'
  # разовый inline-список (переопределяет roster)
  python -m eval.run 'agents=[{label: best, ckpt: outputs/ts/checkpoints/best.pt}, {label: sn, heuristic: sniper}]'

Каталог: каждый agent/<name>.yaml -> узел ``catalog.<name>``; `roster` выбирает имена
(имя не из каталога ищется в ``ckpts`` как ad-hoc путь к чекпойнту).
Спека агента: ``{ckpt, decode?, temperature?}`` (decode=greedy|sample; путь обычно
``${ckpt_dir}/...``) либо ``{heuristic}`` (sniper|full_send|random|hold) либо ``{scripted}``.
Ярлык в таблице = имя в ростере (если в файле нет своего `label`). Hydra меняет cwd на
run-dir — относительные пути чекпойнтов резолвятся к ИСХОДНОМУ cwd.
"""

from __future__ import annotations

# --- bootstrap: корень репо в sys.path (как у sft/train.py) ---
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import random
import time
from dataclasses import asdict, replace
from typing import List

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationToMissingValueError, MissingMandatoryValue

from eval.agents import AgentSpec, spec_from_config
from eval.rating import Standing
from eval.tournament import TournamentResult, run_tournament

_CONFIGS = str(_ROOT / "configs")


# --- печать -------------------------------------------------------------------
def _fmt_standings(standings: List[Standing], mode: str) -> str:
    head = (f"{'#':>2}  {'agent':<14} {'skill':>7} {'mu':>7} {'sigma':>6} "
            f"{'games':>6} {'wins':>5} {'avg_rank':>8}")
    lines = [head, "-" * len(head)]
    for i, s in enumerate(standings, 1):
        lines.append(f"{i:>2}  {s.label:<14} {s.skill:>7.1f} {s.mu:>7.1f} {s.sigma:>6.1f} "
                     f"{s.games:>6} {s.wins:>5} {s.avg_rank:>8.2f}")
    if mode == "4p":
        lines.append("\nраспределение мест (0=1-е):")
        for s in standings:
            places = " ".join(f"{k}:{v}" for k, v in s.placements.items())
            lines.append(f"  {s.label:<14} {places}")
    return "\n".join(lines)


def _fmt_matrix(labels: List[str], matrix: List[List[float]]) -> str:
    head = " " * 10 + " ".join(f"{l[:7]:>7}" for l in labels)
    lines = ["доля эпизодов, где строка финишировала ВЫШЕ столбца:", head]
    for i, lab in enumerate(labels):
        cells = ["    -- " if v != v else f"{v:>7.2f}" for v in matrix[i]]
        lines.append(f"{lab[:9]:<9} " + " ".join(cells))
    return "\n".join(lines)


def _save(out_dir: str, tr: TournamentResult, cfg, seed: int,
          specs: List[AgentSpec]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for r in tr.results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    ckpt_dir = None if OmegaConf.is_missing(cfg, "ckpt_dir") else cfg.ckpt_dir
    summary = {
        "mode": tr.mode, "n_games": tr.n_games, "seed": seed,
        "config": {"mode": cfg.mode, "episodes": cfg.episodes, "steps": cfg.steps,
                   "workers": cfg.workers, "device": cfg.device,
                   "roster": list(cfg.roster), "ckpt_dir": ckpt_dir},
        "agents": [asdict(s) for s in specs],
        "standings": [asdict(s) for s in tr.standings],
        "matrix_labels": tr.matrix_labels, "matrix": tr.matrix,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] результаты -> {out_dir}/ (episodes.jsonl, summary.json)")


def _resolve_name(cfg, name: str):
    """Узел для имени из ростера: сперва каталог (configs/agent/), затем ad-hoc
    чекпойнты `ckpts` (имя -> путь | {ckpt, decode?, temperature?}). Понятная ошибка,
    если имени нет нигде."""
    if name in cfg.catalog:
        return cfg.catalog[name]
    ckpts = OmegaConf.select(cfg, "ckpts", default=None) or {}
    if name in ckpts:
        v = ckpts[name]
        return {"ckpt": v} if isinstance(v, str) else v   # голый путь -> greedy-чекпойнт
    raise SystemExit(f"[eval] агент {name!r} не найден ни в каталоге (configs/agent/), ни в "
                     f"ckpts. каталог: {sorted(cfg.catalog.keys())}"
                     + (f"; ckpts: {sorted(ckpts.keys())}" if ckpts else ""))


def _build_specs(cfg) -> List[AgentSpec]:
    """Спеки агентов из `roster` (имена каталога configs/agent/ или ad-hoc `ckpts`),
    либо из явного inline-списка `agents=[...]` (переопределяет ростер). Ярлык = имя
    в ростере (если в записи нет своего `label`). Пути чекпойнтов резолвим к исходному
    cwd (hydra сменил cwd)."""
    inline = OmegaConf.select(cfg, "agents", default=None)
    named = ([(None, e) for e in inline] if inline
             else [(n, _resolve_name(cfg, n)) for n in cfg.roster])

    specs: List[AgentSpec] = []
    for name, entry in named:
        try:
            s = spec_from_config(entry)         # читает ckpt -> тут всплывёт незаданный ckpt_dir
        except (MissingMandatoryValue, InterpolationToMissingValueError):
            raise SystemExit(
                f"[eval] агент {name!r} — чекпойнт, но не задан ckpt_dir. Запусти с "
                f"`ckpt_dir=outputs/<ts>/checkpoints` (или точечно `catalog.{name}.ckpt=<путь>`).")
        if name and not (isinstance(entry, str) or entry.get("label")):
            s = replace(s, label=str(name))     # ярлык по имени в ростере
        if s.kind == "checkpoint":
            s = replace(s, ckpt_path=to_absolute_path(s.ckpt_path))
        specs.append(s)
    return specs


@hydra.main(version_base=None, config_path=_CONFIGS, config_name="eval")
def main(cfg) -> None:
    seed = cfg.seed
    if seed is None:                              # спавн симметричен -> по умолчанию случайно
        seed = random.randint(0, 2**31 - 1)
    specs = _build_specs(cfg)

    print(f"[eval] режим={cfg.mode} агентов={len(specs)} episodes={cfg.episodes} "
          f"steps={cfg.steps} workers={cfg.workers or 'auto'} seed={seed}")
    for s in specs:
        if s.kind == "checkpoint":
            desc = (f"checkpoint {s.ckpt_path} ({s.decode}"
                    f"{f', T={s.temperature}' if s.decode == 'sample' else ''})")
        elif s.kind == "scripted":
            desc = f"scripted:{s.scripted}"
        else:
            desc = f"heuristic:{s.heuristic}"
        print(f"   {s.label:<14} {desc}")

    t0 = time.time()

    def progress(done, total):
        if done == total or done % max(1, total // 20) == 0:
            print(f"\r[eval] {done}/{total} эпизодов ({time.time()-t0:.0f}с)",
                  end="", file=sys.stderr, flush=True)

    tr = run_tournament(specs, mode=cfg.mode, episodes=cfg.episodes, base_seed=seed,
                        episode_steps=cfg.steps, workers=cfg.workers,
                        device=cfg.device, progress=progress)
    print(file=sys.stderr)
    print(f"\n[eval] {tr.n_games} игр за {time.time()-t0:.0f}с\n")
    print(_fmt_standings(tr.standings, tr.mode))
    print()
    print(_fmt_matrix(tr.matrix_labels, tr.matrix))

    # hydra 1.3 при version_base=None по умолчанию НЕ меняет cwd -> берём run-dir явно
    run_dir = HydraConfig.get().runtime.output_dir
    out_dir = to_absolute_path(cfg.out) if cfg.out else run_dir
    _save(out_dir, tr, cfg, seed, specs)


if __name__ == "__main__":
    main()

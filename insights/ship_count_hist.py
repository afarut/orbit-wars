"""Гистограмма числа кораблей в отправляемых флотах по датасету samples.full_send.

Считает два распределения по всем семплам data/samples.full_send.jsonl:
  * per-fleet  — число кораблей в одном отдельном флоте (move[2]);
  * per-sample — суммарно кораблей, отправленных за один ход (сумма по флотам семпла).
Хвост тяжёлый (до ~18k), поэтому бины логарифмические, ось X в лог-масштабе.
Картинку кладём в insights/ (конвенция проекта — разборы датасета самодостаточны).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "samples.full_send.jsonl"
OUT = ROOT / "insights" / "ship_count_hist.png"

per_fleet = []   # корабли в одном флоте
per_sample = []  # корабли суммарно за ход (только непустые)
n_total = n_empty = 0
with SRC.open() as f:
    for line in f:
        act = json.loads(line)["action"] or []
        n_total += 1
        if not act:
            n_empty += 1
            continue
        tot = 0
        for mv in act:
            per_fleet.append(mv[2])
            tot += mv[2]
        per_sample.append(tot)

pf = np.array(per_fleet, dtype=float)
ps = np.array(per_sample, dtype=float)

# Лог-бины от 1 до максимума
bins = np.logspace(0, np.log10(max(pf.max(), ps.max())) + 0.05, 50)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, data, title, color in (
    (axes[0], pf, "Кораблей в одном флоте (per-fleet)", "#3a7ca5"),
    (axes[1], ps, "Кораблей за один ход, сумма (per-sample)", "#d1495b"),
):
    ax.hist(data, bins=bins, color=color, edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("число кораблей (лог-шкала)")
    ax.set_ylabel("частота (число событий)")
    ax.set_title(title)
    med, mean = np.median(data), data.mean()
    ax.axvline(med, color="black", ls="--", lw=1, label=f"медиана {med:.0f}")
    ax.axvline(mean, color="black", ls=":", lw=1, label=f"среднее {mean:.0f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

share_act = 100 * (n_total - n_empty) / n_total
fig.suptitle(
    f"Orbit Wars — распределение числа отправляемых кораблей "
    f"(samples.full_send: {n_total:,} семплов, {n_empty/n_total*100:.0f}% hold, "
    f"{len(pf):,} флотов)".replace(",", " "),
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=130)
print("saved:", OUT)
print(f"семплов {n_total}, hold {n_empty} ({100-share_act:.1f}%), флотов {len(pf)}")
print(f"per-fleet:  медиана {np.median(pf):.0f}, среднее {pf.mean():.1f}, max {pf.max():.0f}")
print(f"per-sample: медиана {np.median(ps):.0f}, среднее {ps.mean():.1f}, max {ps.max():.0f}")

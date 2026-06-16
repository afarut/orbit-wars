"""Гистограмма ЧИСЛА ФЛОТОВ в полёте на состояние — по датасету samples.full_send.

Считаем НЕ корабли, а флоты: один запуск = один флот-объект (state.fleets),
флот из 60 кораблей = 1 флот. Метрика на семпл — len(state.fleets):
сколько отдельных флотов одновременно летит по доске в этом состоянии.
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
OUT = ROOT / "insights" / "fleet_count_hist.png"

counts = []
with SRC.open() as f:
    for line in f:
        counts.append(len(json.loads(line)["state"].get("fleets") or []))
c = np.array(counts)

med, mean = np.median(c), c.mean()
p99 = np.percentile(c, 99)
CAP = 120  # хвост за CAP схлопываем в последний бин (он тонкий: p99=112)
clipped = np.minimum(c, CAP)
bins = np.arange(0, CAP + 2, 2)

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.hist(clipped, bins=bins, color="#4c72b0", edgecolor="white", linewidth=0.3)
ax.set_xlabel("число флотов в полёте за состояние, len(state.fleets)")
ax.set_ylabel("частота (число семплов)")
ax.axvline(med, color="black", ls="--", lw=1.2, label=f"медиана {med:.0f}")
ax.axvline(mean, color="black", ls=":", lw=1.2, label=f"среднее {mean:.1f}")
ax.axvline(p99, color="#c44e52", ls="-.", lw=1, label=f"p99 {p99:.0f}")
share0 = 100 * (c == 0).mean()
over = 100 * (c > CAP).mean()
ax.text(0.98, 0.7,
        f"0 флотов: {share0:.0f}% семплов\nмакс: {c.max()} флотов\nхвост >{CAP}: {over:.2f}% (в посл. бине)",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", fc="#fff5e6", ec="#cccccc"))
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.set_title(
    f"Orbit Wars — сколько флотов одновременно летит (samples.full_send, "
    f"{len(c):,} семплов)".replace(",", " "))
fig.tight_layout()
fig.savefig(OUT, dpi=130)

print("saved:", OUT)
print(f"семплов {len(c)}, с 0 флотов {share0:.1f}%, медиана {med:.0f}, среднее {mean:.2f}, max {c.max()}")
for q in (50, 75, 90, 95, 99):
    print(f"  p{q}: {np.percentile(c, q):.0f}")

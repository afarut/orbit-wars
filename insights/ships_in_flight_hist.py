"""Гистограмма числа кораблей, которые РЕАЛЬНО ЛЕТЯТ (в полёте) — по датасету samples.full_send.

Источник — НЕ action (это намерение эксперта отправить), а state.fleets:
каждый летящий флот закодирован как (id, owner, x, y, angle, from_planet_id, ships),
последнее поле ships — сколько кораблей физически в пути к цели.

Считаем два распределения по всем семплам data/samples.full_send.jsonl:
  * per-fleet  — кораблей в одном летящем флоте (Fleet.ships);
  * per-state  — суммарно кораблей в полёте в данном состоянии (сумма по всем флотам семпла).
Хвост тяжёлый, поэтому бины логарифмические, ось X в лог-масштабе.
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
OUT = ROOT / "insights" / "ships_in_flight_hist.png"

SHIPS_IDX = 6  # индекс поля ships в кортеже Fleet

per_fleet = []   # кораблей в одном летящем флоте
per_state = []   # кораблей в полёте суммарно за состояние
n_total = n_nofleets = 0
with SRC.open() as f:
    for line in f:
        fleets = json.loads(line)["state"].get("fleets") or []
        n_total += 1
        if not fleets:
            n_nofleets += 1
        tot = 0
        for fl in fleets:
            per_fleet.append(fl[SHIPS_IDX])
            tot += fl[SHIPS_IDX]
        per_state.append(tot)

pf = np.array(per_fleet, dtype=float)
ps = np.array(per_state, dtype=float)
psn = ps[ps > 0]  # состояния, где хоть кто-то летит

bins = np.logspace(0, np.log10(max(pf.max(), ps.max())) + 0.05, 50)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, data, title, color in (
    (axes[0], pf, "Кораблей в одном летящем флоте (per-fleet)", "#2a9d8f"),
    (axes[1], psn, "Кораблей в полёте за состояние (per-state)", "#e76f51"),
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

fig.suptitle(
    f"Orbit Wars — корабли в полёте (state.fleets), датасет samples.full_send: "
    f"{n_total:,} семплов, {n_nofleets/n_total*100:.0f}% без летящих флотов, "
    f"{len(pf):,} флотов".replace(",", " "),
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=130)

print("saved:", OUT)
print(f"семплов {n_total}, без флотов {n_nofleets} ({n_nofleets/n_total*100:.1f}%), флотов всего {len(pf)}")
print(f"per-fleet:  медиана {np.median(pf):.0f}, среднее {pf.mean():.1f}, max {pf.max():.0f}")
print(f"per-state:  медиана {np.median(psn):.0f}, среднее {psn.mean():.1f}, max {psn.max():.0f}")
for q in (50, 75, 90, 95, 99):
    print(f"  per-fleet p{q}: {np.percentile(pf, q):.0f}   per-state p{q}: {np.percentile(psn, q):.0f}")

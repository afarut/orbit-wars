# CLAUDE.md — producer-orbit-wars-utils

Guidance for Claude Code when working in this directory.

## What this is

`orbit_lite` — a self-contained **heuristic** agent for the Kaggle **Orbit Wars** competition,
written with `torch` + the standard library only. It is a *forward-simulator + greedy flow-diff
planner* ("speed-first flow-diff producer"): it forecasts the game state assuming no action, scores
hypothetical launches by their effect on each player's net ship flow, and greedily commits the best
non-conflicting launches.

This is a **different approach** from the parent repo (`../`), which is an SFT / behavioral-cloning
neural net (`model.py` + `core/`). Nothing here is trained — it is pure rule-based search faithful to
the game engine. The two agents are independent; this package does not import from the parent repo.

Comments and docstrings in the code are written in **English** here; keep that when editing. (Note:
the parent repo's convention is Russian — this package is the exception.)

## Status / incompleteness (important)

This directory holds only the `orbit_lite` package plus `USAGE.md`. Two pieces referenced by the code
and docs are **not present**:

- **`main.py`** — the Kaggle entry point with `agent(obs) -> [[from_planet_id, angle, ships], ...]`.
  `USAGE.md` tells the notebook to load it from `/kaggle/input/orbit-lite/main.py`, but it is missing.
- **The top-level orchestrator** — `planner_core.py` supplies the building blocks (shortlist, score,
  select, regroup, payload), but the `plan(obs) -> payload` function that sequences them
  (`build_target_shortlist → score_candidates → _greedy_select → _plan_regroup →
  entries_to_sparse_payload`) lives in a driver module that was not included.
- A config dataclass referenced by `planner_core` (`config.max_offensive_targets`,
  `regroup_*`, `roi_threshold`, etc.) is also external to this archive.

If asked to run the agent end-to-end, the driver + `main.py` + config must be reconstructed first.

## The planning pipeline (as assembled from these blocks)

Each turn the agent conceptually does:

1. **Forecast** a "do-nothing" projection of every planet's owner/ship count over a horizon `H` (~20),
   plus future planet/comet positions and tracked in-flight fleets (`movement.py`).
2. **Build a target shortlist** (`planner_core.build_target_shortlist`): nearest enemy/neutral planets
   (offensive) ∪ own planets the projection shows flipping to an enemy (defensive, by urgency).
3. **Score candidate launches** by the *competitive* metric `Δnet_me − Σ Δnet_opponents`, computed
   with the exact sparse flow projector (`garrison_launch.sparse_launch_flow_delta`).
4. **Greedily select** non-conflicting waves (`planner_core._greedy_select`): best score first, above an
   ROI threshold, one wave per target, debiting a per-planet ship budget, with a source/reinforce mutex.
5. **Regroup leftovers** (`planner_core._plan_regroup`): marshal uncommitted ships along a pressure
   gradient toward more-threatened owned planets.
6. **Emit the sparse payload** (`planner_core.entries_to_sparse_payload`) → decoded to a move list by
   `adapter.sparse_action_row_to_moves`.

## Module map

Runtime / submission layer (numpy-free; `torch` + stdlib):

- `adapter.py` — obs-dict ↔ tensor bridge; sparse payload → `[from_planet_id, angle, ships]` move list,
  with ownership / ship-count validation.
- `constants.py` — engine physics (`BOARD_SIZE=100`, `SUN_RADIUS=10`, `MAX_SHIP_SPEED=6`), comet schedule,
  and early-termination thresholds (calibrated on 535 replays).
- `obs.py` — parse raw `[P,7]`/`[F,7]` tensors into the named `ParsedObs` (ownership masks, orbital
  params). Field indices live *only* here.
- `geometry.py` — fleet-speed formula `1+(MAX_SHIP_SPEED-1)·(log(ships)/log(1000))^1.5` via a LUT with a
  per-(device,dtype) cache (avoids a CUDA host-sync per call).
- `movement.py` (largest) — the forward predictor: orbital mechanics, comet paths, fleet tracking by
  engine fleet-id, swept-circle collision (planets / sun / OOB), and the production→combat recurrence
  that yields `PlanetGarrisonStatus`. Handles fleet-id reconciliation (the engine assigns IDs the agent
  can't know at action time → stash-and-match against the next obs).
- `distance_cache.py` — cross-time distances `dist(s@0, t@k)` for moving-target reachability.
- `movement_aiming.py` / `aiming.py` — swept-pair hit test; obs `step` → orbit phase index (`max(0, step-1)`).
- `intercept_aim.py` — lead-angle aim at an orbiting target: continuous fixed-point intercept time +
  analytic first-contact verification matching the engine's verdict exactly.
- `garrison_launch.py` — exact per-player net-ship flow projector ("if I launch these, how does each
  player's produced − combat-lost change?"); recomputes only the planets a launch touches.
- `movement_step.py` — `LaunchEntries` / `PlannedLaunches` tables, concat, duplicate-angle
  disambiguation, binding launches to the movement cache.
- `planner_core.py` — the heuristics: target scoring, `capture_floor` (ships needed to take a target),
  `safe_drain` (never leave a source losable within the horizon), `_greedy_select`, `_plan_regroup`,
  payload formatting.

## Conventions to preserve when editing

- **CPU≡CUDA determinism**: all argmax/topk use ascending-index tie-breaks (`_stable_argmax`,
  `_stable_topk_indices`) so the agent picks identical moves on any device. Do not introduce
  nondeterministic reductions.
- **Engine faithfulness**: collision order (planet hit resolves before OOB/sun in the same step),
  lowest-slot same-step tie-break, and the launch surface offset (`0.1`) all mirror the real engine so
  the forecast matches simulation. Changing physics here silently desyncs the planner.
- **Single-game shapes**: planets are `[P,7]`, no batch axis. Some docstrings say `[*prefix, ...]`, but
  the implementation is single-game; the "prefix" is only candidate/launch reshaping.
- **Frame conventions** (off-by-one hot-spot): `k=0` is the observation frame, `k=1..H` future steps;
  `fleet_buckets` index `j` ↔ arrival step `k=j+1` (eta=1 → index 0); garrison caches carry an extra
  `k=0` slot (`H+1` time axis).

## No test harness here

There is no `smoke_test.py` / pytest in this directory. When touching the physics or planner, validate
against the parent repo's engine (`kaggle_environments` `orbit_wars` 1.0.9, per `../CLAUDE.md`) or
reconstruct the missing driver to exercise the pipeline end-to-end.

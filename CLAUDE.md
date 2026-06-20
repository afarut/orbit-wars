# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An imitation-learning (SFT / behavioral cloning) bot for the Kaggle **Orbit Wars** competition. A
set-transformer policy/value net is trained on public leaderboard replays to imitate strong players,
then submitted as a `act(obs) -> moves` agent. See `orbit_wars_rules.md` and `insights/*.md` for the
game mechanics the features and decoder encode.

Comments and docstrings are written in **Russian** and kept **brief** — match that when editing.

## Submission boundary (important)

Two halves of the repo that must not blur together:

- **Runtime / submission**: `model.py` + `core/` (numpy + torch only). This is what runs inside the
  Kaggle environment at inference. `core/__init__.py` is the public surface; `model.PolicyValueNet`
  lives at top level and imports `core`. `core/geo_lite.py` is a thin facade over the `orbit_lite`
  package (`producer-orbit-wars-utils/`, torch-only) and is the **sole** importer of it *within the
  submission* — so `orbit_lite/` must be bundled into the submission alongside `model.py` (the facade
  adds its parent dir to `sys.path`). (Offline `agents/` also import `orbit_lite` directly, but are
  never shipped.) The facade supplies the launch geometry (lead angle / angle→planet / launch
  validation); the legacy numpy tools (`core/intercept.py`, `core/utils.validate_launch`) stay because
  `core/features.py` and `dataprep` still use them.
- **Offline-only**: `sft/`, `dataprep/`, `configs/`, `eval/`, `agents/`. These pull in `hydra`,
  `torch.distributed`, `kaggle`, `kaggle_environments`, `trueskill`, datasets — needed for
  training/ETL/evaluation, never shipped in the submission. `agents/` holds standalone heuristic bots
  (`orbit_lite` flow-diff planners ported from Kaggle notebooks) used as eval opponents.

## Environment & commands

No `requirements.txt` / `pyproject.toml`; deps live in the committed `.venv` (Python 3.12, torch CPU,
hydra-core, omegaconf, numpy, tensorboard, kaggle, tqdm). **Not a git repo.** Always invoke with the
venv interpreter: `.venv/bin/python`.

```bash
# Architecture smoke tests (no kaggle_environments needed): forward shapes/masks, set-invariance,
# intercept correctness vs simulated oracle, act() validity, validate_launch vs brute-force oracle
.venv/bin/python smoke_test.py

# SFT pipeline correctness (labels round-trip, mask invariant, overfit-one-batch trainability)
.venv/bin/python -m sft.check --path data/sft.full_send.jsonl     # or a small /tmp/sft.smoke.jsonl

# Train (Hydra). Auto-picks CUDA / MPS / CPU.
.venv/bin/python -m sft.train                                     # single process
.venv/bin/torchrun --standalone --nproc_per_node=N sft/train.py   # multi-GPU DDP
.venv/bin/python -m sft.train train.batch_size=512 data.w_hold=0.05   # CLI overrides (Hydra dot-path)

tensorboard --logdir outputs                                      # runs written per-launch under outputs/
```

Runs land in `outputs/<timestamp>/` (Hydra cwd) with `checkpoints/` (best/last/epochNN `.pt`) and
`tb/` underneath. Checkpoints embed `model_cfg` + `feature_cfg` so weights load into
`PolicyValueNet.act` without guessing shapes.

There is no pytest suite — `smoke_test.py` and `sft.check` are the test harness; run them after touching
`core/`, `model.py`, or the dataset/loss code.

## Model architecture (`model.py` + `core/features.py`)

A **set-transformer over heterogeneous entity tokens**, decoded as a source→target edge problem:

1. `core.features.encode(obs)` turns a raw obs dict into padded per-type feature tensors (planets,
   comets, fleets) plus a single sun token and a global "side" feature vector. It also returns
   `places` (decode metadata, incl. a ready `intercept.Target`) and `owned_idx`.
2. Per-type MLP encoders project each entity to `d_model` and add a learned type-embedding; everything
   concatenates into one token set fed through a `TransformerEncoder` (permutation-invariant; padding
   masked via `src_key_padding_mask`).
3. Planet+comet hidden states are the "places". `mlp_from`/`mlp_to` produce a scaled dot-product score
   matrix `S[from, to]`, with a `hold` column appended → logits `[B, M, M+1]`, softmax over the `to`
   axis. Self-target diagonal and padding columns are masked to `-inf`. Value head reads the global token.
4. `act()` decode: each owned place argmaxes one target (or hold); `num_ships` baseline is the **whole
   garrison** (~70% of expert salvos are "send everything"); the launch **angle** comes from
   `core.geo_lite.GeoEngine.intercept` (orbit_lite lead-angle) so moving targets are led correctly.

**Feature-dim contract**: `PLANET_FEAT_DIM=20`, `COMET_FEAT_DIM=25`, `FLEET_FEAT_DIM=10`,
`GLOBAL_FEAT_DIM=11` in `core/features.py` are hard-wired into the encoder input sizes in `model.py`.
`PLANET_FEAT_DIM` assumes `len(FeatureConfig.horizons)==3` — changing horizons breaks the contract;
the engine deliberately does not surface `horizons` as a trainable knob (`sft/engine.py:_feature_cfg`).

`core/intercept.py` is a self-contained numpy lead-angle tool (static / orbit / comet targets;
logarithmic `fleet_speed` 1→6). `core/utils.py` has `build_mlp` and `validate_launch` (vectorized
straight-line collision check against sun / other planets / out-of-bounds). These still back
`core/features.py`'s feature math, so they remain — but the **runtime launch geometry** (decode angle,
eval sniper, ETL angle→planet) now goes through `core/geo_lite.py` (`orbit_lite`).

**`core/geo_lite.py` `initial_planets` shim (important)**: `orbit_lite` reconstructs each orbit's phase
from `initial_planets` assuming they are the *game-step-0* positions (`angle = a0 + angvel·(step-1)`,
rotation about `(50,50)`). This repo's replay states instead store `initial_planets == current planets`,
which would make orbit_lite's forecast overshoot every orbiting target by `angvel·(step-1)` (~40 board
units at high steps → wrong intercept/resolution). The facade therefore rebuilds the true `a0` by
rotating current planets **back** by `angvel·(step-1)` before handing the obs to orbit_lite. It is
idempotent (a genuine game-initial obs yields the same `a0`) and validated by `smoke_test.test_geo_lite`
(1-step orbit forecast error ≈ 0).

## Training data flow

`sft/dataset.py` reads `data/sft.full_send.jsonl`. Each move is encoded **on the fly** in DataLoader
workers; targets are built layout-independently (per source: destination-planet / `HOLD` / `IGNORE`)
and only mapped to place indices in `collate`, which pads **dynamically to the per-batch max** of each
entity type (model is shape-driven, so no fixed 40/16/256 padding at train time). Train/val split is
**by episode** (`meta.episode_id`) — adjacent moves correlate, so a per-move split would leak.

`sft/loss.py`: weighted cross-entropy over sources. Class count is **dynamic** (`M_b+1` per batch), so a
standard per-class `weight=` vector is meaningless — instead only the `hold` column (always the last,
`hold_idx = M_b`) is down-weighted by `w_hold` (≈0.074, the send/hold ratio; dataset is ~93% hold by
source). The value head is not trained (`value_weight=0`, winner-only data); DDP runs with
`find_unused_parameters=True` to tolerate its dead gradients.

## Offline ETL (`dataprep/`)

Four sequential stages, all offline (never in the submission). Legacy single-file parser lives in
`dataprep/legacy/` (reference only):

```bash
# 1. Download public replays from the Kaggle leaderboard (polite ~1 req/sec, resumable, dedup by id)
.venv/bin/python -m dataprep.download --top 50 --out replays/

# 2. Parse replays -> {state, action, meta} samples
.venv/bin/python -m dataprep.convert --in "replays/*.json" --out data/samples.jsonl --who winner

# 3. Filter by action class (default keep = full_send)
.venv/bin/python -m dataprep.filter --in data/samples.jsonl --out data/samples.full_send.jsonl --keep full_send

# 4. Convert each launch angle -> destination planet id (core.geo_lite.GeoEngine.planet_at_angle)
.venv/bin/python -m dataprep.preprocess --in data/samples.full_send.jsonl --out data/sft.full_send.jsonl
#    --horizon N  (forecast horizon for the orbit_lite resolver; default 150, covers slow long flights)
```

Output of step 4 is the training file: `{state, sends:[[from_id,dest_id]], unresolved:[...], meta}`,
where `unresolved` sources (angle that didn't map back to a planet, ~0%) become `IGNORE` in training.
Step 4 was migrated from `core.utils.planet_at_angle` (numpy) to `core.geo_lite.GeoEngine` (orbit_lite,
via the `initial_planets` shim above): ~99% identical labels to the old resolver, the rest are cases the
engine-faithful orbit_lite resolves that the old one dropped. **Because labels can shift, re-run step 4
and retrain** when adopting it. It builds a torch movement cache per state, so it is slower than the old
numpy path (fine for a one-time offline job).

**Frame-shift gotcha** (`dataprep/convert.py`): kaggle_environments stores `obs[t]` *after* applying
`action[t]`, so `action[t]` was decided from `obs[t-1]`. The converter pairs `(state_t, action_{t+1})`;
the naive same-frame pairing yields ~57% "sent more ships than garrison" violations vs 0% when shifted.

The `full_send` filter exists because the model's decoder always sends the whole garrison — training
only on hold-or-send-everything moves keeps labels consistent with that decode (`insights/` documents
the empirical fractional-shipcount rounding the bot otherwise ignores).

## Local evaluation (`eval/`)

Offline tournament harness that pits checkpoints and heuristics against each other on the **real**
Kaggle engine — `make('orbit_wars')`, env **1.0.9**, the same version as the downloaded replays. The
engine is **not** in the repo and **not** in the committed `.venv`: install it with
`.venv/bin/pip install kaggle-environments==1.30.1 trueskill` (the kaggle-environments install drags in
heavy transitive deps — jax/transformers/litellm — which is fine for offline use but bloats `.venv`).

```bash
# 1v1 round-robin: checkpoint (greedy) vs the same checkpoint sampled vs heuristics
.venv/bin/python -m eval \
    --agents best=outputs/<ts>/checkpoints/best.pt best_T=outputs/<ts>/checkpoints/best.pt:sample:0.7 \
             sniper=heuristic:sniper rng=heuristic:random \
    --mode 1v1 --episodes 25 --out eval_runs/run1

.venv/bin/python -m eval --agents a=ckptA.pt b=ckptB.pt sniper=heuristic:sniper hold=heuristic:hold \
    --mode 4p --episodes 20      # 4-player free-for-all
```

Design: every bot is a `callable(obs, config) -> moves` (`eval/agents.py`). A checkpoint is loaded via
`PolicyValueNet.load(path)` (restores `ModelConfig`/`FeatureConfig` from the `.pt`); its **decode mode**
is an interface parameter — `greedy` (argmax, as in submission) or `sample:T` (sample from
`softmax(logits/T)`; `act()` grew a `decode=`/`temperature=` knob so the decode logic stays in one
place). Heuristics: `sniper` (send `target.ships+1` to weakest reachable capturable target via
`core.geo_lite.GeoEngine.validate_launch`), `full_send`, `random`, `hold`. Scripted bots (the third
kind): hand-written `orbit_lite` planners in `agents/` (`agents.SCRIPTED_AGENTS`), referenced as
`label=scripted:<name>` (e.g. `scripted:apex_master`) in `--agents` or as `{scripted: <name>}` in a pool
yaml — see `configs/pool/scripted.yaml`. Mode = number of seats (2 or 4). `eval/runner.py`
runs one episode and derives full **placement** from final ship counts (the engine reward only flags the
winner, which FFA TrueSkill can't use). `eval/tournament.py` does round-robin with cyclic seat rotation
(map is 4-fold symmetric → all rotations cancel positional bias), fixed seeds (same maps for every
matchup), and parallelism over episodes (`multiprocessing` fork; each checkpoint loads once per worker).
`eval/rating.py` reports TrueSkill (μ/σ, Kaggle-scale μ₀=600) plus an "A finished above B" matrix.
Timeouts are deliberately raised in the runner config so we compare policy quality, not CPU inference
speed.

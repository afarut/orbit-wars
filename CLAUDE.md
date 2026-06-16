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
  lives at top level and imports `core`.
- **Offline-only**: `sft/`, `dataprep/`, `configs/`. These pull in `hydra`, `torch.distributed`,
  `kaggle`, datasets — needed for training/ETL, never shipped in the submission.

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
   `intercept.intercept_angle` so moving targets are led correctly.

**Feature-dim contract**: `PLANET_FEAT_DIM=20`, `COMET_FEAT_DIM=25`, `FLEET_FEAT_DIM=10`,
`GLOBAL_FEAT_DIM=11` in `core/features.py` are hard-wired into the encoder input sizes in `model.py`.
`PLANET_FEAT_DIM` assumes `len(FeatureConfig.horizons)==3` — changing horizons breaks the contract;
the engine deliberately does not surface `horizons` as a trainable knob (`sft/engine.py:_feature_cfg`).

`core/intercept.py` is a self-contained numpy lead-angle tool (static / orbit / comet targets;
logarithmic `fleet_speed` 1→6). `core/utils.py` has `build_mlp` and `validate_launch` (vectorized
straight-line collision check against sun / other planets / out-of-bounds).

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

# 4. Convert each launch angle -> destination planet id (inverse of core.utils.planet_at_angle)
.venv/bin/python -m dataprep.preprocess --in data/samples.full_send.jsonl --out data/sft.full_send.jsonl
```

Output of step 4 is the training file: `{state, sends:[[from_id,dest_id]], unresolved:[...], meta}`,
where `unresolved` sources (angle that didn't map back to a planet, ~0.2%) become `IGNORE` in training.

**Frame-shift gotcha** (`dataprep/convert.py`): kaggle_environments stores `obs[t]` *after* applying
`action[t]`, so `action[t]` was decided from `obs[t-1]`. The converter pairs `(state_t, action_{t+1})`;
the naive same-frame pairing yields ~57% "sent more ships than garrison" violations vs 0% when shifted.

The `full_send` filter exists because the model's decoder always sends the whole garrison — training
only on hold-or-send-everything moves keeps labels consistent with that decode (`insights/` documents
the empirical fractional-shipcount rounding the bot otherwise ignores).

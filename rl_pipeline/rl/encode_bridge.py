"""Единый encode-кодпат для RL через Rust VecEnv.encode_features (rayon, без GIL).

Фичи считаются в Rust bit-for-bit как core.features.encode (проверено parity-тестом),
поэтому RL и Kaggle-inference видят идентичный вход. Decode строит intercept.Target из
Rust place_meta (f64) и зовёт ТОТ ЖЕ intercept.intercept_angle, что и model.act.

Поток:
  bobs, meta = batch_encode_rust(vecenv, player, device)   # Rust фичи + decode-метаданные
  out = model(bobs)
  moves, dest, frac, owned, lp = decode_batch(out, meta, player)
  vecenv.step_p0_ids(moves)  /  step_ids(...)
  phi = compute_phi_rust(meta)   # Φ=prod_adv, отдельно от входа модели
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core import intercept
from core.intercept import Target
from core.utils import sun_block_mask
from .batch_obs import BatchObs

_FRAC_VALUES = (0.25, 0.50, 0.75, 1.00)
F_MAX_P = 40  # как FeatureConfig.max_planets; place index планеты < 40, кометы >= 40


def batch_encode_rust(vecenv, player: int, device: Optional[torch.device] = None):
    """Rust VecEnv.encode_features(player) → (BatchObs, meta-dict).

    meta-dict содержит numpy: place_meta[N,56,8], comet_paths[N,16,L,2], comet_plen[N,16],
    comet_pidx[N,16], phi[N], max_path. Их хранит rollout для decode и шейпинга.
    """
    d = vecenv.encode_features(player)
    bobs = BatchObs(
        torch.from_numpy(np.asarray(d["planet_feats"])),
        torch.from_numpy(np.asarray(d["planet_mask"])).bool(),
        torch.from_numpy(np.asarray(d["comet_feats"])),
        torch.from_numpy(np.asarray(d["comet_mask"])).bool(),
        torch.from_numpy(np.asarray(d["fleet_feats"])),
        torch.from_numpy(np.asarray(d["fleet_mask"])).bool(),
        torch.from_numpy(np.asarray(d["global_feats"])),
    )
    if device is not None:
        bobs = bobs.to(device)
    return bobs, d


def batch_encode_canon_rust(vecenv, player: int, device: Optional[torch.device] = None):
    """Как batch_encode_rust, но КАНОН для player (Rust encode_features_canon, бит-точно к
    model._canonicalize). Возвращает (BatchObs, meta-dict). meta["phi_canon"][N] — φ на env,
    прибавляется к выходному углу decode (canon-кадр -> мировой). Снимает python-encode у self-play."""
    d = vecenv.encode_features_canon(player)
    bobs = BatchObs(
        torch.from_numpy(np.asarray(d["planet_feats"])),
        torch.from_numpy(np.asarray(d["planet_mask"])).bool(),
        torch.from_numpy(np.asarray(d["comet_feats"])),
        torch.from_numpy(np.asarray(d["comet_mask"])).bool(),
        torch.from_numpy(np.asarray(d["fleet_feats"])),
        torch.from_numpy(np.asarray(d["fleet_mask"])).bool(),
        torch.from_numpy(np.asarray(d["global_feats"])),
    )
    if device is not None:
        bobs = bobs.to(device)
    return bobs, d


def slice_batch(bobs, meta, idx):
    """Срез BatchObs (torch [N,...]) и meta (numpy [N,...]) по списку env-индексов idx."""
    ii = list(idx)
    sb = BatchObs(bobs.planet_feats[ii], bobs.planet_mask[ii], bobs.comet_feats[ii],
                  bobs.comet_mask[ii], bobs.fleet_feats[ii], bobs.fleet_mask[ii], bobs.global_feats[ii])
    sm = {}
    for k, v in meta.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == bobs.planet_feats.shape[0]:
            sm[k] = v[ii]
        elif hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and len(v) == bobs.planet_feats.shape[0]:
            sm[k] = np.asarray(v)[ii]
        else:
            sm[k] = v
    return sb, sm


def compute_phi_rust(meta: dict, device: Optional[torch.device] = None) -> torch.Tensor:
    """Φ(s)=prod_adv [N] из Rust (посчитано в encode_features для player)."""
    t = torch.from_numpy(np.asarray(meta["phi"])).float()
    return t.to(device) if device is not None else t


def _target_from_meta(mrow, cpaths_b, cplen_b, cpidx_b, place_idx: int) -> Target:
    """Построить intercept.Target из строки place_meta (f64) + comet_paths."""
    kind = float(mrow[6])
    x = float(mrow[3]); y = float(mrow[4]); av = float(mrow[7])
    if kind == 2.0:  # comet
        slot = place_idx - F_MAX_P
        L = int(cplen_b[slot]); pidx = int(cpidx_b[slot])
        if L > 0:
            path = np.asarray(cpaths_b[slot, :L, :], dtype=np.float64)
            return Target(pos=(x, y), kind="comet", path=path, path_index=pidx)
        return Target(pos=(x, y), kind="static")
    if kind == 1.0:  # orbit
        return Target(pos=(x, y), kind="orbit", center=(50.0, 50.0), angular_velocity=av)
    return Target(pos=(x, y), kind="static")


def _safe_probs(p: torch.Tensor) -> torch.Tensor:
    bad = ~p.isfinite().all(dim=-1, keepdim=True) | (p.sum(dim=-1, keepdim=True) < 1e-9)
    return torch.where(bad, torch.full_like(p, 1.0 / p.shape[-1]), p)


def log_prob_from_logits(
    logits: torch.Tensor, frac_logits: torch.Tensor,
    dest_acts: torch.Tensor, frac_acts: torch.Tensor,
    hold_idx: int, owned_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """log_prob [B] + entropy [B] (mean по owned). Общая для rollout и PPO-пересчёта."""
    B, M = logits.shape[:2]
    log_pi = F.log_softmax(logits.clamp(min=-1e9), dim=-1)
    log_frac = F.log_softmax(frac_logits.clamp(min=-1e9), dim=-1)
    pi = log_pi.exp().nan_to_num(0.0)
    frac_pi = log_frac.exp().nan_to_num(0.0)

    dest_lp = log_pi.gather(2, dest_acts.unsqueeze(2)).squeeze(2)
    frac_lp = log_frac.gather(2, frac_acts.clamp(0).unsqueeze(2)).squeeze(2)

    is_send = (dest_acts != hold_idx) & owned_mask
    dest_lp_safe = torch.where(owned_mask, dest_lp, torch.zeros_like(dest_lp))
    frac_lp_safe = torch.where(is_send, frac_lp, torch.zeros_like(frac_lp))
    n_owned = owned_mask.float().sum(dim=1).clamp(min=1.0)
    lp_slot = dest_lp_safe + frac_lp_safe          # [B, M] — per-action log prob
    lp = lp_slot.sum(dim=1) / n_owned              # скаляр (для совместимости/семплинга)

    ent_dest = torch.where(owned_mask, -(pi * log_pi).nan_to_num(0.0).sum(dim=-1),
                           torch.zeros(B, M, device=logits.device))
    ent_frac = torch.where(is_send, -(frac_pi * log_frac).nan_to_num(0.0).sum(dim=-1),
                           torch.zeros(B, M, device=logits.device))
    ent_slot = ent_dest + ent_frac                 # [B, M] — per-action энтропия
    ent = ent_slot.sum(dim=1) / n_owned
    return lp, ent, lp_slot, ent_slot


def owned_from_meta(meta: dict, player: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """owned_mask [N,M] bool: valid & owner==player & ships>0 (векторно из place_meta)."""
    pm = np.asarray(meta["place_meta"])  # [N, M, 8]
    valid = pm[:, :, 0] > 0.5
    owner = pm[:, :, 2].astype(np.int64)
    ships = pm[:, :, 5]
    owned = valid & (owner == int(player)) & (ships > 0)
    t = torch.from_numpy(owned)
    return t.to(device) if device is not None else t


def block_mask_from_meta(meta: dict, device: Optional[torch.device] = None) -> torch.Tensor:
    """[N,M,M] bool sun-окклюзия из Rust place_meta (тот же sun_block_mask, что в act).

    place_meta[N,M,8]: [0]=valid, [3]=x, [4]=y, [6]=kind(1=orbit,2=comet,else static).
    is_static == kind∉{1,2} — соответствует _target_from_meta (parity с act-путём)."""
    pm = torch.as_tensor(np.asarray(meta["place_meta"]), dtype=torch.float32)
    if device is not None:
        pm = pm.to(device)
    valid = pm[:, :, 0] > 0.5
    pos = pm[:, :, 3:5]
    kind = pm[:, :, 6]
    is_static = (kind != 1.0) & (kind != 2.0)
    return sun_block_mask(pos, is_static, valid)


@torch.no_grad()
def decode_batch(out: dict, meta: dict, player: int = 0, deterministic: bool = False):
    """Сэмпл действий из логитов + ходы по Rust-метаданным.

    Возвращает (moves_per_env, dest_acts[N,M], frac_acts[N,M], owned_mask[N,M], log_prob[N]).
    moves_per_env[i] = list of (from_planet_id, angle, ships). Угол через intercept_angle
    (тот же, что model.act — decode bit-exact к inference, проверено).
    """
    logits = out["logits"]; frac_logits = out["frac_logits"]
    N, M, C = logits.shape
    hold_idx = M
    device = logits.device

    owned_mask = owned_from_meta(meta, player, device)

    log_pi = F.log_softmax(logits.clamp(min=-1e9), dim=-1)
    log_frac = F.log_softmax(frac_logits.clamp(min=-1e9), dim=-1)
    if deterministic:
        dest_acts = log_pi.argmax(dim=-1)
        frac_acts = log_frac.argmax(dim=-1)
    else:
        dest_acts = torch.multinomial(_safe_probs(log_pi.exp()).reshape(N * M, C), 1).reshape(N, M)
        frac_acts = torch.multinomial(_safe_probs(log_frac.exp()).reshape(N * M, 4), 1).reshape(N, M)

    log_prob, _, lp_slot, _ = log_prob_from_logits(
        logits, frac_logits, dest_acts, frac_acts, hold_idx, owned_mask)

    pm = np.asarray(meta["place_meta"])
    cp = np.asarray(meta["comet_paths"]); cl = np.asarray(meta["comet_plen"]); ci = np.asarray(meta["comet_pidx"])
    owned_np = owned_mask.cpu().numpy()
    dest_cpu = dest_acts.cpu().numpy(); frac_cpu = frac_acts.cpu().numpy()

    moves_per_env: List[List[tuple]] = []
    for b in range(N):
        mv: List[tuple] = []
        for i in np.nonzero(owned_np[b])[0]:
            j = int(dest_cpu[b, i])
            if j == hold_idx or pm[b, j, 0] <= 0.5:
                continue
            fb = int(frac_cpu[b, i])
            ships = max(1, int(pm[b, i, 5] * _FRAC_VALUES[fb]))
            sx = float(pm[b, i, 3]); sy = float(pm[b, i, 4])
            tgt = _target_from_meta(pm[b, j], cp[b], cl[b], ci[b], j)
            angle, _eta, _hit = intercept.intercept_angle((sx, sy), tgt, ships)
            mv.append((int(pm[b, i, 1]), float(angle), int(ships)))
        moves_per_env.append(mv)

    return moves_per_env, dest_acts, frac_acts, owned_mask, log_prob, lp_slot

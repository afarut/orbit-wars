"""Сборка роллаутов из VecEnv (encode-кодпат) + вычисление GAE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from core import features
from core.features import FeatureConfig
from .batch_obs import BatchObs
from .encode_bridge import (
    batch_encode_rust, block_mask_from_meta, compute_phi_rust, decode_batch,
    log_prob_from_logits,
)

# re-export для ppo.py
__all__ = ["Rollout", "collect", "compute_gae", "log_prob_from_logits"]


@dataclass
class Rollout:
    """Буфер T шагов × N env."""
    obs_planet_feats:  torch.Tensor   # [T, N, max_planets, 20]
    obs_planet_mask:   torch.Tensor   # [T, N, max_planets]
    obs_comet_feats:   torch.Tensor   # [T, N, max_comets, 25]
    obs_comet_mask:    torch.Tensor   # [T, N, max_comets]
    obs_fleet_feats:   torch.Tensor   # [T, N, max_fleets, 10]
    obs_fleet_mask:    torch.Tensor   # [T, N, max_fleets]
    obs_global_feats:  torch.Tensor   # [T, N, 11]

    dest_acts:   torch.Tensor    # [T, N, M]  long  — hold_idx если hold/не_owned
    frac_acts:   torch.Tensor    # [T, N, M]  long
    owned_mask:  torch.Tensor    # [T, N, M]  bool
    block_mask:  torch.Tensor    # [T, N, M, M] bool — sun-окклюзия (та же, что в act)
    log_probs:   torch.Tensor    # [T, N]     float (скаляр, для логов)
    log_probs_slot: torch.Tensor # [T, N, M]  float — per-action old log prob (для per-action PPO)
    values:      torch.Tensor    # [T, N]     float
    rewards:     torch.Tensor    # [T, N]     float
    dones:       torch.Tensor    # [T, N]     bool
    phi:         torch.Tensor    # [T, N]     float — Φ(s_t)=prod_adv (шейпинг-потенциал)
    hold_idx:    int

    advantages: Optional[torch.Tensor] = None
    returns:    Optional[torch.Tensor] = None


def collect(
    model,
    vecenv,
    opponent_pool,
    T: int,
    device: torch.device,
    feat_cfg: FeatureConfig,
    num_agents: int = 2,
    deterministic: bool = False,
) -> Tuple[Rollout, "BatchObs", torch.Tensor]:
    """Собрать T шагов (p0 — наша модель, оппоненты — Rust-агенты внутри VecEnv).

    Среда должна быть уже сброшена (reset) до первого вызова — VecEnv stateful.
    Возвращает (rollout, last_bobs, last_phi) — вход модели и Φ на s_T для бутстрапа.
    """
    N = vecenv.num_envs
    mp, mc, mf = feat_cfg.max_planets, feat_cfg.max_comets, feat_cfg.max_fleets
    M = mp + mc
    hold_idx = M

    pf  = torch.zeros(T, N, mp, features.PLANET_FEAT_DIM)
    pm  = torch.zeros(T, N, mp, dtype=torch.bool)
    cf  = torch.zeros(T, N, mc, features.COMET_FEAT_DIM)
    cm  = torch.zeros(T, N, mc, dtype=torch.bool)
    ff  = torch.zeros(T, N, mf, features.FLEET_FEAT_DIM)
    fm  = torch.zeros(T, N, mf, dtype=torch.bool)
    gf  = torch.zeros(T, N, features.GLOBAL_FEAT_DIM)
    dest_buf  = torch.full((T, N, M), hold_idx, dtype=torch.long)
    frac_buf  = torch.zeros(T, N, M, dtype=torch.long)
    owned_buf = torch.zeros(T, N, M, dtype=torch.bool)
    block_buf = torch.zeros(T, N, M, M, dtype=torch.bool)   # sun-окклюзия (для PPO-пересчёта)
    lp_buf    = torch.zeros(T, N)
    lp_slot_buf = torch.zeros(T, N, M)
    val_buf   = torch.zeros(T, N)
    rew_buf   = torch.zeros(T, N)
    done_buf  = torch.zeros(T, N, dtype=torch.bool)
    phi_buf   = torch.zeros(T, N)

    model.eval()
    kinds = opponent_pool.sample_kinds(N)
    vecenv.set_opponents(opponent_pool.flat_kinds(kinds))

    rew_np  = np.zeros((N, num_agents), dtype=np.int32)
    done_np = np.zeros(N, dtype=np.uint8)
    done_arr = np.zeros(N, dtype=bool)

    for t in range(T):
        if done_arr.any():
            opponent_pool.update(kinds, rew_np, done_np)
            kinds = opponent_pool.sample_kinds(N)
            vecenv.set_opponents(opponent_pool.flat_kinds(kinds))

        bobs, meta = batch_encode_rust(vecenv, 0, device)
        blk_t = block_mask_from_meta(meta, device)        # sun-окклюзия, та же что в act

        with torch.no_grad():
            out = model(bobs, block_mask=blk_t)
        moves, dest_t, frac_t, owned_t, lp_t, lp_slot_t = decode_batch(out, meta, player=0, deterministic=deterministic)

        pf[t] = bobs.planet_feats.cpu()
        pm[t] = bobs.planet_mask.cpu()
        cf[t] = bobs.comet_feats.cpu()
        cm[t] = bobs.comet_mask.cpu()
        ff[t] = bobs.fleet_feats.cpu()
        fm[t] = bobs.fleet_mask.cpu()
        gf[t] = bobs.global_feats.cpu()
        dest_buf[t]  = dest_t.cpu()
        frac_buf[t]  = frac_t.cpu()
        owned_buf[t] = owned_t.cpu()
        block_buf[t] = blk_t.cpu()
        lp_buf[t]    = lp_t.cpu()
        lp_slot_buf[t] = lp_slot_t.cpu()
        val_buf[t]   = out["value"].cpu()
        phi_buf[t]   = compute_phi_rust(meta)

        rew_np, done_np = vecenv.step_p0_ids([list(m) for m in moves])

        done_arr = done_np.astype(bool)
        rew_buf[t]  = torch.from_numpy(rew_np[:, 0].astype(np.float32))
        done_buf[t] = torch.from_numpy(done_arr)

    model.train()
    last_bobs, last_meta = batch_encode_rust(vecenv, 0, device)
    last_phi = compute_phi_rust(last_meta)
    rollout = Rollout(
        obs_planet_feats=pf, obs_planet_mask=pm,
        obs_comet_feats=cf, obs_comet_mask=cm,
        obs_fleet_feats=ff, obs_fleet_mask=fm,
        obs_global_feats=gf,
        dest_acts=dest_buf, frac_acts=frac_buf,
        owned_mask=owned_buf, block_mask=block_buf,
        log_probs=lp_buf, log_probs_slot=lp_slot_buf, values=val_buf,
        rewards=rew_buf, dones=done_buf,
        phi=phi_buf, hold_idx=hold_idx,
    )
    return rollout, last_bobs, last_phi


def compute_gae(
    rollout: Rollout,
    last_value: torch.Tensor,          # [N]
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    shaping_alpha: float = 0.0,
    last_phi: Optional[torch.Tensor] = None,  # [N] Φ(s_T)=prod_adv на last obs
    margin_reward: bool = True,
    win_weight: float = 0.0,
) -> Rollout:
    T, N = rollout.rewards.shape

    # Терминальная награда = ГИБРИД: маржа доминирования (prod_adv ∈[-1,1]) + win_weight·(±1
    # ПОБЕДА ПО КОРАБЛЯМ). ±1 — это engine scores по СУММЕ ships (планеты+флоты): победитель
    # (max ships, max>0) +1, иначе −1; ничья по кораблям → оба топа +1 (tie-break движка).
    # prod_adv даёт гладкий градиент (проиграть с меньшим отрывом / доминировать), а ±1 —
    # явный сигнал самого критерия победы (по кораблям), которого в чистом prod_adv нет.
    # win_weight=0 → прежнее поведение (только маржа). Не-терминальные шаги = 0 (Rust даёт reward только на done).
    if margin_reward:
        win_pm1 = rollout.rewards                          # на done = ±1 (победа по кораблям), 0 иначе
        term = rollout.phi + win_weight * win_pm1
        rollout.rewards = torch.where(rollout.dones, term, torch.zeros_like(rollout.rewards))

    if shaping_alpha > 0.0:
        # potential-based shaping: F_t = γ·Φ(s_{t+1}) - Φ(s_t), Φ=prod_adv (rollout.phi).
        # Φ хранится отдельно от входа модели (global_feats[:,9] остаётся ship_adv).
        phi = rollout.phi                          # [T, N]
        phi_next = torch.empty_like(phi)
        phi_next[:T - 1] = phi[1:]
        phi_next[T - 1]  = last_phi if last_phi is not None else phi[T - 1]
        # обнуляем шейпинг на терминальных шагах: s_{t+1} уже новый эпизод
        not_done = (~rollout.dones).float()
        rollout.rewards = rollout.rewards + shaping_alpha * (gamma * phi_next - phi) * not_done

    adv = torch.zeros_like(rollout.rewards)
    last_gae = torch.zeros(N)

    for t in reversed(range(T)):
        not_done = (~rollout.dones[t]).float()
        next_val = last_value if t == T - 1 else rollout.values[t + 1]
        delta = rollout.rewards[t] + gamma * next_val * not_done - rollout.values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        adv[t] = last_gae

    rollout.advantages = adv
    rollout.returns = adv + rollout.values
    return rollout

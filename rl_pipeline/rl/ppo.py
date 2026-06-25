"""PPO update step."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .batch_obs import BatchObs
from .rollout import Rollout, log_prob_from_logits


def ppo_update(
    model,
    opt,
    rollout: Rollout,
    device: torch.device,
    *,
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
    clip_eps: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.01,   # early-stop если kl > target_kl (0 = выкл)
) -> Dict[str, float]:
    T, N = rollout.rewards.shape

    adv_flat = rollout.advantages.reshape(-1).to(device)
    adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    def flat(x): return x.reshape(T * N, *x.shape[2:]).to(device)

    pf  = flat(rollout.obs_planet_feats)
    pm  = flat(rollout.obs_planet_mask)
    cf  = flat(rollout.obs_comet_feats)
    cm  = flat(rollout.obs_comet_mask)
    ff  = flat(rollout.obs_fleet_feats)
    fm  = flat(rollout.obs_fleet_mask)
    gf  = flat(rollout.obs_global_feats)
    dest_all  = flat(rollout.dest_acts)    # [T*N, M]
    frac_all  = flat(rollout.frac_acts)    # [T*N, M]
    owned_all = flat(rollout.owned_mask)   # [T*N, M]  ← из сохранённой маски
    block_all = flat(rollout.block_mask)   # [T*N, M, M]  sun-окклюзия (та же, что в collect/act)
    lp_old    = flat(rollout.log_probs)    # [T*N]  (скаляр, для логов)
    lp_slot_old = flat(rollout.log_probs_slot)  # [T*N, M]  per-action old log prob
    val_old   = flat(rollout.values)       # [T*N]
    ret_all   = flat(rollout.returns)      # [T*N]
    adv_all   = adv_norm

    BT = T * N
    hold_idx = rollout.hold_idx

    metrics = {k: 0.0 for k in
               ("pg_loss", "val_loss", "ent", "total_loss", "approx_kl", "clip_frac", "grad_norm")}
    n_upd = 0

    model.eval()  # eval: deterministic logits, ratio от изменения весов, не dropout
    early_stop = False
    for _ in range(ppo_epochs):
        if early_stop:
            break
        perm = torch.randperm(BT, device=device)
        for start in range(0, BT, minibatch_size):
            idx = perm[start:start + minibatch_size]
            if idx.shape[0] < 2:
                continue

            bobs = BatchObs(pf[idx], pm[idx], cf[idx], cm[idx],
                            ff[idx], fm[idx], gf[idx])
            out = model(bobs, block_mask=block_all[idx])
            logits      = out["logits"]
            frac_logits = out["frac_logits"]
            value_new   = out["value"]
            M_b = logits.shape[1]
            h_idx = M_b

            _lp_new, _ent, lp_slot_new, ent_slot = log_prob_from_logits(
                logits, frac_logits,
                dest_all[idx, :M_b], frac_all[idx, :M_b],
                h_idx, owned_all[idx, :M_b],
            )

            # PER-ACTION PPO surrogate: ratio и clip на КАЖДОЕ действие (не на mean по
            # owned). Mean по owned давал ratio≈1 → градиент/действие делился на n_owned
            # → обучение в ~10× медленнее. Здесь полный градиент на действие.
            owned_b = owned_all[idx, :M_b]                         # [mb, M]
            adv_b = adv_all[idx].unsqueeze(1)                      # [mb, 1]
            log_ratio = (lp_slot_new - lp_slot_old[idx, :M_b]).clamp(-10, 10)  # [mb, M]
            ratio = torch.exp(log_ratio)
            surr = torch.min(
                ratio * adv_b,
                ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_b,
            )                                                     # [mb, M]
            # усреднение по всем owned-действиям в минибатче (как в рабочем rl5-прогоне,
            # давшем прорыв vs ProducerLite). Per-env усреднение — потенц. улучшение, но
            # НЕ менять под resume рабочего прогона (иначе динамика сдвинется).
            n_act = owned_b.float().sum().clamp(min=1.0)
            pg_loss = -(surr * owned_b).sum() / n_act
            ent = (ent_slot * owned_b).sum() / n_act

            v_clip    = val_old[idx] + (value_new - val_old[idx]).clamp(-clip_eps, clip_eps)
            val_loss  = torch.max(
                F.mse_loss(value_new, ret_all[idx]),
                F.mse_loss(v_clip,    ret_all[idx]),
            )

            loss = pg_loss + value_coeff * val_loss - entropy_coeff * ent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item() \
                if max_grad_norm > 0 else 0.0
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            opt.step()

            with torch.no_grad():
                om = owned_b.float()
                denom = om.sum().clamp(min=1.0)
                approx_kl = (((log_ratio.exp() - 1 - log_ratio) * om).sum() / denom).item()
                clip_frac = ((((ratio - 1).abs() > clip_eps).float() * om).sum() / denom).item()

            metrics["pg_loss"]    += pg_loss.item()
            metrics["val_loss"]   += val_loss.item()
            metrics["ent"]        += ent.item()
            metrics["total_loss"] += loss.item()
            metrics["approx_kl"]  += approx_kl
            metrics["clip_frac"]  += clip_frac
            metrics["grad_norm"]  += grad_norm
            n_upd += 1

            if target_kl > 0 and approx_kl > 1.5 * target_kl:
                early_stop = True
                break

    model.train()
    if n_upd:
        for k in metrics:
            metrics[k] /= n_upd

    return metrics

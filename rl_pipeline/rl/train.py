"""RL тренировочный цикл: PPO + Rust-агенты + self-play checkpoint pool.

Запуск:
  python -m rl.train sft_ckpt=checkpoints/best.pt
  python -m rl.train sft_ckpt=checkpoints/best.pt rl.selfplay_ratio=0.5 device=cuda
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_model(cfg, device):
    from model import PolicyValueNet, ModelConfig as PolicyConfig
    from core.features import FeatureConfig
    mcfg = PolicyConfig(**cfg.model)
    model = PolicyValueNet(mcfg).to(device)
    feat_cfg = FeatureConfig()
    ckpt_path = cfg.get("sft_ckpt", None)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt.get("model", ckpt))
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            log.info("Missing keys (value/frac heads — ожидаемо): %s", missing[:4])
        if ckpt.get("feature_cfg"):
            feat_cfg = FeatureConfig(**ckpt["feature_cfg"])
        log.info("Loaded SFT checkpoint: %s", ckpt_path)
    else:
        log.warning("No SFT checkpoint — training from scratch")
    return model, feat_cfg


def _reinit_frac_head(model, bin3_bias: float = 1.5, weight_scale: float = 0.01) -> None:
    """Разморозить frac-голову для RL-exploration.

    SFT обучался только на full_send → frac-голова схлопнута на бин 3 (100%) с энтропией
    ~0.001: нулевая дисперсия → policy-gradient по frac мёртв, RL не научится частичным
    отправкам. Переинициализируем ТОЛЬКО последний Linear(·,4): малые веса + мягкий bias
    к бину 3. Старт ≈ [0.13, 0.13, 0.13, 0.60] (bias=1.5) — есть exploration, но разумный
    full-send приор. Трунк/dest/value головы НЕ трогаем (они полезны из SFT).
    """
    import torch.nn as nn
    last = None
    for m in model.mlp_frac:
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        return
    with torch.no_grad():
        last.weight.mul_(weight_scale)
        last.bias.zero_()
        last.bias[3] = bin3_bias
    log.info("Re-initialised frac head (bin3_bias=%.2f, weight_scale=%.3f)", bin3_bias, weight_scale)


def _make_optimizer(model, cfg):
    opt_name = str(cfg.rl.get("optimizer", "adamw")).lower()
    lr = float(cfg.rl.lr)
    wd = float(cfg.rl.get("weight_decay", 1e-2))
    if opt_name == "muon":
        from .muon import build_muon_optimizer
        return build_muon_optimizer(model,
            muon_lr=float(cfg.rl.get("muon_lr", lr)),
            adamw_lr=float(cfg.rl.get("adamw_lr", lr / 10)),
            weight_decay=wd, ddp=False)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _make_vecenv(num_envs: int, num_agents: int, seed: int):
    # seed применяется через reset(seeds); ctor-параметры — только физика (ship_speed и т.д.),
    # поэтому seed сюда НЕ передаём (раньше он ошибочно шёл в ship_speed).
    try:
        import ow_rs
    except ImportError:
        raise ImportError("ow_rs not installed. pip install <path>/ow_rs-*.whl")
    return ow_rs.VecEnv(num_envs, num_agents)


def _cat_rollouts(r1, r2):
    """Склеить два Rollout по оси N (batch envs)."""
    import torch
    from .rollout import Rollout

    def cat(a, b): return torch.cat([a, b], dim=1)

    out = Rollout(
        obs_planet_feats=cat(r1.obs_planet_feats, r2.obs_planet_feats),
        obs_planet_mask=cat(r1.obs_planet_mask, r2.obs_planet_mask),
        obs_comet_feats=cat(r1.obs_comet_feats, r2.obs_comet_feats),
        obs_comet_mask=cat(r1.obs_comet_mask, r2.obs_comet_mask),
        obs_fleet_feats=cat(r1.obs_fleet_feats, r2.obs_fleet_feats),
        obs_fleet_mask=cat(r1.obs_fleet_mask, r2.obs_fleet_mask),
        obs_global_feats=cat(r1.obs_global_feats, r2.obs_global_feats),
        dest_acts=cat(r1.dest_acts, r2.dest_acts),
        frac_acts=cat(r1.frac_acts, r2.frac_acts),
        owned_mask=cat(r1.owned_mask, r2.owned_mask),
        block_mask=cat(r1.block_mask, r2.block_mask),
        log_probs=cat(r1.log_probs, r2.log_probs),
        log_probs_slot=cat(r1.log_probs_slot, r2.log_probs_slot),
        values=cat(r1.values, r2.values),
        rewards=cat(r1.rewards, r2.rewards),
        dones=cat(r1.dones, r2.dones),
        phi=cat(r1.phi, r2.phi),
        hold_idx=r1.hold_idx,
    )
    if r1.advantages is not None and r2.advantages is not None:
        out.advantages = cat(r1.advantages, r2.advantages)
        out.returns    = cat(r1.returns,    r2.returns)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def run(cfg: Any) -> None:
    from .rollout import compute_gae
    from .ppo import ppo_update
    from .selfplay import CheckpointPool, collect_mixed

    try:
        from omegaconf import OmegaConf
        log.info("\n" + OmegaConf.to_yaml(cfg))
    except Exception:
        log.info("config: %s", dict(cfg))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Device: %s", device)

    model, feat_cfg = _load_model(cfg, device)

    # Resume с RL-чекпоинта (model+opt+step) — продолжить обучение, а не стартовать с SFT.
    resume_ckpt = cfg.get("resume_ckpt", None)
    init_ckpt = cfg.get("init_ckpt", None)
    resume_state = None
    inited = False
    if resume_ckpt and Path(resume_ckpt).exists():
        resume_state = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(resume_state["model"])
        log.info("RESUME с %s (step=%d)", resume_ckpt, int(resume_state.get("step", 0)))
    elif init_ckpt and Path(init_ckpt).exists():
        # INIT: только веса, свежий optimizer, step=0 (новый трек с готовой модели).
        ck = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(ck.get("model_state", ck.get("model")))
        inited = True
        log.info("INIT с %s (weights-only, свежий opt, step=0)", init_ckpt)

    # frac-reinit ТОЛЬКО при старте с SFT (при resume/init frac уже обучен — не трогать).
    reinit_bias = cfg.rl.get("reinit_frac_bias", 1.5)
    if reinit_bias is not None and resume_state is None and not inited:
        _reinit_frac_head(model, bin3_bias=float(reinit_bias))

    opt   = _make_optimizer(model, cfg)
    if resume_state is not None and "opt" in resume_state:
        try:
            opt.load_state_dict(resume_state["opt"])
            log.info("RESUME: optimizer state восстановлен")
        except Exception as e:
            log.warning("RESUME: opt state не восстановлен (%s) — старт с нуля для opt", e)

    # rl8: 2 ведра — 50% 1v1 / 50% FFA. В collect_mixed КАЖДЫЙ оппонент-слот КАЖДОГО env
    # независимо: pl_prob → Rust-PL, иначе self-чекпоинт (равномерно 1/n). Любой микс в env,
    # случайные агенты на случайных слотах — нет фикс-нарезки algo/self.
    N_total = int(cfg.rl.num_envs)
    ffa_ratio = float(cfg.rl.get("ffa_ratio", 0.5))
    pl_prob   = float(cfg.rl.get("pl_prob", 0.5))   # вероятность PL на слот (1-pl_prob = self)
    N_ffa = int(N_total * ffa_ratio)
    N_1v1 = N_total - N_ffa

    seed = int(cfg.get("seed", 42))
    env_1v1 = _make_vecenv(N_1v1, 2, seed)             if N_1v1 > 0 else None
    env_ffa = _make_vecenv(N_ffa, 4, seed + 20000)     if N_ffa > 0 else None

    # Self-play пул: 3 статичных (из sp_static_paths) + trailing-self (4 сейва назад)
    from model import PolicyValueNet, ModelConfig as PolicyConfig
    def _model_factory():
        return PolicyValueNet(PolicyConfig(**cfg.model))

    sp_static_paths = list(cfg.rl.get("sp_static_paths", []))
    sp_pool = CheckpointPool(
        model_factory=_model_factory,
        device=device,
        static_paths=sp_static_paths,
        trail_back=int(cfg.rl.get("sp_trail_back", 4)),
        temperature=float(cfg.rl.get("sp_temperature", 0.5)),
        ema_alpha=float(cfg.rl.get("sp_ema_alpha", 0.05)),
    )

    sp_ckpt_interval = int(cfg.rl.get("sp_ckpt_interval", 50_000))

    # wandb
    use_wandb = bool(cfg.get("wandb", False))
    if use_wandb:
        import wandb
        wandb.init(project=cfg.get("wandb_project", "orbit-rl"),
                   config=dict(cfg))

    save_dir = Path(cfg.rl.get("save_dir", "checkpoints/rl"))
    save_dir.mkdir(parents=True, exist_ok=True)

    T             = int(cfg.rl.rollout_steps)
    total         = int(cfg.rl.total_steps)
    shaping_alpha = float(cfg.rl.get("shaping_alpha", 0.0))

    global_step = int(resume_state.get("step", 0)) if resume_state is not None else 0
    t_start = time.time()
    log.info("RL(rl8): N_1v1=%d N_ffa=%d pl_prob=%.2f T=%d (per-slot случайный микс PL/self)",
             N_1v1, N_ffa, pl_prob, T)

    # VecEnv stateful — сбрасываем один раз; непересекающиеся диапазоны seed.
    if env_1v1 is not None:
        env_1v1.reset(list(range(seed, seed + N_1v1)))
    if env_ffa is not None:
        env_ffa.reset(list(range(seed + 200_000, seed + 200_000 + N_ffa)))

    def _last_value(last_bobs):
        with torch.no_grad():
            return model(last_bobs)["value"].cpu()

    win_weight = float(cfg.rl.get("win_weight", 0.0))
    def _gae(r, last_bobs, last_phi):
        return compute_gae(r, _last_value(last_bobs),
                           float(cfg.rl.gamma), float(cfg.rl.gae_lambda),
                           shaping_alpha=shaping_alpha, last_phi=last_phi.cpu(),
                           win_weight=win_weight)

    while global_step < total:
        rollout = None
        wr_1v1 = wr_ffa = None
        def _add(r):
            nonlocal rollout
            rollout = _cat_rollouts(rollout, r) if rollout is not None else r

        # ── 1v1 (per-slot PL/self микс) ───────────────────────────────────────
        if env_1v1 is not None:
            r, lb, lp, w, e = collect_mixed(model, env_1v1, sp_pool, T, device, feat_cfg,
                                            num_agents=2, pl_prob=pl_prob)
            wr_1v1 = (w / e) if e > 0 else None
            _add(_gae(r, lb, lp))

        # ── FFA (per-slot PL/self микс) ───────────────────────────────────────
        if env_ffa is not None:
            r, lb, lp, w, e = collect_mixed(model, env_ffa, sp_pool, T, device, feat_cfg,
                                            num_agents=4, pl_prob=pl_prob)
            wr_ffa = (w / e) if e > 0 else None
            _add(_gae(r, lb, lp))

        if rollout is None:
            continue

        # ── PPO update ────────────────────────────────────────────────────────
        metrics = ppo_update(
            model, opt, rollout, device,
            ppo_epochs=int(cfg.rl.ppo_epochs),
            minibatch_size=int(cfg.rl.minibatch_size),
            clip_eps=float(cfg.rl.clip_eps),
            value_coeff=float(cfg.rl.value_coeff),
            entropy_coeff=float(cfg.rl.entropy_coeff),
            max_grad_norm=float(cfg.rl.max_grad_norm),
            target_kl=float(cfg.rl.get("target_kl", 0.01)),
        )

        global_step += T * N_total
        sps = global_step / (time.time() - t_start)

        ep_rew = rollout.rewards[rollout.dones].mean().item() if rollout.dones.any() else 0.0
        log.info(
            "step=%d  pg=%.4f  val=%.4f  ent=%.3f  kl=%.5f  clip=%.3f  gnorm=%.4f  ep_rew=%.2f  sps=%.0f",
            global_step, metrics["pg_loss"], metrics["val_loss"],
            metrics["ent"], metrics["approx_kl"], metrics["clip_frac"],
            metrics["grad_norm"], ep_rew, sps,
        )
        if wr_1v1 is not None:
            log.info("p0_wr: 1v1=%.2f", wr_1v1)
        if wr_ffa is not None:
            log.info("p0_wr: ffa=%.2f", wr_ffa)
        if sp_pool is not None:
            log.info(sp_pool.log_line())

        if use_wandb:
            import wandb
            wandb.log({"step": global_step, **metrics,
                       "ep_rew": ep_rew, "sps": sps})

        # ── trailing-self: снимок текущей модели каждые sp_ckpt_interval ──────
        if sp_pool is not None and global_step % sp_ckpt_interval < T * N_total:
            sp_pool.snapshot_self(model, global_step)

        # ── чекпойнты ─────────────────────────────────────────────────────────
        save_every = int(cfg.rl.get("save_every", 50_000))
        if global_step % save_every < T * N_total:
            ckpt_path = save_dir / f"rl_{global_step:09d}.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": global_step,
                        "sp_pool": [{"label": e["label"], "wr": e["win_rate"]}
                                    for e in sp_pool.candidates()] if sp_pool else [],
                        }, ckpt_path)
            log.info("Saved: %s", ckpt_path)

    log.info("Done. Total steps: %d", global_step)
    final = save_dir / "rl_final.pt"
    torch.save({"model": model.state_dict(), "step": global_step}, final)

    if use_wandb:
        import wandb
        wandb.finish()


def main() -> None:
    """Hydra-точка входа (для локального запуска с hydra)."""
    import hydra
    hydra.main(config_path="../configs", config_name="rl_train", version_base=None)(run)()


if __name__ == "__main__":
    main()

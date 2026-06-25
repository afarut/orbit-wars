"""Self-play с пулом прошлых чекпойнтов.

Храним до max_ckpts чекпойнтов. Из них выбираем top_k по loss_rate
(против кого мы чаще проигрываем — тот чаще попадается).

Для self-play используем VecEnv.step_ids([N][A] списки (id,angle,ships)) — обе стороны из Python.

Интеграция с основным обучением:
  1. collect_mixed() возвращает Rollout только для player-0 (наш агент).
  2. В конце каждого rollout pool.maybe_add_checkpoint() решает, стоит ли добавить
     текущий чекпойнт (каждые ckpt_interval шагов).
"""
from __future__ import annotations

import copy
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from core import features
from core.features import FeatureConfig
from .batch_obs import BatchObs
from .encode_bridge import (
    batch_encode_rust, batch_encode_canon_rust, block_mask_from_meta,
    compute_phi_rust, decode_batch, slice_batch,
)
from .rollout import Rollout

log = logging.getLogger(__name__)


class CheckpointPool:
    """Пул спарринг-оппонентов для self-play (rl8):

      • 3 СТАТИЧНЫХ чекпоинта (грузятся из файлов на старте) — НИКОГДА не вытесняются;
      • TRAILING-SELF — снимок текущей модели ``trail_back`` сейвов назад (дек длины
        ``trail_back+1``: когда дек полон, ``_trail[0]`` = снимок ровно trail_back назад).
        Пока снимков < trail_back+1 — trailing-self нет.

    Семпл оппонента для 1v1-self: из {3 статичных + trailing-self}, взвешенно по loss_rate
    (чаще тот, кому мы чаще проигрываем). 3 статичных также идут как 3 слота FFA-self.
    """

    def __init__(
        self,
        model_factory,
        device: torch.device,
        static_paths: Optional[List] = None,
        trail_back: int = 4,
        temperature: float = 0.5,
        ema_alpha: float = 0.05,
        min_games: int = 10,
    ):
        self.model_factory = model_factory
        self.device = device
        self.trail_back = trail_back
        self.temperature = temperature
        self.ema_alpha = ema_alpha
        self.min_games = min_games

        self._static: List[Dict] = []          # неудаляемые спарринги
        self._trail: deque = deque(maxlen=trail_back + 1)  # снимки self
        for p in (static_paths or []):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            self._static.append({
                "state": ck.get("model_state", ck.get("model")),
                "label": Path(p).stem, "win_rate": 0.5, "games": 0, "static": True,
            })
            log.info("CheckpointPool: static спарринг '%s'", Path(p).stem)

    # ── снимки self (trailing) ────────────────────────────────────────────────

    def snapshot_self(self, model: torch.nn.Module, global_step: int) -> None:
        """Снять слепок текущей модели в trailing-дек (каждые sp_ckpt_interval)."""
        self._trail.append({"state": copy.deepcopy(model.state_dict()),
                            "step": global_step, "label": f"trail_{global_step}",
                            "win_rate": 0.5, "games": 0, "static": False})
        log.info("CheckpointPool: snapshot self step=%d (trail=%d/%d)",
                 global_step, len(self._trail), self.trail_back + 1)

    def _trailing_entry(self) -> Optional[Dict]:
        """Снимок ровно trail_back сейвов назад (есть только когда дек полон)."""
        return self._trail[0] if len(self._trail) == self._trail.maxlen else None

    # ── выбор оппонента ───────────────────────────────────────────────────────

    def candidates(self) -> List[Dict]:
        """3 статичных + trailing-self (если есть)."""
        c = list(self._static)
        tr = self._trailing_entry()
        if tr is not None:
            c.append(tr)
        return c

    def sample_entry(self) -> Optional[Dict]:
        """Один оппонент из кандидатов, взвешенно по loss_rate (1−win_rate)."""
        cands = self.candidates()
        if not cands:
            return None
        if all(e["games"] < self.min_games for e in cands):
            probs = np.ones(len(cands)) / len(cands)
        else:
            lr = np.array([1.0 - e["win_rate"] for e in cands])
            logits = lr / max(self.temperature, 1e-6)
            logits -= logits.max()
            p = np.exp(logits)
            probs = p / p.sum()
        return cands[int(np.random.choice(len(cands), p=probs))]

    def static_entries(self) -> List[Dict]:
        """3 статичных (для 3 слотов FFA-self)."""
        return list(self._static)

    def load_model(self, entry: Dict) -> torch.nn.Module:
        m = self.model_factory()
        m.load_state_dict(entry["state"])
        m.to(self.device)
        m.eval()
        return m

    def update(self, entry: Dict, wins: float, n: int) -> None:
        if n == 0:
            return
        wr = wins / n
        entry["games"] += n
        alpha = min(self.ema_alpha, 1.0 / (entry["games"] + 1))
        entry["win_rate"] = (1 - alpha) * entry["win_rate"] + alpha * wr

    def log_line(self) -> str:
        parts = [f"{e['label']} wr={e['win_rate']:.2f}({e['games']}g)" for e in self.candidates()]
        return "selfplay: " + ("  ".join(parts) if parts else "no opponents")

    def __len__(self):
        return len(self.candidates())


# ── collect для self-play envs ────────────────────────────────────────────────

def collect_selfplay(
    model: torch.nn.Module,
    opp_models,
    vecenv,
    T: int,
    device: torch.device,
    feat_cfg: FeatureConfig,
    deterministic: bool = False,
    num_agents: int = 2,
) -> Tuple[Rollout, "BatchObs", torch.Tensor, float, int]:
    """Собрать T шагов; ВСЕ игроки — Python-модели (encode-кодпат, vecenv.step_ids).

    ``opp_models`` — список моделей-оппонентов длины ``num_agents-1`` (по одной на слот
    1..num_agents-1). 1v1: [opp]; FFA: [opp1, opp2, opp3]. Каждый оппонент ходит ЧЕРЕЗ КАНОН
    (``act_many(observation_dicts(slot))``) — фичи побитово как при p0-обучении, иначе модель
    ломается на не-p0 слотах. p0 (наш агент) маскирует sun-окклюзию (block_mask), как в collect.

    Среда должна быть уже сброшена до первого вызова (VecEnv stateful).
    Возвращает (rollout, last_bobs, last_phi, wins, total_eps) для p0.
    """
    if not isinstance(opp_models, (list, tuple)):
        opp_models = [opp_models]
    assert len(opp_models) == num_agents - 1, \
        f"нужно {num_agents-1} оппонентов, дано {len(opp_models)}"
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
    block_buf = torch.zeros(T, N, M, M, dtype=torch.bool)   # sun-окклюзия p0 (для PPO-пересчёта)
    lp_buf    = torch.zeros(T, N)
    lp_slot_buf = torch.zeros(T, N, M)
    val_buf   = torch.zeros(T, N)
    rew_buf   = torch.zeros(T, N)
    done_buf  = torch.zeros(T, N, dtype=torch.bool)
    phi_buf   = torch.zeros(T, N)

    wins, total_eps = 0, 0
    model.eval()
    for om in opp_models:
        om.eval()

    rew_np  = np.zeros((N, num_agents), dtype=np.int32)
    done_np = np.zeros(N, dtype=np.uint8)
    done_arr = np.zeros(N, dtype=bool)

    for t in range(T):
        if done_arr.any():
            wins  += float((rew_np[done_arr, 0] > 0).sum())
            total_eps += int(done_arr.sum())

        bobs0, meta0 = batch_encode_rust(vecenv, 0, device)
        blk0 = block_mask_from_meta(meta0, device)        # sun-окклюзия p0, та же что в act/collect

        with torch.no_grad():
            out0 = model(bobs0, block_mask=blk0)

        moves0, dest_t, frac_t, owned_t, lp_t, lp_slot_t = decode_batch(out0, meta0, player=0, deterministic=deterministic)
        # Оппоненты на слотах 1..n-1 играют ЧЕРЕЗ КАНОН (act_many): фичи побитово как при
        # p0-обучении (иначе инвалид на не-p0 слотах). Сэмплят (deterministic=False) — иначе
        # детерминированного оппонента легко вызубрить. Питон-encode на каждый obs оппонента.
        opp_moves = [
            opp_models[s - 1].act_many(vecenv.observation_dicts(s), feat_cfg,
                                       deterministic=deterministic)
            for s in range(1, num_agents)
        ]

        pf[t] = bobs0.planet_feats.cpu()
        pm[t] = bobs0.planet_mask.cpu()
        cf[t] = bobs0.comet_feats.cpu()
        cm[t] = bobs0.comet_mask.cpu()
        ff[t] = bobs0.fleet_feats.cpu()
        fm[t] = bobs0.fleet_mask.cpu()
        gf[t] = bobs0.global_feats.cpu()
        dest_buf[t]  = dest_t.cpu()
        frac_buf[t]  = frac_t.cpu()
        owned_buf[t] = owned_t.cpu()
        block_buf[t] = blk0.cpu()
        lp_buf[t]    = lp_t.cpu()
        lp_slot_buf[t] = lp_slot_t.cpu()
        val_buf[t]   = out0["value"].cpu()
        phi_buf[t]   = compute_phi_rust(meta0)

        # действия всех игроков по planet-id: [p0, slot1, ..., slot_{n-1}] на env
        all_acts = [[list(moves0[e])] + [list(opp_moves[s][e]) for s in range(num_agents - 1)]
                    for e in range(N)]
        rew_np, done_np = vecenv.step_ids(all_acts)

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
    return rollout, last_bobs, last_phi, wins, total_eps


# ── collect_mixed: per-slot случайный микс PL / self (rl8) ────────────────────

def collect_mixed(
    model: torch.nn.Module,
    vecenv,
    sp_pool,
    T: int,
    device: torch.device,
    feat_cfg: FeatureConfig,
    num_agents: int,
    pl_prob: float = 0.5,
    deterministic: bool = False,
) -> Tuple[Rollout, "BatchObs", torch.Tensor, float, int]:
    """Собрать T шагов; КАЖДЫЙ оппонент-слот КАЖДОГО env независимо: w.p. ``pl_prob`` —
    Rust-ProducerLite (``producer_lite_moves``, бит-идентичен внутреннему PL), иначе —
    self-чекпоинт РАВНОМЕРНО из пула (1/n) через ``act_many`` (КАНОН per-slot). Любой микс
    в одном env (3/0, 2/1, 0/3), случайные агенты на случайных слотах. Нет фикс-нарезки.

    Назначение бросается ОДИН раз на роллаут (фикс на T шагов, переживает авто-ресеты).
    p0 — наш агент (с block_mask). Возвращает (rollout, last_bobs, last_phi, wins, total_eps).
    """
    N = vecenv.num_envs
    mp, mc, mf = feat_cfg.max_planets, feat_cfg.max_comets, feat_cfg.max_fleets
    M = mp + mc
    hold_idx = M

    pf = torch.zeros(T, N, mp, features.PLANET_FEAT_DIM); pm = torch.zeros(T, N, mp, dtype=torch.bool)
    cf = torch.zeros(T, N, mc, features.COMET_FEAT_DIM);  cm = torch.zeros(T, N, mc, dtype=torch.bool)
    ff = torch.zeros(T, N, mf, features.FLEET_FEAT_DIM);  fm = torch.zeros(T, N, mf, dtype=torch.bool)
    gf = torch.zeros(T, N, features.GLOBAL_FEAT_DIM)
    dest_buf  = torch.full((T, N, M), hold_idx, dtype=torch.long)
    frac_buf  = torch.zeros(T, N, M, dtype=torch.long)
    owned_buf = torch.zeros(T, N, M, dtype=torch.bool)
    block_buf = torch.zeros(T, N, M, M, dtype=torch.bool)
    lp_buf = torch.zeros(T, N); lp_slot_buf = torch.zeros(T, N, M)
    val_buf = torch.zeros(T, N); rew_buf = torch.zeros(T, N)
    done_buf = torch.zeros(T, N, dtype=torch.bool); phi_buf = torch.zeros(T, N)

    cands = sp_pool.candidates() if sp_pool is not None else []
    # назначение per (slot, env): None=PL, иначе индекс кандидата. Self-модели грузим лениво.
    assign = [[None] * N for _ in range(num_agents)]   # assign[p][e]
    self_models: Dict[int, torch.nn.Module] = {}
    import random as _random
    for p in range(1, num_agents):
        for e in range(N):
            if (not cands) or _random.random() < pl_prob:
                assign[p][e] = None
            else:
                ci = _random.randrange(len(cands))     # РАВНОМЕРНО 1/n
                assign[p][e] = ci
                if ci not in self_models:
                    self_models[ci] = sp_pool.load_model(cands[ci])

    model.eval()
    rew_np = np.zeros((N, num_agents), dtype=np.int32)
    done_arr = np.zeros(N, dtype=bool)
    wins = total_eps = 0

    for t in range(T):
        if done_arr.any():
            wins += float((rew_np[done_arr, 0] > 0).sum()); total_eps += int(done_arr.sum())

        bobs0, meta0 = batch_encode_rust(vecenv, 0, device)
        blk0 = block_mask_from_meta(meta0, device)
        with torch.no_grad():
            out0 = model(bobs0, block_mask=blk0)
        moves0, dest_t, frac_t, owned_t, lp_t, lp_slot_t = decode_batch(
            out0, meta0, player=0, deterministic=deterministic)

        # ходы оппонентов по слотам
        slot_moves = [[None] * N for _ in range(num_agents)]
        for p in range(1, num_agents):
            pl_envs = [e for e in range(N) if assign[p][e] is None]
            self_envs = [e for e in range(N) if assign[p][e] is not None]
            for e in pl_envs:
                slot_moves[p][e] = [(int(x[0]), float(x[1]), int(x[2]))
                                    for x in vecenv.producer_lite_moves(e, p)]
            if self_envs:
                # Rust canon-encode для слота p (бит-точно к act_many, но БЕЗ python per-obs):
                # один батч-encode на все env слота, forward по группам чекпоинтов, decode_batch,
                # угол + φ_canon (canon-кадр -> мировой). Это и есть ускорение sps.
                bobs_p, meta_p = batch_encode_canon_rust(vecenv, p, device)
                phic = np.asarray(meta_p["phi_canon"])
                by_ci: Dict[int, List[int]] = {}
                for e in self_envs:
                    by_ci.setdefault(assign[p][e], []).append(e)
                for ci, envs in by_ci.items():
                    sb, sm = slice_batch(bobs_p, meta_p, envs)
                    with torch.no_grad():
                        out = self_models[ci](sb, block_mask=block_mask_from_meta(sm, device))
                    mv = decode_batch(out, sm, player=0, deterministic=deterministic)[0]
                    for k, e in enumerate(envs):
                        ph = float(phic[e])
                        slot_moves[p][e] = [(int(x[0]), float(x[1]) + ph, int(x[2])) for x in mv[k]]

        pf[t]=bobs0.planet_feats.cpu(); pm[t]=bobs0.planet_mask.cpu()
        cf[t]=bobs0.comet_feats.cpu();  cm[t]=bobs0.comet_mask.cpu()
        ff[t]=bobs0.fleet_feats.cpu();  fm[t]=bobs0.fleet_mask.cpu()
        gf[t]=bobs0.global_feats.cpu()
        dest_buf[t]=dest_t.cpu(); frac_buf[t]=frac_t.cpu(); owned_buf[t]=owned_t.cpu()
        block_buf[t]=blk0.cpu(); lp_buf[t]=lp_t.cpu(); lp_slot_buf[t]=lp_slot_t.cpu()
        val_buf[t]=out0["value"].cpu(); phi_buf[t]=compute_phi_rust(meta0)

        all_acts = [[list(moves0[e])] + [slot_moves[p][e] for p in range(1, num_agents)]
                    for e in range(N)]
        rew_np, done_np = vecenv.step_ids(all_acts)
        done_arr = done_np.astype(bool)
        rew_buf[t] = torch.from_numpy(rew_np[:, 0].astype(np.float32))
        done_buf[t] = torch.from_numpy(done_arr)

    model.train()
    last_bobs, last_meta = batch_encode_rust(vecenv, 0, device)
    last_phi = compute_phi_rust(last_meta)
    rollout = Rollout(
        obs_planet_feats=pf, obs_planet_mask=pm, obs_comet_feats=cf, obs_comet_mask=cm,
        obs_fleet_feats=ff, obs_fleet_mask=fm, obs_global_feats=gf,
        dest_acts=dest_buf, frac_acts=frac_buf, owned_mask=owned_buf, block_mask=block_buf,
        log_probs=lp_buf, log_probs_slot=lp_slot_buf, values=val_buf,
        rewards=rew_buf, dones=done_buf, phi=phi_buf, hold_idx=hold_idx,
    )
    return rollout, last_bobs, last_phi, wins, total_eps

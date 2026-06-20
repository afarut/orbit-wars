r"""Тренировочный движок SFT: выбор устройства, DDP/torchrun, цикл train/eval, чекпойнты.

Устройство-агностичен: сам выбирает CUDA (Nvidia) / MPS (Mac) / CPU. Под ``torchrun``
поднимает DDP (nccl на CUDA, gloo иначе); без него — одиночный процесс. AMP включается
только на CUDA. Чекпойнт пишет только rank 0; в него кладём ``ModelConfig`` и
``FeatureConfig``, чтобы веса грузились в боевой ``PolicyValueNet.act`` без догадок.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from core.features import FeatureConfig
from model import ModelConfig, PolicyValueNet

from . import loss as L
from .dataset import SftDataset, build_index, collate, split_indices


# --- распределёнка / устройство ----------------------------------------------
def _dist_info() -> Tuple[int, int, int]:
    """(rank, local_rank, world_size) из env torchrun (по умолчанию одиночный процесс)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world_size


def _pick_device(local_rank: int) -> Tuple[torch.device, str]:
    """Выбор устройства и DDP-бэкенда: CUDA->nccl, иначе (MPS/CPU)->gloo."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), "nccl"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps"), "gloo"
    return torch.device("cpu"), "gloo"


def _resolve_path(path: str) -> str:
    """Относительный path -> к исходному cwd (Hydra переключает рабочую директорию)."""
    if os.path.isabs(path):
        return path
    try:
        from hydra.utils import get_original_cwd
        return os.path.join(get_original_cwd(), path)
    except Exception:
        return path


def _seed(seed: int, rank: int) -> None:
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def _lr_lambda(warmup: int, total: int):
    """Линейный warmup -> косинусный спад до 0."""
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if total <= warmup:
            return 1.0
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return fn


# --- конфиги из OmegaConf -----------------------------------------------------
def _feature_cfg(cfg) -> FeatureConfig:
    d = cfg.data
    # horizons НЕ трогаем: PLANET_FEAT_DIM=20 завязан на len(horizons)==3 (контракт с model)
    return FeatureConfig(max_planets=int(d.max_planets),
                         max_comets=int(d.max_comets),
                         max_fleets=int(d.max_fleets))


def _model_cfg(cfg) -> ModelConfig:
    m = cfg.model
    return ModelConfig(d_model=int(m.d_model), d_k=int(m.d_k), n_layers=int(m.n_layers),
                       n_heads=int(m.n_heads), ffn=int(m.ffn), dropout=float(m.dropout),
                       enc_hidden=int(m.enc_hidden), head_hidden=int(m.head_hidden),
                       n_frac_buckets=int(m.get("n_frac_buckets", 4)))


def _make_writer(cfg):
    """SummaryWriter в tb_dir (относительно hydra run-dir); None если выключен/нет пакета."""
    if not bool(cfg.train.get("tensorboard", True)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[engine] tensorboard не установлен — логи TB пропущены "
              "(pip install tensorboard)")
        return None
    tb_dir = os.path.abspath(cfg.train.get("tb_dir", "tb"))
    os.makedirs(tb_dir, exist_ok=True)
    print(f"[engine] tensorboard -> {tb_dir}")
    return SummaryWriter(tb_dir)


# --- основной вход ------------------------------------------------------------
def run(cfg) -> None:
    rank, local_rank, world_size = _dist_info()
    ddp = world_size > 1
    device, backend = _pick_device(local_rank)
    if ddp:
        dist.init_process_group(backend=backend)
    is_main = rank == 0
    _seed(int(cfg.data.seed), rank)

    if is_main:
        from hydra_utils import print_cfg     # общий хелпер печати конфига (офлайн)
        print_cfg(cfg, "sft")

    fcfg = _feature_cfg(cfg)
    mcfg = _model_cfg(cfg)
    if is_main:
        print(f"[engine] device={device} backend={backend} world_size={world_size} ddp={ddp}")

    # --- данные (hydra меняет cwd -> относительный path резолвим к исходному) ---
    data_path = _resolve_path(cfg.data.path)
    idx = build_index(data_path)
    limit = int(cfg.data.limit) if cfg.data.get("limit") else None
    train_lines, val_lines = split_indices(
        idx, val_frac=float(cfg.data.val_frac), seed=int(cfg.data.seed),
        limit=limit, hold_subsample=float(cfg.data.hold_subsample))
    if is_main:
        print(f"[engine] train={len(train_lines):,} val={len(val_lines):,} ходов")

    train_ds = SftDataset(data_path, train_lines, idx, cfg=fcfg)
    val_ds = SftDataset(data_path, val_lines, idx, cfg=fcfg)

    pin = device.type == "cuda"
    nw = int(cfg.data.num_workers)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None
    bs = int(cfg.train.batch_size)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=nw, pin_memory=pin, drop_last=True,
        collate_fn=collate, persistent_workers=(nw > 0))
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, sampler=val_sampler, num_workers=nw,
        pin_memory=pin, drop_last=False, collate_fn=collate,
        persistent_workers=(nw > 0)) if len(val_ds) else None

    # --- модель / оптимизатор ---
    model = PolicyValueNet(mcfg).to(device)
    if ddp:
        dev_ids = [local_rank] if device.type == "cuda" else None
        # value-голова не участвует в лоссе при value_weight=0 -> её параметры без
        # градиента; find_unused_parameters позволяет DDP это переварить.
        model = DDP(model, device_ids=dev_ids, find_unused_parameters=True)
    core = model.module if ddp else model

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr),
                            weight_decay=float(cfg.train.weight_decay))
    epochs = int(cfg.train.epochs)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(int(cfg.train.warmup_steps), total_steps))

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    w_hold = float(cfg.data.w_hold)
    # голова числа кораблей: вес терма + веса классов (балансировка перекоса к 100%)
    w_frac = float(cfg.train.get("w_frac", 1.0))
    fw = cfg.train.get("frac_weights", None)
    frac_weights = (torch.tensor([float(x) for x in fw], dtype=torch.float32, device=device)
                    if fw else None)
    grad_clip = float(cfg.train.grad_clip)
    log_every = int(cfg.train.log_every)

    ckpt_dir = os.path.abspath(cfg.train.ckpt_dir)
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[engine] чекпойнты -> {ckpt_dir}")
    metrics_log = os.path.join(ckpt_dir, "metrics.jsonl")
    best_send_acc = -1.0

    # --- TensorBoard (только rank 0) ---
    writer = _make_writer(cfg) if is_main else None

    global_step = 0
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for bi, (obs, labels, frac_labels, hold_idx) in enumerate(train_loader):
            obs = obs.to(device)
            labels = labels.to(device)
            frac_labels = frac_labels.to(device)
            opt.zero_grad(set_to_none=True)
            # пары (источник, экспертная цель) для головы числа кораблей -> внутрь forward
            frac_pairs, frac_tgt = L.frac_pairs_from(labels, frac_labels)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(obs, frac_pairs=frac_pairs)
                lloss = L.policy_loss(out["logits"], labels, hold_idx, w_hold=w_hold)
                floss = L.fraction_loss(out["frac_logits"], frac_tgt, weight=frac_weights)
                loss = lloss + w_frac * floss
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1

            if is_main and log_every and global_step % log_every == 0:
                m = L.policy_metrics(out["logits"].detach(), labels, hold_idx)
                fa, fn = L.fraction_acc(out["frac_logits"].detach(), frac_tgt)
                lr = sched.get_last_lr()[0]
                print(f"  e{epoch} s{global_step} loss={loss.item():.4f} "
                      f"send_acc={m['send_acc']:.3f} hold_acc={m['hold_acc']:.3f} "
                      f"frac_acc={fa:.3f} lr={lr:.2e}")
                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/policy_loss", lloss.item(), global_step)
                    writer.add_scalar("train/frac_loss", floss.item(), global_step)
                    writer.add_scalar("train/acc", m["acc"], global_step)
                    writer.add_scalar("train/send_acc", m["send_acc"], global_step)
                    writer.add_scalar("train/hold_acc", m["hold_acc"], global_step)
                    writer.add_scalar("train/frac_acc", fa, global_step)
                    writer.add_scalar("train/lr", lr, global_step)

        # --- eval ---
        if val_loader is not None:
            vm = _evaluate(model, val_loader, device, hold_w=w_hold, ddp=ddp)
            if is_main:
                print(f"[eval e{epoch}] loss={vm['loss']:.4f} acc={vm['acc']:.3f} "
                      f"send_acc={vm['send_acc']:.3f} hold_acc={vm['hold_acc']:.3f} "
                      f"frac_acc={vm['frac_acc']:.3f} "
                      f"hold_P={vm['hold_precision']:.3f} hold_R={vm['hold_recall']:.3f} "
                      f"(send={vm['n_send']:,} hold={vm['n_hold']:,} frac={vm['n_frac']:,})")
                with open(metrics_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": epoch, **vm}) + "\n")
                if writer is not None:
                    for k, v in vm.items():
                        writer.add_scalar(f"val/{k}", v, global_step)
                if vm["send_acc"] >= best_send_acc:
                    best_send_acc = vm["send_acc"]
                    _save_ckpt(os.path.join(ckpt_dir, "best.pt"), core, mcfg, fcfg, epoch, vm)
        if is_main:
            _save_ckpt(os.path.join(ckpt_dir, "last.pt"), core, mcfg, fcfg, epoch, None)
            if bool(cfg.train.get("save_each_epoch", False)):
                ep_metrics = vm if val_loader is not None else None
                _save_ckpt(os.path.join(ckpt_dir, f"epoch{epoch:02d}.pt"),
                           core, mcfg, fcfg, epoch, ep_metrics)

    if writer is not None:
        writer.close()
    if ddp:
        dist.destroy_process_group()


@torch.no_grad()
def _evaluate(model, loader, device, hold_w: float, ddp: bool) -> Dict[str, float]:
    """Прогон по val: агрегируем счётчики (по DDP — all_reduce SUM) -> метрики."""
    model.eval()
    acc = torch.zeros(len(L.COUNT_KEYS), dtype=torch.float64, device=device)
    facc = torch.zeros(2, dtype=torch.float64, device=device)   # [frac_correct, frac_valid]
    hold_idx = 0
    for obs, labels, frac_labels, hold_idx in loader:
        obs = obs.to(device)
        labels = labels.to(device)
        frac_labels = frac_labels.to(device)
        frac_pairs, frac_tgt = L.frac_pairs_from(labels, frac_labels)
        out = model(obs, frac_pairs=frac_pairs)
        acc += L.policy_counts(out["logits"], labels, hold_idx, w_hold=hold_w)
        facc += L.frac_counts(out["frac_logits"], frac_tgt)
    if ddp:
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(facc, op=dist.ReduceOp.SUM)
    model.train()
    metrics = L.counts_to_metrics(acc)
    metrics["frac_acc"] = float(facc[0] / facc[1]) if facc[1] else 0.0
    metrics["n_frac"] = int(facc[1].item())
    return metrics


def _save_ckpt(path: str, core: PolicyValueNet, mcfg: ModelConfig,
               fcfg: FeatureConfig, epoch: int, metrics: Optional[dict]) -> None:
    """Сохранить веса + конфиги (для загрузки в PolicyValueNet.act)."""
    torch.save({
        "model_state": core.state_dict(),
        "model_cfg": asdict(mcfg),
        "feature_cfg": asdict(fcfg),
        "epoch": epoch,
        "metrics": metrics,
    }, path)

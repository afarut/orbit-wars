r"""Взвешенный policy cross-entropy + метрики для SFT Orbit Wars.

Число классов ДИНАМИЧЕСКОЕ (``M_b+1`` зависит от батча), поэтому стандартный
``weight=``-вектор по классам неприменим (номер класса = разные планеты в разных
батчах). Понижаем вес ТОЛЬКО у hold (это всегда последняя колонка ``hold_idx = M_b``),
веса целевых планет не трогаем — ровно то, что нужно при перекосе ~98% hold по источникам.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .dataset import IGNORE_INDEX


def policy_loss(logits: torch.Tensor, labels: torch.Tensor, hold_idx: int,
                w_hold: float = 1.0) -> torch.Tensor:
    """Взвешенный CE по источникам. logits [B,M,M+1], labels [B,M] (-100 = ignore)."""
    n_classes = logits.shape[-1]
    flat_logits = logits.reshape(-1, n_classes)
    flat_labels = labels.reshape(-1)

    ce = F.cross_entropy(flat_logits, flat_labels,
                         reduction="none", ignore_index=IGNORE_INDEX)  # [B*M]
    valid = (flat_labels != IGNORE_INDEX).float()
    # вес: hold-таргеты -> w_hold, остальные (планеты) -> 1.0
    w = torch.where(flat_labels == hold_idx,
                    torch.full_like(ce, w_hold), torch.ones_like(ce))
    denom = (w * valid).sum().clamp_min(1.0)
    return (ce * w * valid).sum() / denom


@torch.no_grad()
def policy_metrics(logits: torch.Tensor, labels: torch.Tensor,
                   hold_idx: int) -> Dict[str, float]:
    """accuracy по источникам + отдельно send/hold + precision/recall hold."""
    pred = logits.argmax(dim=-1).reshape(-1)
    lab = labels.reshape(-1)
    valid = lab != IGNORE_INDEX
    correct = (pred == lab) & valid

    is_hold = valid & (lab == hold_idx)
    is_send = valid & (lab != hold_idx)
    pred_hold = valid & (pred == hold_idx)

    def _ratio(num: torch.Tensor, den: torch.Tensor) -> float:
        d = int(den.sum().item())
        return float(num.sum().item()) / d if d else 0.0

    tp_hold = pred_hold & (lab == hold_idx)
    return {
        "acc": _ratio(correct, valid),
        "send_acc": _ratio(correct & is_send, is_send),
        "hold_acc": _ratio(correct & is_hold, is_hold),
        "hold_precision": _ratio(tp_hold, pred_hold),
        "hold_recall": _ratio(tp_hold, is_hold),
        "n_send": int(is_send.sum().item()),
        "n_hold": int(is_hold.sum().item()),
    }


# порядок счётчиков для агрегации (в т.ч. all_reduce по DDP)
COUNT_KEYS = ("loss_num", "loss_den", "correct", "valid",
              "send_correct", "send", "hold_correct", "hold", "tp_hold", "pred_hold")


@torch.no_grad()
def policy_counts(logits: torch.Tensor, labels: torch.Tensor, hold_idx: int,
                  w_hold: float = 1.0) -> torch.Tensor:
    """Сырые счётчики батча (порядок COUNT_KEYS) — суммируемы и all_reduce-абельны."""
    n_classes = logits.shape[-1]
    flat = labels.reshape(-1)
    ce = F.cross_entropy(logits.reshape(-1, n_classes), flat,
                         reduction="none", ignore_index=IGNORE_INDEX)
    valid = flat != IGNORE_INDEX
    w = torch.where(flat == hold_idx, torch.full_like(ce, w_hold), torch.ones_like(ce))

    pred = logits.argmax(dim=-1).reshape(-1)
    correct = (pred == flat) & valid
    is_hold = valid & (flat == hold_idx)
    is_send = valid & (flat != hold_idx)
    pred_hold = valid & (pred == hold_idx)
    tp_hold = pred_hold & (flat == hold_idx)

    return torch.tensor([
        (ce * w * valid.float()).sum(), (w * valid.float()).sum(),
        correct.sum(), valid.sum(),
        (correct & is_send).sum(), is_send.sum(),
        (correct & is_hold).sum(), is_hold.sum(),
        tp_hold.sum(), pred_hold.sum(),
    ], dtype=torch.float64, device=logits.device)


def counts_to_metrics(c: torch.Tensor) -> Dict[str, float]:
    """Свернуть агрегированные счётчики (COUNT_KEYS) в метрики."""
    d = {k: float(c[i].item()) for i, k in enumerate(COUNT_KEYS)}

    def _r(a: str, b: str) -> float:
        return d[a] / d[b] if d[b] else 0.0

    return {
        "loss": _r("loss_num", "loss_den"),
        "acc": _r("correct", "valid"),
        "send_acc": _r("send_correct", "send"),
        "hold_acc": _r("hold_correct", "hold"),
        "hold_precision": _r("tp_hold", "pred_hold"),
        "hold_recall": _r("tp_hold", "hold"),
        "n_send": int(d["send"]),
        "n_hold": int(d["hold"]),
    }

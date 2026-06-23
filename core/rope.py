r"""2D axial RoPE для внимания set-transformer (torch-only, входит в сабмишн).

Вращает q/k каждого токена на угол, линейный по его координатам (x, y) на поле.
Тогда q·k зависит только от ОТНОСИТЕЛЬНОЙ позиции пары токенов (трансляционно-
эквивариантное внимание в 2D). Перестановочная инвариантность по множеству
сохраняется: поворот привязан к координате токена, а не к его индексу. Голова
значений (v) не трогается — RoPE правит только скоры внимания.

`RoPEEncoderLayer`/`RoPEEncoder` зеркалят `nn.TransformerEncoderLayer`
(`norm_first=True`, gelu, та же раскладка dropout, без финального norm — как у
текущего бейзлайна), но q/k поворачиваются перед `F.scaled_dot_product_attention`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def axial_rope_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    """Обратные частоты НА ОСЬ: head_dim делится пополам (x|y), на каждую ось
    head_dim//2 каналов -> head_dim//4 частотных пар. Возвращает [head_dim//4]."""
    if head_dim % 4 != 0:
        raise ValueError(
            f"head_dim={head_dim} должен делиться на 4 (axial RoPE: половина на ось, пары)")
    axis_dim = head_dim // 2
    j = torch.arange(0, axis_dim, 2, dtype=torch.float32)
    return theta ** (-j / axis_dim)          # [axis_dim//2] = [head_dim//4]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1) по последней оси (rotate-half, как GPT-NeoX)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_1d(t: torch.Tensor, ang_half: torch.Tensor) -> torch.Tensor:
    """Повернуть пары канала t на углы ang_half. t: [B,H,N,axis_dim];
    ang_half: [B,N,axis_dim//2]. cos/sin считаются в fp32, кастятся в dtype t."""
    ang = torch.cat((ang_half, ang_half), dim=-1)        # [B,N,axis_dim]
    cos = ang.cos()[:, None].to(t.dtype)                 # [B,1,N,axis_dim] (broadcast по головам)
    sin = ang.sin()[:, None].to(t.dtype)
    return t * cos + _rotate_half(t) * sin


def apply_axial_rope(q, k, pos, inv_freq, apply_mask=None):
    """Повернуть q,k по 2D-координатам токенов (axial: первая половина head_dim
    крутится по x, вторая — по y).

    q,k: [B,H,N,head_dim]; pos: [B,N,2] (x,y в единицах поля); inv_freq: [head_dim//4];
    apply_mask: [B,N] bool — False обнуляет угол (поворот=identity, для непозиционного
    CLS). Возвращает (q_rot, k_rot)."""
    axis_dim = q.shape[-1] // 2
    # углы по осям: [B,N,head_dim//4]; считаем в fp32 для устойчивости
    inv_freq = inv_freq.to(torch.float32)
    ang_x = pos[..., 0:1].to(torch.float32) * inv_freq
    ang_y = pos[..., 1:2].to(torch.float32) * inv_freq
    if apply_mask is not None:
        m = apply_mask[..., None].to(torch.float32)      # [B,N,1]
        ang_x = ang_x * m
        ang_y = ang_y * m
    q_rot = torch.cat((_rope_1d(q[..., :axis_dim], ang_x),
                       _rope_1d(q[..., axis_dim:], ang_y)), dim=-1)
    k_rot = torch.cat((_rope_1d(k[..., :axis_dim], ang_x),
                       _rope_1d(k[..., axis_dim:], ang_y)), dim=-1)
    return q_rot, k_rot


class RoPEEncoderLayer(nn.Module):
    """Слой энкодера (norm_first, gelu) с axial-RoPE в self-attention.

    Зеркалит `nn.TransformerEncoderLayer`, но q/k формируются явно и вращаются по
    координатам токенов перед SDPA. `inv_freq` — non-persistent buffer (не в
    state_dict, девайс-корректен)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, theta: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} не делится на nhead={nhead}")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.attn_dropout = float(dropout)

        self.in_proj = nn.Linear(d_model, 3 * d_model)       # упакованные q,k,v
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)                   # внутри FFN
        self.dropout1 = nn.Dropout(dropout)                  # после attention
        self.dropout2 = nn.Dropout(dropout)                  # после FFN
        self.register_buffer(
            "inv_freq", axial_rope_inv_freq(self.head_dim, theta), persistent=False)

    def _sa(self, x, positions, apply_mask, key_padding_mask):
        B, N, D = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        # [B,N,D] -> [B,H,N,head_dim]
        shape = lambda t: t.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        q, k = apply_axial_rope(q, k, positions, self.inv_freq, apply_mask)
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask [B,N] True=игнорировать -> bool-маска SDPA True=участвует
            attn_mask = (~key_padding_mask)[:, None, None, :]   # [B,1,1,N]
        drop = self.attn_dropout if self.training else 0.0
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop)
        o = o.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(o)

    def _ff(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

    def forward(self, x, positions, apply_mask, key_padding_mask=None):
        x = x + self.dropout1(self._sa(self.norm1(x), positions, apply_mask, key_padding_mask))
        x = x + self.dropout2(self._ff(self.norm2(x)))
        return x


class RoPEEncoder(nn.Module):
    """Стек `RoPEEncoderLayer`. Координаты/apply_mask общие для всех слоёв."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 num_layers: int, dropout: float, theta: float):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout, theta)
            for _ in range(num_layers)
        ])

    def forward(self, x, positions, apply_mask, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, positions, apply_mask, src_key_padding_mask)
        return x

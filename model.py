r"""Policy/value сеть для Orbit Wars.

Архитектура (соответствует наброску схемы):

    планеты  кометы  флоты  солнце  глобал("доп фичи")
       |       |       |      |        |
    build_mlp энкодеры (каждый -> d_model) + обучаемый type-эмбеддинг
       \_______ конкат в единый набор токенов _______/
                          |
              Transformer encoder (self-attention, set-инвариантный)
                          |
              хидден «мест» (планеты + кометы)
                     /          \
                 mlp_from      mlp_to        -> from_emb, to_emb
                     \    X (dot)   /
                  S[from, to]  (+ колонка hold)
                          |
              softmax по оси `to`  -> распределение для каждого источника
                          |
                +  value-голова с глобального токена

Декод (``act``): каждая своя планета выбирает одну цель (или `hold`) через argmax;
``num_ships`` выбирает голова ``mlp_frac`` — 4-классовый бакет доли гарнизона
{25,50,75,100}%, обусловленный ПАРОЙ (источник, цель): конкат хидденов `h_src`⊕`h_tgt`
(факторизация `p(куда)·p(сколько|куда)`); угол запуска берётся из
:class:`core.geo_lite.GeoEngine` (обёртка над ``orbit_lite``, чтобы движущиеся цели
брались с правильным упреждением).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import features, geo_lite
from core.utils import build_mlp
from core.features import EncodedObs, FeatureConfig


@dataclass
class ModelConfig:
    d_model: int = 128
    d_k: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ffn: int = 512
    dropout: float = 0.0
    enc_hidden: int = 128       # ширина скрытого слоя per-type энкодеров
    head_hidden: int = 128      # ширина скрытого слоя голов from/to/value/frac
    n_frac_buckets: int = 4     # голова числа кораблей: бакеты доли {25,50,75,100}%
    capture_buffer: int = 2     # (не используется) num_ships = min(garrison, target.ships+1+buffer)


# доли гарнизона по бакетам и обратный декод bucket -> целое число кораблей.
# Округляем долю вниз (floor): не пере-засылаем, ошибка < 1 корабля.
# 100%->весь гарнизон, 25/50/75%->floor(доля·g).
def bucket_to_ships(bucket: int, garrison: int) -> int:
    """Бакет доли {0:25,1:50,2:75,3:100}% -> целое число кораблей из гарнизона."""
    if bucket >= 3:
        return garrison
    if bucket == 0:
        n = int(math.floor(0.25 * garrison))
    elif bucket == 1:
        n = int(math.floor(0.50 * garrison))
    else:
        n = int(math.floor(0.75 * garrison))
    return max(1, min(garrison, n))


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # per-type энкодеры: сырые фичи -> токен-эмбеддинги d_model
        self.enc_planet = build_mlp(features.PLANET_FEAT_DIM, [cfg.enc_hidden], d, out_norm=True)
        self.enc_comet = build_mlp(features.COMET_FEAT_DIM, [cfg.enc_hidden], d, out_norm=True)
        self.enc_fleet = build_mlp(features.FLEET_FEAT_DIM, [cfg.enc_hidden], d, out_norm=True)
        self.enc_global = build_mlp(features.GLOBAL_FEAT_DIM, [cfg.enc_hidden], d, out_norm=True)

        # обучаемые type-эмбеддинги + константный токен-солнце
        self.type_planet = nn.Parameter(torch.zeros(d))
        self.type_comet = nn.Parameter(torch.zeros(d))
        self.type_fleet = nn.Parameter(torch.zeros(d))
        self.type_sun = nn.Parameter(torch.zeros(d))
        self.type_global = nn.Parameter(torch.zeros(d))
        self.sun_token = nn.Parameter(torch.zeros(d))
        for p in (self.type_planet, self.type_comet, self.type_fleet,
                  self.type_sun, self.type_global, self.sun_token):
            nn.init.normal_(p, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.ffn,
            dropout=cfg.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False)

        # головы рёбер + value-голова
        self.mlp_from = build_mlp(d, [cfg.head_hidden], cfg.d_k)
        self.mlp_to = build_mlp(d, [cfg.head_hidden], cfg.d_k)
        self.mlp_hold = build_mlp(d, [cfg.head_hidden], 1)
        self.mlp_value = build_mlp(d, [cfg.head_hidden], 1)
        # голова числа кораблей: вход 2*d (конкат h_src ⊕ h_tgt), выход — бакеты доли
        self.mlp_frac = build_mlp(2 * d, [cfg.head_hidden], cfg.n_frac_buckets)

    # -- forward ---------------------------------------------------------------
    def forward(self, enc: EncodedObs, frac_pairs=None) -> Dict[str, torch.Tensor]:
        """Возвращает dict: ``logits`` [B,M,M+1], ``pi`` [B,M,M+1], ``value`` [B],
        ``h_place`` [B,M,d], ``place_mask`` [B,M].

        ``frac_pairs=(b_idx, s_idx, t_idx)`` (teacher forcing) — если переданы, голова
        числа кораблей считается ВНУТРИ forward (важно для DDP: иначе градиенты
        ``mlp_frac`` не синхронизируются) и кладётся в ``frac_logits`` [N,4]."""
        planet_tok = self.enc_planet(enc.planet_feats) + self.type_planet
        comet_tok = self.enc_comet(enc.comet_feats) + self.type_comet
        fleet_tok = self.enc_fleet(enc.fleet_feats) + self.type_fleet
        B = planet_tok.shape[0]
        sun_tok = (self.sun_token + self.type_sun).expand(B, 1, -1)
        global_tok = (self.enc_global(enc.global_feats) + self.type_global).unsqueeze(1)

        x = torch.cat([planet_tok, comet_tok, fleet_tok, sun_tok, global_tok], dim=1)

        # маска паддинга ключей: True == игнорировать. Солнце и глобал есть всегда.
        always = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        pad_mask = torch.cat(
            [~enc.planet_mask, ~enc.comet_mask, ~enc.fleet_mask, always, always], dim=1
        )
        h = self.encoder(x, src_key_padding_mask=pad_mask)

        P = planet_tok.shape[1]
        C = comet_tok.shape[1]
        h_planet = h[:, :P]
        h_comet = h[:, P:P + C]
        h_global = h[:, -1]                      # хидден глобального/CLS токена

        h_place = torch.cat([h_planet, h_comet], dim=1)        # [B, M, d]
        place_mask = torch.cat([enc.planet_mask, enc.comet_mask], dim=1)  # [B, M]
        M = h_place.shape[1]

        from_emb = self.mlp_from(h_place)        # [B, M, d_k]
        to_emb = self.mlp_to(h_place)            # [B, M, d_k]
        scores = torch.matmul(from_emb, to_emb.transpose(1, 2)) / math.sqrt(self.cfg.d_k)
        hold = self.mlp_hold(h_place)            # [B, M, 1]

        neg_inf = float("-inf")
        # маскируем паддинг-колонки целей (колонка hold остаётся валидной)
        col_pad = (~place_mask).unsqueeze(1)                 # [B, 1, M]
        scores = scores.masked_fill(col_pad, neg_inf)
        # маскируем self-target по диагонали
        eye = torch.eye(M, dtype=torch.bool, device=h.device).unsqueeze(0)
        scores = scores.masked_fill(eye, neg_inf)

        logits = torch.cat([scores, hold], dim=-1)           # [B, M, M+1]
        pi = F.softmax(logits, dim=-1)                       # softmax по `to` (+hold)
        value = self.mlp_value(h_global).squeeze(-1)         # [B]
        out = {"logits": logits, "pi": pi, "value": value,
               "h_place": h_place, "place_mask": place_mask}
        if frac_pairs is not None:
            # teacher forcing: гейзерим хиддены источника и ЭКСПЕРТНОЙ цели, конкат -> mlp_frac
            b_idx, s_idx, t_idx = frac_pairs
            h_src = h_place[b_idx, s_idx]                     # [N, d]
            h_tgt = h_place[b_idx, t_idx]                     # [N, d]
            out["frac_logits"] = self.mlp_frac(torch.cat([h_src, h_tgt], dim=-1))  # [N, 4]
        return out

    # -- загрузка чекпойнта -----------------------------------------------------
    @classmethod
    def load(cls, path: str, map_location="cpu"):
        """Загрузить чекпойнт sft.engine -> (готовая сеть в eval-режиме, FeatureConfig).

        Восстанавливает ``ModelConfig``/``FeatureConfig`` из самого чекпойнта, поэтому
        форму угадывать не нужно. Возвращает кортеж, т.к. ``act`` ждёт ``cfg``.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        mcfg = ModelConfig(**ckpt["model_cfg"])
        fcfg = FeatureConfig(**ckpt["feature_cfg"])
        net = cls(mcfg)
        net.load_state_dict(ckpt["model_state"])
        net.to(map_location)
        net.eval()
        return net, fcfg

    # -- декод -----------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs, cfg: FeatureConfig = FeatureConfig(), *,
            decode: str = "greedy", temperature: float = 1.0,
            generator: "torch.Generator | None" = None) -> List[list]:
        """obs -> список [from_planet_id, angle_rad, num_ships] (по одному на источник).

        ``decode``: ``"greedy"`` — argmax по `to` (как в сабмишне); ``"sample"`` —
        сэмпл из ``softmax(logits / temperature)`` (стохастическая политика для eval).
        Колонка hold всегда валидна, так что у каждого источника есть ≥1 вариант.
        """
        device = next(self.parameters()).device
        enc = features.encode(obs, cfg=cfg, device=device)
        out = self.forward(enc)
        logits = out["logits"][0]                # [M, M+1]
        h_place = out["h_place"][0]              # [M, d]
        hold_idx = logits.shape[1] - 1

        moves: List[list] = []
        geo = None                               # geo_lite-движок строим лениво (один раз за act)
        for i in enc.owned_idx:
            if decode == "sample":
                probs = torch.softmax(logits[i] / max(1e-6, temperature), dim=-1)
                j = int(torch.multinomial(probs, 1, generator=generator).item())
            else:
                j = int(torch.argmax(logits[i]).item())
            if j == hold_idx:
                continue
            src = enc.places[i]
            tgt = enc.places[j]
            if tgt is None or src is None:
                continue
            garrison = int(src.ships)
            if garrison <= 0:
                continue
            # число кораблей: бакет доли, обусловленный парой (источник i -> цель j)
            fl = self.mlp_frac(torch.cat([h_place[i], h_place[j]], dim=-1))  # [4]
            if decode == "sample":
                bprobs = torch.softmax(fl / max(1e-6, temperature), dim=-1)
                bucket = int(torch.multinomial(bprobs, 1, generator=generator).item())
            else:
                bucket = int(torch.argmax(fl).item())
            num_ships = bucket_to_ships(bucket, garrison)
            if geo is None:
                geo = geo_lite.GeoEngine(obs, player=enc.player, device=device)
            angle, _eta, _hit = geo.intercept(src.id, tgt.id, num_ships)
            moves.append([src.id, float(angle), num_ships])
        return moves

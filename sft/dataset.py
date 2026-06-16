r"""SFT-датасет + collate с динамическим паддингом батча по каждой сущности.

Читает `data/sft.full_send.jsonl` (выход dataprep/preprocess.py). Каждый ход кодируется
:func:`core.features.encode` НА ЛЕТУ (распараллеливается DataLoader-воркерами), а
таргеты строятся layout-независимо (на источник: планета-назначение / hold / ignore) и
переводятся в индексы мест уже в `collate_fn`, когда известны batch-максимумы.

Динамический паддинг: модель `PolicyValueNet.forward` shape-driven (M берётся из форм
тензоров), поэтому каждую сущность паддим до МАКСИМУМА В БАТЧЕ, а не до 40/16/256.

Сплит train/val — ПО ЭПИЗОДАМ (meta.episode_id): соседние ходы одной партии сильно
коррелируют, поэлементный сплит протёк бы.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from core.features import FeatureConfig, encode

# спец-метки таргета источника (до перевода в индекс места в collate)
HOLD = "HOLD"
IGNORE = "IGNORE"
IGNORE_INDEX = -100


# --- индекс файла (offset+episode на строку), кэш в сайдкар --------------------
def _index_path(path: str) -> str:
    return path + ".idx.json"


def build_index(path: str, max_lines: Optional[int] = None) -> Dict[str, list]:
    """Построить (или загрузить из кэша) индекс строк: байт-офсеты, episode_id, has_action.

    Кэш в ``<path>.idx.json`` валиден, пока совпадают размер и mtime файла.
    ``max_lines`` — прочитать только первые N строк (для быстрых проверок); такой
    частичный индекс кэш НЕ трогает (ни чтение, ни запись), чтобы не портить полный.
    """
    stat = os.stat(path)
    cache = _index_path(path)
    if max_lines is None and os.path.exists(cache):
        try:
            with open(cache, "r", encoding="utf-8") as f:
                idx = json.load(f)
            if idx.get("size") == stat.st_size and idx.get("mtime") == int(stat.st_mtime):
                return idx
        except (json.JSONDecodeError, OSError):
            pass  # битый кэш -> пересоберём

    offsets, episodes, has_action = [], [], []
    with open(path, "rb") as f:
        off = f.tell()
        line = f.readline()
        while line:
            if max_lines is not None and len(offsets) >= max_lines:
                break
            if line.strip():
                rec = json.loads(line)
                offsets.append(off)
                episodes.append(rec["meta"].get("episode_id"))
                has_action.append(bool(rec.get("sends") or rec.get("unresolved")))
            off = f.tell()
            line = f.readline()

    idx = {"size": stat.st_size, "mtime": int(stat.st_mtime),
           "offsets": offsets, "episodes": episodes, "has_action": has_action}
    if max_lines is None:
        try:
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(idx, f)
        except OSError:
            pass  # кэш не обязателен
    return idx


def split_indices(idx: Dict[str, list], val_frac: float, seed: int = 0,
                  limit: Optional[int] = None,
                  hold_subsample: float = 1.0) -> Tuple[List[int], List[int]]:
    """Разбить строки на train/val ПО ЭПИЗОДАМ. Возвращает списки индексов строк.

    hold_subsample < 1.0 — доля сохраняемых hold-only ходов (детерминированно по rng),
    чтобы при желании ослабить перекос ещё и прореживанием (по умолчанию 1.0 — все).
    """
    episodes = idx["episodes"]
    has_action = idx["has_action"]
    n = len(episodes) if limit is None else min(limit, len(episodes))
    rng = np.random.default_rng(seed)

    uniq = sorted({episodes[i] for i in range(n)}, key=lambda e: (e is None, e))
    perm = rng.permutation(len(uniq))
    n_val = int(round(len(uniq) * val_frac))
    val_eps = {uniq[perm[k]] for k in range(n_val)}

    train, val = [], []
    for i in range(n):
        if hold_subsample < 1.0 and not has_action[i] and rng.random() > hold_subsample:
            continue
        (val if episodes[i] in val_eps else train).append(i)
    return train, val


class SftDataset(Dataset):
    """Один элемент = (trimmed-фичи хода, список source-таргетов). Кодирует obs на лету."""

    def __init__(self, path: str, line_indices: List[int], idx: Dict[str, list],
                 cfg: FeatureConfig = FeatureConfig()):
        self.path = path
        self.line_indices = line_indices
        self.offsets = idx["offsets"]
        self.cfg = cfg
        self._fh = None  # файловый хэндл открываем лениво (свой на каждый воркер)

    def __len__(self) -> int:
        return len(self.line_indices)

    def _read(self, line_no: int) -> dict:
        if self._fh is None:
            self._fh = open(self.path, "rb")
        self._fh.seek(self.offsets[line_no])
        return json.loads(self._fh.readline())

    def __getitem__(self, i: int) -> dict:
        rec = self._read(self.line_indices[i])
        enc = encode(rec["state"], cfg=self.cfg, device=None)

        n_p = int(enc.planet_mask.sum().item())
        n_c = int(enc.comet_mask.sum().item())
        n_f = int(enc.fleet_mask.sum().item())
        mp = self.cfg.max_planets

        # id -> (kind, local_idx) по непустым местам (planets: 0..mp-1, comets: mp..)
        id2place: Dict[int, Tuple[str, int]] = {}
        for j, pl in enumerate(enc.places):
            if pl is None:
                continue
            id2place[int(pl.id)] = ("planet", j) if j < mp else ("comet", j - mp)

        sends = {int(a): int(b) for a, b in rec.get("sends", [])}
        unresolved = {int(x) for x in rec.get("unresolved", [])}

        # source-таргеты: (src_kind, src_local, tgt) где tgt = ('planet'|'comet',l)|HOLD|IGNORE
        sources: List[Tuple[str, int, object]] = []
        for j in enc.owned_idx:
            pl = enc.places[j]
            src_kind, src_local = ("planet", j) if j < mp else ("comet", j - mp)
            sid = int(pl.id)
            if sid in sends:
                dst = sends[sid]
                tgt = id2place.get(dst, IGNORE)   # назначение усечено -> ignore
            elif sid in unresolved:
                tgt = IGNORE                       # угол не восстановился -> не учим
            else:
                tgt = HOLD
            sources.append((src_kind, src_local, tgt))

        return {
            "planet_feats": enc.planet_feats[0, :n_p].clone(),   # [n_p, 20]
            "comet_feats": enc.comet_feats[0, :n_c].clone(),     # [n_c, 25]
            "fleet_feats": enc.fleet_feats[0, :n_f].clone(),     # [n_f, 10]
            "global_feats": enc.global_feats[0].clone(),         # [11]
            "n_p": n_p, "n_c": n_c, "n_f": n_f,
            "sources": sources,
        }


def _pad_stack(items: List[torch.Tensor], n_max: int, dim: int):
    """Сложить список [n_i, dim] в [B, n_max, dim] (нулевой паддинг) + bool-маску [B, n_max]."""
    B = len(items)
    out = torch.zeros(B, n_max, dim, dtype=torch.float32)
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    for b, t in enumerate(items):
        n = t.shape[0]
        if n:
            out[b, :n] = t
            mask[b, :n] = True
    return out, mask


def collate(batch: List[dict]) -> Tuple["BatchObs", torch.Tensor, int]:
    """Динамический паддинг по каждой сущности + сборка labels [B, M_b] (-100 = ignore)."""
    P_b = max(x["n_p"] for x in batch)
    C_b = max(x["n_c"] for x in batch)
    F_b = max(x["n_f"] for x in batch)
    M_b = P_b + C_b
    hold_idx = M_b

    planet_feats, planet_mask = _pad_stack([x["planet_feats"] for x in batch], P_b, 20)
    comet_feats, comet_mask = _pad_stack([x["comet_feats"] for x in batch], C_b, 25)
    fleet_feats, fleet_mask = _pad_stack([x["fleet_feats"] for x in batch], F_b, 10)
    global_feats = torch.stack([x["global_feats"] for x in batch], dim=0)  # [B, 11]

    labels = torch.full((len(batch), M_b), IGNORE_INDEX, dtype=torch.long)
    for b, x in enumerate(batch):
        for src_kind, src_local, tgt in x["sources"]:
            src_g = src_local if src_kind == "planet" else P_b + src_local
            if tgt == IGNORE:
                continue
            if tgt == HOLD:
                labels[b, src_g] = hold_idx
            else:
                tk, tl = tgt
                labels[b, src_g] = tl if tk == "planet" else P_b + tl

    obs = BatchObs(planet_feats, planet_mask, comet_feats, comet_mask,
                   fleet_feats, fleet_mask, global_feats)
    return obs, labels, hold_idx


class BatchObs:
    """Утиный аналог EncodedObs для forward: те же поля-тензоры (без places/owned_idx).

    PolicyValueNet.forward читает только *_feats и *_mask, поэтому остальное не нужно.
    """

    __slots__ = ("planet_feats", "planet_mask", "comet_feats", "comet_mask",
                 "fleet_feats", "fleet_mask", "global_feats")

    def __init__(self, planet_feats, planet_mask, comet_feats, comet_mask,
                 fleet_feats, fleet_mask, global_feats):
        self.planet_feats = planet_feats
        self.planet_mask = planet_mask
        self.comet_feats = comet_feats
        self.comet_mask = comet_mask
        self.fleet_feats = fleet_feats
        self.fleet_mask = fleet_mask
        self.global_feats = global_feats

    def to(self, device: torch.device) -> "BatchObs":
        return BatchObs(
            self.planet_feats.to(device), self.planet_mask.to(device),
            self.comet_feats.to(device), self.comet_mask.to(device),
            self.fleet_feats.to(device), self.fleet_mask.to(device),
            self.global_feats.to(device),
        )

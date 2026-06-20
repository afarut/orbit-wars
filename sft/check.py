r"""Быстрые проверки корректности SFT-пайплайна (без полноценного обучения).

1) Метки: для каждого send-источника таргет из Dataset указывает на ту же планету-
   назначение, что лежит в ``sends`` (т.е. id<->место смаппены верно).
2) Инвариант масок: после collate logits[label] для всех валидных источников КОНЕЧНЫ
   (метка не попала на замаскированную колонку — self/паддинг).
3) Обучаемость: оверфит ОДНОГО батча за ~150 шагов -> send_acc уходит вверх.

Запуск:  python -m sft.check --path /tmp/sft.smoke.jsonl
"""
from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import torch

from core.features import FeatureConfig, encode
from model import ModelConfig, PolicyValueNet

from . import loss as L
from .dataset import (HOLD, IGNORE, SftDataset, build_index, collate,
                      ship_bucket, split_indices)


def check_labels(path: str, cfg: FeatureConfig, n_samples: int = 300,
                 max_lines: int = 4000) -> None:
    """Сверка таргетов Dataset с содержимым sends (id->место->id round-trip)."""
    idx = build_index(path, max_lines=max_lines)
    mp = cfg.max_planets
    ds = SftDataset(path, list(range(len(idx["offsets"]))), idx, cfg=cfg)

    checked = miss = frac_checked = 0
    for i in range(min(n_samples, len(ds))):
        rec = ds._read(ds.line_indices[i])
        sends, send_ships = {}, {}
        for row in rec.get("sends", []):
            sends[int(row[0])] = int(row[1])
            if len(row) > 2:
                send_ships[int(row[0])] = int(row[2])
        if not sends:
            continue
        enc = encode(rec["state"], cfg=cfg, device=None)
        # (kind, local) -> id под раскладкой places; id -> гарнизон (для round-trip доли)
        id_of = {}
        ships_of = {}
        for j, pl in enumerate(enc.places):
            if pl is not None:
                id_of[("planet", j) if j < mp else ("comet", j - mp)] = int(pl.id)
                ships_of[int(pl.id)] = pl.ships

        item = ds[i]
        src_targets, src_buckets = {}, {}
        for src_kind, src_local, tgt, frac_bucket in item["sources"]:
            fid = id_of[(src_kind, src_local)]
            src_targets[fid] = tgt
            src_buckets[fid] = frac_bucket

        for from_id, dest_id in sends.items():
            tgt = src_targets.get(from_id)
            assert tgt is not None, f"источник {from_id} не среди owned"
            if tgt in (HOLD, IGNORE):
                miss += 1
                continue
            mapped = id_of[tgt]
            assert mapped == dest_id, f"метка {mapped} != dest {dest_id} (src {from_id})"
            checked += 1
            # round-trip бакета доли: метка датасета == пересчёт ship_bucket(ships, гарнизон)
            if from_id in send_ships:
                exp = ship_bucket(send_ships[from_id], ships_of[from_id])
                got = src_buckets[from_id]
                assert got == exp, f"бакет {got} != {exp} (src {from_id}, ships {send_ships[from_id]})"
                frac_checked += 1
    print(f"[labels] сверено send-меток: {checked}; бакетов доли: {frac_checked}; "
          f"hold/ignore у send-источников: {miss} — OK")


def check_mask_invariant(path: str, cfg: FeatureConfig, max_lines: int = 4000) -> None:
    """logits на позиции метки должны быть конечны (метка не на замаскированной колонке)."""
    idx = build_index(path, max_lines=max_lines)
    tr, _ = split_indices(idx, val_frac=0.0, limit=400)
    ds = SftDataset(path, tr, idx, cfg=cfg)
    batch = [ds[i] for i in range(min(64, len(ds)))]
    obs, labels, frac_labels, hold_idx = collate(batch)

    model = PolicyValueNet(ModelConfig())
    frac_pairs, frac_tgt = L.frac_pairs_from(labels, frac_labels)
    with torch.no_grad():
        out = model(obs, frac_pairs=frac_pairs)
    logits = out["logits"]
    B, M, C = logits.shape
    assert C == M + 1 and M == hold_idx, f"формы: {logits.shape}, hold_idx={hold_idx}"

    valid = labels != -100
    gathered = logits.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    bad = (~torch.isfinite(gathered)) & valid
    assert bad.sum().item() == 0, f"{int(bad.sum())} меток попали на -inf колонку!"

    # голова числа кораблей: forma [N,4], N == число валидных бакетов доли
    n_frac = int((frac_labels != -100).sum())
    assert tuple(out["frac_logits"].shape) == (n_frac, 4), \
        f"frac_logits {tuple(out['frac_logits'].shape)} != ({n_frac}, 4)"
    print(f"[mask] logits {tuple(logits.shape)}, валидных меток {int(valid.sum())}, "
          f"frac_logits ({n_frac}, 4) — OK")


def check_overfit_one_batch(path: str, cfg: FeatureConfig, steps: int = 250,
                            batch_size: int = 96, max_lines: int = 4000) -> None:
    """Оверфит ОДНОГО батча: send_acc должен дойти до ~1.0 (метки/лосс обучаемы).

    Печатает кривую send_acc/hold_acc, чтобы видеть, что модель реально учится слать,
    а не просто залипает в hold. w_hold=1.0 (без понижения) — чисто тест плумбинга.
    """
    idx = build_index(path, max_lines=max_lines)
    tr, _ = split_indices(idx, val_frac=0.0, limit=2000)
    ds = SftDataset(path, tr, idx, cfg=cfg)
    batch = [ds[i] for i in range(min(batch_size, len(ds)))]
    obs, labels, frac_labels, hold_idx = collate(batch)
    frac_pairs, frac_tgt = L.frac_pairs_from(labels, frac_labels)

    model = PolicyValueNet(ModelConfig())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    # берём ЛУЧШИЙ acc по кривой: на высоком lr/крошечном батче возможен поздний
    # разнос (overshoot AdamW) — тест про обучаемость «дошёл ли вверх», не про стабильность.
    best_send, best_frac, any_frac = 0.0, 0.0, False
    for step in range(steps + 1):
        out = model(obs, frac_pairs=frac_pairs)
        loss = (L.policy_loss(out["logits"], labels, hold_idx, w_hold=1.0)
                + L.fraction_loss(out["frac_logits"], frac_tgt))
        if step % 50 == 0:
            m = L.policy_metrics(out["logits"], labels, hold_idx)
            facc, fn = L.fraction_acc(out["frac_logits"], frac_tgt)
            best_send = max(best_send, m["send_acc"])
            if fn:
                best_frac, any_frac = max(best_frac, facc), True
            print(f"[overfit] step {step:3d}  loss={loss.item():.3f}  "
                  f"send_acc={m['send_acc']:.3f}  hold_acc={m['hold_acc']:.3f}  "
                  f"frac_acc={facc:.3f}  (n_send={m['n_send']} n_frac={fn})")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert best_send > 0.6, \
        "send_acc не вырос — возможен баг в метках/лоссе"
    assert not any_frac or best_frac > 0.6, \
        "frac_acc не вырос — возможен баг в метках/лоссе головы числа кораблей"


def main() -> None:
    ap = argparse.ArgumentParser(description="Быстрые проверки SFT-пайплайна")
    ap.add_argument("--path", default="data/sft.full_send.jsonl")
    ap.add_argument("--steps", type=int, default=250, help="шагов оверфита одного батча")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--max-lines", type=int, default=4000,
                    help="сколько первых строк читать в индекс (быстрее, без полного парса)")
    args = ap.parse_args()
    cfg = FeatureConfig()
    check_labels(args.path, cfg, max_lines=args.max_lines)
    check_mask_invariant(args.path, cfg, max_lines=args.max_lines)
    check_overfit_one_batch(args.path, cfg, steps=args.steps,
                            batch_size=args.batch_size, max_lines=args.max_lines)
    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")


if __name__ == "__main__":
    main()

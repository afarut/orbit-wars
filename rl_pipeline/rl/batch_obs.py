"""Контейнер obs для PolicyValueNet.forward (поля-тензоры, без places/owned_idx)."""
from __future__ import annotations

import torch


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

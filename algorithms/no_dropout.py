"""No modular dropout -- all branches always active (baseline for multi-branch)."""

import torch
from .base import ModularDropout


class NoDropout(ModularDropout):
    """All branches always on with equal weight."""

    @property
    def expected_mask_sum(self):
        return self.num_modules

    def get_mask(self, category_ids, training=True, **kwargs):
        B = category_ids.size(0)
        return torch.ones(B, self.num_modules, device=category_ids.device)

"""Random modular dropout — category-AGNOSTIC Bernoulli masking baseline.

Used by the E3 ablation as the "no assignment matrix A" point: each branch is
kept with the same probability for every input, regardless of superclass.
Combined with the denom_mode toggle this gives the rand_fixed / rand_naive
configurations that isolate (i) the role of A from (ii) the role of the
fixed-denominator merge.

Revived from an archived early implementation with the addition of
the denom_mode flag introduced for E3.
"""

import torch
from .base import ModularDropout


class RandomDropout(ModularDropout):
    """Category-agnostic Bernoulli masking.

    Each branch is independently kept with probability `keep_prob = 1 - drop_prob`,
    irrespective of the input category.

    denom_mode='fixed'    -> expected_mask_sum = K * keep_prob  (constant)
    denom_mode='adaptive' -> expected_mask_sum = None           (model uses ÷Σm_k)
    """

    def __init__(self, num_modules, num_categories, drop_prob=0.5,
                 denom_mode='fixed'):
        super().__init__(num_modules, num_categories)
        self.keep_prob = 1.0 - drop_prob
        if denom_mode not in ('fixed', 'adaptive'):
            raise ValueError(f"denom_mode must be 'fixed' or 'adaptive', got {denom_mode}")
        self.denom_mode = denom_mode
        self.use_adaptive_denom = (denom_mode == 'adaptive')

    @property
    def expected_mask_sum(self):
        if self.denom_mode == 'fixed':
            return self.num_modules * self.keep_prob
        return None

    def get_mask(self, category_ids, training=True, **kwargs):
        B = category_ids.size(0)
        device = category_ids.device

        if training:
            return torch.bernoulli(
                torch.full((B, self.num_modules), self.keep_prob, device=device))
        return torch.full((B, self.num_modules), self.keep_prob, device=device)

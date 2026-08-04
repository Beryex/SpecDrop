"""Stochastic SpecDrop: same fixed assignment as Soft SpecDrop, but with
Bernoulli mask sampling at training time.

Used by the E3 fixed-vs-stochastic-denominator ablation to isolate the
mask-stochasticity factor (binary sampling) from the merge-denominator
factor (constant S vs per-sample Σm_k). Two configs:

    denom_mode='fixed'    : ÷ S = p_active + (K-1)*p_inactive  (Theorem 2)
    denom_mode='adaptive' : ÷ Σ_k m_k                         (Jensen bias)

At eval, the soft probability matrix is returned (no sampling) — exactly the
expectation of the Bernoulli draws — so merged_eval = E[merged_train] for
the fixed-denominator case (T2 in tests/test_fixed_denominator_invariant.py).
"""

import torch
from .soft_specdrop import SoftSpecDrop


class StochasticSpecDrop(SoftSpecDrop):
    """Bernoulli-sampled category-conditioned routing.

    Identical assignment matrix and probability schedule to SoftSpecDrop, but
    training uses binary Bernoulli draws. Eval returns the soft expected probs
    (matching SoftSpecDrop) for like-for-like comparison.
    """

    def __init__(self, num_modules, num_categories,
                 p_active=0.9, p_inactive=0.1,
                 assignment='round_robin',
                 warmup_ratio=0.0, total_epochs=200,
                 denom_mode='fixed',
                 assignment_seed=42):
        self.assignment_seed = assignment_seed
        super().__init__(num_modules, num_categories,
                         p_active=p_active, p_inactive=p_inactive,
                         assignment=assignment,
                         warmup_ratio=warmup_ratio, total_epochs=total_epochs)
        if denom_mode not in ('fixed', 'adaptive'):
            raise ValueError(f"denom_mode must be 'fixed' or 'adaptive', got {denom_mode}")
        self.denom_mode = denom_mode
        self.use_adaptive_denom = (denom_mode == 'adaptive')

    @property
    def expected_mask_sum(self):
        """Fixed: S = p_active + (K-1)*p_inactive. Adaptive: None (model uses ÷Σm_k)."""
        if self.denom_mode == 'fixed':
            return self._mask_scale
        return None

    def get_mask(self, category_ids, training=True, **kwargs):
        """Training: Bernoulli(soft_probs). Eval: soft_probs (expectation)."""
        soft = super().get_mask(category_ids, training=training, **kwargs)
        if training:
            return torch.bernoulli(soft)
        return soft

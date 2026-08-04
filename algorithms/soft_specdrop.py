"""Soft SpecDrop: fixed category-conditioned routing with soft weights.

Same fixed assignment matrix as SpecDrop, but uses the probability matrix
directly as soft weights during BOTH training and inference (no Bernoulli
sampling). This removes the stochastic masking variance while preserving
the category-conditioned structure.

Supports optional warmup schedule: start with uniform weights (all branches
equal), cosine-anneal to target (p_active, p_inactive). This lets the model
learn good representations before specialization pressure is applied.
The mask_scale stays constant throughout — only the weight distribution changes.

Per-category routing extension (2026-04-18):
    When `frac_per_category` is supplied, each category c gets its own
    (p_a^c, p_i^c) via a linear gap mapping:
        gap_c = g * ((1 - frac_c) / (1 - 1/M))**beta,  g = p_a - p_i
        p_a^c = S/K + gap_c * (K-1)/K
        p_i^c = S/K - gap_c / K
    S = p_a^c + (K-1) p_i^c is preserved for every c (verified in tests),
    so the fixed-denominator merge and all downstream theory are unchanged.
    With balanced frac (1/M each), (p_a^c, p_i^c) = (p_a, p_i) bit-identical,
    so CIFAR behavior is unchanged when `frac_per_category=None`.

Key properties:
    - Zero learnable routing params (same as SpecDrop)
    - Only standard CE loss (no MI loss, no load-balance loss)
    - Deterministic: same input always produces same mask
    - Constant mask_scale: train-test consistency preserved during warmup
"""

import math
import torch
from .base import ModularDropout


class SoftSpecDrop(ModularDropout):
    """Fixed category-conditioned soft routing with optional warmup.

    For sample with category c, branch k receives weight:
        w_k(c) = A[c,k] * p_active(t, c) + (1 - A[c,k]) * p_inactive(t, c)

    Without warmup: p_active/p_inactive are constant.
    With warmup: cosine anneal from uniform (S/K, S/K) to (p_active, p_inactive)
    over warmup_epochs, where S = p_active + (K-1)*p_inactive is constant.

    When `frac_per_category` is None (default), (p_active, p_inactive) are
    category-independent scalars — original behavior. When supplied, they
    become per-category targets (p_active_c, p_inactive_c) derived from the
    linear gap mapping described at the top of this file.
    """

    def __init__(self, num_modules, num_categories,
                 p_active=0.9, p_inactive=0.1,
                 assignment='round_robin',
                 warmup_ratio=0.0, total_epochs=200,
                 assignment_seed=42,
                 frac_per_category=None, amplification_beta=1.0,
                 warmup_schedule='cosine',
                 warmup_unit='epoch'):
        super().__init__(num_modules, num_categories)
        self.p_active_final = p_active
        self.p_inactive_final = p_inactive
        self.warmup_ratio = warmup_ratio
        self.warmup_epochs = int(warmup_ratio * total_epochs)
        self.current_epoch = 0
        # Used only when assignment='random' (E4 ablation); ignored for round_robin.
        self.assignment_seed = assignment_seed
        # Warmup schedule shape: 'cosine' (default, CIFAR + main results) or
        # 'linear' (standard LR-warmup convention from Transformer / BERT / GPT).
        # Both share endpoints S/K → (p_a, p_i); only the interpolation form
        # differs. Default cosine preserves CIFAR bit-identity.
        if warmup_schedule not in ('cosine', 'linear'):
            raise ValueError(
                f"warmup_schedule must be 'cosine' or 'linear', got {warmup_schedule!r}")
        self.warmup_schedule = warmup_schedule
        # Warmup granularity:
        #   'epoch' (default) — progress measured by current_epoch/warmup_epochs.
        #     Coarse: on 10-epoch NLP run, only 10 step-changes (stair curve).
        #     Preserves CIFAR-era bit-identity when default.
        #   'step'  — progress measured by current_step/total_warmup_steps.
        #     Fine: on 100M × 10ep ≈ 30k steps, smooth per-step cosine,
        #     matching the per-step LR schedule already in trainer_nlp.
        # Requires `set_total_steps(total_steps)` to have been called by the
        # trainer (idempotent, no-op for warmup_unit='epoch').
        if warmup_unit not in ('epoch', 'step'):
            raise ValueError(
                f"warmup_unit must be 'epoch' or 'step', got {warmup_unit!r}")
        self.warmup_unit = warmup_unit
        self.current_step = 0
        self.total_warmup_steps = 0  # populated by set_total_steps() for 'step'

        # Build assignment matrix A (M x K) — identical to SpecDrop
        self.assignment = self._build_assignment(assignment)

        # Constant mask scale: S = p_active + (K-1) * p_inactive
        self._mask_scale = p_active + (num_modules - 1) * p_inactive
        # Uniform starting point that preserves scale: S / K
        self._p_uniform = self._mask_scale / num_modules

        # Per-category target weights. When frac_per_category is None, default
        # to uniform 1/M (→ gap_c = g for every c → scalar behavior preserved).
        self.frac_per_category = frac_per_category
        self.amplification_beta = float(amplification_beta)
        self._p_active_target, self._p_inactive_target = self._compute_per_category_targets(
            frac_per_category, amplification_beta)

    def _compute_per_category_targets(self, frac_per_category, beta):
        """Build (M,) tensors of per-category (p_a^c, p_i^c) targets.

        If frac_per_category is None, returns scalar-broadcast tensors so the
        behavior is bit-identical to the pre-extension scalar path.
        """
        M, K = self.num_categories, self.num_modules
        S = self._mask_scale
        g = self.p_active_final - self.p_inactive_final

        if frac_per_category is None:
            fracs = torch.full((M,), 1.0 / M, dtype=torch.float32)
        else:
            assert len(frac_per_category) == M, (
                f"frac_per_category length {len(frac_per_category)} != "
                f"num_categories {M}")
            fracs = torch.tensor(list(frac_per_category), dtype=torch.float32)
            s = float(fracs.sum())
            assert abs(s - 1.0) < 1e-4, (
                f"frac_per_category must sum to 1.0 (got {s:.6f}); "
                f"pass raw fractions, not counts.")
            # Clamp tiny negatives from upstream rounding.
            fracs = fracs.clamp(min=0.0)

        # gap_c = g * [(1 - frac_c) / (1 - 1/M)]**beta
        denom = 1.0 - 1.0 / M
        base = (1.0 - fracs) / denom  # (M,), balanced → 1.0, small-frac → >1.0
        ratio = base.clamp(min=0.0) ** beta
        gap_c = g * ratio  # (M,)

        # Compute raw targets then clamp to [0, 1] for validity.
        #
        # Clamp is lossy in general (can break S = p_a^c + (K-1) p_i^c),
        # but it IS safe at pa=1.0: there S=1, and any over-cap category
        # clamps to (p_a=1, p_i=0), whose row sum = 1 = S. Elsewhere the
        # clamp must leave row sum == S within fp tolerance, or we raise —
        # that way any accidental constraint violation at intermediate pa
        # (e.g. via β>1) still surfaces instead of silently corrupting S.
        p_active_c = (S / K + gap_c * (K - 1) / K).clamp(0.0, 1.0)
        p_inactive_c = (S / K - gap_c / K).clamp(0.0, 1.0)

        row_sum = p_active_c + (K - 1) * p_inactive_c  # (M,)
        max_dev = float((row_sum - S).abs().max())
        if max_dev > 1e-4:
            offenders = [
                (i, float(row_sum[i]))
                for i in range(M) if abs(float(row_sum[i]) - S) > 1e-4
            ]
            raise ValueError(
                f"Per-category clamp broke S invariance (S={S:.4f}, max |Δ|="
                f"{max_dev:.4f}); offenders (category_id, row_sum): {offenders}. "
                f"This typically means amplification_beta>1 pushed gap_c past "
                f"the feasible region at a pa where S!=1 (clamp is only safe "
                f"at pa=1). Reduce β or lower p_active.")
        return p_active_c, p_inactive_c

    def _build_assignment(self, strategy):
        M, K = self.num_categories, self.num_modules
        A = torch.zeros(M, K)
        if strategy == 'round_robin':
            for c in range(M):
                A[c, c % K] = 1.0
        elif strategy == 'random':
            # Explicit RNG, not global numpy state, for reproducible assignments
            # even if the algorithm is instantiated after an RNG-consuming call.
            import numpy as np
            rng = np.random.default_rng(getattr(self, 'assignment_seed', 42))
            perm = rng.permutation(M)
            for c in range(M):
                A[c, perm[c] % K] = 1.0
        else:
            raise ValueError(f"Unknown assignment strategy: {strategy}")
        return A

    def set_total_steps(self, total_steps):
        """Called by the trainer when warmup_unit='step' to communicate the
        total-step budget. Idempotent for warmup_unit='epoch'.

        `total_warmup_steps = warmup_ratio × total_steps` defines the cosine/
        linear ramp denominator for step-based progress. Same mechanism as
        warmup_epochs under epoch-based warmup.
        """
        self.total_warmup_steps = int(self.warmup_ratio * total_steps)

    def _warmup_progress(self):
        """Return float ∈ [0, 1] indicating warmup completion fraction.

        Branches on warmup_unit. 'epoch' uses current_epoch/warmup_epochs (legacy);
        'step' uses current_step/total_warmup_steps (new, smooth per-step curve).
        Returns 1.0 (i.e., warmup complete) if the respective denominator is ≤ 0.
        """
        if self.warmup_unit == 'step':
            if self.total_warmup_steps <= 0:
                return 1.0
            return min(self.current_step / self.total_warmup_steps, 1.0)
        # 'epoch' — validated in __init__
        if self.warmup_epochs <= 0:
            return 1.0
        return min(self.current_epoch / self.warmup_epochs, 1.0)

    def _current_probs(self):
        """Return per-category (p_active_c, p_inactive_c) tensors at current
        warmup progress.

        Shapes: (M,) tensors. When frac_per_category is None, all entries are
        equal to the original scalar values (so downstream code is agnostic).
        """
        progress = self._warmup_progress()
        if progress >= 1.0:
            return self._p_active_target, self._p_inactive_target

        if self.warmup_schedule == 'cosine':
            alpha = 0.5 * (1 - math.cos(math.pi * progress))
        else:  # 'linear' — validated in __init__
            alpha = progress

        p_active = self._p_uniform + (self._p_active_target - self._p_uniform) * alpha
        p_inactive = self._p_uniform + (self._p_inactive_target - self._p_uniform) * alpha
        return p_active, p_inactive

    @property
    def expected_mask_sum(self):
        """Constant: p_active + (K-1) * p_inactive. Does NOT change with warmup
        or with per-category targets (property A: S^c = S for every c)."""
        return self._mask_scale

    def get_assignment_summary(self):
        summary = {}
        for k in range(self.num_modules):
            cats = (self.assignment[:, k] == 1).nonzero(as_tuple=True)[0].tolist()
            summary[f'module_{k}'] = cats
        return summary

    def get_mask(self, category_ids, training=True, **kwargs):
        """Return soft weights — same in training and inference.

        During warmup, weights transition from uniform to specialized.
        With per-category targets, each category_id indexes its own row of
        the prob_matrix.
        """
        device = category_ids.device
        p_active_c, p_inactive_c = self._current_probs()  # (M,) each
        A = self.assignment.to(device)
        pa = p_active_c.to(device).view(-1, 1)     # (M, 1)
        pi = p_inactive_c.to(device).view(-1, 1)   # (M, 1)
        prob_matrix = A * pa + (1 - A) * pi         # (M, K)
        return prob_matrix[category_ids]           # (B, K)

    def state_dict(self):
        return {'current_epoch': self.current_epoch}

    def load_state_dict(self, state):
        self.current_epoch = state.get('current_epoch', 0)

    def on_epoch_end(self, branches, epoch):
        """Update epoch counter for warmup schedule."""
        self.current_epoch = epoch + 1
        p_active_c, p_inactive_c = self._current_probs()
        # Scalar status for log convenience: if frac_per_category is None, all
        # entries are identical; otherwise we report mean to keep the log
        # lightweight.
        return {
            'p_active': round(float(p_active_c.mean()), 4),
            'p_inactive': round(float(p_inactive_c.mean()), 4),
            'warmup_progress': min(self.current_epoch / max(self.warmup_epochs, 1), 1.0),
        }

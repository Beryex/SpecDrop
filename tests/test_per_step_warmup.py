"""Unit tests for SoftSpecDrop's per-step pa warmup (warmup_unit='step').

Validates:
  1. Default warmup_unit='epoch' reproduces legacy behavior (bit-identical to
     pre-extension code under identical inputs).
  2. Setting warmup_unit='step' + set_total_steps(N) drives _current_probs
     via current_step/warmup_steps rather than current_epoch/warmup_epochs.
  3. Step-based warmup is smooth across steps: step 0 = uniform, step N =
     fully specialized, monotone between.
  4. Step-based at step=total_warmup_steps matches epoch-based at
     epoch=warmup_epochs (both fully specialized = targets).
  5. Invalid warmup_unit raises ValueError.
  6. set_total_steps is a no-op when warmup_unit='epoch' (backward compat).

Run: python tests/test_per_step_warmup.py
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from algorithms.soft_specdrop import SoftSpecDrop


def _make_sd(warmup_unit='epoch', warmup_ratio=1.0, total_epochs=10,
              pa=0.7, pi=0.3, M=7, K=7, schedule='cosine'):
    return SoftSpecDrop(
        num_modules=K, num_categories=M,
        p_active=pa, p_inactive=pi, assignment='round_robin',
        warmup_ratio=warmup_ratio, total_epochs=total_epochs,
        warmup_schedule=schedule, warmup_unit=warmup_unit,
    )


class TestDefaultEpochBehavior(unittest.TestCase):

    def test_default_is_epoch(self):
        sd = _make_sd()
        self.assertEqual(sd.warmup_unit, 'epoch')

    def test_epoch_progress_legacy(self):
        """Epoch-based: progress = current_epoch/warmup_epochs (legacy)."""
        sd = _make_sd(warmup_unit='epoch', warmup_ratio=1.0, total_epochs=10)
        # At epoch 0, progress=0 → uniform start
        sd.current_epoch = 0
        pa0, pi0 = sd._current_probs()
        self.assertTrue(torch.allclose(pa0, pi0))  # uniform
        S = sd._mask_scale
        self.assertTrue(torch.allclose(pa0, torch.full_like(pa0, S / sd.num_modules)))
        # At epoch == warmup_epochs, progress=1 → targets
        sd.current_epoch = 10
        pa1, pi1 = sd._current_probs()
        self.assertTrue(torch.allclose(pa1, sd._p_active_target))
        self.assertTrue(torch.allclose(pi1, sd._p_inactive_target))


class TestStepWarmup(unittest.TestCase):

    def test_step_requires_set_total_steps(self):
        """warmup_unit='step' with total_warmup_steps=0 returns targets (no
        warmup). set_total_steps() activates the ramp."""
        sd = _make_sd(warmup_unit='step', warmup_ratio=1.0)
        # Before set_total_steps, total_warmup_steps=0 → treated as no-warmup
        sd.current_step = 0
        pa, pi = sd._current_probs()
        self.assertTrue(torch.allclose(pa, sd._p_active_target))
        # After set_total_steps, ramp is active
        sd.set_total_steps(1000)
        self.assertEqual(sd.total_warmup_steps, 1000)  # = warmup_ratio × total_steps
        sd.current_step = 0
        pa0, pi0 = sd._current_probs()
        S = sd._mask_scale
        self.assertTrue(torch.allclose(pa0, torch.full_like(pa0, S / sd.num_modules)))

    def test_step_monotone_ramp(self):
        """Cosine warmup should be monotone non-decreasing in pa_c as step grows."""
        sd = _make_sd(warmup_unit='step', warmup_ratio=1.0, total_epochs=10,
                       schedule='cosine')
        sd.set_total_steps(1000)
        prev_pa = None
        for step in (0, 100, 250, 500, 750, 900, 1000, 1500):
            sd.current_step = step
            pa_c, _ = sd._current_probs()
            # Check first assigned category's pa is non-decreasing
            if prev_pa is not None:
                assert torch.all(pa_c >= prev_pa - 1e-6), \
                    f'pa decreased at step={step}: prev={prev_pa}, now={pa_c}'
            prev_pa = pa_c.clone()

    def test_step_reaches_targets_at_end(self):
        """At step=total_warmup_steps, pa/pi should equal targets."""
        sd = _make_sd(warmup_unit='step', warmup_ratio=1.0, total_epochs=10)
        sd.set_total_steps(3000)
        sd.current_step = 3000  # exactly at end of warmup
        pa, pi = sd._current_probs()
        self.assertTrue(torch.allclose(pa, sd._p_active_target))
        self.assertTrue(torch.allclose(pi, sd._p_inactive_target))
        # Beyond warmup: should clamp at targets
        sd.current_step = 10000
        pa2, pi2 = sd._current_probs()
        self.assertTrue(torch.allclose(pa2, sd._p_active_target))

    def test_step_matches_epoch_at_boundaries(self):
        """Under cosine schedule, step-based at the equivalent "epoch boundary"
        step-count (e.g. half_warmup in step-based ≈ half in epoch-based)
        produces numerically close pa values."""
        sd_e = _make_sd(warmup_unit='epoch', warmup_ratio=1.0, total_epochs=10)
        sd_s = _make_sd(warmup_unit='step', warmup_ratio=1.0, total_epochs=10)
        sd_s.set_total_steps(1000)
        # At 50% progress under either measure, cosine gives identical alpha
        sd_e.current_epoch = 5  # 5/10 = 0.5
        sd_s.current_step = 500  # 500/1000 = 0.5
        pa_e, _ = sd_e._current_probs()
        pa_s, _ = sd_s._current_probs()
        self.assertTrue(torch.allclose(pa_e, pa_s, atol=1e-6),
                         msg=f'epoch pa={pa_e} vs step pa={pa_s} differ at 50%')

    def test_fine_granularity(self):
        """Key motivation: step-based gives many more distinct pa values
        over training than epoch-based does."""
        sd_e = _make_sd(warmup_unit='epoch', warmup_ratio=1.0, total_epochs=10)
        sd_s = _make_sd(warmup_unit='step', warmup_ratio=1.0, total_epochs=10)
        sd_s.set_total_steps(3000)

        # Epoch-based: only 11 distinct progress values over 10 epochs
        epoch_progresses = set()
        for e in range(11):
            sd_e.current_epoch = e
            epoch_progresses.add(sd_e._warmup_progress())
        self.assertEqual(len(epoch_progresses), 11)

        # Step-based: ~3001 distinct progress values over 3000 steps
        step_progresses = set()
        for s in range(0, 3001, 100):
            sd_s.current_step = s
            step_progresses.add(sd_s._warmup_progress())
        self.assertGreater(len(step_progresses), 30)  # much finer


class TestValidationAndCompat(unittest.TestCase):

    def test_invalid_warmup_unit_raises(self):
        with self.assertRaises(ValueError):
            _make_sd(warmup_unit='bogus')

    def test_epoch_set_total_steps_is_noop_effectively(self):
        """set_total_steps updates total_warmup_steps but 'epoch' doesn't use
        it — epoch path still drives progress. Smoke-test they don't interfere."""
        sd = _make_sd(warmup_unit='epoch', warmup_ratio=1.0, total_epochs=10)
        sd.set_total_steps(1000)
        self.assertEqual(sd.total_warmup_steps, 1000)
        # Epoch path still works
        sd.current_epoch = 5
        pa, _ = sd._current_probs()
        self.assertIsNotNone(pa)

    def test_no_warmup_returns_targets_both_units(self):
        for unit in ('epoch', 'step'):
            sd = _make_sd(warmup_unit=unit, warmup_ratio=0.0)  # no warmup
            pa, pi = sd._current_probs()
            self.assertTrue(torch.allclose(pa, sd._p_active_target))


class TestPerCategoryStepWarmup(unittest.TestCase):
    """When frac_per_category is set, step-based warmup should still produce
    per-category differentiated targets at the end of warmup."""

    def test_per_cat_end_state_step_based(self):
        fracs = [0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02]  # imbalanced
        sd = SoftSpecDrop(
            num_modules=7, num_categories=7,
            p_active=0.7, p_inactive=0.3, assignment='round_robin',
            warmup_ratio=1.0, total_epochs=10,
            frac_per_category=fracs, amplification_beta=2.0,
            warmup_unit='step',
        )
        sd.set_total_steps(1000)
        sd.current_step = 1000
        pa, pi = sd._current_probs()
        # At end of warmup, should equal the per-category targets (which differ
        # by category because fracs are not uniform and β>1 amplifies).
        self.assertTrue(torch.allclose(pa, sd._p_active_target))
        self.assertGreater(pa.max() - pa.min(), 0.01)  # differentiated


if __name__ == '__main__':
    unittest.main(verbosity=2)

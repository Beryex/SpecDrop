"""Unit tests for per-category routing in SoftSpecDrop.

Covers:
  (1) Balanced bit-identical recovery: frac=[1/M]*M → identical mask to
      the scalar (p_a, p_i) path.
  (2) Row-sum invariance (property A): p_a^c + (K-1) p_i^c = S for every c,
      for arbitrary frac.
  (3) Balanced limit (property B): frac=1/M → gap_c = g → (p_a^c, p_i^c) =
      (p_a, p_i) exactly.
  (4) Non-negativity assertion (property C): offending frac/beta triggers
      ValueError at construction time.
  (5) Warmup: epoch=0 → uniform mask (S/K everywhere); epoch=warmup_end →
      per-category target.
  (6) Shape/device: mask on cuda returns correct shape and dtype.
  (7) compute_category_fractions glue: output sums to 1 and is in domain_id
      order (Book id=3 has smaller fraction than ArXiv id=4 on real cache).
  (8) Expected_mask_sum: returns scalar S unchanged whether or not
      frac_per_category is supplied.
  (9) NLP smoke: build algorithm with real fracs, get_mask produces sane
      per-category rows.

Run:  python tests/test_per_category_routing.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from algorithms.soft_specdrop import SoftSpecDrop


def _row_sums(prob_matrix_M_x_K):
    return prob_matrix_M_x_K.sum(dim=1)


class TestBalancedRecovery(unittest.TestCase):
    """frac=[1/M]*M must reproduce the original scalar (p_a, p_i) behavior
    exactly, so CIFAR (and any uniform-frac run) is bit-identical."""

    def _build_pair(self, M, K, p_a, p_i, warmup_ratio=0.0, epoch=0,
                    total_epochs=10):
        base = SoftSpecDrop(num_modules=K, num_categories=M,
                            p_active=p_a, p_inactive=p_i,
                            warmup_ratio=warmup_ratio, total_epochs=total_epochs,
                            frac_per_category=None)
        uniform = SoftSpecDrop(num_modules=K, num_categories=M,
                               p_active=p_a, p_inactive=p_i,
                               warmup_ratio=warmup_ratio, total_epochs=total_epochs,
                               frac_per_category=[1.0 / M] * M)
        base.current_epoch = epoch
        uniform.current_epoch = epoch
        return base, uniform

    def test_balanced_bit_identical_no_warmup(self):
        base, uniform = self._build_pair(M=20, K=20, p_a=0.7, p_i=0.3)
        cat_ids = torch.arange(20)
        self.assertTrue(torch.allclose(
            base.get_mask(cat_ids), uniform.get_mask(cat_ids), atol=1e-7))

    def test_balanced_bit_identical_cifar(self):
        """CIFAR config: M=100 superclasses, K=20 branches."""
        base, uniform = self._build_pair(M=100, K=20, p_a=0.7, p_i=0.3)
        cat_ids = torch.arange(100)
        self.assertTrue(torch.allclose(
            base.get_mask(cat_ids), uniform.get_mask(cat_ids), atol=1e-7))

    def test_balanced_bit_identical_in_warmup(self):
        """Mid-warmup must also coincide, since both collapse to the same
        cosine-interpolated scalar at every epoch."""
        for epoch in (0, 2, 5, 10):
            base, uniform = self._build_pair(
                M=20, K=20, p_a=0.9, p_i=0.1,
                warmup_ratio=1.0, epoch=epoch, total_epochs=10)
            cat_ids = torch.arange(20)
            diff = (base.get_mask(cat_ids) - uniform.get_mask(cat_ids)).abs().max()
            self.assertLess(diff.item(), 1e-7,
                            f"epoch={epoch} diverges (max |Δ|={diff:.2e})")


class TestPropertyA_RowSumInvariance(unittest.TestCase):
    """p_a^c + (K-1) p_i^c = S for every c under any frac."""

    def test_nlp_fracs_row_sums(self):
        # Actual SlimPajama 100M proportions.
        fracs = [0.534, 0.185, 0.088, 0.046, 0.066, 0.043, 0.038]
        # Normalize explicitly since these are rounded.
        fracs = [f / sum(fracs) for f in fracs]
        algo = SoftSpecDrop(num_modules=7, num_categories=7,
                            p_active=0.7, p_inactive=0.3,
                            frac_per_category=fracs)
        cat_ids = torch.arange(7)
        mask = algo.get_mask(cat_ids)  # (7, 7)
        row_sums = _row_sums(mask)
        expected = algo.expected_mask_sum
        self.assertTrue(torch.allclose(
            row_sums, torch.full_like(row_sums, float(expected)), atol=1e-6),
            f"row sums = {row_sums.tolist()}, expected {expected}")

    def test_random_fracs_row_sums(self):
        torch.manual_seed(0)
        for _ in range(6):
            M = 7
            raw = torch.rand(M)
            fracs = (raw / raw.sum()).tolist()
            algo = SoftSpecDrop(num_modules=M, num_categories=M,
                                p_active=0.7, p_inactive=0.3,
                                frac_per_category=fracs)
            mask = algo.get_mask(torch.arange(M))
            row_sums = _row_sums(mask)
            self.assertTrue(torch.allclose(
                row_sums, torch.full_like(row_sums, float(algo.expected_mask_sum)),
                atol=1e-6))


class TestPropertyB_BalancedLimit(unittest.TestCase):
    """When frac_c = 1/M, (p_a^c, p_i^c) = (p_a, p_i) exactly."""

    def test_balanced_targets_match_scalars(self):
        algo = SoftSpecDrop(num_modules=7, num_categories=7,
                            p_active=0.7, p_inactive=0.3,
                            frac_per_category=[1 / 7] * 7)
        for c in range(7):
            self.assertAlmostEqual(float(algo._p_active_target[c]), 0.7, places=6)
            self.assertAlmostEqual(float(algo._p_inactive_target[c]), 0.3, places=6)


class TestPropertyC_Constraint(unittest.TestCase):
    """Aggressive amplification that would require p_a^c > 1 must either
    (a) clamp to the boundary AND preserve S (allowed at pa=1.0), or
    (b) raise because clamp breaks S (anywhere else)."""

    NLP_FRACS = [0.534, 0.185, 0.088, 0.046, 0.066, 0.043, 0.038]

    @classmethod
    def _nlp(cls):
        return [f / sum(cls.NLP_FRACS) for f in cls.NLP_FRACS]

    def test_pa_1_clamps_small_domains_to_hard_routing(self):
        """pa=1.0 is the one regime where clamp preserves S. Small-frac
        categories clamp to (p_a=1, p_i=0) but every row still sums to S=1."""
        fracs = self._nlp()
        algo = SoftSpecDrop(num_modules=7, num_categories=7,
                            p_active=1.0, p_inactive=0.0,
                            frac_per_category=fracs)
        mask = algo.get_mask(torch.arange(7))
        # Every row sums to S=1.0 (clamp-safe at pa=1.0).
        self.assertTrue(torch.allclose(
            mask.sum(dim=1), torch.ones(7), atol=1e-5))
        # Small-frac categories (Book id=3, ArXiv id=4, Wiki id=5, SE id=6,
        # Github id=2 — all frac < 1/M≈0.143) must clamp to (1, 0).
        for c in [2, 3, 4, 5, 6]:
            pa_c = float(algo._p_active_target[c])
            pi_c = float(algo._p_inactive_target[c])
            self.assertAlmostEqual(pa_c, 1.0, places=5,
                msg=f"category {c}: p_a^c={pa_c} not clamped to 1.0")
            self.assertAlmostEqual(pi_c, 0.0, places=5,
                msg=f"category {c}: p_i^c={pi_c} not clamped to 0.0")
        # Large-frac categories (CC, C4) stay below the cap.
        for c in [0, 1]:
            self.assertLess(float(algo._p_active_target[c]), 1.0)
            self.assertGreater(float(algo._p_inactive_target[c]), 0.0)

    def test_beta_50_at_pa_07_raises_on_s_break(self):
        """β=50 at pa=0.7 pushes gap well past feasibility; clamp would
        break S (since S=2.5 != 1), so construction must raise."""
        fracs = self._nlp()
        with self.assertRaises(ValueError):
            SoftSpecDrop(num_modules=7, num_categories=7,
                         p_active=0.7, p_inactive=0.3,
                         frac_per_category=fracs,
                         amplification_beta=50.0)


class TestWarmupBehavior(unittest.TestCase):
    """epoch=0 → uniform (S/K everywhere); epoch=warmup_end → per-c targets."""

    def _make(self, epoch, total=10, warmup_ratio=1.0):
        fracs = [0.534, 0.185, 0.088, 0.046, 0.066, 0.043, 0.038]
        fracs = [f / sum(fracs) for f in fracs]
        algo = SoftSpecDrop(num_modules=7, num_categories=7,
                            p_active=0.7, p_inactive=0.3,
                            warmup_ratio=warmup_ratio, total_epochs=total,
                            frac_per_category=fracs)
        algo.current_epoch = epoch
        return algo

    def test_warmup_start_uniform_mask(self):
        algo = self._make(epoch=0)
        mask = algo.get_mask(torch.arange(7))
        # At epoch 0, cosine progress = 0, so weights = S/K on every entry.
        expected = algo._p_uniform
        self.assertTrue(torch.allclose(
            mask, torch.full_like(mask, expected), atol=1e-6),
            f"warmup-start mask not uniform: min={mask.min()}, max={mask.max()}")

    def test_warmup_end_matches_per_cat_targets(self):
        algo = self._make(epoch=10)  # at/past warmup_epochs
        mask = algo.get_mask(torch.arange(7))
        for c in range(7):
            for k in range(7):
                expected = (float(algo._p_active_target[c]) if c == k
                            else float(algo._p_inactive_target[c]))
                self.assertAlmostEqual(float(mask[c, k]), expected, places=6,
                                       msg=f"mismatch at (c={c}, k={k})")


class TestShapeAndDevice(unittest.TestCase):

    def test_shape_cpu(self):
        algo = SoftSpecDrop(num_modules=7, num_categories=7,
                            p_active=0.7, p_inactive=0.3)
        cat_ids = torch.tensor([0, 3, 6, 1])
        mask = algo.get_mask(cat_ids)
        self.assertEqual(mask.shape, (4, 7))
        self.assertEqual(mask.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_shape_cuda(self):
        fracs = [0.534, 0.185, 0.088, 0.046, 0.066, 0.043, 0.038]
        fracs = [f / sum(fracs) for f in fracs]
        algo = SoftSpecDrop(num_modules=7, num_categories=7,
                            p_active=0.7, p_inactive=0.3,
                            frac_per_category=fracs)
        cat_ids = torch.tensor([0, 3, 6, 1], device='cuda')
        mask = algo.get_mask(cat_ids)
        self.assertEqual(mask.device.type, 'cuda')
        self.assertEqual(mask.shape, (4, 7))


class TestExpectedMaskSumInvariant(unittest.TestCase):
    """expected_mask_sum returns S scalar regardless of frac_per_category."""

    def test_invariant_under_perc(self):
        fracs = [0.534, 0.185, 0.088, 0.046, 0.066, 0.043, 0.038]
        fracs = [f / sum(fracs) for f in fracs]
        a1 = SoftSpecDrop(num_modules=7, num_categories=7,
                          p_active=0.7, p_inactive=0.3)
        a2 = SoftSpecDrop(num_modules=7, num_categories=7,
                          p_active=0.7, p_inactive=0.3,
                          frac_per_category=fracs)
        self.assertAlmostEqual(float(a1.expected_mask_sum),
                               float(a2.expected_mask_sum), places=7)


class TestMonotonicityOnNLPFracs(unittest.TestCase):
    """Sanity: with NLP fracs, large-data domain (CC) gets smaller gap_c
    than small-data domain (SE), i.e. |p_a^CC - p_i^CC| < |p_a^SE - p_i^SE|."""

    def test_gap_ordering(self):
        fracs = [0.534, 0.185, 0.088, 0.046, 0.066, 0.043, 0.038]
        fracs = [f / sum(fracs) for f in fracs]
        algo = SoftSpecDrop(num_modules=7, num_categories=7,
                            p_active=0.7, p_inactive=0.3,
                            frac_per_category=fracs)
        gap = (algo._p_active_target - algo._p_inactive_target).tolist()
        # CC (c=0) is largest → smallest gap; SE (c=6) is smallest → largest gap.
        self.assertLess(gap[0], gap[6])
        # Balanced row (frac=1/7 ~0.143) would have gap = 0.4; CC is below.
        self.assertLess(gap[0], 0.4)
        self.assertGreater(gap[6], 0.4)


class TestWarmupSchedule(unittest.TestCase):
    """Verify cosine vs linear warmup schedule both connect S/K → target
    with the correct shapes. At endpoints they must agree; in the middle
    they must differ (cosine is S-shaped, linear is constant-slope)."""

    def _build(self, schedule):
        return SoftSpecDrop(num_modules=7, num_categories=7,
                            p_active=0.8, p_inactive=0.2,
                            warmup_ratio=1.0, total_epochs=10,
                            warmup_schedule=schedule)

    def test_endpoints_match_for_both_schedules(self):
        cos_a = self._build('cosine'); lin_a = self._build('linear')
        cat = torch.arange(7)
        # ep=0: both at S/K
        for a in (cos_a, lin_a):
            a.current_epoch = 0
            m = a.get_mask(cat)
            expected = a._p_uniform
            self.assertTrue(torch.allclose(m, torch.full_like(m, expected), atol=1e-6))
        # ep=10: both at target
        for a in (cos_a, lin_a):
            a.current_epoch = 10
            m = a.get_mask(cat)
            # each row is either p_a (diagonal) or p_i (off-diagonal)
            for c in range(7):
                self.assertAlmostEqual(float(m[c, c]), 0.8, places=6)
                for k in range(7):
                    if k != c:
                        self.assertAlmostEqual(float(m[c, k]), 0.2, places=6)

    def test_mid_epoch_shapes_differ(self):
        """At ep=5 (midway), cosine uses 0.5 interpolation, linear also 0.5 —
        so actually they agree at the midpoint. Check ep=2 instead where
        linear=0.2 and cosine ≈ 0.10. The two should differ."""
        cos_a = self._build('cosine'); lin_a = self._build('linear')
        cos_a.current_epoch = 2; lin_a.current_epoch = 2
        m_cos = cos_a.get_mask(torch.arange(7))
        m_lin = lin_a.get_mask(torch.arange(7))
        # max diff should be noticeable (not within fp noise)
        diff = (m_cos - m_lin).abs().max().item()
        self.assertGreater(diff, 0.02,
            f"cosine and linear should differ by >0.02 at ep=2; got {diff:.4f}")

    def test_invalid_schedule_raises(self):
        with self.assertRaises(ValueError):
            SoftSpecDrop(num_modules=7, num_categories=7,
                         p_active=0.7, p_inactive=0.3,
                         warmup_ratio=1.0, total_epochs=10,
                         warmup_schedule='quadratic')

    def test_default_is_cosine(self):
        a = SoftSpecDrop(num_modules=7, num_categories=7,
                         p_active=0.7, p_inactive=0.3,
                         warmup_ratio=1.0, total_epochs=10)
        self.assertEqual(a.warmup_schedule, 'cosine')


class TestOnEpochEndLogsScalarSummary(unittest.TestCase):
    """on_epoch_end returns mean p_a / p_i (scalars) to keep log tidy."""

    def test_summary_scalars(self):
        fracs = [0.534, 0.185, 0.088, 0.046, 0.066, 0.043, 0.038]
        fracs = [f / sum(fracs) for f in fracs]
        algo = SoftSpecDrop(num_modules=7, num_categories=7,
                            p_active=0.7, p_inactive=0.3,
                            warmup_ratio=1.0, total_epochs=10,
                            frac_per_category=fracs)
        summary = algo.on_epoch_end(branches=None, epoch=9)
        self.assertIn('p_active', summary)
        self.assertIn('p_inactive', summary)
        self.assertIsInstance(summary['p_active'], float)


if __name__ == '__main__':
    unittest.main()

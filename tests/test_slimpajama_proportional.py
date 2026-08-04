"""Unit tests for compute_proportional_ffn_dims and find_tokenize_cache helpers.

Validates:
  (1) ffn_dims are returned in domain_id order (NOT sorted by size), so
      ffn_dims[k] matches branch k's assigned domain.
  (2) Branch 0 (CommonCrawl, largest domain) gets the largest ffn when data
      is SlimPajama-like.
  (3) Small-id large-data + large-id small-data pairs (e.g. Book id=3 vs
      ArXiv id=4) preserve the domain_id order regardless of relative size.
  (4) Total sum ≈ target_ffn within K (rounding error).
  (5) End-to-end: ffn_dims_per_branch from helper feeds MultiBranchTransformerLM
      correctly (branch k → NonUniformParallelFFN.ffns[k].fc1.weight has that
      branch's ffn_dim).
  (6) find_tokenize_cache glob-finds a cache and picks largest token budget.

Run:  python tests/test_slimpajama_proportional.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from data.slimpajama import (
    DOMAIN_TO_ID, DOMAIN_NAMES, NUM_DOMAINS,
    compute_proportional_ffn_dims, find_tokenize_cache,
    split_ffn_budget_for_se,
)


def _make_fake_cache(path, counts_by_id):
    """Write a minimal fake tokenized cache with the given per-id counts."""
    domain_ids = []
    for domain_id, cnt in counts_by_id.items():
        domain_ids.extend([domain_id] * cnt)
    input_ids = torch.randint(0, 100, (len(domain_ids), 16), dtype=torch.long)
    torch.save({
        'input_ids': input_ids,
        'domain_ids': torch.tensor(domain_ids, dtype=torch.long),
    }, path)


class TestComputeProportional(unittest.TestCase):

    def test_domain_id_order_not_size_order(self):
        """Critical regression guard: ffn_dims must be indexed by domain_id,
        not sorted by data size. Book (id=3) is SMALLER than ArXiv (id=4),
        so ffn_dims[3] must be < ffn_dims[4] — the list is NOT monotone."""
        # Actual-ish SlimPajama-6B 100M proportions:
        # CC>C4>Github>ArXiv>Book>Wiki>SE   (size order)
        # but domain_ids are CC=0, C4=1, Github=2, Book=3, ArXiv=4, Wiki=5, SE=6
        counts = {
            0: 104_282,  # CC
            1: 36_215,   # C4
            2: 17_259,   # Github
            3: 8_901,    # Book   ← id 3, data smaller than id 4
            4: 12_869,   # ArXiv  ← id 4, data larger than id 3
            5: 8_369,    # Wiki
            6: 7_418,    # SE
        }
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, 'fake_cache.pt')
            _make_fake_cache(cache_path, counts)
            ffn_dims, mapping = compute_proportional_ffn_dims(
                cache_path, total_ffn=1540, num_branches=7)

            # ffn_dims must be in domain_id order
            self.assertEqual(mapping[0]['domain_name'], 'RedPajamaCommonCrawl')
            self.assertEqual(mapping[1]['domain_name'], 'RedPajamaC4')
            self.assertEqual(mapping[2]['domain_name'], 'RedPajamaGithub')
            self.assertEqual(mapping[3]['domain_name'], 'RedPajamaBook')
            self.assertEqual(mapping[4]['domain_name'], 'RedPajamaArXiv')
            self.assertEqual(mapping[5]['domain_name'], 'RedPajamaWikipedia')
            self.assertEqual(mapping[6]['domain_name'], 'RedPajamaStackExchange')

            # Book (idx 3) smaller than ArXiv (idx 4) by data → ffn[3] < ffn[4]
            self.assertLess(ffn_dims[3], ffn_dims[4],
                             f"ffn_dims[3]={ffn_dims[3]} (Book) should be < "
                             f"ffn_dims[4]={ffn_dims[4]} (ArXiv) because Book<ArXiv by data")

    def test_branch_0_gets_largest_ffn(self):
        """CommonCrawl (id=0, largest) should map to biggest branch."""
        counts = {0: 500, 1: 200, 2: 100, 3: 50, 4: 80, 5: 40, 6: 30}
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, 'fake.pt')
            _make_fake_cache(cache_path, counts)
            ffn_dims, mapping = compute_proportional_ffn_dims(
                cache_path, total_ffn=1000, num_branches=7)

            self.assertEqual(ffn_dims[0], max(ffn_dims))
            self.assertEqual(ffn_dims[6], min(ffn_dims))

    def test_total_sum_close_to_target(self):
        """Sum of ffn_dims should be within K of target (rounding)."""
        counts = {k: 100 * (k + 1) for k in range(7)}  # arbitrary non-uniform
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, 'fake.pt')
            _make_fake_cache(cache_path, counts)
            for target in [1000, 1540, 2000]:
                ffn_dims, _ = compute_proportional_ffn_dims(
                    cache_path, total_ffn=target, num_branches=7)
                diff = abs(sum(ffn_dims) - target)
                self.assertLess(diff, 7,
                                 f"sum {sum(ffn_dims)} drifted > K from target {target}")

    def test_missing_domain_id_gets_zero(self):
        """If a domain_id has 0 chunks, its branch gets ffn=0 (not crash)."""
        # Domain 3 (Book) missing entirely
        counts = {0: 100, 1: 50, 2: 30, 4: 20, 5: 10, 6: 5}  # no id 3
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, 'fake.pt')
            _make_fake_cache(cache_path, counts)
            ffn_dims, mapping = compute_proportional_ffn_dims(
                cache_path, total_ffn=1000, num_branches=7)
            self.assertEqual(ffn_dims[3], 0)
            self.assertEqual(mapping[3]['count'], 0)
            self.assertEqual(mapping[3]['domain_name'], 'RedPajamaBook')

    def test_mapping_dict_structure(self):
        counts = {0: 100, 1: 50}
        # pad to 7 domains
        for k in range(2, 7):
            counts[k] = 10
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, 'fake.pt')
            _make_fake_cache(cache_path, counts)
            ffn_dims, mapping = compute_proportional_ffn_dims(
                cache_path, total_ffn=1000, num_branches=7)
            self.assertEqual(len(mapping), 7)
            for k, m in enumerate(mapping):
                self.assertEqual(m['branch'], k)
                self.assertEqual(m['domain_id'], k)
                self.assertEqual(m['ffn_dim'], ffn_dims[k])
                self.assertIn('domain_name', m)
                self.assertIn('count', m)
                self.assertIn('frac', m)


class TestFindTokenizeCache(unittest.TestCase):

    def test_finds_single_cache(self):
        with tempfile.TemporaryDirectory() as td:
            inner = os.path.join(td, 'slimpajama')
            os.makedirs(inner)
            path = os.path.join(inner, 'tokenized_train_seq512_tok100000000_vocabABCD12.pt')
            open(path, 'w').close()
            result = find_tokenize_cache(data_dir=inner, split='train',
                                          max_tokens=100_000_000, max_seq_len=512)
            self.assertEqual(result, path)

    def test_prefers_largest_token_budget(self):
        with tempfile.TemporaryDirectory() as td:
            inner = os.path.join(td, 'slimpajama')
            os.makedirs(inner)
            p100 = os.path.join(inner, 'tokenized_train_seq512_tok100000000_vocabAA.pt')
            p500 = os.path.join(inner, 'tokenized_train_seq512_tok500000000_vocabAA.pt')
            open(p100, 'w').close()
            open(p500, 'w').close()
            # Without explicit max_tokens, pick 500M (larger)
            result = find_tokenize_cache(data_dir=inner, split='train',
                                          max_tokens=None, max_seq_len=512)
            self.assertEqual(result, p500)
            # With explicit max_tokens=100M, pick 100M
            result2 = find_tokenize_cache(data_dir=inner, split='train',
                                           max_tokens=100_000_000, max_seq_len=512)
            self.assertEqual(result2, p100)

    def test_raises_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                find_tokenize_cache(data_dir=td, split='train',
                                     max_tokens=100_000_000, max_seq_len=512)


class TestSplitFfnBudgetForSE(unittest.TestCase):
    """split_ffn_budget_for_se keeps dense-equivalent total invariant across
    SE ratios, within rounding tolerance."""

    def test_se_zero_all_to_branches(self):
        tb, se = split_ffn_budget_for_se(1540, 0.0, num_branches=7)
        self.assertEqual(tb, 1540)
        self.assertEqual(se, 0)

    def test_se_one_splits_one_eighth(self):
        tb, se = split_ffn_budget_for_se(1540, 1.0, num_branches=7)
        # avg_branch = 1540 / 8 = 192.5 → SE ≈ 192, branches ≈ 1348
        # Rounding: 1540 × 7 / 8 = 1347.5 → 1348  ;  1540 / 8 = 192.5 → 192
        self.assertEqual(tb, 1348)
        self.assertEqual(se, 192)
        self.assertLess(abs((tb + se) - 1540), 3,
                         f"total diverged: {tb}+{se}={tb+se}")

    def test_se_four(self):
        tb, se = split_ffn_budget_for_se(1540, 4.0, num_branches=7)
        # 1540 × 7/11 = 980 ; 1540 × 4/11 = 560
        self.assertEqual(tb, 980)
        self.assertEqual(se, 560)
        self.assertEqual(tb + se, 1540)

    def test_se_half(self):
        tb, se = split_ffn_budget_for_se(1540, 0.5, num_branches=7)
        # 1540 × 7/7.5 ≈ 1437.33 → 1437 ; 1540 × 0.5/7.5 ≈ 102.67 → 103
        self.assertEqual(tb, 1437)
        self.assertEqual(se, 103)
        self.assertLess(abs((tb + se) - 1540), 2)

    def test_budget_invariant_across_ratios(self):
        """For every SE ratio in ablation, branches + SE ≈ total within K."""
        for ratio in (0.0, 0.5, 1.0, 2.0, 4.0):
            tb, se = split_ffn_budget_for_se(1540, ratio, num_branches=7)
            self.assertLess(abs((tb + se) - 1540), 7,
                             f"ratio={ratio}: tb+se={tb+se} diverged from 1540")

    def test_se_ratio_equals_se_over_avg_branch(self):
        """Ratio semantics: SE_ratio = SE_dim / average_branch_ffn, both in
        dimensions and in parameters. User-verified invariant."""
        for ratio in (0.5, 1.0, 4.0):
            tb, se = split_ffn_budget_for_se(1540, ratio, num_branches=7)
            avg_branch_ffn = tb / 7
            observed_ratio = se / avg_branch_ffn
            # Rounding from int conversion can shift ratio by up to ~1/K.
            # Assert observed ratio within 0.01 (much tighter than 1/K=0.14).
            self.assertAlmostEqual(
                observed_ratio, ratio, places=2,
                msg=f"ratio={ratio}: SE_dim={se} / avg_branch={avg_branch_ffn:.3f} "
                    f"= {observed_ratio:.4f}, expected {ratio} ± 0.01")

    def test_se_params_vs_avg_branch_params(self):
        """Ratio invariant in parameter count (not just ffn dim):
        SE_params / avg(branch_params) = SE_ratio, since FFN params scale
        linearly with ffn_dim at fixed hidden_dim."""
        hidden = 384  # matches NLP config
        for ratio in (0.5, 1.0, 4.0):
            tb, se = split_ffn_budget_for_se(1540, ratio, num_branches=7)
            avg_branch_ffn = tb / 7
            avg_branch_params = 2 * hidden * avg_branch_ffn  # fc1 + fc2
            se_params = 2 * hidden * se
            observed = se_params / avg_branch_params
            self.assertAlmostEqual(
                observed, ratio, places=2,
                msg=f"ratio={ratio}: SE params / avg branch params = {observed:.4f}, "
                    f"expected {ratio} ± 0.01")


class TestEndToEndModelMapping(unittest.TestCase):
    """Verify ffn_dims from helper → MultiBranchTransformerLM: branch k's
    FFN module has the exact ffn_dim corresponding to domain k."""

    def test_branch_ffn_dims_match_allocation(self):
        counts = {
            0: 104_282, 1: 36_215, 2: 17_259, 3: 8_901,
            4: 12_869, 5: 8_369, 6: 7_418,
        }
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, 'fake.pt')
            _make_fake_cache(cache_path, counts)
            ffn_dims, mapping = compute_proportional_ffn_dims(
                cache_path, total_ffn=1540, num_branches=7)

        from models.transformer_lm import MultiBranchTransformerLM
        model = MultiBranchTransformerLM(
            vocab_size=100, hidden_dim=32, num_layers=1, num_heads=4,
            num_branches=7, ffn_dim_per_branch=ffn_dims,
            max_seq_len=16, dropout=0.0,
        )

        # Reach the parallel_ffn of the only block
        block = model.blocks[0]
        pffn = block.parallel_ffn
        # Must be NonUniformParallelFFN (list dispatch)
        from models.transformer_lm import NonUniformParallelFFN
        self.assertIsInstance(pffn, NonUniformParallelFFN)

        # Each branch k's fc1.weight should have ffn_dim matching ffn_dims[k]
        for k in range(7):
            w_shape = pffn.ffns[k].fc1.weight.shape
            self.assertEqual(w_shape[0], ffn_dims[k],
                              f"branch {k} (domain={mapping[k]['domain_name']}) "
                              f"expected ffn_dim {ffn_dims[k]}, got fc1 shape {w_shape}")

        # Spot-check the critical ones
        self.assertEqual(pffn.ffns[0].fc1.weight.shape[0], max(ffn_dims),
                          'CommonCrawl branch 0 must have the largest ffn')
        # Book (id 3) < ArXiv (id 4)
        self.assertLess(pffn.ffns[3].fc1.weight.shape[0],
                         pffn.ffns[4].fc1.weight.shape[0],
                         'Book branch 3 should be smaller than ArXiv branch 4 '
                         '(despite Book having a smaller domain_id)')


class TestNonUniformWithSharedExpert(unittest.TestCase):
    """Full Phase L setup: non-uniform branches + SE under fixed dense budget.
    Verifies total model param count stays within ±2% of dense reference."""

    def test_nonuniform_plus_se_param_budget(self):
        from models.transformer_lm import MultiBranchTransformerLM

        counts = {0: 104_282, 1: 36_215, 2: 17_259, 3: 8_901,
                  4: 12_869, 5: 8_369, 6: 7_418}
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, 'fake.pt')
            _make_fake_cache(cache_path, counts)

            # Dense reference: ffn=1536, ~30.14M params
            # Build uniform for comparison (small-scale version, 2 layers to speed up)
            model_u = MultiBranchTransformerLM(
                vocab_size=200, hidden_dim=32, num_layers=2, num_heads=4,
                num_branches=7, ffn_dim_per_branch=220,
                max_seq_len=16, dropout=0.0,
            )
            params_u = sum(p.numel() for p in model_u.parameters())

            for se_ratio in (0.0, 0.5, 1.0, 4.0):
                tb, se_dim = split_ffn_budget_for_se(1540, se_ratio, num_branches=7)
                ffn_dims, _ = compute_proportional_ffn_dims(
                    cache_path, total_ffn=tb, num_branches=7)
                model = MultiBranchTransformerLM(
                    vocab_size=200, hidden_dim=32, num_layers=2, num_heads=4,
                    num_branches=7, ffn_dim_per_branch=ffn_dims,
                    shared_expert_dim=se_dim,
                    max_seq_len=16, dropout=0.0,
                )
                p = sum(x.numel() for x in model.parameters())
                rel = abs(p - params_u) / params_u
                self.assertLess(rel, 0.02,
                                 f"SE_ratio={se_ratio}: params={p:,} vs uniform {params_u:,} "
                                 f"({rel:.2%}) exceeds 2% budget")

    def test_nonuniform_plus_se_forward(self):
        """Model with non-uniform branches + SE should forward with branch_mask."""
        from models.transformer_lm import MultiBranchTransformerLM
        import torch

        counts = {0: 500, 1: 200, 2: 100, 3: 50, 4: 80, 5: 40, 6: 30}
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, 'fake.pt')
            _make_fake_cache(cache_path, counts)
            ffn_dims, _ = compute_proportional_ffn_dims(cache_path, 800, 7)

        model = MultiBranchTransformerLM(
            vocab_size=50, hidden_dim=16, num_layers=1, num_heads=4,
            num_branches=7, ffn_dim_per_branch=ffn_dims,
            shared_expert_dim=24,  # SE present
            max_seq_len=8, dropout=0.0,
        )
        model.eval()
        model.mask_scale = 2.5

        x = torch.randint(0, 50, (2, 6))
        mask = torch.rand(2, 7)
        with torch.no_grad():
            logits = model(x, branch_mask=mask)
        self.assertEqual(logits.shape, (2, 6, 50))


if __name__ == '__main__':
    unittest.main(verbosity=2)

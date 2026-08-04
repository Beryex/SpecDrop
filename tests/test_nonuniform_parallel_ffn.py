"""Unit tests for NonUniformParallelFFN and dispatch logic in transformer_lm.

Covers:
  (1) Backward compat: scalar ffn_dim_per_branch still uses ParallelFFN.
  (2) Shape: NonUniformParallelFFN.forward → (B, T, K, D) regardless of
      per-branch ffn_dim.
  (3) Bit-equivalence: with all ffn_dims equal AND weights copied from
      ParallelFFN, NonUniformParallelFFN produces identical output in eval
      mode (no dropout).
  (4) End-to-end: MultiBranchTransformerLM with ffn_dims_per_branch list
      builds, forwards with branch_mask, and has the expected total param
      count close to dense-equivalent 1536.

Run:  python tests/test_nonuniform_parallel_ffn.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from models.transformer_lm import (
    ParallelFFN,
    NonUniformParallelFFN,
    MoETransformerBlock,
    SharedExpertMoETransformerBlock,
    SandwichMoETransformerBlock,
    MultiBranchTransformerLM,
    _build_parallel_ffn,
)


class TestDispatch(unittest.TestCase):
    """_build_parallel_ffn chooses correct class based on input type."""

    def test_scalar_builds_parallel_ffn(self):
        m = _build_parallel_ffn(hidden_dim=16, ffn_width=24, num_branches=4, dropout=0.0)
        self.assertIsInstance(m, ParallelFFN)
        # ParallelFFN uses stacked params: w1 shape (K, FF, D)
        self.assertEqual(m.w1.shape, (4, 24, 16))

    def test_list_builds_nonuniform(self):
        m = _build_parallel_ffn(
            hidden_dim=16, ffn_width=[24, 32, 8, 48], num_branches=4, dropout=0.0)
        self.assertIsInstance(m, NonUniformParallelFFN)
        self.assertEqual(m.ffn_dims, [24, 32, 8, 48])
        # Each branch has its own FFN with its own fc1 / fc2 shapes.
        for k, expected_dim in enumerate([24, 32, 8, 48]):
            self.assertEqual(m.ffns[k].fc1.weight.shape, (expected_dim, 16))
            self.assertEqual(m.ffns[k].fc2.weight.shape, (16, expected_dim))

    def test_list_length_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            _build_parallel_ffn(
                hidden_dim=16, ffn_width=[24, 32], num_branches=4, dropout=0.0)


class TestNonUniformForwardShape(unittest.TestCase):
    """NonUniformParallelFFN returns (B, T, K, D) regardless of internal dims."""

    def test_forward_shape_varied_dims(self):
        torch.manual_seed(0)
        B, T, D = 2, 5, 16
        ffn_dims = [24, 48, 12, 64, 8]  # K=5, all different
        m = NonUniformParallelFFN(D, ffn_dims, dropout=0.0)
        m.eval()
        x = torch.randn(B, T, D)
        out = m(x)
        self.assertEqual(out.shape, (B, T, 5, D))

    def test_forward_shape_uniform_dims(self):
        torch.manual_seed(0)
        B, T, D = 2, 5, 16
        m = NonUniformParallelFFN(D, [24] * 7, dropout=0.0)
        m.eval()
        x = torch.randn(B, T, D)
        out = m(x)
        self.assertEqual(out.shape, (B, T, 7, D))


class TestBitEquivalence(unittest.TestCase):
    """When weights are manually copied from ParallelFFN to NonUniform with same
    uniform dims, outputs must match within fp32 precision in eval mode."""

    def test_copy_weights_match_output(self):
        torch.manual_seed(42)
        K, D, FF = 4, 16, 24
        B, T = 2, 5

        parallel = ParallelFFN(D, FF, K, dropout=0.0)
        nonuniform = NonUniformParallelFFN(D, [FF] * K, dropout=0.0)

        # Copy weights from parallel stacked params into per-branch FFN modules.
        for k in range(K):
            nonuniform.ffns[k].fc1.weight.data.copy_(parallel.w1[k])
            nonuniform.ffns[k].fc1.bias.data.copy_(parallel.b1[k])
            nonuniform.ffns[k].fc2.weight.data.copy_(parallel.w2[k])
            nonuniform.ffns[k].fc2.bias.data.copy_(parallel.b2[k])

        parallel.eval()
        nonuniform.eval()

        x = torch.randn(B, T, D)
        with torch.no_grad():
            out_p = parallel(x)
            out_n = nonuniform(x)
        self.assertEqual(out_p.shape, out_n.shape)
        self.assertTrue(torch.allclose(out_p, out_n, atol=1e-5, rtol=1e-5),
                         f"max diff = {(out_p - out_n).abs().max().item():.2e}")


class TestBlockIntegration(unittest.TestCase):
    """Backward compat: scalar input to MoE blocks behaves exactly like before.
    New: list input switches to non-uniform dispatch without breaking forward."""

    def _forward_ok(self, block, B=2, T=5, D=16, K=4):
        x = torch.randn(B, T, D)
        mask = torch.ones(B, K)
        # Fixed-denom merge needs mask_scale (any positive number for shape check).
        out = block(x, branch_mask=mask, mask_scale=float(K))
        self.assertEqual(out.shape, (B, T, D))

    def test_moe_block_scalar_backward_compat(self):
        torch.manual_seed(0)
        block = MoETransformerBlock(hidden_dim=16, num_heads=4, num_branches=4,
                                     ffn_dim_per_branch=24, dropout=0.0)
        block.eval()
        self.assertIsInstance(block.parallel_ffn, ParallelFFN)
        self._forward_ok(block)

    def test_moe_block_list_non_uniform(self):
        torch.manual_seed(0)
        block = MoETransformerBlock(hidden_dim=16, num_heads=4, num_branches=4,
                                     ffn_dim_per_branch=[32, 16, 48, 8], dropout=0.0)
        block.eval()
        self.assertIsInstance(block.parallel_ffn, NonUniformParallelFFN)
        self._forward_ok(block)

    def test_shared_expert_block_scalar_backward_compat(self):
        torch.manual_seed(0)
        block = SharedExpertMoETransformerBlock(
            hidden_dim=16, num_heads=4, num_branches=4,
            ffn_dim_per_branch=24, shared_expert_dim=24, dropout=0.0)
        block.eval()
        self.assertIsInstance(block.parallel_ffn, ParallelFFN)
        self._forward_ok(block)

    def test_shared_expert_block_list_non_uniform(self):
        torch.manual_seed(0)
        block = SharedExpertMoETransformerBlock(
            hidden_dim=16, num_heads=4, num_branches=4,
            ffn_dim_per_branch=[32, 16, 48, 8], shared_expert_dim=24, dropout=0.0)
        block.eval()
        self.assertIsInstance(block.parallel_ffn, NonUniformParallelFFN)
        self._forward_ok(block)

    def test_sandwich_block_scalar_backward_compat(self):
        torch.manual_seed(0)
        block = SandwichMoETransformerBlock(
            hidden_dim=16, num_heads=4, num_branches=4,
            ffn_dim_per_branch=24, sandwich_dim=12, dropout=0.0)
        block.eval()
        # Sandwich still uses ModuleList (it always did); branches have uniform dim.
        self.assertEqual(len(block.branches), 4)
        for b in block.branches:
            self.assertEqual(b.fc1.weight.shape, (24, 16))
        self._forward_ok(block)

    def test_sandwich_block_list_non_uniform(self):
        torch.manual_seed(0)
        block = SandwichMoETransformerBlock(
            hidden_dim=16, num_heads=4, num_branches=4,
            ffn_dim_per_branch=[32, 16, 48, 8], sandwich_dim=12, dropout=0.0)
        block.eval()
        self.assertEqual(len(block.branches), 4)
        expected = [32, 16, 48, 8]
        for k, b in enumerate(block.branches):
            self.assertEqual(b.fc1.weight.shape, (expected[k], 16))
        self._forward_ok(block)


class TestEndToEndLM(unittest.TestCase):
    """MultiBranchTransformerLM with ffn_dims_per_branch list."""

    def test_build_and_forward(self):
        torch.manual_seed(0)
        # Data-proportional allocation for SlimPajama-6B 100M (from actual cache):
        # [822, 286, 136, 70, 101, 66, 58], sum=1539
        ffn_dims = [822, 286, 136, 70, 101, 66, 58]
        model = MultiBranchTransformerLM(
            vocab_size=100, hidden_dim=32, num_layers=2, num_heads=4,
            num_branches=7, ffn_dim_per_branch=ffn_dims,
            max_seq_len=16, dropout=0.0,
        )
        model.eval()

        # Forward without mask.
        x = torch.randint(0, 100, (2, 8))
        logits = model(x)
        self.assertEqual(logits.shape, (2, 8, 100))

        # Forward with branch_mask.
        model.mask_scale = 2.5  # set a fixed-denom value
        mask = torch.rand(2, 7)
        logits2 = model(x, branch_mask=mask)
        self.assertEqual(logits2.shape, (2, 8, 100))

    def test_param_counts_match_allocation(self):
        """Total FFN params under non-uniform allocation ≈ uniform allocation
        with same total ffn width (1540 vs 1539)."""
        torch.manual_seed(0)
        ffn_dims_nu = [822, 286, 136, 70, 101, 66, 58]  # sum 1539
        model_nu = MultiBranchTransformerLM(
            vocab_size=100, hidden_dim=32, num_layers=1, num_heads=4,
            num_branches=7, ffn_dim_per_branch=ffn_dims_nu,
            max_seq_len=16, dropout=0.0,
        )
        model_u = MultiBranchTransformerLM(
            vocab_size=100, hidden_dim=32, num_layers=1, num_heads=4,
            num_branches=7, ffn_dim_per_branch=220,  # 220*7 = 1540
            max_seq_len=16, dropout=0.0,
        )
        p_nu = sum(p.numel() for p in model_nu.parameters())
        p_u = sum(p.numel() for p in model_u.parameters())
        # FFN params dominate the difference. Allow ~1% tolerance.
        self.assertAlmostEqual(p_nu, p_u, delta=p_u * 0.01,
                                msg=f"non-uniform ({p_nu}) vs uniform ({p_u}) diverge >1%")


class TestSoftSpecDropIntegration(unittest.TestCase):
    """Verify build_algorithm / mask_scale wiring still works with non-uniform
    — mask broadcast (B, 1, K, 1) only depends on K, independent of ffn_dims."""

    def test_mask_merge_correct(self):
        torch.manual_seed(0)
        model = MultiBranchTransformerLM(
            vocab_size=50, hidden_dim=16, num_layers=1, num_heads=4,
            num_branches=4, ffn_dim_per_branch=[24, 12, 36, 8],
            max_seq_len=8, dropout=0.0,
        )
        model.eval()
        model.mask_scale = 2.5

        x = torch.randint(0, 50, (2, 6))

        # With mask all-zeros except one branch → only that branch contributes.
        mask_only_b0 = torch.zeros(2, 4)
        mask_only_b0[:, 0] = 1.0
        logits0 = model(x, branch_mask=mask_only_b0)

        mask_only_b2 = torch.zeros(2, 4)
        mask_only_b2[:, 2] = 1.0
        logits2 = model(x, branch_mask=mask_only_b2)

        # Different single-branch activations must yield different logits
        # (would be a bug if the mask path silently ignored branch index).
        self.assertFalse(torch.allclose(logits0, logits2),
                          "Mask[:, 0]=1 and mask[:, 2]=1 produced identical output — "
                          "branch routing is not working correctly.")


if __name__ == '__main__':
    unittest.main(verbosity=2)

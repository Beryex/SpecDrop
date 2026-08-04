#!/usr/bin/env python
"""Test that ParallelFFN (einsum) is equivalent to K separate FFN modules.

Tests:
  1. ParallelFFN vs K separate FFNs — forward equivalence
  2. Gradient equivalence
  3. Full MoETransformerBlock equivalence (einsum vs for-loop)
  4. Speed comparison (ParallelFFN vs for-loop)

Usage:
    python tests/test_parallel_ffn.py
"""

import sys
import os
import time
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_lm import FFN, ParallelFFN, MoETransformerBlock

torch.manual_seed(42)


def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# For-loop reference implementations (what einsum replaces)
# ---------------------------------------------------------------------------

class ForLoopParallelFFN(nn.Module):
    """K separate FFN modules with for-loop — the naive baseline."""

    def __init__(self, hidden_dim, ffn_dim, num_branches, dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            FFN(hidden_dim, ffn_dim, dropout) for _ in range(num_branches)
        ])

    def forward(self, x):
        """Returns (B, T, K, D) — same interface as ParallelFFN."""
        return torch.stack([branch(x) for branch in self.branches], dim=2)


class ForLoopMoEBlock(nn.Module):
    """MoETransformerBlock but using for-loop FFN instead of einsum."""

    def __init__(self, hidden_dim, num_heads, num_branches, ffn_dim_per_branch,
                 dropout=0.1):
        super().__init__()
        self.num_branches = num_branches
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.branches = nn.ModuleList([
            FFN(hidden_dim, ffn_dim_per_branch, dropout)
            for _ in range(num_branches)
        ])

    def forward(self, x, branch_mask=None, mask_scale=None, attn_mask=None):
        B, T, D = x.shape
        K = self.num_branches
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, is_causal=True)
        x = x + self.drop(h)
        h = self.ln2(x)
        branch_outputs = torch.stack(
            [branch(h) for branch in self.branches], dim=2)
        if branch_mask is not None:
            mask = branch_mask.view(B, 1, K, 1)
            branch_outputs = branch_outputs * mask
            if mask_scale is not None:
                merged = branch_outputs.sum(dim=2) / mask_scale
            else:
                mask_sum = branch_mask.sum(dim=1).view(B, 1, 1).clamp(min=1e-8)
                merged = branch_outputs.sum(dim=2) / mask_sum
        else:
            merged = branch_outputs.mean(dim=2)
        x = x + merged
        return x


def copy_weights_to_parallel_ffn(forloop_ffn, parallel_ffn):
    """Copy weights from K separate FFNs to ParallelFFN (einsum format)."""
    K = len(forloop_ffn.branches)
    with torch.no_grad():
        for k in range(K):
            # fc1: nn.Linear(D, FF) → weight (FF, D), bias (FF)
            parallel_ffn.w1.data[k] = forloop_ffn.branches[k].fc1.weight.data.clone()
            parallel_ffn.b1.data[k] = forloop_ffn.branches[k].fc1.bias.data.clone()
            # fc2: nn.Linear(FF, D) → weight (D, FF), bias (D)
            parallel_ffn.w2.data[k] = forloop_ffn.branches[k].fc2.weight.data.clone()
            parallel_ffn.b2.data[k] = forloop_ffn.branches[k].fc2.bias.data.clone()


def copy_weights_forloop_to_moe_block(forloop_block, einsum_block):
    """Copy weights from ForLoopMoEBlock to MoETransformerBlock."""
    # Shared layers (attention, layer norms)
    einsum_block.ln1.load_state_dict(forloop_block.ln1.state_dict())
    einsum_block.attn.load_state_dict(forloop_block.attn.state_dict())
    einsum_block.drop.load_state_dict(forloop_block.drop.state_dict())
    einsum_block.ln2.load_state_dict(forloop_block.ln2.state_dict())
    # FFN branches → ParallelFFN
    K = len(forloop_block.branches)
    pffn = einsum_block.parallel_ffn
    with torch.no_grad():
        for k in range(K):
            pffn.w1.data[k] = forloop_block.branches[k].fc1.weight.data.clone()
            pffn.b1.data[k] = forloop_block.branches[k].fc1.bias.data.clone()
            pffn.w2.data[k] = forloop_block.branches[k].fc2.weight.data.clone()
            pffn.b2.data[k] = forloop_block.branches[k].fc2.bias.data.clone()
    # Copy dropout state
    pffn.dropout.load_state_dict(forloop_block.branches[0].dropout.state_dict())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parallel_ffn_equivalence():
    """ParallelFFN (einsum) matches K separate FFNs."""
    D, FF, K = 256, 512, 7

    torch.manual_seed(42)
    forloop = ForLoopParallelFFN(D, FF, K, dropout=0.0)
    einsum = ParallelFFN(D, FF, K, dropout=0.0)
    copy_weights_to_parallel_ffn(forloop, einsum)

    forloop.eval()
    einsum.eval()

    for B, T in [(1, 16), (4, 64), (8, 128)]:
        torch.manual_seed(123)
        x = torch.randn(B, T, D)
        with torch.no_grad():
            out_fl = forloop(x)   # (B, T, K, D)
            out_ei = einsum(x)    # (B, T, K, D)
        diff = (out_fl - out_ei).abs().max().item()
        assert diff < 1e-5, f"FFN mismatch B={B} T={T}: {diff:.2e}"
        print(f"  B={B:>3} T={T:>3}: max_diff={diff:.2e} ✓")


def test_parallel_ffn_gradient():
    """Gradients match between einsum and for-loop."""
    D, FF, K = 256, 512, 7

    torch.manual_seed(42)
    forloop = ForLoopParallelFFN(D, FF, K, dropout=0.0)
    einsum = ParallelFFN(D, FF, K, dropout=0.0)
    copy_weights_to_parallel_ffn(forloop, einsum)

    forloop.train()
    einsum.train()

    torch.manual_seed(999)
    x = torch.randn(4, 32, D)

    out_fl = forloop(x).sum()
    out_fl.backward()

    out_ei = einsum(x).sum()
    out_ei.backward()

    # Compare gradients on w1 (the main parameter)
    # forloop branch k's fc1.weight.grad vs einsum w1.grad[k]
    for k in range(K):
        fl_grad = forloop.branches[k].fc1.weight.grad
        ei_grad = einsum.w1.grad[k]
        diff = (fl_grad - ei_grad).abs().max().item()
        if k < 3:
            print(f"  Branch {k} w1 grad diff: {diff:.2e}")
        assert diff < 1e-4, f"Gradient mismatch at branch {k}: {diff:.2e}"

    print(f"  All {K} branches gradient check passed ✓")


def test_moe_block_equivalence():
    """Full MoETransformerBlock (einsum) matches for-loop version."""
    D, K, FF, heads = 256, 7, 512, 4

    torch.manual_seed(42)
    forloop_block = ForLoopMoEBlock(D, heads, K, FF, dropout=0.0)
    einsum_block = MoETransformerBlock(D, heads, K, FF, dropout=0.0)
    copy_weights_forloop_to_moe_block(forloop_block, einsum_block)

    forloop_block.eval()
    einsum_block.eval()

    for B, T in [(2, 32), (4, 64), (8, 128)]:
        torch.manual_seed(456)
        x = torch.randn(B, T, D)
        mask = torch.rand(B, K)
        mask_scale = 2.8
        attn_mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)

        with torch.no_grad():
            out_fl = forloop_block(x, branch_mask=mask, mask_scale=mask_scale,
                                   attn_mask=attn_mask)
            out_ei = einsum_block(x, branch_mask=mask, mask_scale=mask_scale,
                                  attn_mask=attn_mask)

        diff = (out_fl - out_ei).abs().max().item()
        assert diff < 1e-5, f"Block mismatch B={B} T={T}: {diff:.2e}"
        print(f"  B={B:>3} T={T:>3}: max_diff={diff:.2e} ✓")


def test_param_count():
    """ParallelFFN has same param count as K separate FFNs."""
    D, FF, K = 256, 512, 7

    forloop = ForLoopParallelFFN(D, FF, K)
    einsum = ParallelFFN(D, FF, K)

    fl_params = sum(p.numel() for p in forloop.parameters())
    ei_params = sum(p.numel() for p in einsum.parameters())
    assert fl_params == ei_params, f"Param mismatch: forloop={fl_params}, einsum={ei_params}"
    print(f"  K={K}, D={D}, FF={FF}: {fl_params:,} params each ✓")


def test_speed_comparison():
    """Benchmark einsum vs for-loop at realistic NLP scale."""
    device = _get_device()
    print(f"  Device: {device}")

    configs = [
        # (label, D, FF, K, B, T)
        ("K=7 D=256 FF=512 B=8 T=128", 256, 512, 7, 8, 128),
        ("K=7 D=256 FF=512 B=16 T=256", 256, 512, 7, 16, 256),
    ]

    n_warmup = 5
    n_timed = 20

    for label, D, FF, K, B, T in configs:
        torch.manual_seed(42)
        forloop = ForLoopParallelFFN(D, FF, K, dropout=0.0).to(device).eval()
        einsum = ParallelFFN(D, FF, K, dropout=0.0).to(device).eval()
        copy_weights_to_parallel_ffn(forloop, einsum)

        x = torch.randn(B, T, D, device=device)

        # Warm up
        with torch.no_grad():
            for _ in range(n_warmup):
                forloop(x)
                einsum(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # Time for-loop
        times_fl = []
        with torch.no_grad():
            for _ in range(n_timed):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                forloop(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times_fl.append(time.perf_counter() - t0)

        # Time einsum
        times_ei = []
        with torch.no_grad():
            for _ in range(n_timed):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                einsum(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times_ei.append(time.perf_counter() - t0)

        mean_fl = sum(times_fl) / len(times_fl) * 1000
        mean_ei = sum(times_ei) / len(times_ei) * 1000
        speedup = mean_fl / mean_ei if mean_ei > 0 else float('inf')

        print(f"\n  [{label}]")
        print(f"    For-loop: {mean_fl:>8.2f} ms")
        print(f"    Einsum:   {mean_ei:>8.2f} ms")
        print(f"    Speedup:  {speedup:>8.1f}x")

    # Forward + backward benchmark
    print(f"\n  --- Forward + Backward (full MoE block) ---")
    D, K, FF, heads, B, T = 256, 7, 512, 4, 8, 128

    torch.manual_seed(42)
    forloop_block = ForLoopMoEBlock(D, heads, K, FF, dropout=0.0).to(device).train()
    einsum_block = MoETransformerBlock(D, heads, K, FF, dropout=0.0).to(device).train()
    copy_weights_forloop_to_moe_block(forloop_block, einsum_block)

    x = torch.randn(B, T, D, device=device)
    mask = torch.rand(B, K, device=device)
    attn_mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)

    # Warm up
    for block in [forloop_block, einsum_block]:
        for _ in range(n_warmup):
            block.zero_grad()
            out = block(x, branch_mask=mask, mask_scale=2.8, attn_mask=attn_mask)
            out.sum().backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()

    for name, block in [("For-loop", forloop_block), ("Einsum", einsum_block)]:
        times = []
        for _ in range(n_timed):
            block.zero_grad()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = block(x, branch_mask=mask, mask_scale=2.8, attn_mask=attn_mask)
            out.sum().backward()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        mean_t = sum(times) / len(times) * 1000
        print(f"    {name:>10}: {mean_t:>8.2f} ms (fwd+bwd, K={K} D={D} B={B} T={T})")

    # --- Einsum + AMP + torch.compile ---
    if device.type == 'cuda':
        from torch.amp import autocast, GradScaler
        print(f"\n  --- Einsum + AMP + compile (fwd+bwd, full MoE block) ---")

        torch.manual_seed(42)
        ei_amp = MoETransformerBlock(D, heads, K, FF, dropout=0.0).to(device).train()
        copy_weights_forloop_to_moe_block(forloop_block, ei_amp)
        try:
            ei_compiled = torch.compile(ei_amp)
        except Exception:
            ei_compiled = ei_amp

        scaler = GradScaler('cuda')
        opt = torch.optim.AdamW(ei_compiled.parameters(), lr=1e-4)

        # Warmup (extra for compile JIT)
        for _ in range(5):
            opt.zero_grad()
            with autocast('cuda'):
                out = ei_compiled(x, branch_mask=mask, mask_scale=2.8, attn_mask=attn_mask)
            scaler.scale(out.sum()).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()

        times_amp = []
        for _ in range(n_timed):
            opt.zero_grad()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast('cuda'):
                out = ei_compiled(x, branch_mask=mask, mask_scale=2.8, attn_mask=attn_mask)
            scaler.scale(out.sum()).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            times_amp.append(time.perf_counter() - t0)
        mean_amp = sum(times_amp) / len(times_amp) * 1000
        print(f"    Einsum+AMP+compile: {mean_amp:>8.2f} ms (K={K} D={D} B={B} T={T})")


def test_shared_expert_block_uses_parallel_ffn():
    """Verify SharedExpertMoETransformerBlock uses ParallelFFN (not for-loop)."""
    from models.transformer_lm import SharedExpertMoETransformerBlock
    block = SharedExpertMoETransformerBlock(256, 4, 7, 512, shared_expert_dim=220)
    assert hasattr(block, 'parallel_ffn'), "Missing parallel_ffn attribute"
    assert isinstance(block.parallel_ffn, ParallelFFN), "parallel_ffn is not ParallelFFN"
    assert not hasattr(block, 'branches'), "Still has branches ModuleList (should be removed)"
    print(f"  SharedExpertMoETransformerBlock uses ParallelFFN ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        ("Parameter Count", test_param_count),
        ("ParallelFFN Forward Equivalence", test_parallel_ffn_equivalence),
        ("ParallelFFN Gradient Equivalence", test_parallel_ffn_gradient),
        ("MoETransformerBlock Equivalence", test_moe_block_equivalence),
        ("SharedExpert Uses ParallelFFN", test_shared_expert_block_uses_parallel_ffn),
        ("Speed Comparison", test_speed_comparison),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f" {name}")
        print(f"{'='*60}")
        try:
            fn()
            passed += 1
            print(f" ✓ PASSED")
        except Exception as e:
            failed += 1
            print(f" ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f" Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)

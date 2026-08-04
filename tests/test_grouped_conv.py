#!/usr/bin/env python
"""Test that GroupedMultiBranchResNet110 is equivalent to MultiBranchResNet110.

Tests:
  1. Parameter count matches exactly
  2. Forward pass equivalence (eval mode) — multiple configs
  3. Forward pass equivalence (train mode)
  4. Various input shapes and mask configurations
  5. Gradient equivalence (backward pass)
  6. Weight conversion roundtrip (separate → grouped → separate)
  7. Speed comparison (forward + backward)

Usage:
    python tests/test_grouped_conv.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from models.multi_branch import (
    MultiBranchResNet110,
    GroupedMultiBranchResNet110,
    convert_resnet110_forloop_to_grouped,
    convert_resnet110_grouped_to_forloop,
)

# Reproducibility
torch.manual_seed(42)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ── Test configs ──────────────────────────────────────────────────────────

# (K, num_blocks, base_ch, branch_ch, shared_expert, se_blocks)
CONFIGS = [
    # Small: quick sanity check
    (4, 3, 16, [16, 32, 64], False, None),
    # Our actual config without SE
    (20, 3, 16, [4, 7, 14], False, None),
    # Our actual config with SE (se_blocks = num_blocks)
    (20, 3, 16, [3, 7, 14], True, None),
    # SE with different capacity (se_blocks=2, fewer than num_blocks=3)
    (10, 3, 16, None, True, 2),
]

# Use a small num_blocks (3) for fast testing. The real config (18) is tested
# only in the speed benchmark to keep unit tests fast.


def _make_pair(cfg):
    """Create matched for-loop and grouped models from config tuple."""
    K, nb, bc, bch, se, se_blk = cfg
    kwargs = dict(
        num_branches=K, num_blocks=nb, num_classes=100,
        base_channels=bc, branch_channels=bch,
        shared_expert=se, shared_expert_blocks=se_blk,
    )
    torch.manual_seed(42)
    forloop = MultiBranchResNet110(**kwargs)
    grouped = convert_resnet110_forloop_to_grouped(forloop)
    return forloop, grouped


def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── Tests ─────────────────────────────────────────────────────────────────

def test_param_count():
    """Both models have identical parameter counts for all configs."""
    for cfg in CONFIGS:
        fl, gr = _make_pair(cfg)
        fl_p = sum(p.numel() for p in fl.parameters())
        gr_p = sum(p.numel() for p in gr.parameters())
        assert fl_p == gr_p, (
            f"Param mismatch for cfg {cfg}: forloop={fl_p}, grouped={gr_p}")
        print(f"  K={cfg[0]:>2}, SE={str(cfg[4]):>5}: {fl_p:,} params ✓")


def test_forward_equivalence_eval():
    """Forward outputs match in eval mode (deterministic BN with running stats)."""
    for cfg in CONFIGS:
        fl, gr = _make_pair(cfg)
        fl.eval()
        gr.eval()
        K = cfg[0]

        torch.manual_seed(123)
        x = torch.randn(8, 3, 32, 32)
        mask = torch.rand(8, K)
        fl.mask_scale = 2.8
        gr.mask_scale = 2.8

        with torch.no_grad():
            out_fl = fl(x, mask)
            out_gr = gr(x, mask)

        diff = (out_fl - out_gr).abs().max().item()
        assert torch.allclose(out_fl, out_gr, atol=1e-5, rtol=1e-5), (
            f"Eval forward mismatch for cfg {cfg}: max_diff={diff:.2e}")
        print(f"  K={cfg[0]:>2}, SE={str(cfg[4]):>5}: max_diff={diff:.2e} ✓")


def test_forward_equivalence_train():
    """Forward outputs match in train mode (BN uses batch statistics)."""
    for cfg in CONFIGS:
        fl, gr = _make_pair(cfg)
        fl.train()
        gr.train()
        K = cfg[0]

        torch.manual_seed(456)
        x = torch.randn(16, 3, 32, 32)
        mask = torch.rand(16, K)
        fl.mask_scale = 2.8
        gr.mask_scale = 2.8

        out_fl = fl(x, mask)
        out_gr = gr(x, mask)

        diff = (out_fl - out_gr).abs().max().item()
        assert torch.allclose(out_fl, out_gr, atol=1e-5, rtol=1e-5), (
            f"Train forward mismatch for cfg {cfg}: max_diff={diff:.2e}")
        print(f"  K={cfg[0]:>2}, SE={str(cfg[4]):>5}: max_diff={diff:.2e} ✓")


def test_forward_various_inputs():
    """Test with various batch sizes and mask configurations."""
    K, nb, bc, bch, se, se_blk = 20, 3, 16, [4, 7, 14], False, None
    fl, gr = _make_pair((K, nb, bc, bch, se, se_blk))
    fl.eval()
    gr.eval()

    cases = [
        ("B=1, mask=None", 1, None, None),
        ("B=1, mask=ones", 1, lambda b: torch.ones(b, K), 2.8),
        ("B=4, mask=rand", 4, lambda b: torch.rand(b, K), 2.8),
        # Fixed-denominator: when mask_scale is provided, silent per-sample
        # normalization is no longer an option (would re-introduce Jensen bias).
        ("B=32, mask=binary", 32, lambda b: (torch.rand(b, K) > 0.5).float(), 2.8),
        ("B=64, mask=None", 64, None, None),
        ("B=128, mask=soft_specdrop",
         128,
         lambda b: torch.where(
             torch.arange(K).unsqueeze(0).expand(b, -1) == (torch.arange(b) % K).unsqueeze(1),
             torch.tensor(0.9), torch.tensor(0.1)),
         2.8),
    ]

    for name, B, mask_fn, ms in cases:
        torch.manual_seed(789)
        x = torch.randn(B, 3, 32, 32)
        mask = mask_fn(B) if mask_fn is not None else None
        fl.mask_scale = ms
        gr.mask_scale = ms

        with torch.no_grad():
            out_fl = fl(x, mask)
            out_gr = gr(x, mask)

        diff = (out_fl - out_gr).abs().max().item()
        assert torch.allclose(out_fl, out_gr, atol=1e-5, rtol=1e-5), (
            f"Mismatch for '{name}': max_diff={diff:.2e}")
        print(f"  {name:>35}: max_diff={diff:.2e} ✓")


def test_gradient_equivalence():
    """Backward pass produces identical gradients on shared stem and FC."""
    K, nb, bc, bch = 20, 3, 16, [4, 7, 14]
    fl, gr = _make_pair((K, nb, bc, bch, False, None))
    fl.train()
    gr.train()
    fl.mask_scale = 2.8
    gr.mask_scale = 2.8

    torch.manual_seed(999)
    x = torch.randn(16, 3, 32, 32)
    mask = torch.rand(16, K)
    target = torch.randint(0, 100, (16,))

    # Forward + backward on both
    loss_fl = nn.CrossEntropyLoss()(fl(x, mask), target)
    loss_fl.backward()

    loss_gr = nn.CrossEntropyLoss()(gr(x, mask), target)
    loss_gr.backward()

    # CUDA float32 accumulation order can differ between grouped and separate
    # convolutions, so use a relaxed tolerance for gradient checks.
    grad_tol = 1e-3

    # Compare loss values
    loss_diff = abs(loss_fl.item() - loss_gr.item())
    print(f"  Loss diff: {loss_diff:.2e}")
    assert loss_diff < grad_tol, f"Loss mismatch: {loss_diff:.2e}"

    # Compare gradients on shared stem (conv1)
    g_fl = fl.conv1.weight.grad
    g_gr = gr.conv1.weight.grad
    grad_diff_stem = (g_fl - g_gr).abs().max().item()
    print(f"  Stem conv1 grad diff: {grad_diff_stem:.2e}")
    assert grad_diff_stem < grad_tol, f"Stem grad mismatch: {grad_diff_stem:.2e}"

    # Compare gradients on FC
    g_fl_fc = fl.fc.weight.grad
    g_gr_fc = gr.fc.weight.grad
    grad_diff_fc = (g_fl_fc - g_gr_fc).abs().max().item()
    print(f"  FC weight grad diff: {grad_diff_fc:.2e}")
    assert grad_diff_fc < grad_tol, f"FC grad mismatch: {grad_diff_fc:.2e}"

    # Compare branch gradients: forloop branch 0 conv1 vs grouped layer1[0] slice
    fl_b0_grad = fl.branches_l1[0][0].conv1.weight.grad  # (bc1, w, 3, 3)
    gr_l1_grad = gr.layer1[0].conv1.weight.grad  # (K*bc1, w, 3, 3)
    C = fl_b0_grad.size(0)
    gr_b0_grad = gr_l1_grad[:C]  # branch 0's slice
    branch_grad_diff = (fl_b0_grad - gr_b0_grad).abs().max().item()
    print(f"  Branch 0 layer1 conv1 grad diff: {branch_grad_diff:.2e}")
    assert branch_grad_diff < grad_tol, f"Branch grad mismatch: {branch_grad_diff:.2e}"

    print(f"  All gradient checks passed ✓")


def test_weight_conversion_roundtrip():
    """separate → grouped → separate preserves weights exactly."""
    for cfg in CONFIGS:
        torch.manual_seed(42)
        fl_orig = MultiBranchResNet110(
            num_branches=cfg[0], num_blocks=cfg[1], num_classes=100,
            base_channels=cfg[2], branch_channels=cfg[3],
            shared_expert=cfg[4], shared_expert_blocks=cfg[5],
        )

        # forloop → grouped → forloop
        grouped = convert_resnet110_forloop_to_grouped(fl_orig)
        fl_back = convert_resnet110_grouped_to_forloop(grouped)

        # Compare all parameters
        sd_orig = fl_orig.state_dict()
        sd_back = fl_back.state_dict()
        assert set(sd_orig.keys()) == set(sd_back.keys()), "State dict keys mismatch"

        max_diff = 0.0
        for key in sd_orig:
            d = (sd_orig[key].float() - sd_back[key].float()).abs().max().item()
            max_diff = max(max_diff, d)

        assert max_diff < 1e-6, (
            f"Roundtrip weight mismatch for cfg {cfg}: max_diff={max_diff:.2e}")
        print(f"  K={cfg[0]:>2}, SE={str(cfg[4]):>5}: max_diff={max_diff:.2e} ✓")


def test_speed_comparison():
    """Benchmark grouped conv vs for-loop at realistic scale."""
    device = _get_device()
    print(f"  Device: {device}")

    configs_to_bench = [
        # (label, K, num_blocks, batch_size, branch_channels, SE)
        ("K=20 nb=18 B=128 (ResNet-110)", 20, 18, 128, [4, 7, 14], False),
        ("K=20 nb=18 B=128 +SE (ResNet-110)", 20, 18, 128, [3, 7, 14], True),
    ]

    for label, K, nb, B, bch, se in configs_to_bench:
        torch.manual_seed(42)
        fl = MultiBranchResNet110(
            num_branches=K, num_blocks=nb, num_classes=100,
            base_channels=16, branch_channels=bch, shared_expert=se,
        ).to(device).eval()
        gr = convert_resnet110_forloop_to_grouped(fl).to(device).eval()
        fl.mask_scale = 2.8
        gr.mask_scale = 2.8

        x = torch.randn(B, 3, 32, 32, device=device)
        mask = torch.rand(B, K, device=device)

        # Warm up
        n_warmup = 3
        n_timed = 10
        with torch.no_grad():
            for _ in range(n_warmup):
                fl(x, mask)
                gr(x, mask)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # Time for-loop version
        times_fl = []
        with torch.no_grad():
            for _ in range(n_timed):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                fl(x, mask)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times_fl.append(time.perf_counter() - t0)

        # Time grouped version
        times_gr = []
        with torch.no_grad():
            for _ in range(n_timed):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                gr(x, mask)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times_gr.append(time.perf_counter() - t0)

        mean_fl = sum(times_fl) / len(times_fl) * 1000
        mean_gr = sum(times_gr) / len(times_gr) * 1000
        speedup = mean_fl / mean_gr if mean_gr > 0 else float('inf')

        print(f"\n  [{label}]")
        print(f"    For-loop: {mean_fl:>8.1f} ms")
        print(f"    Grouped:  {mean_gr:>8.1f} ms")
        print(f"    Speedup:  {speedup:>8.1f}x")

    # Also benchmark forward+backward at full scale
    print(f"\n  --- Forward + Backward (ResNet-110) ---")
    K, nb, B, bch = 20, 18, 128, [4, 7, 14]
    torch.manual_seed(42)
    fl = MultiBranchResNet110(
        num_branches=K, num_blocks=nb, num_classes=100,
        base_channels=16, branch_channels=bch,
    ).to(device).train()
    gr = convert_resnet110_forloop_to_grouped(fl).to(device).train()
    fl.mask_scale = 2.8
    gr.mask_scale = 2.8

    x = torch.randn(B, 3, 32, 32, device=device)
    mask = torch.rand(B, K, device=device)
    target = torch.randint(0, 100, (B,), device=device)
    criterion = nn.CrossEntropyLoss()

    # Warm up
    for model in [fl, gr]:
        for _ in range(n_warmup):
            model.zero_grad()
            loss = criterion(model(x, mask), target)
            loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()

    fwd_bwd_results = {}
    for name, model in [("For-loop", fl), ("Grouped", gr)]:
        times = []
        for _ in range(n_timed):
            model.zero_grad()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = criterion(model(x, mask), target)
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        mean_t = sum(times) / len(times) * 1000
        fwd_bwd_results[name] = mean_t
        print(f"    {name:>10}: {mean_t:>8.1f} ms (fwd+bwd, K={K} nb={nb} B={B})")
    print(f"    {'Speedup':>10}: {fwd_bwd_results['For-loop']/fwd_bwd_results['Grouped']:>8.1f}x")

    # --- Grouped + AMP + torch.compile (realistic training speed) ---
    if device.type == 'cuda':
        from torch.amp import autocast, GradScaler
        print(f"\n  --- Grouped + AMP + compile (fwd+bwd, ResNet-110) ---")
        amp_configs = [
            ("no SE", [4, 7, 14], False),
            ("+SE",   [3, 7, 14], True),
        ]
        for label, bch_amp, se_flag in amp_configs:
            torch.manual_seed(42)
            gr_amp = GroupedMultiBranchResNet110(
                num_branches=K, num_blocks=18, num_classes=100,
                base_channels=16, branch_channels=bch_amp,
                shared_expert=se_flag,
            ).to(device).train()
            gr_amp.mask_scale = 2.8
            try:
                gr_compiled = torch.compile(gr_amp)
            except Exception:
                gr_compiled = gr_amp

            scaler = GradScaler('cuda')
            x_b = torch.randn(B, 3, 32, 32, device=device)
            mask_b = torch.rand(B, K, device=device)
            tgt_b = torch.randint(0, 100, (B,), device=device)

            # Warmup (extra for compile JIT)
            for _ in range(5):
                gr_compiled.zero_grad()
                with autocast('cuda'):
                    loss = criterion(gr_compiled(x_b, mask_b), tgt_b)
                scaler.scale(loss).backward()
                scaler.step(torch.optim.SGD(gr_compiled.parameters(), lr=0.01))
                scaler.update()
            torch.cuda.synchronize()

            times_amp = []
            opt = torch.optim.SGD(gr_compiled.parameters(), lr=0.01)
            for _ in range(n_timed):
                opt.zero_grad()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with autocast('cuda'):
                    loss = criterion(gr_compiled(x_b, mask_b), tgt_b)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                torch.cuda.synchronize()
                times_amp.append(time.perf_counter() - t0)
            mean_amp = sum(times_amp) / len(times_amp) * 1000
            vs_fl = fwd_bwd_results['For-loop'] / mean_amp
            print(f"    Grouped+AMP+compile ({label}): {mean_amp:>8.1f} ms  (vs for-loop: {vs_fl:.1f}x)")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ("Parameter Count", test_param_count),
        ("Forward Equivalence (eval)", test_forward_equivalence_eval),
        ("Forward Equivalence (train)", test_forward_equivalence_train),
        ("Various Inputs & Masks", test_forward_various_inputs),
        ("Gradient Equivalence", test_gradient_equivalence),
        ("Weight Conversion Roundtrip", test_weight_conversion_roundtrip),
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

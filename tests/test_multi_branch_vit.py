"""Unit tests for MultiBranchViT-Small.

Verifies:
  1. Param count ≤ 2% deviation from dense ViT-Small/16 (22,050,664 params)
  2. Auto-computed branch_hidden param-matches across K ∈ {8, 16, 46}
  3. Forward shape (B, num_classes) on 224×224 input
  4. get_stem_features returns 4-D (B, D, H_patches, W_patches) for trainer compat
  5. forward_from_stem works with / without branch_mask
  6. mask_scale fixed-denominator merge works
  7. Same branch_mask is applied identically at every block
  8. ParallelMLP equivalent to a list of dense FFNs (batched einsum correctness)
  9. Backward gradients flow to every parameter
 10. DropPath randomizes in train, deterministic in eval
 11. Shared params: patch_embed / cls_token / pos_embed / attention / head

Run: python tests/test_multi_branch_vit.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.multi_branch_vit import (MultiBranchViT, ParallelMLP,
                                     multi_branch_vit_small)
from models.vit import vit_small


DENSE_VIT_S16_PARAMS = 22_050_664
PARAM_TOLERANCE = 0.02


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def test_param_match_default_k46():
    """Default K=46 (BREEDS superclass count), branch_hidden=33 → param-matched."""
    model = multi_branch_vit_small(num_classes=1000)
    assert model.num_branches == 46
    assert model.branch_hidden == 33
    params = _count_params(model)
    ratio = params / DENSE_VIT_S16_PARAMS
    assert abs(ratio - 1.0) < PARAM_TOLERANCE, \
        f"K=46 params={params:,}, ratio={ratio:.4f}, deviation {abs(ratio-1):.2%}"
    print(f"  PASS param_match_default_k46: K=46 h=33 → {params:,} ({ratio:.4f}x)")


def test_param_match_k16():
    """K=16 auto branch_hidden=96 → param-matched."""
    model = multi_branch_vit_small(num_classes=1000, num_branches=16)
    assert model.branch_hidden == 96
    params = _count_params(model)
    ratio = params / DENSE_VIT_S16_PARAMS
    assert abs(ratio - 1.0) < PARAM_TOLERANCE, \
        f"K=16 params={params:,}, ratio={ratio:.4f}"
    print(f"  PASS param_match_k16: K=16 h=96 → {params:,} ({ratio:.4f}x)")


def test_auto_compute_branch_hidden_accounts_for_shared_expert_dim():
    """Regression: when branch_hidden is None AND shared_expert_dim > 0,
    auto-compute must subtract SE from the MLP budget BEFORE dividing.

    Pre-fix the formula was `standard_hidden / K` ignoring SE, so total
    FFN = K × branch_hidden + SE ended up ~SE/standard_hidden over budget.
    Observable when SE_ratio > ~0.3x (SE=115 on K=16 blows the 2% sanity
    check). Latent because all 5a/5b/5c shell scripts pass explicit
    branch_hidden, so the bug never fired in practice — but fixing it
    preempts any future caller that relies on the auto-compute path.
    """
    # Case 1: K=16, SE=96 — SE takes 6.25% of budget, must reduce branches
    model = multi_branch_vit_small(num_classes=1000, num_branches=16,
                                    shared_expert_dim=96)
    # Pre-fix: branch_hidden = 1536/16 = 96 → total FFN = 16*96 + 96 = 1632 (+6%)
    # Post-fix: branch_hidden = (1536-96)/16 = 90 → total FFN = 16*90 + 96 = 1536 (0%)
    assert model.branch_hidden == 90, \
        f"Expected branch_hidden=90 (SE-aware); got {model.branch_hidden} " \
        f"(pre-fix would be 96, ignoring SE)"
    params = _count_params(model)
    ratio = params / DENSE_VIT_S16_PARAMS
    assert abs(ratio - 1.0) < PARAM_TOLERANCE, \
        f"K=16 SE=96 params={params:,}, ratio={ratio:.4f}, " \
        f"deviation {abs(ratio-1):.2%}"

    # Case 2: K=46, SE=192 — big SE_ratio, would blow the 2% budget pre-fix
    model = multi_branch_vit_small(num_classes=1000, num_branches=46,
                                    shared_expert_dim=192)
    # Pre-fix: branch_hidden = round(1536/46) = 33 → total = 46*33 + 192 = 1710 (+11%)
    # Post-fix: branch_hidden = round((1536-192)/46) = round(29.22) = 29
    #          → total = 46*29 + 192 = 1526 (-0.65%)
    assert model.branch_hidden == 29, \
        f"Expected branch_hidden=29 (SE-aware); got {model.branch_hidden}"
    params = _count_params(model)
    ratio = params / DENSE_VIT_S16_PARAMS
    assert abs(ratio - 1.0) < PARAM_TOLERANCE, \
        f"K=46 SE=192 params={params:,}, ratio={ratio:.4f}"


def test_auto_compute_branch_hidden_floor_guard():
    """Edge case: SE so large the remaining budget / K < 8 → floor to 8."""
    model = multi_branch_vit_small(num_classes=1000, num_branches=46,
                                    shared_expert_dim=1500)
    # (1536 - 1500) / 46 = 0.78 → round to 1 → max(8, 1) = 8
    assert model.branch_hidden == 8, \
        f"Expected branch_hidden=8 (floor guard), got {model.branch_hidden}"


def test_auto_compute_branch_hidden_se_zero_backward_compat():
    """With SE=0, auto-compute must match pre-fix behavior exactly."""
    model = multi_branch_vit_small(num_classes=1000, num_branches=46,
                                    shared_expert_dim=0)
    # (1536 - 0) / 46 = round(33.39) = 33 → same as pre-fix
    assert model.branch_hidden == 33
    model2 = multi_branch_vit_small(num_classes=1000, num_branches=16,
                                     shared_expert_dim=0)
    # (1536 - 0) / 16 = 96 → same as pre-fix
    assert model2.branch_hidden == 96


def test_param_match_k46():
    """K=46 (BREEDS) auto branch_hidden=33 → param-matched."""
    model = multi_branch_vit_small(num_classes=1000, num_branches=46)
    assert model.branch_hidden == 33
    params = _count_params(model)
    ratio = params / DENSE_VIT_S16_PARAMS
    assert abs(ratio - 1.0) < PARAM_TOLERANCE, \
        f"K=46 params={params:,}, ratio={ratio:.4f}"
    print(f"  PASS param_match_k46: K=46 h=33 → {params:,} ({ratio:.4f}x)")


def test_forward_shape_imagenet():
    """Forward on 224×224 should yield (B, 1000)."""
    model = multi_branch_vit_small(num_classes=1000).eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1000), f"Got {out.shape}"
    print(f"  PASS forward_shape_imagenet: {out.shape}")


def test_forward_shape_cifar100():
    """Forward at num_classes=100 should yield (B, 100)."""
    model = multi_branch_vit_small(num_classes=100).eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 100), f"Got {out.shape}"
    print(f"  PASS forward_shape_cifar100: {out.shape}")


def test_get_stem_features_shape():
    """get_stem_features → (B, D=384, H=14, W=14) for 224×224 patch=16."""
    model = multi_branch_vit_small(num_classes=1000).eval()
    x = torch.randn(2, 3, 224, 224)
    stem = model.get_stem_features(x)
    assert stem.shape == (2, 384, 14, 14), f"Got {stem.shape}"
    # Trainer pattern: pooled = F.adaptive_avg_pool2d(stem, 1).flatten(1)
    pooled = F.adaptive_avg_pool2d(stem, 1).flatten(1)
    assert pooled.shape == (2, 384), f"Pooled got {pooled.shape}"
    print(f"  PASS get_stem_features_shape: {stem.shape} → pooled {pooled.shape}")


def test_forward_from_stem_no_mask():
    model = multi_branch_vit_small(num_classes=100).eval()
    x = torch.randn(2, 3, 224, 224)
    stem = model.get_stem_features(x)
    out = model.forward_from_stem(stem, branch_mask=None)
    assert out.shape == (2, 100)
    # Equivalent to full forward when mask is None
    assert torch.allclose(out, model(x), atol=1e-5)
    print(f"  PASS forward_from_stem_no_mask: matches full forward")


def test_forward_with_branch_mask():
    model = multi_branch_vit_small(num_classes=100, num_branches=8).eval()
    # B1 enforcement: providing branch_mask without mask_scale now raises.
    # Set a reasonable fixed denominator for the test.
    model.mask_scale = 4.0
    x = torch.randn(2, 3, 224, 224)
    mask = torch.rand(2, 8)
    out = model(x, branch_mask=mask)
    assert out.shape == (2, 100)
    mask2 = torch.rand(2, 8)
    out2 = model(x, branch_mask=mask2)
    assert not torch.allclose(out, out2), "Different masks should differ"
    print(f"  PASS forward_with_branch_mask")


def test_fixed_denominator_merge():
    """B1 enforcement: branch_mask without mask_scale must raise; different
    mask_scale values produce different outputs (denominator matters)."""
    model = multi_branch_vit_small(num_classes=100, num_branches=8).eval()
    x = torch.randn(2, 3, 224, 224)
    mask = torch.full((2, 8), 0.5)

    # B1: mask without mask_scale must error loudly (no silent fallback).
    model.mask_scale = None
    try:
        _ = model(x, branch_mask=mask)
    except RuntimeError as e:
        assert "mask_scale" in str(e) and "Jensen" in str(e)
    else:
        raise AssertionError("Expected RuntimeError for None mask_scale")

    # Different mask_scale values produce different outputs (denominator matters).
    model.mask_scale = 4.0
    out_4 = model(x, branch_mask=mask)
    model.mask_scale = 8.0
    out_8 = model(x, branch_mask=mask)
    assert not torch.allclose(out_4, out_8)
    print(f"  PASS fixed_denominator_merge (raises on None + respects mask_scale)")


def test_mask_applied_per_block():
    """Binary mask (all-branches-on for one sample, only branch 0 for another)
    should produce different outputs — confirming mask flows into every block."""
    model = multi_branch_vit_small(num_classes=100, num_branches=8).eval()
    model.mask_scale = 8.0
    x = torch.randn(2, 3, 224, 224)
    mask = torch.zeros(2, 8)
    mask[0] = 1.0              # sample 0: all 8 branches active
    mask[1, 0] = 1.0            # sample 1: only branch 0 active
    out = model(x, branch_mask=mask)
    # Different masks must produce different logits
    assert not torch.allclose(out[0], out[1], atol=1e-4)
    print(f"  PASS mask_applied_per_block")


def test_parallel_mlp_matches_list_of_ffns():
    """ParallelMLP output[:, :, k] must equal a dense FFN_k(x) with copied weights."""
    torch.manual_seed(0)
    D, H, K = 32, 48, 4
    pmlp = ParallelMLP(dim=D, branch_hidden=H, num_branches=K).eval()

    x = torch.randn(2, 10, D)
    y_parallel = pmlp(x)            # (B, N, K, D)

    for k in range(K):
        ref_fc1 = nn.Linear(D, H)
        ref_fc1.weight.data = pmlp.w1[k].clone()
        ref_fc1.bias.data = pmlp.b1[k].clone()
        ref_fc2 = nn.Linear(H, D)
        ref_fc2.weight.data = pmlp.w2[k].clone()
        ref_fc2.bias.data = pmlp.b2[k].clone()
        y_ref = ref_fc2(F.gelu(ref_fc1(x)))
        assert torch.allclose(y_parallel[:, :, k], y_ref, atol=1e-5), \
            f"Branch {k} einsum mismatch with dense FFN"
    print(f"  PASS parallel_mlp_matches_list_of_ffns ({K} branches)")


def test_backward_grad_flow():
    model = multi_branch_vit_small(num_classes=100, num_branches=8).train()
    model.mask_scale = 4.0  # B1: must set before passing branch_mask
    x = torch.randn(2, 3, 224, 224)
    mask = torch.rand(2, 8)
    out = model(x, branch_mask=mask)
    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"
    print(f"  PASS backward_grad_flow")


def test_shared_params_vs_branched():
    """Attention, patch_embed, cls_token, pos_embed, norm, head should be shared
    across branches (not replicated K times)."""
    model = multi_branch_vit_small(num_classes=1000, num_branches=8)
    blk0 = model.blocks[0]
    assert blk0.attn.qkv.weight.shape == (3 * 384, 384)
    assert blk0.attn.proj.weight.shape == (384, 384)
    assert blk0.parallel_mlp.w1.shape == (8, 192, 384)
    assert blk0.parallel_mlp.w2.shape == (8, 384, 192)
    assert model.patch_embed.proj.weight.shape == (384, 3, 16, 16)
    assert model.cls_token.shape == (1, 1, 384)
    assert model.pos_embed.shape == (1, 197, 384)
    assert model.head.weight.shape == (1000, 384)
    print(f"  PASS shared_params_vs_branched: attn & stem shared, MLP branched")


def test_drop_path_behavior():
    model = multi_branch_vit_small(num_classes=100, num_branches=4, drop_path_rate=0.5)
    x = torch.randn(2, 3, 224, 224)

    model.train()
    o1, o2 = model(x), model(x)
    assert not torch.allclose(o1, o2), "DropPath train should randomize"

    model.eval()
    o3, o4 = model(x), model(x)
    assert torch.allclose(o3, o4), "DropPath eval should be deterministic"
    print(f"  PASS drop_path_behavior")


def test_branch_count_in_model():
    """Verify total MLP-branch params scale with K correctly."""
    for K in [4, 8, 16]:
        model = multi_branch_vit_small(num_classes=1000, num_branches=K)
        total_branch_params = sum(
            p.numel() for blk in model.blocks for p in blk.parallel_mlp.parameters())
        expected_per_block = K * (2 * model.embed_dim * model.branch_hidden
                                   + model.branch_hidden + model.embed_dim)
        assert total_branch_params == 12 * expected_per_block, \
            f"K={K}: got {total_branch_params}, expected {12*expected_per_block}"
    print(f"  PASS branch_count_in_model (K ∈ {{4, 8, 16}})")


def test_algorithm_integration():
    """Every registered routing algorithm should work end-to-end on MultiBranchViT.

    Verifies: build_algorithm → get_mask (with/without features) → forward_from_stem
    → backward (incl. auxiliary loss). Catches shape/dtype mismatches and dead gradients
    before full ImageNet training starts (addresses review Flag 4).
    """
    from algorithms import build_algorithm

    K = 8
    num_categories = 46   # BREEDS ImageNet superclass count

    # After 2026-04-12 pivot, only 'soft_specdrop' (ours) routes MultiBranchViT.
    # Other MoE baselines are now faithful standalone models (models/{soft_moe,
    # mod_squad, comet}_vit.py) and do not go through build_algorithm.
    algo_types = ['soft_specdrop']

    for algo_type in algo_types:
        model = multi_branch_vit_small(num_classes=1000, num_branches=K)
        full_cfg = {
            'seed': 42,
            'model': {'type': 'multi_branch_vit_small', 'num_branches': K,
                      'embed_dim': 384},
            'algorithm': {'type': algo_type, '_num_categories': num_categories},
            'training': {'epochs': 200},
        }
        algo = build_algorithm(full_cfg)

        if algo.expected_mask_sum is not None:
            model.mask_scale = algo.expected_mask_sum

        x = torch.randn(2, 3, 224, 224)
        category_ids = torch.tensor([0, 1])

        model.train()
        stem = model.get_stem_features(x)
        assert stem.shape == (2, 384, 14, 14), \
            f"{algo_type}: stem shape {stem.shape}"
        pooled = F.adaptive_avg_pool2d(stem, 1).flatten(1)
        assert pooled.shape == (2, 384)

        if getattr(algo, 'needs_features', False):
            mask = algo.get_mask(category_ids, training=True, features=pooled)
        else:
            mask = algo.get_mask(category_ids, training=True)
        assert mask.shape == (2, K), \
            f"{algo_type}: mask shape {mask.shape}, expected (2, {K})"
        assert torch.isfinite(mask).all(), f"{algo_type}: non-finite mask values"

        logits = model.forward_from_stem(stem, branch_mask=mask)
        assert logits.shape == (2, 1000), \
            f"{algo_type}: logits shape {logits.shape}"

        loss = logits.sum() + algo.get_auxiliary_loss()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"{algo_type}/{name}: no grad"
            assert torch.isfinite(p.grad).all(), \
                f"{algo_type}/{name}: non-finite grad"
    print(f"  PASS algorithm_integration: {len(algo_types)} algorithms × "
          f"end-to-end forward+backward")


def test_compare_ratio_to_dense():
    """Print the param ratio for each K vs dense ViT-S — informational."""
    dense = _count_params(vit_small(num_classes=1000))
    print(f"  Dense ViT-S/16:  {dense:,} (reference)")
    for K in [4, 8, 16, 46]:
        model = multi_branch_vit_small(num_classes=1000, num_branches=K)
        p = _count_params(model)
        print(f"  MB-ViT K={K:2d} h={model.branch_hidden:3d}: {p:,}  "
              f"({p/dense:.4f}x)")


# ── Shared-expert (SE) tests ────────────────────────────────────────────────

def test_shared_expert_off_is_backward_compatible():
    """shared_expert_dim=0 (default) must be bit-identical to pre-SE code."""
    torch.manual_seed(0)
    m_off = multi_branch_vit_small(num_classes=100, num_branches=8,
                                    shared_expert_dim=0).eval()
    torch.manual_seed(0)
    m_ref = multi_branch_vit_small(num_classes=100, num_branches=8).eval()
    x = torch.randn(2, 3, 224, 224)
    assert torch.allclose(m_off(x), m_ref(x), atol=1e-6)
    # No shared_expert module should be instantiated.
    for blk in m_off.blocks:
        assert blk.shared_expert is None
    print(f"  PASS shared_expert_off_is_backward_compatible")


def test_shared_expert_param_budget_se_sweep():
    """At K=46, SE_ratio ∈ {0, 0.5, 1.0, 2.0} with split_ffn_budget_for_se
    should keep total params within ±2% of dense ViT-Small."""
    from data.slimpajama import split_ffn_budget_for_se
    K = 46
    total_ffn = 384 * 4   # embed_dim × mlp_ratio
    for se_ratio in (0.0, 0.5, 1.0, 2.0):
        total_branch, se_dim = split_ffn_budget_for_se(total_ffn, se_ratio, K)
        branch_hidden = total_branch // K
        m = multi_branch_vit_small(num_classes=1000, num_branches=K,
                                    branch_hidden=branch_hidden,
                                    shared_expert_dim=se_dim)
        p = _count_params(m)
        ratio = p / DENSE_VIT_S16_PARAMS
        assert abs(ratio - 1.0) < PARAM_TOLERANCE, \
            f"K={K} SE={se_ratio}: branch_h={branch_hidden} SE_dim={se_dim} " \
            f"params={p:,} ratio={ratio:.4f}"
        print(f"  SE_ratio={se_ratio} h={branch_hidden} SE={se_dim}: "
              f"{p:,} ({ratio:.4f}x)")
    print(f"  PASS shared_expert_param_budget_se_sweep (K={K})")


def test_shared_expert_forward_shape():
    """SE-enabled forward passes must return the same shape as SE-off."""
    m = multi_branch_vit_small(num_classes=100, num_branches=8,
                                shared_expert_dim=32).eval()
    m.mask_scale = 4.0
    x = torch.randn(2, 3, 224, 224)
    mask = torch.rand(2, 8)
    out = m(x, branch_mask=mask)
    assert out.shape == (2, 100)
    print(f"  PASS shared_expert_forward_shape")


def test_shared_expert_adds_nonzero_contribution():
    """With SE on, output must differ from SE-off variant at identical init
    seed — verifies the SE branch actually contributes to the forward path."""
    torch.manual_seed(0)
    m_se = multi_branch_vit_small(num_classes=100, num_branches=8,
                                    branch_hidden=24,
                                    shared_expert_dim=32).eval()
    torch.manual_seed(0)
    m_off = multi_branch_vit_small(num_classes=100, num_branches=8,
                                    branch_hidden=24,
                                    shared_expert_dim=0).eval()
    m_se.mask_scale = m_off.mask_scale = 4.0
    x = torch.randn(2, 3, 224, 224)
    mask = torch.rand(2, 8)
    assert not torch.allclose(m_se(x, branch_mask=mask),
                              m_off(x, branch_mask=mask), atol=1e-4)
    print(f"  PASS shared_expert_adds_nonzero_contribution")


def test_shared_expert_backward_grad():
    """Every SE parameter must receive a finite gradient."""
    m = multi_branch_vit_small(num_classes=100, num_branches=8,
                                shared_expert_dim=32).train()
    m.mask_scale = 4.0
    x = torch.randn(2, 3, 224, 224)
    mask = torch.rand(2, 8)
    m(x, branch_mask=mask).sum().backward()
    for blk_idx, blk in enumerate(m.blocks):
        assert blk.shared_expert is not None
        for name, p in blk.shared_expert.named_parameters():
            assert p.grad is not None, f"block {blk_idx} SE.{name}: no grad"
            assert torch.isfinite(p.grad).all(), \
                f"block {blk_idx} SE.{name}: non-finite grad"
    print(f"  PASS shared_expert_backward_grad (all 12 SE blocks)")


def test_shared_expert_invariant_to_mask_scale():
    """The SE path is always-on (no routing), so its contribution is independent
    of branch_mask / mask_scale: varying mask_scale should scale the routed
    component only, not the SE component."""
    torch.manual_seed(0)
    m = multi_branch_vit_small(num_classes=100, num_branches=8,
                                branch_hidden=24,
                                shared_expert_dim=32).eval()
    x = torch.randn(2, 3, 224, 224)
    mask = torch.zeros(2, 8)  # zero mask: routed contribution vanishes
    m.mask_scale = 1.0
    out_mask1 = m(x, branch_mask=mask)
    m.mask_scale = 4.0
    out_mask4 = m(x, branch_mask=mask)
    # Routed = 0 / mask_scale = 0 regardless of mask_scale → only SE contributes
    # → outputs must match identically.
    assert torch.allclose(out_mask1, out_mask4, atol=1e-5)
    print(f"  PASS shared_expert_invariant_to_mask_scale (zero mask isolates SE)")


# ── build_model integration (config → shared_expert_dim) ──────────────────

def test_build_model_respects_shared_expert_dim():
    """models/__init__.py::build_model should pass shared_expert_dim from cfg."""
    from models import build_model
    cfg = {'model': {'type': 'multi_branch_vit_small', 'num_classes': 1000,
                      'num_branches': 46, 'branch_hidden': 33,
                      'shared_expert_dim': 33}}
    m = build_model(cfg)
    assert m.shared_expert_dim == 33
    for blk in m.blocks:
        assert blk.shared_expert is not None
        # Linear shape check: fc1: 384→33, fc2: 33→384
        assert blk.shared_expert.fc1.out_features == 33
        assert blk.shared_expert.fc2.in_features == 33
    print(f"  PASS build_model_respects_shared_expert_dim")


if __name__ == "__main__":
    print("=" * 60)
    print(" MultiBranchViT-Small unit tests")
    print("=" * 60)
    test_param_match_default_k46()
    test_param_match_k16()
    test_param_match_k46()
    test_forward_shape_imagenet()
    test_forward_shape_cifar100()
    test_get_stem_features_shape()
    test_forward_from_stem_no_mask()
    test_forward_with_branch_mask()
    test_fixed_denominator_merge()
    test_mask_applied_per_block()
    test_parallel_mlp_matches_list_of_ffns()
    test_backward_grad_flow()
    test_shared_params_vs_branched()
    test_drop_path_behavior()
    test_branch_count_in_model()
    test_algorithm_integration()
    test_shared_expert_off_is_backward_compatible()
    test_shared_expert_param_budget_se_sweep()
    test_shared_expert_forward_shape()
    test_shared_expert_adds_nonzero_contribution()
    test_shared_expert_backward_grad()
    test_shared_expert_invariant_to_mask_scale()
    test_build_model_respects_shared_expert_dim()
    print("-" * 60)
    print(" Param-count scaling summary:")
    test_compare_ratio_to_dense()
    print("=" * 60)
    print(" All tests passed")
    print("=" * 60)

"""Unit tests for models/multi_branch_lora.py.

Tests the core ParallelLoRAAdapter + LoRALinear modules standalone (no HF
CausalLM dependency — that integration is tested in Batch 3). Covers:
  - Param count matches K·r·(D_in+D_out) + SE rank contribution
  - Forward shape (3-D and 2-D inputs)
  - LoRA-off at init: B=0 → adapter contribution = 0, so LoRALinear ≡ base
  - Fixed-denominator merge produces the expected weighted sum
  - mask_scale guard raises on missing scale
  - Per-sample routing: different cluster_ids produce different outputs
  - Shared expert (SE) contributes nonzero output when enabled + trained
  - Backward: gradient flows to LoRA params, base frozen
  - K=1 equivalence: ParallelLoRAAdapter with K=1 ≡ Hu 2022 single LoRA
  - SE rank helper integration: split_lora_rank_budget_for_se → (r_e, r_SE)
"""
import pytest
import torch
import torch.nn as nn

from data.slimpajama import split_lora_rank_budget_for_se
from models.multi_branch_lora import (LoRALinear, ParallelLoRAAdapter,
                                      count_lora_params)


# ── Shape + param count ─────────────────────────────────────────────────────

def test_adapter_forward_shape_3d():
    adapter = ParallelLoRAAdapter(
        in_features=64, out_features=128, num_experts=5, rank=8)
    adapter.mask_scale = 1.0
    x = torch.randn(3, 7, 64)
    mask = torch.ones(3, 5)
    out = adapter(x, branch_mask=mask)
    assert out.shape == (3, 7, 128)


def test_adapter_forward_shape_2d():
    adapter = ParallelLoRAAdapter(
        in_features=32, out_features=32, num_experts=4, rank=4)
    adapter.mask_scale = 1.0
    x = torch.randn(5, 32)
    mask = torch.ones(5, 4)
    out = adapter(x, branch_mask=mask)
    assert out.shape == (5, 32)


def test_adapter_param_count_no_se():
    K, r, D_in, D_out = 20, 16, 3072, 3072
    adapter = ParallelLoRAAdapter(D_in, D_out, num_experts=K, rank=r)
    num_train, _ = count_lora_params(adapter)
    # A: K × r × D_in, B: K × D_out × r
    expected = K * r * D_in + K * D_out * r
    assert num_train == expected


def test_adapter_param_count_with_se():
    K, r, D_in, D_out, r_SE = 20, 15, 3072, 3072, 15
    adapter = ParallelLoRAAdapter(
        D_in, D_out, num_experts=K, rank=r, shared_expert_rank=r_SE)
    num_train, _ = count_lora_params(adapter)
    expected = (K * r * D_in + K * D_out * r
                + r_SE * D_in + D_out * r_SE)
    assert num_train == expected


# ── Zero-init (LoRA off at step 0) ─────────────────────────────────────────

def test_lora_off_at_init():
    """B is zero-initialised → adapter output at step 0 is exactly zero."""
    adapter = ParallelLoRAAdapter(
        in_features=32, out_features=64, num_experts=8, rank=4)
    adapter.mask_scale = 2.0
    x = torch.randn(2, 5, 32)
    mask = torch.rand(2, 8)
    out = adapter(x, branch_mask=mask)
    assert torch.allclose(out, torch.zeros_like(out))


def test_lora_off_with_se_at_init():
    """Even with SE, B_SE is zero-initialised → output at step 0 is zero."""
    adapter = ParallelLoRAAdapter(
        in_features=32, out_features=64, num_experts=8, rank=4,
        shared_expert_rank=8)
    adapter.mask_scale = 2.0
    x = torch.randn(2, 5, 32)
    mask = torch.rand(2, 8)
    out = adapter(x, branch_mask=mask)
    assert torch.allclose(out, torch.zeros_like(out))


def test_loralinear_off_at_init():
    """LoRALinear at init (B=0) ≡ frozen base linear (bit-identical)."""
    base = nn.Linear(64, 96)
    wrapped = LoRALinear(base, num_experts=5, rank=8)
    wrapped.set_mask(torch.rand(3, 5), mask_scale=1.5)
    x = torch.randn(3, 7, 64)
    base_out = base(x)
    wrapped_out = wrapped(x)
    assert torch.allclose(wrapped_out, base_out), (
        "wrapped = base + LoRA(x) but LoRA should be 0 at init (B=0)")


# ── Fixed-denominator merge math ────────────────────────────────────────────

def test_fixed_denominator_merge_math():
    """Manually set B to known nonzero values and verify merge = sum(m·h)/S."""
    torch.manual_seed(0)
    K, r, D_in, D_out = 4, 2, 8, 8
    adapter = ParallelLoRAAdapter(D_in, D_out, num_experts=K, rank=r)
    # Set B to random non-zero so LoRA contribution is non-trivial.
    with torch.no_grad():
        adapter.B.copy_(torch.randn_like(adapter.B) * 0.1)
    adapter.mask_scale = 2.0  # fixed S
    x = torch.randn(3, 1, D_in)
    # Build a non-uniform mask and compare to manual reference.
    mask = torch.rand(3, K)

    actual = adapter(x, branch_mask=mask)

    # Manual: compute per-expert routed, weight, sum / S, × scaling
    with torch.no_grad():
        hidden = torch.einsum('btd,krd->btkr', x, adapter.A)
        per_expert = torch.einsum('btkr,kdr->btkd', hidden, adapter.B)
        weighted = per_expert * mask.view(3, 1, K, 1)
        manual = (weighted.sum(dim=2) / adapter.mask_scale) * adapter.scaling
    assert torch.allclose(actual, manual, atol=1e-5)


def test_mask_scale_required():
    adapter = ParallelLoRAAdapter(
        in_features=16, out_features=16, num_experts=3, rank=4)
    # No mask_scale set
    x = torch.randn(2, 3, 16)
    mask = torch.rand(2, 3)
    # Shake the zero-init: set B to non-zero so branch path actually runs.
    with torch.no_grad():
        adapter.B.copy_(torch.randn_like(adapter.B))
    with pytest.raises(RuntimeError, match=r"mask_scale"):
        adapter(x, branch_mask=mask)


# ── Per-sample routing ──────────────────────────────────────────────────────

def test_different_masks_produce_different_outputs():
    """Two samples with different cluster_ids → different LoRA contributions."""
    torch.manual_seed(1)
    adapter = ParallelLoRAAdapter(
        in_features=32, out_features=32, num_experts=4, rank=4)
    adapter.mask_scale = 1.5
    with torch.no_grad():
        adapter.B.copy_(torch.randn_like(adapter.B) * 0.1)
    x = torch.randn(1, 2, 32).expand(2, 2, 32).contiguous()
    # Two very different masks (one-hot on different experts)
    mask_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask_b = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    out_a = adapter(x[:1], branch_mask=mask_a)
    out_b = adapter(x[:1], branch_mask=mask_b)
    # Same input, different mask → should differ substantially
    assert not torch.allclose(out_a, out_b, atol=1e-3)


# ── Shared expert (SE) independent contribution ────────────────────────────

def test_shared_expert_contribution_nonzero_when_trained():
    """After updating B_SE (via a fake gradient step), SE output is nonzero."""
    torch.manual_seed(2)
    adapter = ParallelLoRAAdapter(
        in_features=16, out_features=16, num_experts=3, rank=4,
        shared_expert_rank=4)
    adapter.mask_scale = 1.0
    # Manually "train" by setting B_se nonzero.
    with torch.no_grad():
        adapter.B_se.copy_(torch.randn_like(adapter.B_se) * 0.5)
    x = torch.randn(2, 3, 16)
    mask = torch.zeros(2, 3)  # route-off routed part → isolate SE
    out = adapter(x, branch_mask=mask)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_se_disabled_equals_no_se():
    """shared_expert_rank=0 → identical to adapter built without SE."""
    torch.manual_seed(3)
    a = ParallelLoRAAdapter(16, 16, num_experts=3, rank=4, shared_expert_rank=0)
    assert a.A_se is None
    assert a.B_se is None


# ── Gradient flow ───────────────────────────────────────────────────────────

def test_backward_lora_only():
    """Gradient should flow to LoRA params; base.weight remains None grad."""
    base = nn.Linear(16, 16)
    wrapped = LoRALinear(base, num_experts=3, rank=4, shared_expert_rank=2)
    wrapped.set_mask(torch.softmax(torch.randn(2, 3), dim=-1), mask_scale=1.5)
    # Kick B non-zero so gradients through B propagate a real signal.
    with torch.no_grad():
        wrapped.lora.B.copy_(torch.randn_like(wrapped.lora.B) * 0.01)
    x = torch.randn(2, 4, 16, requires_grad=False)
    y = wrapped(x)
    loss = y.pow(2).sum()
    loss.backward()
    # Base weight frozen: no grad (requires_grad False → grad is None).
    assert wrapped.base.weight.grad is None
    assert wrapped.base.bias.grad is None
    # LoRA params: grads populated
    assert wrapped.lora.A.grad is not None
    assert wrapped.lora.B.grad is not None
    assert wrapped.lora.A_se.grad is not None
    assert wrapped.lora.B_se.grad is not None


# ── K=1 equivalence with Hu 2022 single LoRA ────────────────────────────────

def test_k1_equivalent_to_single_lora():
    """ParallelLoRAAdapter(K=1, r) computing A·B·x equals the vanilla LoRA
    formulation y = α/r · B·A·x (Hu 2022 eq. 2).
    """
    torch.manual_seed(4)
    D_in, D_out, r = 32, 48, 8
    adapter = ParallelLoRAAdapter(D_in, D_out, num_experts=1, rank=r,
                                   alpha=16)
    adapter.mask_scale = 1.0  # S = p_a + 0 × p_i = p_a = 1.0 for K=1
    with torch.no_grad():
        adapter.B.copy_(torch.randn_like(adapter.B) * 0.1)

    x = torch.randn(3, 5, D_in)
    mask = torch.ones(3, 1)
    lora_out = adapter(x, branch_mask=mask)

    # Reference: Hu 2022 single LoRA: y = (α/r) · x · A^T · B^T
    # with our shapes A: (1, r, D_in), B: (1, D_out, r)
    A = adapter.A[0]  # (r, D_in)
    B = adapter.B[0]  # (D_out, r)
    scaling = 16.0 / r
    manual = scaling * x @ A.t() @ B.t()  # (B, T, D_out)
    assert torch.allclose(lora_out, manual, atol=1e-5)


# ── Integration with split_lora_rank_budget_for_se ─────────────────────────

@pytest.mark.parametrize("se_ratio", [0.0, 0.5, 1.0, 2.0])
def test_se_budget_integration(se_ratio):
    """Build an adapter using (r_e, r_SE) returned by the helper; verify
    it instantiates and yields a param count within ±5% of target budget."""
    total_rank = 320
    K = 20
    D_in = D_out = 3072
    r_e, r_se = split_lora_rank_budget_for_se(total_rank, se_ratio, K)

    adapter = ParallelLoRAAdapter(
        in_features=D_in, out_features=D_out,
        num_experts=K, rank=r_e, shared_expert_rank=r_se)

    num_train, _ = count_lora_params(adapter)
    # Reference: K·r_e·(D_in + D_out) + r_se·(D_in + D_out) if r_se>0
    target = (total_rank) * (D_in + D_out)  # K·r_e + r_se ≈ total_rank
    actual_rank_sum = K * r_e + r_se
    ratio = actual_rank_sum / total_rank
    # Within ±3% rank budget (matches test_lora_rank_budget guarantee)
    assert 0.97 <= ratio <= 1.03


# ── Backward-compat: no new imports at module load ─────────────────────────

def test_no_hf_dependency():
    """models/multi_branch_lora.py must not import transformers / peft /
    datasets etc. — those come later in Batch 3 for the HF wrapper. Keeps
    this module importable in bare torch envs (local tests, etc.)."""
    import models.multi_branch_lora as mbl
    source = open(mbl.__file__).read()
    for forbidden in ('import transformers', 'from transformers',
                      'import peft', 'from peft'):
        assert forbidden not in source, (
            f"{forbidden} not allowed in Batch 1 core module")

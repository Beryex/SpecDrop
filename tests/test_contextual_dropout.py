"""Unit tests for Contextual Dropout ResNet-110 (Fan et al., ICLR 2021).

Verifies:
  1. Param count ≈ ResNet-110 (only small overhead from context networks, <5%)
  2. Forward shape (B, num_classes)
  3. Eval mode: deterministic (z = E[z] = 1, identity transform)
  4. Train mode: stochastic (different forwards give different outputs)
  5. Context networks are per-block (54 modules: 18 × 3 layer groups)
  6. Context network structure: Linear(C, C/γ) → LeakyReLU → Linear(C/γ, C)
  7. Backward gradients flow (including into context networks)
  8. Non-finite guard: no NaN/Inf in outputs

Run: python tests/test_contextual_dropout.py
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.contextual_dropout import (ContextualDropoutResNet,
                                       contextual_dropout_resnet110,
                                       ContextualDropout2d,
                                       ContextualBasicBlock)
from models.resnet import ResNet


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def test_param_count_small_overhead():
    """Context networks add ~25K params; should be within 5% of dense ResNet-110."""
    ctxd = contextual_dropout_resnet110(num_classes=100)
    ref = ResNet(num_blocks=18, num_classes=100, base_channels=16)
    p_ctx = _count_params(ctxd)
    p_ref = _count_params(ref)
    overhead = (p_ctx - p_ref) / p_ref
    assert 0 < overhead < 0.05, \
        f"overhead {overhead:.2%} outside expected 0-5%"
    print(f"  PASS param_count_small_overhead: "
          f"{p_ctx:,} vs ResNet-110 {p_ref:,} (+{overhead:.2%})")


def test_forward_shape():
    model = contextual_dropout_resnet110(num_classes=100).eval()
    x = torch.randn(2, 3, 32, 32)
    logits, kl = model(x)
    assert logits.shape == (2, 100)
    assert kl.item() == 0.0, "eval KL must be 0"
    print(f"  PASS forward_shape: {logits.shape}, eval kl={kl.item()}")


def test_eval_deterministic():
    """At eval, z = 1 (expected value) → output is deterministic."""
    model = contextual_dropout_resnet110(num_classes=100).eval()
    x = torch.randn(2, 3, 32, 32)
    l1, _ = model(x)
    l2, _ = model(x)
    assert torch.allclose(l1, l2, atol=1e-6), \
        "Eval mode must be deterministic (z=E[z]=1)"
    print(f"  PASS eval_deterministic")


def test_train_stochastic():
    """In train mode, reparameterized Gaussian sampling → different outputs per forward."""
    torch.manual_seed(0)
    model = contextual_dropout_resnet110(num_classes=100).train()
    x = torch.randn(2, 3, 32, 32)
    l1, kl1 = model(x)
    l2, kl2 = model(x)
    assert not torch.allclose(l1, l2), \
        "Train mode must be stochastic (z ~ N(1, var))"
    assert kl1.item() > 0 and kl2.item() > 0, \
        "Train mode should produce non-zero KL (variational posterior)"
    print(f"  PASS train_stochastic (kl ~ {kl1.item():.4f})")


def test_context_net_structure():
    """Each block has a ContextualDropout2d with Linear→LeakyReLU→Linear."""
    model = contextual_dropout_resnet110(num_classes=100)
    n_ctx = 0
    for m in model.modules():
        if isinstance(m, ContextualDropout2d):
            n_ctx += 1
            assert isinstance(m.fc1, nn.Linear)
            assert isinstance(m.act, nn.LeakyReLU)
            assert isinstance(m.fc2, nn.Linear)
            # Input dim == output dim == channels; bottleneck == channels / reduction
            assert m.fc1.in_features == m.fc2.out_features == m.channels
            assert m.fc1.out_features == m.fc2.in_features
            assert m.fc1.out_features <= m.channels  # bottleneck
    # 3 layer groups × 18 blocks = 54 context dropout modules
    assert n_ctx == 54, f"expected 54 ContextualDropout2d, got {n_ctx}"
    print(f"  PASS context_net_structure: {n_ctx} context-dropout modules "
          f"(3 groups × 18 blocks)")


def test_identity_at_eval_on_feature_map():
    """ContextualDropout2d must be exact identity in eval mode."""
    ctxd = ContextualDropout2d(channels=16).eval()
    x = torch.randn(2, 16, 8, 8)
    out = ctxd(x)
    assert torch.equal(out, x), "eval must be identity (z=1)"
    print(f"  PASS identity_at_eval_on_feature_map")


def test_backward_grad_flow():
    """All params (backbone + context nets) must receive finite gradients."""
    model = contextual_dropout_resnet110(num_classes=100).train()
    x = torch.randn(2, 3, 32, 32)
    logits, kl = model(x)
    (logits.sum() + kl).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    # Confirm context network params are included
    ctx_params = [n for n, _ in model.named_parameters() if 'ctx_drop' in n]
    assert len(ctx_params) > 0, "no context-dropout params found"
    print(f"  PASS backward_grad_flow: {len(ctx_params)} ctx-dropout params updated")


def test_no_nan_in_output():
    """Extreme context logits (large α) should still yield finite outputs.

    Gaussian variance = σ(α)/(1-σ(α)) can blow up for α → ∞. We clip σ to avoid NaN.
    """
    torch.manual_seed(0)
    model = contextual_dropout_resnet110(num_classes=100).train()
    x = torch.randn(4, 3, 32, 32) * 10.0   # amplified inputs → large activations
    logits, kl = model(x)
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"
    assert torch.isfinite(kl).all(), "KL contains NaN/Inf"
    print(f"  PASS no_nan_in_output")


def test_context_dropout_applies_per_channel():
    """ContextualDropout2d's mask is per-channel (not per-spatial-location)."""
    torch.manual_seed(42)
    ctxd = ContextualDropout2d(channels=8).train()
    x = torch.ones(1, 8, 4, 4)
    out = ctxd(x)
    # After multiplication, each of the 8 channels has a single scalar z;
    # so within a channel, all 16 spatial locations should be identical.
    for c in range(8):
        ch_values = out[0, c].flatten()
        assert torch.allclose(ch_values, ch_values[0].expand_as(ch_values), atol=1e-6), \
            f"channel {c} is not spatially constant"
    print(f"  PASS context_dropout_applies_per_channel")


if __name__ == "__main__":
    print("=" * 60)
    print(" Contextual Dropout ResNet-110 unit tests")
    print("=" * 60)
    test_param_count_small_overhead()
    test_forward_shape()
    test_eval_deterministic()
    test_train_stochastic()
    test_context_net_structure()
    test_identity_at_eval_on_feature_map()
    test_backward_grad_flow()
    test_no_nan_in_output()
    test_context_dropout_applies_per_channel()
    print("=" * 60)
    print(" All tests passed")
    print("=" * 60)

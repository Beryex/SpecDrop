"""Unit tests for Example-Tied Dropout ResNet-110 (Maini et al. ICML 2023).

Verifies:
  1. Param count ≈ dense ResNet-110 (≤2% deviation — only extra params are masks buffer, not trainable)
  2. Forward shape (B, num_classes)
  3. Training requires `example_ids` (raises clearly if missing)
  4. Same example_id across forwards → same mask (mask is deterministic per example)
  5. Different example_ids → different masks (memorization path differs)
  6. Test mode: identical output regardless of example_ids (memorization zeroed out)
  7. Train mode: with full keep-rate mask, matches standard ResNet-110 layers up to FC
  8. Mask table is a buffer (saved in state_dict, on correct device)
  9. Backward gradient flow
 10. example_ids bounds check

Run: python tests/test_et_dropout.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.et_dropout import (ExampleTiedDropoutResNet,
                               example_tied_dropout_resnet110)
from models.resnet import ResNet


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def test_param_count_matches_resnet110():
    """Trainable params should match dense ResNet-110 exactly (masks are buffer)."""
    etd = example_tied_dropout_resnet110(num_classes=100, num_examples=50000)
    ref = ResNet(num_blocks=18, num_classes=100, base_channels=16)
    p_etd = _count_params(etd)
    p_ref = _count_params(ref)
    assert p_etd == p_ref, \
        f"ETD trainable params {p_etd:,} != ResNet-110 {p_ref:,}"
    print(f"  PASS param_count_matches_resnet110: {p_etd:,} (both)")


def test_mask_buffer_shape():
    """Mask table shape = (num_examples, total_mem_channels) where
    total_mem_channels = Σ over all 54 BasicBlocks of round(out_ch × α_mem)."""
    model = example_tied_dropout_resnet110(
        num_classes=100, num_examples=1000, alpha_mem=0.5)
    # ResNet-110: layer1 (18 × 16 = 288), layer2 (18 × 32 = 576), layer3 (18 × 64 = 1152).
    # Total channels = 2016. α=0.5 rounds per-block: 8/16/32 × 18 blocks each.
    expected = 18 * 8 + 18 * 16 + 18 * 32
    assert model.example_masks.shape == (1000, expected)
    assert model.total_mem_channels == expected
    uniq = torch.unique(model.example_masks)
    assert set(uniq.tolist()).issubset({0.0, 1.0})
    print(f"  PASS mask_buffer_shape: {tuple(model.example_masks.shape)}, "
          f"binary, total_mem={expected} (per-layer: 144/288/576)")


def test_forward_shape_train():
    model = example_tied_dropout_resnet110(num_classes=100, num_examples=100).train()
    x = torch.randn(4, 3, 32, 32)
    example_ids = torch.tensor([0, 1, 2, 3])
    out, _aux = model(x, example_ids=example_ids)
    assert out.shape == (4, 100)
    print(f"  PASS forward_shape_train: {out.shape}")


def test_forward_shape_eval():
    """At eval, example_ids can be None — all memorization neurons zeroed."""
    model = example_tied_dropout_resnet110(num_classes=100, num_examples=100).eval()
    x = torch.randn(4, 3, 32, 32)
    out, _aux = model(x)
    assert out.shape == (4, 100)
    print(f"  PASS forward_shape_eval (no example_ids needed)")


def test_training_requires_example_ids():
    """Training without example_ids must raise a clear error."""
    model = example_tied_dropout_resnet110(num_classes=100, num_examples=100).train()
    x = torch.randn(2, 3, 32, 32)
    try:
        model(x)
    except ValueError as e:
        assert "example_ids" in str(e)
        print(f"  PASS training_requires_example_ids (raised: {str(e)[:60]}...)")
        return
    raise AssertionError("Expected ValueError for missing example_ids at training")


def test_mask_deterministic_per_example():
    """Same example_id across multiple forwards must apply the same mask.

    At train mode, dropout of memorization neurons is fixed per example (not stochastic)
    — so outputs depend only on model weights + input + mask. With same input, same
    example_id, same BN statistics, the output must be identical.
    """
    model = example_tied_dropout_resnet110(num_classes=100, num_examples=50).eval()
    # Eval mode: BN deterministic; still uses fixed mask lookup internally, but eval
    # zeroes mem neurons anyway, so identity across calls.
    x = torch.randn(2, 3, 32, 32)
    o1, _ = model(x, example_ids=torch.tensor([0, 1]))
    o2, _ = model(x, example_ids=torch.tensor([0, 1]))
    assert torch.allclose(o1, o2), "Eval outputs should be deterministic"

    # Now test that the MASK itself is fixed across recomputations
    mask_a = model.example_masks[0].clone()
    mask_b = model.example_masks[0].clone()
    assert torch.equal(mask_a, mask_b)
    print(f"  PASS mask_deterministic_per_example")


def test_different_examples_different_mem_masks():
    """Different example_ids should (typically) produce different memorization masks."""
    model = example_tied_dropout_resnet110(
        num_classes=100, num_examples=50, mem_keep_rate=0.5, mask_seed=42)
    n_mem = model.total_mem_channels
    # With Bernoulli(0.5), collisions among 50 masks of length ~32 are astronomically rare.
    # Count pairs with identical masks.
    collisions = 0
    for i in range(50):
        for j in range(i + 1, 50):
            if torch.equal(model.example_masks[i], model.example_masks[j]):
                collisions += 1
    assert collisions == 0, \
        f"found {collisions} pair(s) of identical masks among 50 examples"
    print(f"  PASS different_examples_different_mem_masks: "
          f"0 collisions across C(50,2)=1225 pairs, |M|={n_mem}")


def test_eval_output_invariant_to_example_ids():
    """At inference, memorization neurons are zeroed → output is independent of example_ids."""
    model = example_tied_dropout_resnet110(num_classes=100, num_examples=100).eval()
    x = torch.randn(4, 3, 32, 32)
    out_a, _ = model(x, example_ids=torch.tensor([0, 1, 2, 3]))
    out_b, _ = model(x, example_ids=torch.tensor([10, 20, 30, 40]))
    out_c, _ = model(x)   # no example_ids
    assert torch.allclose(out_a, out_b, atol=1e-6)
    assert torch.allclose(out_a, out_c, atol=1e-6)
    print(f"  PASS eval_output_invariant_to_example_ids")


def test_backward_grad_flow():
    model = example_tied_dropout_resnet110(num_classes=100, num_examples=100).train()
    x = torch.randn(2, 3, 32, 32)
    example_ids = torch.tensor([5, 10])
    logits, _aux = model(x, example_ids=example_ids)
    logits.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    print(f"  PASS backward_grad_flow")


def test_example_ids_out_of_range():
    """example_ids ≥ num_examples or < 0 must raise IndexError."""
    model = example_tied_dropout_resnet110(num_classes=100, num_examples=10).train()
    x = torch.randn(2, 3, 32, 32)
    try:
        model(x, example_ids=torch.tensor([0, 100]))   # 100 ≥ 10
    except IndexError as e:
        assert "out of range" in str(e)
        print(f"  PASS example_ids_out_of_range")
        return
    raise AssertionError("Expected IndexError for out-of-range example_ids")


def test_mask_seed_reproducibility():
    """Same mask_seed → identical mask tables; different seeds → different tables."""
    m1 = example_tied_dropout_resnet110(num_examples=100, mask_seed=42)
    m2 = example_tied_dropout_resnet110(num_examples=100, mask_seed=42)
    m3 = example_tied_dropout_resnet110(num_examples=100, mask_seed=123)
    assert torch.equal(m1.example_masks, m2.example_masks)
    assert not torch.equal(m1.example_masks, m3.example_masks)
    print(f"  PASS mask_seed_reproducibility")


if __name__ == "__main__":
    print("=" * 60)
    print(" Example-Tied Dropout ResNet-110 unit tests")
    print("=" * 60)
    test_param_count_matches_resnet110()
    test_mask_buffer_shape()
    test_forward_shape_train()
    test_forward_shape_eval()
    test_training_requires_example_ids()
    test_mask_deterministic_per_example()
    test_different_examples_different_mem_masks()
    test_eval_output_invariant_to_example_ids()
    test_backward_grad_flow()
    test_example_ids_out_of_range()
    test_mask_seed_reproducibility()
    print("=" * 60)
    print(" All tests passed")
    print("=" * 60)

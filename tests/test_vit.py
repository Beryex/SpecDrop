"""Unit tests for ViT-Small baseline.

Verifies that the implementation matches community standards:
  1. ViT-S/16 ImageNet param count = 22,050,664 (timm vit_small_patch16_224 reference)
  2. Forward pass yields correct (B, num_classes) shape on 224×224 input
  3. forward_features yields (B, embed_dim) after CLS pool + LayerNorm
  4. CIFAR-100 head sizing (num_classes=100) drops exactly (384×900 + 900) params
  5. Backward pass: every parameter receives a finite gradient
  6. Eval mode is deterministic
  7. DropPath/attention/proj dropout randomize in train mode, deterministic in eval
  8. branch_mask kwarg is accepted and ignored (API compatibility with multi-branch)
  9. Configurable img_size/patch_size works (e.g., ViT-Tiny 32×32 patch=4)
 10. If `timm` is available, match vit_small_patch16_224 param count exactly.

Run: python tests/test_vit.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.vit import ViT, vit_small


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def test_vit_small_param_count():
    """ViT-S/16 ImageNet should have 22,050,664 params (timm reference)."""
    model = vit_small(num_classes=1000)
    params = _count_params(model)
    expected = 22_050_664
    assert params == expected, \
        f"ViT-S/16 params = {params:,}, expected {expected:,}"
    print(f"  PASS param_count: {params:,} (={params/1e6:.3f}M)")


def test_vit_small_forward_shape():
    model = vit_small(num_classes=1000).eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1000), f"Expected (2, 1000), got {out.shape}"
    print(f"  PASS forward_shape: {out.shape}")


def test_vit_forward_features_shape():
    """forward_features must return (B, embed_dim) after CLS pool + norm."""
    model = vit_small(num_classes=100).eval()
    x = torch.randn(2, 3, 224, 224)
    feats = model.forward_features(x)
    assert feats.shape == (2, 384), f"Expected (2, 384), got {feats.shape}"
    print(f"  PASS forward_features: {feats.shape}")


def test_vit_small_cifar100_param_count():
    """num_classes=100 should reduce params by exactly (384*900 + 900) = 346,500."""
    m1000 = _count_params(vit_small(num_classes=1000))
    m100 = _count_params(vit_small(num_classes=100))
    diff = m1000 - m100
    expected_diff = 384 * 900 + 900
    assert diff == expected_diff, f"head diff {diff:,} != expected {expected_diff:,}"
    print(f"  PASS cifar100_param_count: 100-class={m100:,} ({m100/1e6:.3f}M)")


def test_backward_grad_flow():
    """Every parameter must receive a finite gradient."""
    model = vit_small(num_classes=100).train()
    x = torch.randn(2, 3, 224, 224)
    loss = model(x).sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"
    print(f"  PASS backward_grad_flow: {sum(1 for _ in model.parameters())} tensors all have grads")


def test_determinism_eval():
    """Eval mode with drop_path=0 + same seed must produce identical outputs."""
    torch.manual_seed(42)
    m1 = vit_small(num_classes=100).eval()
    torch.manual_seed(42)
    m2 = vit_small(num_classes=100).eval()
    x = torch.randn(2, 3, 224, 224)
    out1, out2 = m1(x), m2(x)
    assert torch.allclose(out1, out2), "Same seed should yield same outputs"
    print(f"  PASS determinism_eval")


def test_drop_path_behavior():
    """drop_path_rate>0 randomizes training forwards; eval is deterministic."""
    torch.manual_seed(0)
    model = vit_small(num_classes=100, drop_path_rate=0.5)
    x = torch.randn(2, 3, 224, 224)

    model.train()
    o1, o2 = model(x), model(x)
    assert not torch.allclose(o1, o2), \
        "DropPath in train mode should randomize across forwards"

    model.eval()
    o3, o4 = model(x), model(x)
    assert torch.allclose(o3, o4), \
        "DropPath in eval mode should be deterministic"
    print(f"  PASS drop_path_behavior")


def test_branch_mask_compat():
    """ViT must accept branch_mask kwarg (ignored) for uniform multi-branch API."""
    model = vit_small(num_classes=100).eval()
    x = torch.randn(2, 3, 224, 224)
    out_nomask = model(x)
    out_withmask = model(x, branch_mask=torch.ones(2, 8))
    assert torch.allclose(out_nomask, out_withmask), \
        "branch_mask must not affect dense ViT output"
    print(f"  PASS branch_mask_compat")


def test_configurable_shape():
    """Alternate img_size/patch_size/depth should work (e.g., ViT-Tiny 32×32 patch=4)."""
    model = ViT(img_size=32, patch_size=4, embed_dim=192, depth=4,
                num_heads=3, num_classes=100).eval()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100), f"Expected (2, 100), got {out.shape}"
    assert model.patch_embed.num_patches == 64
    print(f"  PASS configurable_shape: ViT-Tiny 32×32 patch=4 → 64 patches")


def test_arch_spec_vit_s16():
    """Spot-check the architectural hyperparameters match ViT-S/16."""
    m = vit_small(num_classes=1000)
    assert m.embed_dim == 384
    assert len(m.blocks) == 12
    assert m.patch_embed.patch_size == 16
    assert m.patch_embed.num_patches == 196  # (224/16)^2
    assert m.patch_embed.proj.weight.shape == (384, 3, 16, 16)
    assert m.blocks[0].attn.num_heads == 6
    assert m.blocks[0].attn.head_dim == 64
    assert m.blocks[0].attn.qkv.weight.shape == (3 * 384, 384)
    assert m.blocks[0].attn.qkv.bias is not None
    assert m.blocks[0].mlp.fc1.weight.shape == (1536, 384)
    assert m.blocks[0].mlp.fc2.weight.shape == (384, 1536)
    assert m.pos_embed.shape == (1, 197, 384)
    assert m.cls_token.shape == (1, 1, 384)
    print(f"  PASS arch_spec_vit_s16: 384-dim, 12 blocks, 6 heads, patch=16, 196 patches")


def test_match_timm_if_available():
    """Optional: if timm is installed, match vit_small_patch16_224 param count."""
    try:
        import timm
    except ImportError:
        print(f"  SKIP match_timm_if_available (timm not installed)")
        return
    our = _count_params(vit_small(num_classes=1000))
    ref_model = timm.create_model('vit_small_patch16_224',
                                  pretrained=False, num_classes=1000)
    ref = sum(p.numel() for p in ref_model.parameters())
    assert our == ref, f"Ours={our:,}, timm={ref:,} (diff {ref - our:+,})"
    print(f"  PASS match_timm: our {our:,} = timm vit_small_patch16_224 {ref:,}")


if __name__ == "__main__":
    print("=" * 60)
    print(" ViT-Small unit tests")
    print("=" * 60)
    test_vit_small_param_count()
    test_vit_small_forward_shape()
    test_vit_forward_features_shape()
    test_vit_small_cifar100_param_count()
    test_backward_grad_flow()
    test_determinism_eval()
    test_drop_path_behavior()
    test_branch_mask_compat()
    test_configurable_shape()
    test_arch_spec_vit_s16()
    test_match_timm_if_available()
    print("=" * 60)
    print(" All tests passed")
    print("=" * 60)

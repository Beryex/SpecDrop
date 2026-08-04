"""Tests for SoftMoEViT.moe_start_block (Soft MoE placement study).

Covers:
  1. Default moe_start_block=0 is bit-identical to the pre-change module
     (same seed -> same state dict -> same forward output) and state-dict
     compatible (strict load in both directions).
  2. moe_start_block=6 builds 6 dense ViT blocks (full-width 4x MLP) followed
     by 6 SoftMoEBlocks, with correct forward shape / zero aux / backward.
  3. Constructor validation.
  4. Full-scale (ViT-S/16) param counts: default and start=6 both stay within
     the 2% ImageNet param budget (R5 screen config relies on this).

Run:  python tests/test_softmoe_placement.py   (or pytest)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.vit import Block
from models.soft_moe_vit import SoftMoEViT, SoftMoEBlock
from models.vit import PatchEmbed, _trunc_normal_


# Small config for fast CPU tests (depth kept at 12 so moe_start_block=6
# exercises the real half-and-half split).
SMALL = dict(img_size=32, patch_size=16, num_classes=10, embed_dim=64,
             depth=12, num_heads=2, num_experts=4, expert_hidden=64,
             slots_per_expert=1, drop_path_rate=0.1)

VIT_S_REF = 22_050_664  # utils/sanity_check.py REFERENCE_PARAMS['imagenet']


class _PreChangeSoftMoEViT(nn.Module):
    """Verbatim replica of SoftMoEViT as it was BEFORE moe_start_block was
    introduced (every block a SoftMoEBlock, same construction order, same
    init). Used as the bit-identity reference for the default path."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=384, depth=12, num_heads=6,
                 num_experts=32, expert_hidden=None, slots_per_expert=1,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        if expert_hidden is None:
            expert_hidden = max(8, round(4 * embed_dim / num_experts))
        self.expert_hidden = expert_hidden

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            SoftMoEBlock(embed_dim, num_heads, expert_hidden, num_experts,
                         slots_per_expert=slots_per_expert,
                         qkv_bias=qkv_bias, drop=drop_rate,
                         attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        _trunc_normal_(self.pos_embed, std=0.02)
        _trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                _trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits, torch.tensor(0.0, device=x.device)


def _params(m):
    return sum(p.numel() for p in m.parameters())


def test_default_bit_identical_to_prechange():
    torch.manual_seed(1234)
    new = SoftMoEViT(**SMALL)                      # moe_start_block defaults to 0
    torch.manual_seed(1234)
    ref = _PreChangeSoftMoEViT(**SMALL)

    sd_new, sd_ref = new.state_dict(), ref.state_dict()
    assert list(sd_new.keys()) == list(sd_ref.keys()), \
        "default moe_start_block=0 must be state-dict compatible"
    for k in sd_ref:
        assert torch.equal(sd_new[k], sd_ref[k]), \
            f"param {k} not bit-identical after same-seed construction"

    new.eval()
    ref.eval()
    torch.manual_seed(99)
    x = torch.randn(2, 3, SMALL['img_size'], SMALL['img_size'])
    with torch.no_grad():
        logits_new, aux_new = new(x)
        logits_ref, aux_ref = ref(x)
    assert torch.equal(logits_new, logits_ref), \
        "default forward must be BIT-identical to the pre-change module"
    assert aux_new.item() == aux_ref.item() == 0.0
    print(f"  PASS default_bit_identical: {_params(new):,} params "
          f"(== pre-change {_params(ref):,}), forward max|diff| = 0.0")


def test_default_state_dict_cross_load():
    torch.manual_seed(7)
    new = SoftMoEViT(**SMALL)
    torch.manual_seed(8)
    ref = _PreChangeSoftMoEViT(**SMALL)
    # Old checkpoints load into the new class and vice versa (strict).
    new.load_state_dict(ref.state_dict(), strict=True)
    ref.load_state_dict(SoftMoEViT(**SMALL).state_dict(), strict=True)
    print("  PASS state_dict_cross_load: strict load OK in both directions")


def test_start6_structure_and_forward():
    torch.manual_seed(0)
    m = SoftMoEViT(**SMALL, moe_start_block=6)
    assert m.moe_start_block == 6
    n_dense = sum(isinstance(b, Block) for b in m.blocks)
    n_moe = sum(isinstance(b, SoftMoEBlock) for b in m.blocks)
    assert n_dense == 6 and n_moe == 6, f"expected 6+6, got {n_dense}+{n_moe}"
    for i, b in enumerate(m.blocks):
        expected = Block if i < 6 else SoftMoEBlock
        assert isinstance(b, expected), f"block {i} is {type(b).__name__}"
    # Dense blocks are full-width 4x MLP.
    d = SMALL['embed_dim']
    assert m.blocks[0].mlp.fc1.out_features == 4 * d, \
        "dense blocks must use the full-width 4x MLP"
    assert m.blocks[6].moe.experts_fc1.shape == \
        (SMALL['num_experts'], SMALL['expert_hidden'], d)

    m.train()
    x = torch.randn(2, 3, SMALL['img_size'], SMALL['img_size'])
    logits, aux = m(x)
    assert logits.shape == (2, SMALL['num_classes'])
    assert torch.isfinite(logits).all()
    assert aux.item() == 0.0
    logits.sum().backward()
    assert m.blocks[0].mlp.fc1.weight.grad is not None
    assert m.blocks[6].moe.phi.grad is not None

    m0 = SoftMoEViT(**SMALL)
    print(f"  PASS start6_structure: 6 dense + 6 moe, logits {tuple(logits.shape)}, "
          f"params start6={_params(m):,} vs start0={_params(m0):,}")


def test_moe_start_block_validation():
    for bad in (-1, 13):
        try:
            SoftMoEViT(**SMALL, moe_start_block=bad)
        except AssertionError:
            continue
        raise AssertionError(f"moe_start_block={bad} should have raised")
    # Boundary values are legal: 12 = all-dense, 0 = all-moe.
    SoftMoEViT(**SMALL, moe_start_block=12)
    print("  PASS moe_start_block_validation: rejects -1/13, accepts 0..12")


def test_full_scale_param_budget():
    """R0 (start=0) and R5 (start=6, N=32, h=48) must stay within the 2%
    ImageNet param budget enforced by utils/sanity_check.py."""
    torch.manual_seed(0)
    m0 = SoftMoEViT(num_classes=1000, num_experts=32, expert_hidden=48,
                    slots_per_expert=1)
    torch.manual_seed(0)
    m6 = SoftMoEViT(num_classes=1000, num_experts=32, expert_hidden=48,
                    slots_per_expert=1, moe_start_block=6)
    p0, p6 = _params(m0), _params(m6)
    for name, p in (('start0', p0), ('start6', p6)):
        dev = abs(p / VIT_S_REF - 1.0)
        assert dev <= 0.02, f"{name}: {p:,} deviates {dev:.2%} from ViT-S ref"
    print(f"  PASS full_scale_param_budget: start0={p0:,} "
          f"({p0 / VIT_S_REF:.4f}x), start6={p6:,} ({p6 / VIT_S_REF:.4f}x) "
          f"vs ViT-S {VIT_S_REF:,} (both within 2%)")


if __name__ == "__main__":
    print("=" * 60)
    print(" SoftMoEViT moe_start_block placement — unit tests")
    print("=" * 60)
    test_default_bit_identical_to_prechange()
    test_default_state_dict_cross_load()
    test_start6_structure_and_forward()
    test_moe_start_block_validation()
    test_full_scale_param_budget()
    print("=" * 60)
    print(" All placement tests passed")
    print("=" * 60)

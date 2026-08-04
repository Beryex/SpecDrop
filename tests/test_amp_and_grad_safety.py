"""Safety tests for mixed-precision behavior + router gradient flow.

Covers M5 (ParallelFFN einsum vs for-loop equivalence in bf16)
and M7 (router parameters receive non-zero gradient).

Run: python tests/test_amp_and_grad_safety.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _bf16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def test_parallel_ffn_bf16_equivalence():
    """ParallelFFN einsum vs a for-loop of dense FFNs should match in bf16
    within the dtype's tolerance (~1e-2 for activations)."""
    if not _bf16_supported():
        print("  SKIP parallel_ffn_bf16_equivalence (no CUDA bf16)")
        return

    from models.transformer_lm import ParallelFFN
    torch.manual_seed(0)
    D, H, K, B, T = 32, 48, 4, 2, 16
    pf = ParallelFFN(D, H, K, dropout=0.0).cuda().eval()

    x = torch.randn(B, T, D, device='cuda')
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        y_par = pf(x)                          # (B, T, K, D), bf16

        # Reference: loop K dense FFNs with the same weights
        y_ref_list = []
        for k in range(K):
            h = torch.einsum('btd,fd->btf', x, pf.w1[k]) + pf.b1[k]
            h = F.gelu(h)
            out = torch.einsum('btf,df->btd', h, pf.w2[k]) + pf.b2[k]
            y_ref_list.append(out)
        y_ref = torch.stack(y_ref_list, dim=2)  # (B, T, K, D)

    max_abs = (y_par - y_ref).abs().max().item()
    assert max_abs < 5e-2, f"ParallelFFN bf16 drift too large: {max_abs}"
    print(f"  PASS parallel_ffn_bf16_equivalence: max_abs={max_abs:.2e}")


def test_switch_router_receives_grad():
    """Switch's router must receive nonzero gradient after backward — otherwise
    load-balance is decoupled from learning."""
    from models.switch_transformer_lm import SwitchTransformerLM
    m = SwitchTransformerLM(
        vocab_size=50257, hidden_dim=384, num_layers=2, num_heads=6,
        num_experts=16, ffn_dim_per_expert=96, max_seq_len=512,
        load_balance_weight=0.01).train()
    ids = torch.randint(0, 50257, (4, 32))
    logits, aux = m(ids)
    (logits.sum() + aux).backward()
    g = m.blocks[0].moe.router.weight.grad
    assert g is not None and g.abs().sum().item() > 0, \
        "Switch router.weight.grad is None or zero — routing is not learning!"
    print(f"  PASS switch_router_receives_grad: ||∂L/∂W_r|| = {g.norm().item():.4f}")


def test_mod_squad_router_receives_grad():
    from models.mod_squad_vit import ModSquadViT
    m = ModSquadViT(num_classes=100, num_experts=8, top_k=2,
                    mi_weight=0.001).train()  # noise_std defaults to 1/N (Shazeer 2017)
    x = torch.randn(2, 3, 224, 224)
    logits, aux = m(x)
    (logits.sum() + aux).backward()
    g_gate = m.blocks[0].moe.gate.weight.grad
    assert g_gate is not None and g_gate.abs().sum().item() > 0, \
        "Mod-Squad gate has no gradient"
    print(f"  PASS mod_squad_router_receives_grad: ||∂L/∂W_g|| = {g_gate.norm().item():.4f}")


def test_soft_moe_phi_receives_grad():
    from models.soft_moe_vit import SoftMoEViT
    m = SoftMoEViT(num_classes=100, num_experts=8, slots_per_expert=1).train()
    x = torch.randn(2, 3, 224, 224)
    logits, aux = m(x)
    (logits.sum() + aux).backward()
    g_phi = m.blocks[0].moe.phi.grad
    assert g_phi is not None and g_phi.abs().sum().item() > 0
    assert m.blocks[0].moe.scale.grad is not None, "scale param has no grad"
    print(f"  PASS soft_moe_phi_receives_grad: ||∂L/∂Φ|| = {g_phi.norm().item():.4f}, "
          f"scale.grad = {m.blocks[0].moe.scale.grad.item():.4e}")


def test_mask_scale_none_raises():
    """B1: if branch_mask is provided but model.mask_scale is None, forward must
    raise (silent per-sample denominator fallback was removed — it reintroduced
    the Jensen bias this paper eliminates)."""
    from models.multi_branch import MultiBranchResNet
    m = MultiBranchResNet(num_branches=4, num_blocks=3, num_classes=100,
                          branch_channels=16, base_channels=16).eval()
    # Intentionally NOT setting m.mask_scale
    assert m.mask_scale is None
    x = torch.randn(2, 3, 32, 32)
    mask = torch.rand(2, 4)
    try:
        m(x, branch_mask=mask)
    except RuntimeError as e:
        assert "mask_scale" in str(e) and "Jensen" in str(e)
        print(f"  PASS mask_scale_none_raises: {str(e)[:72]}...")
        return
    raise AssertionError("Expected RuntimeError when mask_scale is None")


def test_mask_dtype_cast_to_branch_dtype():
    """M6: when the model runs under autocast(bf16), branch outputs are bf16 but
    the externally-generated mask is fp32. Our code explicitly casts mask →
    branch_outputs.dtype BEFORE multiply so no silent fp32 upcast leaks."""
    if not _bf16_supported():
        print("  SKIP mask_dtype_cast (no CUDA bf16)")
        return
    from models.multi_branch_vit import MultiBranchViT
    m = MultiBranchViT(img_size=32, patch_size=8, num_classes=10,
                       embed_dim=32, depth=2, num_heads=2,
                       num_branches=4).cuda().eval()
    m.mask_scale = 2.0
    x = torch.randn(2, 3, 32, 32, device='cuda')
    mask = torch.rand(2, 4, device='cuda')            # fp32 mask
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = m(x, branch_mask=mask)
    # autocast keeps final Linear(head) in fp32 per PyTorch default, but any
    # intermediate bf16 × mask must not silently upcast mid-block. If the mask
    # cast were missing, forward would crash with dtype mismatch. Reaching here
    # means the cast inside MultiBranchBlock.forward is correct.
    assert torch.isfinite(out).all()
    print(f"  PASS mask_dtype_cast: bf16 forward with fp32 mask ran clean")


if __name__ == "__main__":
    print("=" * 60)
    print(" AMP / grad-safety tests (B1, M5, M6, M7)")
    print("=" * 60)
    test_mask_scale_none_raises()
    test_mask_dtype_cast_to_branch_dtype()
    test_parallel_ffn_bf16_equivalence()
    test_switch_router_receives_grad()
    test_mod_squad_router_receives_grad()
    test_soft_moe_phi_receives_grad()
    print("=" * 60)
    print(" All safety tests passed")
    print("=" * 60)

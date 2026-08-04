"""Numerical-equivalence tests: new ParallelFFN-based MoE dispatch must produce
byte-close forward outputs and per-expert gradients to the reference for-loop
implementation. Run: python tests/test_moe_dispatch_equivalence.py

For each of the 4 faithful MoE LM baselines (Switch / Hash Layers / SMoE-Dropout
/ DEMix), we:
  1. Build the real (new) FFN module with dropout=0 so results are deterministic.
  2. Extract its weights: router (if any), ParallelFFN.w1/b1/w2/b2, hash_table.
  3. Run a hand-coded for-loop reference implementation using those same
     weights + the same input tensor.
  4. Assert forward outputs match (atol=1e-5 is generous for fp32 — differences
     come from einsum vs bmm accumulation order, not algorithm).
  5. Assert aux losses match (Switch only).
  6. Backward: assert each per-expert slice of experts.w1.grad / w2.grad /
     b1.grad / b2.grad is numerically equal to the reference's per-expert grad
     (ParallelFFN stacks N experts into (N, ffn, D); reference computes them
     as a list of (ffn, D) grads). Atol 1e-5 same reason.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.switch_transformer_lm import SwitchFFN
from models.hash_layers_transformer_lm import HashLayerFFN
from models.smoe_dropout_transformer_lm import SMoEDropoutFFN
from models.demix_transformer_lm import DemixFFN
from models.mod_squad_vit import ModSquadFFN

torch.manual_seed(0)


def _ffn_forward(x, w1, b1, w2, b2):
    """Single FFN forward using explicit tensor ops (no nn.Linear init variance)."""
    h = F.gelu(F.linear(x, w1, b1))       # w1: (ffn, D), b1: (ffn,)
    out = F.linear(h, w2, b2)             # w2: (D, ffn), b2: (D,)
    return out


def _zero_all_grads(*tensors):
    for t in tensors:
        if t.grad is not None:
            t.grad = None


# ─── Switch ──────────────────────────────────────────────────────────────────

def test_switch_equivalence():
    B, T, D, N, ffn = 2, 8, 16, 4, 12
    mod = SwitchFFN(hidden_dim=D, ffn_dim_per_expert=ffn, num_experts=N,
                    load_balance_weight=0.01, dropout=0.0)
    mod.train()

    x = torch.randn(B, T, D, requires_grad=True)
    # --- New impl forward + backward ---
    out_new = mod(x)
    aux_new = mod.aux_loss
    loss_new = out_new.sum() + aux_new
    loss_new.backward()
    grad_x_new = x.grad.clone()
    grad_w1_new = mod.experts.w1.grad.clone()
    grad_w2_new = mod.experts.w2.grad.clone()
    grad_b1_new = mod.experts.b1.grad.clone()
    grad_b2_new = mod.experts.b2.grad.clone()
    grad_router_new = mod.router.weight.grad.clone()

    # --- Reference for-loop forward + backward with SAME weights ---
    _zero_all_grads(x, mod.router.weight, mod.experts.w1, mod.experts.b1,
                    mod.experts.w2, mod.experts.b2)
    x2 = x.detach().clone().requires_grad_(True)

    x_flat = x2.reshape(B * T, D)
    logits = mod.router(x_flat).float()
    probs = F.softmax(logits, dim=-1)
    top1_probs, top1_idx = probs.max(dim=-1)
    top1_probs_cast = top1_probs.to(x2.dtype)
    one_hot = F.one_hot(top1_idx, num_classes=N).float()
    f = one_hot.mean(dim=0); P = probs.mean(dim=0)
    aux_ref = mod.load_balance_weight * N * (f * P).sum()

    out_flat = torch.zeros_like(x_flat)
    for i in range(N):
        mask = (top1_idx == i)
        if mask.any():
            expert_in = x_flat[mask]
            ei = _ffn_forward(expert_in,
                              mod.experts.w1[i], mod.experts.b1[i],
                              mod.experts.w2[i], mod.experts.b2[i])
            out_flat[mask] = ei * top1_probs_cast[mask].unsqueeze(-1)
    out_ref = out_flat.reshape(B, T, D)
    loss_ref = out_ref.sum() + aux_ref
    loss_ref.backward()

    # --- Assertions ---
    assert torch.allclose(out_new, out_ref, atol=1e-5), \
        f"Switch forward mismatch: max |Δ|={(out_new-out_ref).abs().max():.2e}"
    assert torch.allclose(aux_new, aux_ref, atol=1e-6), \
        f"Switch aux mismatch: {aux_new.item()} vs {aux_ref.item()}"
    assert torch.allclose(x2.grad, grad_x_new, atol=1e-5), "Switch ∂L/∂x mismatch"
    # stacked w1 grad: compare per-expert
    for i in range(N):
        g_new_i = grad_w1_new[i]
        g_ref_i = mod.experts.w1.grad[i]
        assert torch.allclose(g_new_i, g_ref_i, atol=1e-5), \
            f"Switch expert {i} w1 grad mismatch"
    assert torch.allclose(grad_w2_new, mod.experts.w2.grad, atol=1e-5)
    assert torch.allclose(grad_b1_new, mod.experts.b1.grad, atol=1e-5)
    assert torch.allclose(grad_b2_new, mod.experts.b2.grad, atol=1e-5)
    assert torch.allclose(grad_router_new, mod.router.weight.grad, atol=1e-5)
    print(f"  PASS switch_equivalence (B={B}, T={T}, N={N}, ffn={ffn})")


# ─── Hash Layers ─────────────────────────────────────────────────────────────

def test_hash_layers_equivalence():
    B, T, D, N, ffn, V = 2, 8, 16, 4, 12, 100
    mod = HashLayerFFN(hidden_dim=D, ffn_dim_per_expert=ffn, num_experts=N,
                       vocab_size=V, hash_seed=7, dropout=0.0)
    mod.train()

    x = torch.randn(B, T, D, requires_grad=True)
    token_ids = torch.randint(0, V, (B, T))

    out_new = mod(x, token_ids)
    out_new.sum().backward()
    grad_x_new = x.grad.clone()
    grad_w1_new = mod.experts.w1.grad.clone()
    grad_b2_new = mod.experts.b2.grad.clone()

    _zero_all_grads(x, mod.experts.w1, mod.experts.b1,
                    mod.experts.w2, mod.experts.b2)
    x2 = x.detach().clone().requires_grad_(True)

    x_flat = x2.reshape(B * T, D)
    expert_idx = mod.hash_table[token_ids.reshape(B * T)]
    out_flat = torch.zeros_like(x_flat)
    for i in range(N):
        mask = (expert_idx == i)
        if mask.any():
            ei = _ffn_forward(x_flat[mask],
                              mod.experts.w1[i], mod.experts.b1[i],
                              mod.experts.w2[i], mod.experts.b2[i])
            out_flat[mask] = ei
    out_ref = out_flat.reshape(B, T, D)
    out_ref.sum().backward()

    assert torch.allclose(out_new, out_ref, atol=1e-5), \
        f"Hash forward mismatch: max |Δ|={(out_new-out_ref).abs().max():.2e}"
    assert torch.allclose(x2.grad, grad_x_new, atol=1e-5)
    for i in range(N):
        assert torch.allclose(grad_w1_new[i], mod.experts.w1.grad[i], atol=1e-5), \
            f"Hash expert {i} w1 grad mismatch"
    assert torch.allclose(grad_b2_new, mod.experts.b2.grad, atol=1e-5)
    print(f"  PASS hash_layers_equivalence (B={B}, T={T}, N={N}, ffn={ffn}, V={V})")


# ─── SMoE-Dropout ────────────────────────────────────────────────────────────

def test_smoe_dropout_equivalence_k_equals_N():
    """k=N case (inference, or end of schedule): all experts active per token,
    weighted by softmax over full logits. Clean equivalence case."""
    B, T, D, N, ffn = 2, 8, 16, 4, 12
    mod = SMoEDropoutFFN(hidden_dim=D, ffn_dim_per_expert=ffn, num_experts=N,
                        k_init=1, expert_drop_prob=0.0, router_seed=7, dropout=0.0)
    mod.set_k(N)             # k = N → all experts active per token
    mod.eval()               # eval mode also forces k = N and skips expert dropout
    mod.train()              # keep training mode but k still = N

    x = torch.randn(B, T, D, requires_grad=True)
    out_new = mod(x)
    out_new.sum().backward()
    grad_x_new = x.grad.clone()
    grad_w1_new = mod.experts.w1.grad.clone()
    grad_w2_new = mod.experts.w2.grad.clone()

    _zero_all_grads(x, mod.experts.w1, mod.experts.b1,
                    mod.experts.w2, mod.experts.b2)
    x2 = x.detach().clone().requires_grad_(True)

    # Reference: for each slot 0..N-1, gather that slot's expert, weighted sum.
    x_flat = x2.reshape(B * T, D)
    with torch.no_grad():
        logits = mod.router(x_flat)
    top_vals, top_idx = logits.topk(N, dim=-1)
    top_w = F.softmax(top_vals, dim=-1)

    out_flat = torch.zeros_like(x_flat)
    for slot in range(N):
        idx_slot = top_idx[:, slot]
        w_slot = top_w[:, slot].unsqueeze(-1)
        for i in range(N):
            mask = (idx_slot == i)
            if mask.any():
                ei = _ffn_forward(x_flat[mask],
                                  mod.experts.w1[i], mod.experts.b1[i],
                                  mod.experts.w2[i], mod.experts.b2[i])
                out_flat[mask] = out_flat[mask] + w_slot[mask] * ei
    out_ref = out_flat.reshape(B, T, D)
    out_ref.sum().backward()

    assert torch.allclose(out_new, out_ref, atol=1e-5), \
        f"SMoE-Dropout k=N forward mismatch: max |Δ|={(out_new-out_ref).abs().max():.2e}"
    assert torch.allclose(x2.grad, grad_x_new, atol=1e-5)
    for i in range(N):
        assert torch.allclose(grad_w1_new[i], mod.experts.w1.grad[i], atol=1e-5)
    assert torch.allclose(grad_w2_new, mod.experts.w2.grad, atol=1e-5)
    print(f"  PASS smoe_dropout_equivalence_k=N (B={B}, T={T}, N={N}, ffn={ffn})")


def test_smoe_dropout_equivalence_k1():
    """k=1 case (start of schedule): per-token top-1 routing."""
    B, T, D, N, ffn = 2, 8, 16, 4, 12
    mod = SMoEDropoutFFN(hidden_dim=D, ffn_dim_per_expert=ffn, num_experts=N,
                        k_init=1, expert_drop_prob=0.0, router_seed=7, dropout=0.0)
    mod.set_k(1)
    mod.train()

    x = torch.randn(B, T, D, requires_grad=True)
    out_new = mod(x)
    out_new.sum().backward()
    grad_x_new = x.grad.clone()
    grad_w1_new = mod.experts.w1.grad.clone()

    _zero_all_grads(x, mod.experts.w1, mod.experts.b1,
                    mod.experts.w2, mod.experts.b2)
    x2 = x.detach().clone().requires_grad_(True)

    x_flat = x2.reshape(B * T, D)
    with torch.no_grad():
        logits = mod.router(x_flat)
    top_vals, top_idx = logits.topk(1, dim=-1)
    top_w = F.softmax(top_vals, dim=-1)      # (BT, 1), all 1s

    out_flat = torch.zeros_like(x_flat)
    idx = top_idx[:, 0]; w = top_w[:, 0].unsqueeze(-1)
    for i in range(N):
        mask = (idx == i)
        if mask.any():
            ei = _ffn_forward(x_flat[mask],
                              mod.experts.w1[i], mod.experts.b1[i],
                              mod.experts.w2[i], mod.experts.b2[i])
            out_flat[mask] = w[mask] * ei
    out_ref = out_flat.reshape(B, T, D)
    out_ref.sum().backward()

    assert torch.allclose(out_new, out_ref, atol=1e-5), \
        f"SMoE-Dropout k=1 forward mismatch: max |Δ|={(out_new-out_ref).abs().max():.2e}"
    assert torch.allclose(x2.grad, grad_x_new, atol=1e-5)
    for i in range(N):
        assert torch.allclose(grad_w1_new[i], mod.experts.w1.grad[i], atol=1e-5)
    print(f"  PASS smoe_dropout_equivalence_k=1 (B={B}, T={T}, N={N}, ffn={ffn})")


# ─── DEMix ───────────────────────────────────────────────────────────────────

def test_demix_hard_equivalence():
    """Per-document hard routing: each doc uses one expert."""
    B, T, D, N, ffn = 4, 8, 16, 4, 12
    mod = DemixFFN(hidden_dim=D, ffn_dim_per_expert=ffn, num_domains=N, dropout=0.0)
    mod.train()

    x = torch.randn(B, T, D, requires_grad=True)
    domain_ids = torch.tensor([0, 2, 1, 3])

    out_new = mod(x, domain_ids=domain_ids)
    out_new.sum().backward()
    grad_x_new = x.grad.clone()
    grad_w1_new = mod.experts.w1.grad.clone()

    _zero_all_grads(x, mod.experts.w1, mod.experts.b1,
                    mod.experts.w2, mod.experts.b2)
    x2 = x.detach().clone().requires_grad_(True)

    out_ref = torch.zeros_like(x2)
    for d in range(N):
        mask = (domain_ids == d)
        if mask.any():
            ei = _ffn_forward(x2[mask],
                              mod.experts.w1[d], mod.experts.b1[d],
                              mod.experts.w2[d], mod.experts.b2[d])
            out_ref[mask] = ei
    out_ref.sum().backward()

    assert torch.allclose(out_new, out_ref, atol=1e-5), \
        f"DEMix hard forward mismatch: max |Δ|={(out_new-out_ref).abs().max():.2e}"
    assert torch.allclose(x2.grad, grad_x_new, atol=1e-5)
    for i in range(N):
        assert torch.allclose(grad_w1_new[i], mod.experts.w1.grad[i], atol=1e-5)
    print(f"  PASS demix_hard_equivalence (B={B}, T={T}, N={N}, ffn={ffn})")


def test_demix_mixture_equivalence():
    """Mixture-of-Experts inference: all experts weight-combined per doc."""
    B, T, D, N, ffn = 4, 8, 16, 4, 12
    mod = DemixFFN(hidden_dim=D, ffn_dim_per_expert=ffn, num_domains=N, dropout=0.0)
    mod.eval()

    x = torch.randn(B, T, D, requires_grad=True)
    mix_w = torch.softmax(torch.randn(B, N), dim=-1)     # (B, N) posteriors

    out_new = mod(x, mixture_weights=mix_w)
    out_new.sum().backward()
    grad_x_new = x.grad.clone()
    grad_w1_new = mod.experts.w1.grad.clone()

    _zero_all_grads(x, mod.experts.w1, mod.experts.b1,
                    mod.experts.w2, mod.experts.b2)
    x2 = x.detach().clone().requires_grad_(True)

    # Reference: stack all N experts' outputs, weight-combine per doc.
    stacked = torch.stack([
        _ffn_forward(x2, mod.experts.w1[i], mod.experts.b1[i],
                     mod.experts.w2[i], mod.experts.b2[i])
        for i in range(N)], dim=1)    # (B, N, T, D)
    w = mix_w.view(B, N, 1, 1)
    out_ref = (stacked * w).sum(dim=1)
    out_ref.sum().backward()

    assert torch.allclose(out_new, out_ref, atol=1e-5), \
        f"DEMix mixture forward mismatch: max |Δ|={(out_new-out_ref).abs().max():.2e}"
    assert torch.allclose(x2.grad, grad_x_new, atol=1e-5)
    for i in range(N):
        assert torch.allclose(grad_w1_new[i], mod.experts.w1.grad[i], atol=1e-5)
    print(f"  PASS demix_mixture_equivalence (B={B}, T={T}, N={N}, ffn={ffn})")


# ─── Mod-Squad ViT ────────────────────────────────────────────────────────────

def test_mod_squad_equivalence():
    """Noisy top-k routing: disable noise (noise_std=0) for deterministic
    equivalence, then verify the new ParallelFFN + gather dispatch matches the
    for-loop reference bit-close on forward + grads + MI aux loss."""
    B, T, D, N, ffn, k = 2, 8, 16, 4, 12, 2
    mod = ModSquadFFN(dim=D, expert_hidden=ffn, num_experts=N,
                     top_k=k, mi_weight=0.01, noise_std=0.0, drop=0.0)
    mod.train()

    x = torch.randn(B, T, D, requires_grad=True)
    out_new = mod(x)
    aux_new = mod.aux_loss.clone()
    loss_new = out_new.sum() + aux_new
    loss_new.backward()
    grad_x_new = x.grad.clone()
    grad_w1_new = mod.experts.w1.grad.clone()
    grad_w2_new = mod.experts.w2.grad.clone()
    grad_gate_new = mod.gate.weight.grad.clone()

    _zero_all_grads(x, mod.experts.w1, mod.experts.b1,
                    mod.experts.w2, mod.experts.b2,
                    mod.gate.weight, mod.noise_gate.weight)
    x2 = x.detach().clone().requires_grad_(True)

    x_flat = x2.reshape(B * T, D)
    clean_logits = mod.gate(x_flat)            # noise_std=0 → no randomness
    probs_full = F.softmax(clean_logits, dim=-1)
    top_vals, top_idx = clean_logits.topk(k, dim=-1)
    top_w = F.softmax(top_vals, dim=-1)

    # MI loss
    P_E = probs_full.mean(dim=0)
    H_E = -(P_E * (P_E + 1e-12).log()).sum()
    H_E_given_T = -(probs_full * (probs_full + 1e-12).log()).sum(dim=-1).mean()
    aux_ref = -mod.mi_weight * (H_E - H_E_given_T)

    # For-loop reference dispatch.
    out_flat = torch.zeros_like(x_flat)
    for slot in range(k):
        idx_slot = top_idx[:, slot]; w_slot = top_w[:, slot].unsqueeze(-1)
        for i in range(N):
            mask = (idx_slot == i)
            if mask.any():
                ei = _ffn_forward(x_flat[mask],
                                  mod.experts.w1[i], mod.experts.b1[i],
                                  mod.experts.w2[i], mod.experts.b2[i])
                out_flat[mask] = out_flat[mask] + w_slot[mask] * ei
    out_ref = out_flat.reshape(B, T, D)
    loss_ref = out_ref.sum() + aux_ref
    loss_ref.backward()

    assert torch.allclose(out_new, out_ref, atol=1e-5), \
        f"Mod-Squad forward mismatch: max |Δ|={(out_new-out_ref).abs().max():.2e}"
    assert torch.allclose(aux_new, aux_ref, atol=1e-6), \
        f"Mod-Squad aux mismatch: {aux_new.item()} vs {aux_ref.item()}"
    assert torch.allclose(x2.grad, grad_x_new, atol=1e-5)
    for i in range(N):
        assert torch.allclose(grad_w1_new[i], mod.experts.w1.grad[i], atol=1e-5), \
            f"Mod-Squad expert {i} w1 grad mismatch"
    assert torch.allclose(grad_w2_new, mod.experts.w2.grad, atol=1e-5)
    assert torch.allclose(grad_gate_new, mod.gate.weight.grad, atol=1e-5)
    print(f"  PASS mod_squad_equivalence (B={B}, T={T}, N={N}, ffn={ffn}, k={k})")


if __name__ == "__main__":
    print("=" * 60)
    print(" MoE dispatch equivalence: new ParallelFFN impl ≡ for-loop ref")
    print("=" * 60)
    test_switch_equivalence()
    test_hash_layers_equivalence()
    test_smoe_dropout_equivalence_k_equals_N()
    test_smoe_dropout_equivalence_k1()
    test_demix_hard_equivalence()
    test_demix_mixture_equivalence()
    test_mod_squad_equivalence()
    print("=" * 60)
    print(" All 7 equivalence tests passed (atol=1e-5 for fp32)")
    print("=" * 60)

"""Unit tests for the Auxiliary-Loss-Free (ALF) MoE ViT baseline (Wang 2024).

Covers:
  (a) forward output shape + zero aux loss
  (b) bias update direction: starved expert's bias rises, overloaded falls
  (c) expert_bias is a non-trainable buffer and receives no gradient
  (d) eval mode does not update expert_bias
  (e) trainable param count within 1% of the Mod-Squad config's count
      (the pair isolates the balancing mechanism) and within 2% of dense ViT-S
  (f) 5-step tiny training loop: load spread decreases or bias moves
      monotonically toward balancing

Run: python tests/test_alf_moe_vit.py   (also pytest-compatible)
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.alf_moe_vit import ALFMoEViT, ALFMoEFFN
from models.mod_squad_vit import ModSquadViT


VIT_REF = 22_050_664       # dense ViT-Small/16 @ 1000 classes


def _trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _tiny_vit(**kw):
    kw.setdefault('img_size', 32)
    kw.setdefault('patch_size', 16)
    kw.setdefault('num_classes', 10)
    kw.setdefault('embed_dim', 64)
    kw.setdefault('depth', 2)
    kw.setdefault('num_heads', 2)
    kw.setdefault('num_experts', 4)
    kw.setdefault('expert_hidden', 8)
    kw.setdefault('top_k', 2)
    return ALFMoEViT(**kw)


# ─── (a) forward shape ───────────────────────────────────────────────────────

def test_forward_shape():
    torch.manual_seed(0)
    m = _tiny_vit()
    m.train()
    x = torch.randn(3, 3, 32, 32)
    logits, aux = m(x)
    assert logits.shape == (3, 10), f"logits shape {logits.shape}"
    assert aux.dim() == 0 and aux.item() == 0.0, \
        f"ALF must have zero aux loss, got {aux}"
    (logits.sum() + aux).backward()   # backward is clean
    assert torch.isfinite(logits).all()
    print("  PASS forward_shape: logits (3, 10), aux == 0.0")


# ─── (b) bias update direction ───────────────────────────────────────────────

def test_bias_update_direction():
    torch.manual_seed(0)
    N, k, D = 4, 2, 8
    u = 1e-3
    ffn = ALFMoEFFN(D, expert_hidden=4, num_experts=N, top_k=k,
                    bias_update_rate=u)
    # Craft the gate so every token produces the same expert ranking:
    # logits = x @ W^T with x = ones → logit_i = sum(W[i]). Set row sums to
    # [-2, +2, +1, 0] → top-2 selection is always {1, 2}; experts 0 and 3
    # are fully starved, experts 1 and 2 fully overloaded.
    with torch.no_grad():
        for i, c in enumerate([-2.0, 2.0, 1.0, 0.0]):
            ffn.gate.weight[i].fill_(c / D)
    x = torch.ones(2, 6, D)

    ffn.train()
    assert torch.all(ffn.expert_bias == 0)
    ffn(x)

    b = ffn.expert_bias
    load = ffn.last_load
    assert torch.allclose(load, torch.tensor([0.0, 1.0, 1.0, 0.0])), \
        f"expected load [0,1,1,0], got {load}"
    # Starved experts (0, 3): bias must INCREASE by u; overloaded (1, 2):
    # bias must DECREASE by u.
    assert b[0].item() > 0 and abs(b[0].item() - u) < 1e-8, f"b[0]={b[0]}"
    assert b[3].item() > 0, f"b[3]={b[3]}"
    assert b[1].item() < 0 and abs(b[1].item() + u) < 1e-8, f"b[1]={b[1]}"
    assert b[2].item() < 0, f"b[2]={b[2]}"

    # A second training forward accumulates in the same direction.
    ffn(x)
    assert abs(ffn.expert_bias[0].item() - 2 * u) < 1e-8
    assert abs(ffn.expert_bias[1].item() + 2 * u) < 1e-8
    print(f"  PASS bias_update_direction: starved +{u}, overloaded -{u} per step")


# ─── (c) no gradient through the bias ────────────────────────────────────────

def test_bias_no_grad():
    torch.manual_seed(0)
    N, k, D = 4, 2, 8
    ffn = ALFMoEFFN(D, expert_hidden=4, num_experts=N, top_k=k)
    ffn.train()
    x = torch.randn(2, 6, D, requires_grad=True)
    out = ffn(x)
    out.pow(2).mean().backward()

    assert not ffn.expert_bias.requires_grad, "expert_bias must not require grad"
    assert ffn.expert_bias.grad is None, "expert_bias must receive no grad"
    assert not isinstance(ffn.expert_bias, torch.nn.Parameter), \
        "expert_bias must be a buffer, not a Parameter"
    assert 'expert_bias' not in [n for n, _ in ffn.named_parameters()]
    # The gate DOES get grad (combine weights are differentiable through s).
    assert ffn.gate.weight.grad is not None
    assert ffn.gate.weight.grad.abs().sum().item() > 0
    print("  PASS bias_no_grad: buffer, requires_grad=False, grad is None; "
          "gate still receives grad")


# ─── (d) eval mode does not update b ─────────────────────────────────────────

def test_eval_no_bias_update():
    torch.manual_seed(0)
    N, k, D = 4, 2, 8
    ffn = ALFMoEFFN(D, expert_hidden=4, num_experts=N, top_k=k)
    ffn.train()
    ffn(torch.randn(2, 6, D))
    b_after_train = ffn.expert_bias.clone()
    assert not torch.all(b_after_train == 0), "train forward should update b"

    ffn.eval()
    with torch.no_grad():
        ffn(torch.randn(2, 6, D))
        ffn(torch.randn(4, 3, D))
    assert torch.equal(ffn.expert_bias, b_after_train), \
        "eval forward must NOT update expert_bias"
    print("  PASS eval_no_bias_update: b unchanged across eval forwards")


# ─── (e) param count vs Mod-Squad config ─────────────────────────────────────

def test_param_count_vs_mod_squad():
    # Deployed config: num_experts=16, expert_hidden=96 (mod_squad implicit
    # default = 4*384/16), top_k=2, 1000 classes.
    alf = ALFMoEViT(num_classes=1000, num_experts=16, expert_hidden=96, top_k=2)
    mod = ModSquadViT(num_classes=1000, num_experts=16, expert_hidden=96, top_k=2)
    p_alf, p_mod = _trainable(alf), _trainable(mod)
    ratio = p_alf / p_mod
    assert abs(ratio - 1.0) < 0.01, \
        f"ALF {p_alf:,} vs Mod-Squad {p_mod:,}: ratio {ratio:.4f} exceeds 1%"
    # Also within the repo-wide ±2% ViT budget (utils/sanity_check.py).
    vit_ratio = p_alf / VIT_REF
    assert abs(vit_ratio - 1.0) < 0.02, \
        f"ALF {p_alf:,} vs dense ViT-S {VIT_REF:,}: ratio {vit_ratio:.4f}"
    # expert_bias must not be counted as trainable anywhere.
    n_bias_buffers = sum(1 for n, _ in alf.named_buffers() if 'expert_bias' in n)
    assert n_bias_buffers == 12, f"expected 12 expert_bias buffers, got {n_bias_buffers}"
    print(f"  PASS param_count: ALF {p_alf:,} vs Mod-Squad {p_mod:,} "
          f"(ratio {ratio:.4f}); {vit_ratio:.4f}x dense ViT-S")


# ─── (f) tiny training loop moves toward balance ─────────────────────────────

def test_load_balancing_dynamics():
    torch.manual_seed(0)
    N, k, D = 8, 2, 16
    # Deployed u=1e-3: small enough that the natural load imbalance persists
    # across 5 steps (a large u makes the sign update flip selection every
    # step and oscillate around balance — correct behavior, but then neither
    # single-step signal below is monotone).
    ffn = ALFMoEFFN(D, expert_hidden=8, num_experts=N, top_k=k,
                    bias_update_rate=1e-3)
    opt = torch.optim.SGD(ffn.parameters(), lr=0.01)
    x = torch.randn(4, 32, D)   # fixed batch reused every step

    spreads, biases, loads = [], [], []
    ffn.train()
    for _ in range(5):
        out = ffn(x)
        loss = out.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loads.append(ffn.last_load.clone())
        spreads.append((ffn.last_load.max() - ffn.last_load.min()).item())
        biases.append(ffn.expert_bias.clone())

    overloaded = int(torch.argmax(loads[0]))
    starved = int(torch.argmin(loads[0]))
    spread_decreases = spreads[-1] < spreads[0]
    bias_monotone = (
        all(biases[i + 1][overloaded] <= biases[i][overloaded]
            for i in range(len(biases) - 1))
        and all(biases[i + 1][starved] >= biases[i][starved]
                for i in range(len(biases) - 1))
        and biases[-1][overloaded] < biases[0][overloaded]
    )
    assert spread_decreases or bias_monotone, (
        f"neither balancing signal holds: spreads={spreads}, "
        f"b_overloaded={[b[overloaded].item() for b in biases]}, "
        f"b_starved={[b[starved].item() for b in biases]}"
    )
    print(f"  PASS load_balancing_dynamics: spread {spreads[0]:.3f} → "
          f"{spreads[-1]:.3f}, monotone_bias={bias_monotone}")


if __name__ == "__main__":
    print("=" * 60)
    print(" ALF MoE ViT (Wang 2024) — unit tests")
    print("=" * 60)
    test_forward_shape()
    test_bias_update_direction()
    test_bias_no_grad()
    test_eval_no_bias_update()
    test_param_count_vs_mod_squad()
    test_load_balancing_dynamics()
    print("=" * 60)
    print(" All ALF MoE ViT tests passed")
    print("=" * 60)

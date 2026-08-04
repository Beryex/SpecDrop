"""Unit tests for models/lora_adapters.py + lora_base.py + lora_models.py.

Tests the 5 LoRA adapter variants (Single, MultiBranch-SoftSpecDrop, HydraLoRA,
LoRAMoE, MoCLE) and the shared injection infrastructure (LoRAInjectedLinear,
inject_lora_adapters, freeze_base_params, set_routing_all, sum_aux_losses),
WITHOUT loading a real 3B HF model. Uses a toy transformer-like module to
exercise tree walking + adapter injection + routing flow.

HF-dependent tests (build_lora_model with a real Llama base, actual training
loop) are covered in integration tests — these unit tests run in CI on CPU
in <10 seconds.

Covers:
  - Each adapter: forward shape, zero-init identity, backward to LoRA params
  - LoRAInjectedLinear: base frozen, adapter trainable, set_routing propagates
  - inject_lora_adapters: all matching sites wrapped, others untouched
  - freeze_base_params: only adapter params trainable after freeze
  - sum_aux_losses: accumulates LoRAMoE's balance loss across sites
"""
import pytest
import torch
import torch.nn as nn

from models.lora_adapters import (HydraLoRAAdapter, LoRAMoEAdapter, MoCLEAdapter,
                                   MultiBranchSoftSpecDropAdapter,
                                   SingleLoRAAdapter)
from models.lora_base import (DEFAULT_TARGETS_ALL, LoRAInjectedLinear,
                               freeze_base_params, inject_lora_adapters,
                               set_routing_all, sum_aux_losses)


# ══════════════════════════════════════════════════════════════════════════
# Toy transformer mimicking Llama's module tree (for walker tests).
# ══════════════════════════════════════════════════════════════════════════


class _ToySelfAttn(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)


class _ToyMLP(nn.Module):
    def __init__(self, d=16, h=32):
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)


class _ToyLayer(nn.Module):
    def __init__(self, d=16, h=32):
        super().__init__()
        self.self_attn = _ToySelfAttn(d)
        self.mlp = _ToyMLP(d, h)


class _ToyLLM(nn.Module):
    def __init__(self, d=16, h=32, L=2):
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer(d, h) for _ in range(L)])
        # A decoy Linear that should NOT get wrapped (name not in target_modules).
        self.lm_head = nn.Linear(d, 100, bias=False)


# ══════════════════════════════════════════════════════════════════════════
# 1. SingleLoRAAdapter
# ══════════════════════════════════════════════════════════════════════════

def test_single_lora_forward_shape_and_zero_init():
    adapter = SingleLoRAAdapter(in_features=16, out_features=32, rank=4)
    x = torch.randn(2, 5, 16)
    out = adapter(x)
    assert out.shape == (2, 5, 32)
    # B=0 at init → output is exactly zero.
    assert torch.allclose(out, torch.zeros_like(out))


def test_single_lora_backward():
    adapter = SingleLoRAAdapter(in_features=8, out_features=8, rank=4)
    with torch.no_grad():
        adapter.B.copy_(torch.randn_like(adapter.B) * 0.1)
    x = torch.randn(2, 3, 8)
    y = adapter(x)
    y.pow(2).sum().backward()
    assert adapter.A.grad is not None
    assert adapter.B.grad is not None


# ══════════════════════════════════════════════════════════════════════════
# 2. MultiBranchSoftSpecDropAdapter
# ══════════════════════════════════════════════════════════════════════════

def test_mb_softspecdrop_forward_with_routing():
    adapter = MultiBranchSoftSpecDropAdapter(
        in_features=16, out_features=16, num_experts=4, rank=4)
    x = torch.randn(3, 5, 16)
    mask = torch.rand(3, 4)
    out = adapter(x, routing={'mask': mask, 'mask_scale': 1.5})
    assert out.shape == (3, 5, 16)
    # Zero-init → output is zero (B=0).
    assert torch.allclose(out, torch.zeros_like(out))


def test_mb_softspecdrop_routing_none_uses_uniform():
    adapter = MultiBranchSoftSpecDropAdapter(
        in_features=8, out_features=8, num_experts=3, rank=4)
    x = torch.randn(2, 4, 8)
    out = adapter(x, routing=None)  # fallback to uniform 1/K
    assert out.shape == (2, 4, 8)


# ══════════════════════════════════════════════════════════════════════════
# 3. HydraLoRAAdapter (Tian 2024)
# ══════════════════════════════════════════════════════════════════════════

def test_hydra_lora_forward_and_zero_init():
    adapter = HydraLoRAAdapter(
        in_features=16, out_features=32, num_B_heads=8, rank=8)
    x = torch.randn(2, 5, 16)
    out = adapter(x, routing=None)  # ignores routing
    assert out.shape == (2, 5, 32)
    # B all zeros → output zero at init
    assert torch.allclose(out, torch.zeros_like(out))


def test_hydra_lora_param_count_asymmetric():
    """Asymmetric: 1 A (r × D_in) + N B (N × D_out × r) + gate (N × D_in)."""
    D_in, D_out, r, N = 16, 32, 8, 6
    adapter = HydraLoRAAdapter(
        in_features=D_in, out_features=D_out, num_B_heads=N, rank=r)
    # Only A, B, gate.weight are parameters.
    expected_A = r * D_in
    expected_B = N * D_out * r
    expected_gate = N * D_in  # bias=False
    total = sum(p.numel() for p in adapter.parameters())
    assert total == expected_A + expected_B + expected_gate


def test_hydra_lora_backward():
    adapter = HydraLoRAAdapter(
        in_features=8, out_features=8, num_B_heads=4, rank=4)
    with torch.no_grad():
        adapter.B.copy_(torch.randn_like(adapter.B) * 0.1)
        adapter.gate.weight.copy_(torch.randn_like(adapter.gate.weight) * 0.1)
    x = torch.randn(2, 3, 8)
    out = adapter(x)
    out.pow(2).sum().backward()
    assert adapter.A.grad is not None
    assert adapter.B.grad is not None
    assert adapter.gate.weight.grad is not None


# ══════════════════════════════════════════════════════════════════════════
# 4. LoRAMoEAdapter (Dou 2023)
# ══════════════════════════════════════════════════════════════════════════

def test_loramoe_forward_and_aux_loss():
    adapter = LoRAMoEAdapter(
        in_features=16, out_features=16, num_experts=6, rank=4)
    x = torch.randn(2, 5, 16)
    out = adapter(x)
    assert out.shape == (2, 5, 16)
    # aux_loss populated (coefficient of variation, non-negative).
    assert adapter.last_aux_loss is not None
    assert float(adapter.last_aux_loss) >= 0


def test_loramoe_balance_loss_scalar():
    """Balance loss is a scalar tensor (not a batch)."""
    adapter = LoRAMoEAdapter(
        in_features=8, out_features=8, num_experts=4, rank=4)
    x = torch.randn(3, 2, 8)
    adapter(x)
    assert adapter.last_aux_loss.dim() == 0


# ══════════════════════════════════════════════════════════════════════════
# 5. MoCLEAdapter (Gou 2024)
# ══════════════════════════════════════════════════════════════════════════

def test_mocle_forward_with_cluster_id():
    adapter = MoCLEAdapter(
        in_features=16, out_features=16, num_task_experts=4, rank=4,
        num_clusters=10)
    x = torch.randn(3, 5, 16)
    routing = {'cluster_id': torch.tensor([0, 1, 2])}
    out = adapter(x, routing=routing)
    assert out.shape == (3, 5, 16)
    # Zero-init B + B_u → output zero
    assert torch.allclose(out, torch.zeros_like(out))


def test_mocle_cluster_embedding_init_from_centroids():
    E = 4
    num_clusters = 10
    D_ce = 12
    adapter = MoCLEAdapter(
        in_features=8, out_features=8, num_task_experts=E, rank=2,
        num_clusters=num_clusters, cluster_embed_dim=D_ce)
    centroids = torch.randn(num_clusters, D_ce)
    adapter.init_cluster_embeddings(centroids)
    assert torch.allclose(adapter.cluster_embeddings.weight, centroids)


def test_mocle_different_clusters_different_gates():
    """Different cluster_ids → different gate outputs → different expert selection.

    Constructed deterministically: cluster_embeddings = I_4 one-hots, gate = I_4,
    so cluster k routes to expert k with weight 1. Per-expert B matrices differ,
    so outputs differ across clusters. Disables noise + universal expert to
    isolate the gate path. Uses τ=1.0 so softmax isn't so sharp that multiple
    cluster embeddings collapse to the same argmax.
    """
    torch.manual_seed(0)
    adapter = MoCLEAdapter(
        in_features=8, out_features=8, num_task_experts=4, rank=4,
        num_clusters=4, cluster_embed_dim=4,
        noise_std=0.0, temperature=1.0)
    with torch.no_grad():
        adapter.cluster_embeddings.weight.copy_(torch.eye(4))
        adapter.gate.weight.copy_(torch.eye(4))
        adapter.B.copy_(torch.randn_like(adapter.B) * 0.1)
        adapter.B_u.copy_(torch.zeros_like(adapter.B_u))  # silence universal

    x = torch.randn(1, 2, 8)
    out_0 = adapter(x, routing={'cluster_id': torch.tensor([0])})
    out_1 = adapter(x, routing={'cluster_id': torch.tensor([1])})
    out_2 = adapter(x, routing={'cluster_id': torch.tensor([2])})
    # All three should be pairwise different (different B_k selected).
    assert not torch.allclose(out_0, out_1, atol=1e-5)
    assert not torch.allclose(out_1, out_2, atol=1e-5)
    assert not torch.allclose(out_0, out_2, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════
# LoRAInjectedLinear + injection walker
# ══════════════════════════════════════════════════════════════════════════

def test_lora_injected_linear_wraps_base_frozen():
    base = nn.Linear(16, 32, bias=False)
    adapter = SingleLoRAAdapter(16, 32, rank=4)
    wrapped = LoRAInjectedLinear(base, adapter, site_role='attn', target_name='q_proj')
    # Base is frozen
    for p in wrapped.base.parameters():
        assert p.requires_grad is False
    # Adapter is trainable
    for p in wrapped.adapter.parameters():
        assert p.requires_grad is True


def test_lora_injected_linear_forward_equals_base_at_init():
    base = nn.Linear(16, 32, bias=True)
    adapter = SingleLoRAAdapter(16, 32, rank=4)
    wrapped = LoRAInjectedLinear(base, adapter)
    x = torch.randn(2, 5, 16)
    base_y = base(x)
    wrapped_y = wrapped(x)
    # LoRA off at init (B=0) → equal
    assert torch.allclose(wrapped_y, base_y)


def test_lora_injected_set_routing_propagates():
    base = nn.Linear(16, 16, bias=False)
    adapter = MultiBranchSoftSpecDropAdapter(16, 16, num_experts=3, rank=4)
    wrapped = LoRAInjectedLinear(base, adapter)
    routing = {'mask': torch.rand(2, 3), 'mask_scale': 1.5}
    wrapped.set_routing(routing)
    # forward now uses the routing
    x = torch.randn(2, 4, 16)
    out = wrapped(x)
    assert out.shape == (2, 4, 16)


def test_inject_lora_adapters_wraps_all_targets():
    toy = _ToyLLM(d=16, h=32, L=2)
    wrapped_sites = inject_lora_adapters(
        toy, adapter_factory=lambda lin, name, role: SingleLoRAAdapter(
            lin.in_features, lin.out_features, rank=4),
        target_modules=DEFAULT_TARGETS_ALL)
    # 2 layers × 7 targets = 14 wrapped sites
    assert len(wrapped_sites) == 14
    # lm_head NOT wrapped (not in target_modules)
    assert not isinstance(toy.lm_head, LoRAInjectedLinear)


def test_inject_lora_adapters_respects_target_filter():
    """Pass only MLP targets → only 3 sites per layer get wrapped."""
    toy = _ToyLLM(d=16, h=32, L=2)
    wrapped_sites = inject_lora_adapters(
        toy, adapter_factory=lambda lin, name, role: SingleLoRAAdapter(
            lin.in_features, lin.out_features, rank=4),
        target_modules=('gate_proj', 'up_proj', 'down_proj'))
    assert len(wrapped_sites) == 6  # 2 layers × 3 MLP targets
    # Attn sites untouched
    assert not isinstance(toy.layers[0].self_attn.q_proj, LoRAInjectedLinear)


def test_freeze_base_params_leaves_only_adapter_trainable():
    toy = _ToyLLM(d=16, h=32, L=2)
    inject_lora_adapters(
        toy, adapter_factory=lambda lin, name, role: SingleLoRAAdapter(
            lin.in_features, lin.out_features, rank=4),
        target_modules=DEFAULT_TARGETS_ALL)
    # lm_head still trainable before freeze.
    assert toy.lm_head.weight.requires_grad is True

    n_trainable = freeze_base_params(toy)
    assert n_trainable > 0

    # lm_head now FROZEN; adapter params still trainable.
    assert toy.lm_head.weight.requires_grad is False
    for m in toy.modules():
        if isinstance(m, LoRAInjectedLinear):
            assert m.base.weight.requires_grad is False
            for p in m.adapter.parameters():
                assert p.requires_grad is True


def test_set_routing_all_broadcasts():
    toy = _ToyLLM(d=16, h=32, L=2)
    inject_lora_adapters(
        toy, adapter_factory=lambda lin, name, role: MultiBranchSoftSpecDropAdapter(
            lin.in_features, lin.out_features, num_experts=3, rank=2),
        target_modules=DEFAULT_TARGETS_ALL)
    routing = {'mask': torch.rand(2, 3), 'mask_scale': 2.0}
    set_routing_all(toy, routing)
    for m in toy.modules():
        if isinstance(m, LoRAInjectedLinear):
            assert m.current_routing is routing


def test_sum_aux_losses_accumulates():
    """LoRAMoE adapters populate aux_loss; sum_aux_losses collects them."""
    toy = _ToyLLM(d=16, h=32, L=2)
    inject_lora_adapters(
        toy, adapter_factory=lambda lin, name, role: LoRAMoEAdapter(
            lin.in_features, lin.out_features, num_experts=4, rank=2),
        target_modules=('gate_proj', 'up_proj', 'down_proj'))
    freeze_base_params(toy)
    # Run a forward: each adapter populates last_aux_loss
    x = torch.randn(2, 3, 16)
    for layer in toy.layers:
        _ = layer.mlp.gate_proj(x)
        _ = layer.mlp.up_proj(x)

    total = sum_aux_losses(toy)
    # 2 layers × 3 MLP sites = 6 aux contributions (but only gate/up touched
    # in the loop above, so down_proj aux is 0 from init). Total is finite.
    assert torch.isfinite(total).all()
    assert float(total) >= 0

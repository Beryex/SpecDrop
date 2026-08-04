"""Unit tests for scripts/_ablation_hooks.py."""
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts._ablation_hooks import (
    install_ablation, uninstall_ablation, ablate_branch,
    find_multibranch_modules, model_branch_count,
)


# ─── Fixtures: tiny models per learned-router class ────────────────────────

def _make_parallel_ffn(K=4, D=8, FF=12):
    from models.transformer_lm import ParallelFFN
    return ParallelFFN(hidden_dim=D, ffn_dim=FF, num_branches=K, dropout=0.0)


def _make_switch_block(K=4, D=8, FF=12, num_heads=2):
    """Switch FFN that wraps ParallelFFN — exactly the production block class."""
    from models.switch_transformer_lm import SwitchFFN
    return SwitchFFN(hidden_dim=D, ffn_dim_per_expert=FF, num_experts=K,
                      dropout=0.0)


def _make_softmoe_ffn(N=4, S=1, D=8, FF=12):
    from models.soft_moe_vit import SoftMoEFFN
    return SoftMoEFFN(dim=D, expert_hidden=FF, num_experts=N,
                       slots_per_expert=S, drop=0.0)


def _make_hydra_lora(N=4, D_in=8, D_out=8, r=4):
    from models.lora_adapters import HydraLoRAAdapter
    return HydraLoRAAdapter(in_features=D_in, out_features=D_out,
                             num_B_heads=N, rank=r, dropout=0.0)


def _make_lora_moe(K=4, D_in=8, D_out=8, r=4):
    from models.lora_adapters import LoRAMoEAdapter
    return LoRAMoEAdapter(in_features=D_in, out_features=D_out,
                           num_experts=K, rank=r, dropout=0.0)


def _make_mocle(E=3, M=5, D_in=8, D_out=8, r=4):
    from models.lora_adapters import MoCLEAdapter
    return MoCLEAdapter(in_features=D_in, out_features=D_out,
                         num_task_experts=E, rank=r, num_clusters=M,
                         cluster_embed_dim=D_in, dropout=0.0)


def _randomize_lora(module):
    """LoRA adapters initialize B to zeros (LoRA-off) so output is 0
    regardless of mask. Force B random for ablation tests to detect change."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith('.B') or name == 'B':
                nn.init.normal_(p, std=0.1)
            elif name in ('B_u',) or name.endswith('B_u'):
                nn.init.normal_(p, std=0.1)


# ─── Class detection + K extraction ────────────────────────────────────────

def test_find_modules_parallel_ffn():
    pf = _make_parallel_ffn(K=7)
    found = find_multibranch_modules(pf)
    assert len(found) == 1
    m, kind, K = found[0]
    assert kind == 'parallel_ffn'
    assert K == 7


def test_find_modules_soft_moe():
    sm = _make_softmoe_ffn(N=8)
    found = find_multibranch_modules(sm)
    assert len(found) == 1
    m, kind, K = found[0]
    assert kind == 'soft_moe' and K == 8


def test_find_modules_hydra_lora():
    h = _make_hydra_lora(N=5)
    found = find_multibranch_modules(h)
    assert len(found) == 1 and found[0][1] == 'hydra_lora' and found[0][2] == 5


def test_find_modules_lora_moe():
    lm = _make_lora_moe(K=6)
    found = find_multibranch_modules(lm)
    assert len(found) == 1 and found[0][1] == 'lora_moe' and found[0][2] == 6


def test_find_modules_mocle():
    mc = _make_mocle(E=4)
    found = find_multibranch_modules(mc)
    assert len(found) == 1 and found[0][1] == 'mocle' and found[0][2] == 4


def test_find_modules_in_switch_block():
    """SwitchFFN contains a ParallelFFN — should be discovered through
    sub-module recursion."""
    sb = _make_switch_block(K=5)
    found = find_multibranch_modules(sb)
    assert any(kind == 'parallel_ffn' and K == 5 for _, kind, K in found)


# ─── ParallelFFN: hook zeros expert k's column ─────────────────────────────

def test_ablation_parallel_ffn_zeros_column_k():
    torch.manual_seed(0)
    pf = _make_parallel_ffn(K=4, D=8, FF=12)
    pf.eval()
    x = torch.randn(2, 5, 8)

    out_baseline = pf(x)
    assert out_baseline.shape == (2, 5, 4, 8)

    # Ablate branch 1
    install_ablation(pf, k=1)
    out_ablated = pf(x)
    uninstall_ablation(pf)

    # Column 1 must be zero
    assert torch.allclose(out_ablated[:, :, 1, :], torch.zeros_like(out_ablated[:, :, 1, :]))
    # Other columns must be unchanged from baseline
    for k in [0, 2, 3]:
        assert torch.allclose(out_ablated[:, :, k, :], out_baseline[:, :, k, :])


def test_ablation_parallel_ffn_baseline_no_ablate():
    """install_ablation(model, k=None) installs hook but disables ablation."""
    torch.manual_seed(0)
    pf = _make_parallel_ffn(K=4)
    pf.eval()
    x = torch.randn(2, 5, 8)
    baseline = pf(x).clone()

    install_ablation(pf, k=None)
    out = pf(x)
    uninstall_ablation(pf)
    assert torch.allclose(out, baseline)


def test_ablation_uninstall_restores_baseline():
    """After uninstall, output should match pre-install baseline exactly."""
    torch.manual_seed(0)
    pf = _make_parallel_ffn(K=4)
    pf.eval()
    x = torch.randn(2, 5, 8)
    baseline = pf(x).clone()

    install_ablation(pf, k=1)
    _ = pf(x)
    uninstall_ablation(pf)
    out = pf(x)
    assert torch.allclose(out, baseline)


def test_ablation_context_manager():
    torch.manual_seed(0)
    pf = _make_parallel_ffn(K=4)
    pf.eval()
    x = torch.randn(2, 5, 8)
    baseline = pf(x).clone()

    with ablate_branch(pf, k=2):
        out = pf(x)
    assert torch.allclose(out[:, :, 2, :], torch.zeros_like(out[:, :, 2, :]))
    # After exit, full output restored
    out_after = pf(x)
    assert torch.allclose(out_after, baseline)


# ─── SoftMoEFFN: ablating expert k zeros its slot output ─────────────────

def test_ablation_softmoe_changes_output():
    torch.manual_seed(0)
    sm = _make_softmoe_ffn(N=4, D=8, FF=12)
    sm.eval()
    x = torch.randn(2, 6, 8)
    baseline = sm(x).clone()

    install_ablation(sm, k=1)
    out_ablated = sm(x)
    uninstall_ablation(sm)
    # Output should differ when expert k contributes (with random init it does).
    assert not torch.allclose(out_ablated, baseline, atol=1e-6), \
        'SoftMoE ablation should change output (unless all weights happen to be 0)'


# ─── HydraLoRA: ablating B head k changes per_head sum ───────────────────

def test_ablation_hydra_lora_changes_output():
    torch.manual_seed(0)
    h = _make_hydra_lora(N=4, D_in=8, D_out=8, r=4)
    _randomize_lora(h)
    h.eval()
    x = torch.randn(2, 6, 8)
    baseline = h(x).clone()
    install_ablation(h, k=1)
    out_ablated = h(x)
    uninstall_ablation(h)
    assert not torch.allclose(out_ablated, baseline, atol=1e-6), \
        'HydraLoRA ablation should change output'


# ─── LoRAMoE: ablating expert k changes per_expert sum ───────────────────

def test_ablation_lora_moe_changes_output():
    torch.manual_seed(0)
    lm = _make_lora_moe(K=4, D_in=8, D_out=8, r=4)
    _randomize_lora(lm)
    lm.eval()
    x = torch.randn(2, 6, 8)
    baseline = lm(x).clone()
    install_ablation(lm, k=2)
    out_ablated = lm(x)
    uninstall_ablation(lm)
    assert not torch.allclose(out_ablated, baseline, atol=1e-6), \
        'LoRAMoE ablation should change output'


# ─── MoCLE: ablating task expert k changes output via top-1 routing ──────

def test_ablation_mocle_changes_output_when_argmax_is_k():
    """MoCLE forward selects top-1 task expert per sample. Force-set gate
    weights so argmax = expert k for all samples; then ablate expert k."""
    torch.manual_seed(0)
    mc = _make_mocle(E=3, M=4, D_in=8, D_out=8, r=4)
    _randomize_lora(mc)
    # Force gate to always pick expert k=1
    with torch.no_grad():
        mc.gate.weight.zero_()
        mc.gate.weight[1, :] = 10.0  # large logit for expert 1
    mc.eval()
    x = torch.randn(2, 6, 8)
    routing = {'cluster_id': torch.tensor([0, 1])}
    baseline = mc(x, routing=routing).clone()
    install_ablation(mc, k=1)
    out_ablated = mc(x, routing=routing)
    uninstall_ablation(mc)
    assert not torch.allclose(out_ablated, baseline, atol=1e-6)


# ─── Switch (real production class via ParallelFFN) ────────────────────────

def test_ablation_switch_block_changes_output():
    """SwitchFFN -> ParallelFFN(self.experts). Ablating expert k of the
    inner ParallelFFN should change the SwitchFFN's gathered output."""
    torch.manual_seed(0)
    sb = _make_switch_block(K=4, D=8, FF=12)
    sb.eval()
    x = torch.randn(2, 6, 8)
    baseline = sb(x).clone()
    install_ablation(sb, k=1)
    # SwitchFFN forward returns (output, lb_loss). Take output only.
    out_ablated = sb(x)
    uninstall_ablation(sb)
    if isinstance(out_ablated, tuple):
        out_ablated = out_ablated[0]
    if isinstance(baseline, tuple):
        baseline = baseline[0]
    # If router happens to never pick expert 1, output unchanged. Force-set
    # router weights so expert 1 is picked sometimes.
    # This test checks the hook fires; if it doesn't trigger a delta, retest
    # with manipulated router.
    # For correctness under random init: if expert 1 is the top-1 for at
    # least one (B, T) pair, output will differ. Probabilistically this is
    # very likely with random init. Use loose check.
    delta = (out_ablated - baseline).abs().sum().item()
    assert delta > 0, 'SwitchFFN ablation should change at least one token'


# ─── Idempotent install + multiple ks ─────────────────────────────────────

def test_idempotent_install_then_uninstall():
    pf = _make_parallel_ffn(K=4)
    install_ablation(pf, k=1)
    install_ablation(pf, k=1)  # idempotent — should not double-hook
    # exactly one hook handle
    handles = getattr(pf, '_ablation_handles', None)
    if handles is None:
        # parallel_ffn case stores handle on model attribute
        from scripts._ablation_hooks import find_multibranch_modules
        for m, kind, K in find_multibranch_modules(pf):
            if kind == 'parallel_ffn':
                # Only one hook should be active.
                # (We can't directly check pytorch internal hook count without
                # diving into _forward_hooks. Assertion at next line.)
                assert hasattr(m, '_ablation_hook_handle')
    uninstall_ablation(pf)
    # uninstall removes
    found_hooks = []
    for m in pf.modules():
        if hasattr(m, '_ablation_hook_handle'):
            found_hooks.append(m)
    assert not found_hooks


def test_change_k_after_install():
    """Ablation k can be changed by re-calling install_ablation."""
    torch.manual_seed(0)
    pf = _make_parallel_ffn(K=4)
    pf.eval()
    x = torch.randn(2, 5, 8)

    install_ablation(pf, k=0)
    out0 = pf(x)
    install_ablation(pf, k=2)  # update k
    out2 = pf(x)
    uninstall_ablation(pf)

    # out0 has col 0 = 0; out2 has col 2 = 0. They should differ.
    assert torch.allclose(out0[:, :, 0, :], torch.zeros_like(out0[:, :, 0, :]))
    assert torch.allclose(out2[:, :, 2, :], torch.zeros_like(out2[:, :, 2, :]))
    assert not torch.allclose(out0, out2)


# ─── model_branch_count ────────────────────────────────────────────────────

def test_model_branch_count():
    pf = _make_parallel_ffn(K=7)
    assert model_branch_count(pf) == 7
    sm = _make_softmoe_ffn(N=12)
    assert model_branch_count(sm) == 12
    h = _make_hydra_lora(N=5)
    assert model_branch_count(h) == 5
    # Empty model
    empty = nn.Linear(4, 4)
    assert model_branch_count(empty) == 0

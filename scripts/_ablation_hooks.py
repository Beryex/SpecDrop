"""Ablation hooks for learned-router and native-MoE multibranch models.

Provides `install_ablation(model, k)` / `uninstall_ablation(model)` that
zero out expert/branch k's contribution at the right point inside each
multibranch FFN/adapter forward, regardless of routing mechanism.

Coverage:
- `ParallelFFN` (NLP transformer_lm.py + Mod-Squad ViT): post-hook on its
  (B, T, N, D) output — zeros column k. Implicitly handles SwitchFFN,
  HashLayerFFN, DemixFFN, SMoEDropoutFFN, ModSquadFFN (all use ParallelFFN
  as expert backbone).
- `SoftMoEFFN` (Soft MoE ViT): monkey-patch forward to zero Y_tilde[:, k]
  before combine.
- `HydraLoRAAdapter`: monkey-patch to zero per_head[:, :, k]
- `LoRAMoEAdapter`: monkey-patch to zero per_expert[:, :, k]
- `MoCLEAdapter`: monkey-patch to zero per_expert[:, :, k] for k < E
  (task-expert k); k == E ablates the universal expert.

Two ways to use:

  # Context manager (recommended):
  with ablate_branch(model, k=3):
      out = model(...)

  # Manual install/uninstall:
  install_ablation(model, k=3)
  out = model(...)
  uninstall_ablation(model)

Idempotent on reinstall (each module remembers original forward).
"""
from __future__ import annotations

import contextlib
from typing import Optional


# ─── ParallelFFN: post-hook on (B, T, N, D) output ─────────────────────────

def _make_parallel_ffn_hook(get_k):
    """Returns a forward hook that zeros column k of ParallelFFN output."""
    def _hook(module, inputs, output):
        k = get_k()
        if k is None:
            return output
        # output shape: (B, T, K, D). Zero column k.
        if output.dim() != 4:
            return output
        K = output.shape[2]
        if not (0 <= k < K):
            return output
        out = output.clone()
        out[:, :, k, :] = 0.0
        return out
    return _hook


# ─── SoftMoEFFN: wrap forward, zero Y_tilde[:, k] ─────────────────────────

def _patch_softmoe_forward(module):
    """Replace SoftMoEFFN.forward with an ablation-aware version. Idempotent."""
    if getattr(module, '_orig_forward', None) is not None:
        return  # already patched
    import torch
    import torch.nn.functional as F
    orig = module.forward

    def _new_forward(x):
        k = getattr(module, '_ablate_branch_idx', None)
        if k is None:
            return orig(x)
        # Replicate forward with branch-k zero. We can't avoid recomputation
        # cleanly without intercepting Y_tilde, so re-run the math here.
        B, T, D = x.shape
        N, S = module.num_experts, module.slots_per_expert
        if module.normalize:
            x_n = F.normalize(x, dim=-1)
            phi_n = F.normalize(module.phi, dim=0)
            logits = module.scale * torch.einsum('btd,dns->btns', x_n, phi_n)
        else:
            logits = torch.einsum('btd,dns->btns', x, module.phi)
        D_weights = F.softmax(logits, dim=1)
        X_tilde = torch.einsum('btns,btd->bnsd', D_weights, x)
        hidden = torch.einsum('bnsd,nhd->bnsh', X_tilde, module.experts_fc1) \
                 + module.experts_b1.view(1, N, 1, -1)
        hidden = F.gelu(hidden)
        Y_tilde = torch.einsum('bnsh,ndh->bnsd', hidden, module.experts_fc2) \
                  + module.experts_b2.view(1, N, 1, -1)
        # Ablate expert k by zeroing its slot outputs.
        if 0 <= k < N:
            Y_tilde = Y_tilde.clone()
            Y_tilde[:, k, :, :] = 0.0
        C_weights = F.softmax(logits.reshape(B, T, N * S), dim=-1).reshape(B, T, N, S)
        y = torch.einsum('btns,bnsd->btd', C_weights, Y_tilde)
        return module.drop(y)

    module._orig_forward = orig
    module.forward = _new_forward


def _unpatch_softmoe_forward(module):
    if getattr(module, '_orig_forward', None) is not None:
        module.forward = module._orig_forward
        module._orig_forward = None


# ─── HydraLoRAAdapter: wrap forward, zero per_head[:, :, k] ───────────────

def _patch_hydra_lora_forward(module):
    if getattr(module, '_orig_forward', None) is not None:
        return
    import torch
    import torch.nn.functional as F
    orig = module.forward

    def _new_forward(x, routing=None):
        k = getattr(module, '_ablate_branch_idx', None)
        if k is None:
            return orig(x, routing=routing)
        orig_ndim = x.ndim
        if orig_ndim == 2:
            x = x.unsqueeze(1)
        Bsz, T, D_in = x.shape
        N = module.num_B_heads
        x_drop = module.dropout(x)
        hidden = F.linear(x_drop, module.A)
        per_head = torch.einsum('btr,nor->btno', hidden, module.B)
        if 0 <= k < N:
            per_head = per_head.clone()
            per_head[:, :, k, :] = 0.0
        gate_logits = module.gate(x_drop)
        gate_w = F.softmax(gate_logits, dim=-1)
        merged = (per_head * gate_w.unsqueeze(-1)).sum(dim=2)
        out = merged * module.scaling
        if orig_ndim == 2:
            out = out.squeeze(1)
        return out

    module._orig_forward = orig
    module.forward = _new_forward


def _unpatch_hydra_lora_forward(module):
    if getattr(module, '_orig_forward', None) is not None:
        module.forward = module._orig_forward
        module._orig_forward = None


# ─── LoRAMoEAdapter: wrap forward, zero per_expert[:, :, k] ───────────────

def _patch_lora_moe_forward(module):
    if getattr(module, '_orig_forward', None) is not None:
        return
    import torch
    import torch.nn.functional as F
    orig = module.forward

    def _new_forward(x, routing=None):
        k = getattr(module, '_ablate_branch_idx', None)
        if k is None:
            return orig(x, routing=routing)
        orig_ndim = x.ndim
        if orig_ndim == 2:
            x = x.unsqueeze(1)
        Bsz, T, D_in = x.shape
        K = module.num_experts
        x_drop = module.dropout(x)
        hidden = torch.einsum('btd,krd->btkr', x_drop, module.A)
        per_expert = torch.einsum('btkr,kor->btko', hidden, module.B)
        if 0 <= k < K:
            per_expert = per_expert.clone()
            per_expert[:, :, k, :] = 0.0
        gate_w = F.softmax(module.gate(x_drop) / module.balance_tau, dim=-1)
        merged = (per_expert * gate_w.unsqueeze(-1)).sum(dim=2)
        out = merged * module.scaling
        # Skip aux loss assignment under ablation — eval-only path.
        if orig_ndim == 2:
            out = out.squeeze(1)
        return out

    module._orig_forward = orig
    module.forward = _new_forward


def _unpatch_lora_moe_forward(module):
    if getattr(module, '_orig_forward', None) is not None:
        module.forward = module._orig_forward
        module._orig_forward = None


# ─── MoCLEAdapter: wrap forward, zero per_expert[:, :, k] (task) or universal ─

def _patch_mocle_forward(module):
    if getattr(module, '_orig_forward', None) is not None:
        return
    import torch
    import torch.nn.functional as F
    orig = module.forward

    def _new_forward(x, routing=None):
        k = getattr(module, '_ablate_branch_idx', None)
        if k is None:
            return orig(x, routing=routing)
        orig_ndim = x.ndim
        if orig_ndim == 2:
            x = x.unsqueeze(1)
        Bsz, T, D_in = x.shape
        E = module.num_task_experts
        if routing is None or routing.get('cluster_id', None) is None:
            cluster_ids = torch.zeros(Bsz, dtype=torch.long, device=x.device)
        else:
            cluster_ids = routing['cluster_id']
        x_drop = module.dropout(x)
        hidden_task = torch.einsum('btd,erd->bter', x_drop, module.A)
        per_expert = torch.einsum('bter,eor->bteo', hidden_task, module.B)
        # Ablate task-expert k (k in [0, E)) or universal expert (k == E).
        if 0 <= k < E:
            per_expert = per_expert.clone()
            per_expert[:, :, k, :] = 0.0
        # Cluster embedding lookup
        c_emb = module.cluster_embeddings(cluster_ids)             # (Bsz, ce_dim)
        gate_logits = module.gate(c_emb)                           # (Bsz, E)
        if module.training and module.noise_std > 0:
            noise = torch.randn_like(gate_logits) * module.noise_std
            gate_logits = gate_logits + noise
        gate_logits = gate_logits / max(module.temperature, 1e-6)
        gate_soft = F.softmax(gate_logits, dim=-1)                 # (Bsz, E)
        # Top-1 task expert per sample (Gou Eq. 1).
        argmax = gate_logits.argmax(dim=-1)                        # (Bsz,)
        task_w_top = gate_soft.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)  # (Bsz,)
        # Gather top-1 task expert's per_expert output: (Bsz, T, D_out)
        idx = argmax.view(Bsz, 1, 1, 1).expand(-1, T, 1, per_expert.size(-1))
        task_out = per_expert.gather(2, idx).squeeze(2)
        # Universal expert
        u_hidden = F.linear(x_drop, module.A_u)
        u_out = F.linear(u_hidden, module.B_u)                      # (Bsz, T, D_out)
        if k == E:  # ablate universal
            u_out = torch.zeros_like(u_out)
        # Mix: G_max · task + (1 − G_max) · universal (Gou Eq. 3)
        task_w = task_w_top.view(Bsz, 1, 1)
        merged = task_w * task_out + (1.0 - task_w) * u_out
        out = merged * module.scaling
        if orig_ndim == 2:
            out = out.squeeze(1)
        return out

    module._orig_forward = orig
    module.forward = _new_forward


def _unpatch_mocle_forward(module):
    if getattr(module, '_orig_forward', None) is not None:
        module.forward = module._orig_forward
        module._orig_forward = None


# ─── Module-class dispatch ────────────────────────────────────────────────

def _module_kind(module):
    """Returns the lowercase short-name of the multibranch class, or None."""
    cn = type(module).__name__
    if cn == 'ParallelFFN':       return 'parallel_ffn'
    if cn == 'SoftMoEFFN':        return 'soft_moe'
    if cn == 'HydraLoRAAdapter':  return 'hydra_lora'
    if cn == 'LoRAMoEAdapter':    return 'lora_moe'
    if cn == 'MoCLEAdapter':      return 'mocle'
    return None


def _module_K(module, kind):
    """Effective branch count for a multibranch module."""
    if kind == 'parallel_ffn':  return int(module.num_branches)
    if kind == 'soft_moe':      return int(module.num_experts)
    if kind == 'hydra_lora':    return int(module.num_B_heads)
    if kind == 'lora_moe':      return int(module.num_experts)
    if kind == 'mocle':         return int(module.num_task_experts)  # +1 for universal handled by caller
    return 0


def find_multibranch_modules(model):
    """Returns list of (module, kind, K) for all multibranch modules in model."""
    out = []
    for m in model.modules():
        k = _module_kind(m)
        if k is not None:
            out.append((m, k, _module_K(m, k)))
    return out


def model_branch_count(model):
    """Effective K for the whole model. Returns max K seen across multibranch
    modules (all should agree for a single-method model)."""
    found = find_multibranch_modules(model)
    if not found:
        return 0
    Ks = {K for _, _, K in found}
    return max(Ks)


def install_ablation(model, k: Optional[int]):
    """Install ablation hooks across all multibranch modules in `model`.

    For ParallelFFN modules (which back Switch / Hash / Demix / SMoE-Dropout
    / Mod-Squad), use a forward hook with closure reading `model._ablate_k`.
    For Soft MoE and LoRA adapters, monkey-patch the forward.

    `k`:
      - integer: zero expert/branch k's contribution
      - None: install hooks but disable ablation (matches no-ablation baseline)

    Returns a dict {module_id: handle} that can be passed to uninstall_ablation
    (or just call uninstall_ablation(model) to undo all).
    """
    # Cache k on the model so the closure can read it dynamically.
    model._ablate_k = k

    # Track installed (handle, module-with-monkey-patch) for cleanup.
    if not hasattr(model, '_ablation_handles'):
        model._ablation_handles = []
    if not hasattr(model, '_ablation_patched_modules'):
        model._ablation_patched_modules = []

    for m, kind, K in find_multibranch_modules(model):
        if kind == 'parallel_ffn':
            # Idempotent: skip if already hooked on this module.
            if getattr(m, '_ablation_hook_handle', None) is not None:
                continue
            hook = _make_parallel_ffn_hook(lambda mm=m: getattr(model, '_ablate_k', None))
            handle = m.register_forward_hook(hook)
            m._ablation_hook_handle = handle
            model._ablation_handles.append((m, handle))
        else:
            # Monkey-patch path
            m._ablate_branch_idx = k  # initial value; install_ablation may be called again to update
            if kind == 'soft_moe':
                _patch_softmoe_forward(m)
            elif kind == 'hydra_lora':
                _patch_hydra_lora_forward(m)
            elif kind == 'lora_moe':
                _patch_lora_moe_forward(m)
            elif kind == 'mocle':
                _patch_mocle_forward(m)
            if m not in model._ablation_patched_modules:
                model._ablation_patched_modules.append(m)

    # Update _ablate_branch_idx on monkey-patched modules each call (since
    # they read from this attribute, not from model._ablate_k).
    for m in model._ablation_patched_modules:
        m._ablate_branch_idx = k


def uninstall_ablation(model):
    """Remove all ablation hooks + restore monkey-patched forwards."""
    # Remove hooks (ParallelFFN)
    for m, handle in getattr(model, '_ablation_handles', []):
        try:
            handle.remove()
        except Exception:
            pass
        if hasattr(m, '_ablation_hook_handle'):
            del m._ablation_hook_handle
    model._ablation_handles = []

    # Restore monkey-patched forwards
    for m in getattr(model, '_ablation_patched_modules', []):
        cn = type(m).__name__
        if cn == 'SoftMoEFFN':       _unpatch_softmoe_forward(m)
        elif cn == 'HydraLoRAAdapter': _unpatch_hydra_lora_forward(m)
        elif cn == 'LoRAMoEAdapter':   _unpatch_lora_moe_forward(m)
        elif cn == 'MoCLEAdapter':     _unpatch_mocle_forward(m)
        if hasattr(m, '_ablate_branch_idx'):
            del m._ablate_branch_idx
    model._ablation_patched_modules = []
    if hasattr(model, '_ablate_k'):
        del model._ablate_k


@contextlib.contextmanager
def ablate_branch(model, k: Optional[int]):
    """Context manager: ablation installed on enter, removed on exit.

    Usage:
        with ablate_branch(model, k=3):
            out = model(...)
    """
    install_ablation(model, k)
    try:
        yield model
    finally:
        uninstall_ablation(model)

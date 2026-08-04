"""Per-method LoRA adapter classes for the post-train track.

Each class is a drop-in module accepted by LoRAInjectedLinear: it takes the
frozen linear's input `x` plus a method-specific `routing` kwarg and returns
the LoRA contribution (same shape as `base(x)`).

Adapters implemented here:
  - SingleLoRAAdapter       (Hu 2022 vanilla)
  - MultiBranchSoftSpecDropAdapter  (OURS + no-routing; uses (B, K) soft mask)
  - HydraLoRAAdapter        (Tian 2024 NeurIPS: 1 shared A + N B + learned gate)
  - LoRAMoEAdapter          (Dou 2023 ACL: K experts + softmax gate + balance loss)
  - MoCLEAdapter            (Gou 2024: E task experts + 1 universal,
                             cluster-conditional learned gate, top-1 over tasks)

All adapters zero-init their output (B matrix zeros) so the wrapped linear is
bit-identical to the frozen base at step 0 — critical for `test_lora_off`
assertions across methods.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers for A / B init and scaling ────────────────────────────────────

def _kaiming_A(tensor):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


# ══════════════════════════════════════════════════════════════════════════
# 1. SingleLoRAAdapter — Hu 2022 vanilla
# ══════════════════════════════════════════════════════════════════════════

class SingleLoRAAdapter(nn.Module):
    """Classic single LoRA: y_delta = (α/r) · B·A·x. Routing is ignored."""

    def __init__(self, in_features, out_features, rank, alpha=None, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else 2.0 * rank
        self.scaling = self.alpha / self.rank
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        _kaiming_A(self.A)

    def forward(self, x, routing=None):
        x = self.dropout(x)
        hidden = F.linear(x, self.A)
        out = F.linear(hidden, self.B)
        return out * self.scaling


# ══════════════════════════════════════════════════════════════════════════
# 2. MultiBranchSoftSpecDropAdapter — OURS (+ no-routing variant)
# ══════════════════════════════════════════════════════════════════════════

class MultiBranchSoftSpecDropAdapter(nn.Module):
    """K parallel LoRAs with fixed-denominator merge, driven by external mask.

    This is a thin wrapper around ParallelLoRAAdapter from models.multi_branch_lora
    with the adapter interface `(x, routing=<dict or (mask, mask_scale)>)`. The
    `routing` kwarg carries the algorithm-provided mask + mask_scale:

        routing = {'mask': (B, K) tensor, 'mask_scale': float}

    If `routing` is None (e.g., eval with no explicit routing installed), the
    adapter defaults to uniform 1/K routing — only used for smoke / debug.
    """

    def __init__(self, in_features, out_features, num_experts, rank,
                 alpha=None, dropout=0.0, shared_expert_rank=0):
        super().__init__()
        # Reuse ParallelLoRAAdapter's einsum math from Batch 1.
        from .multi_branch_lora import ParallelLoRAAdapter
        self.inner = ParallelLoRAAdapter(
            in_features=in_features, out_features=out_features,
            num_experts=num_experts, rank=rank, alpha=alpha,
            dropout=dropout, shared_expert_rank=shared_expert_rank,
        )
        self.num_experts = num_experts

    def forward(self, x, routing=None):
        if routing is not None:
            self.inner.mask_scale = routing.get('mask_scale', None)
            mask = routing.get('mask', None)
        else:
            mask = None
        return self.inner(x, branch_mask=mask)


# ══════════════════════════════════════════════════════════════════════════
# 3. HydraLoRAAdapter — Tian 2024 NeurIPS (asymmetric 1A + N B + learned gate)
# ══════════════════════════════════════════════════════════════════════════

class HydraLoRAAdapter(nn.Module):
    """Asymmetric LoRA: single shared A matrix + N distinct B matrices, merged
    via a learnable softmax gate (Eq. 2 of Tian et al. 2024 NeurIPS).

        y_delta = (α/r) · Σ_i softmax(W_g x)_i · B_i · A · x

    A is shared across experts, {B_i} are distinct. W_g is a trainable linear
    layer from input dim → N logits. Routing is ignored (gate is learned).
    """

    def __init__(self, in_features, out_features, num_B_heads, rank,
                 alpha=None, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_B_heads = num_B_heads
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else 2.0 * rank
        self.scaling = self.alpha / self.rank
        # Single shared A
        self.A = nn.Parameter(torch.empty(rank, in_features))
        # N B matrices, stacked for batched einsum
        self.B = nn.Parameter(torch.zeros(num_B_heads, out_features, rank))
        # Learnable gate over N heads
        self.gate = nn.Linear(in_features, num_B_heads, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        _kaiming_A(self.A)
        # Zero-init gate → softmax = uniform at step 0, combined with B=0 still
        # gives zero delta. Not strictly necessary but matches "LoRA off at init".
        nn.init.zeros_(self.gate.weight)

    def forward(self, x, routing=None):
        """
        x: (B, T, D_in) or (B, D_in).
        routing is ignored — gate is learned.
        """
        orig_ndim = x.ndim
        if orig_ndim == 2:
            x = x.unsqueeze(1)
        B, T, D_in = x.shape
        N = self.num_B_heads
        x_drop = self.dropout(x)
        # Shared A: (B, T, D_in) · (r, D_in) → (B, T, r)
        hidden = F.linear(x_drop, self.A)
        # N B matrices: (B, T, r) · (N, D_out, r) → (B, T, N, D_out)
        per_head = torch.einsum('btr,nor->btno', hidden, self.B)
        # Learned gate: (B, T, D_in) → (B, T, N), softmax over N
        gate_logits = self.gate(x_drop)
        gate_w = F.softmax(gate_logits, dim=-1)
        # Merge: (B, T, N, D_out) × (B, T, N, 1) → sum → (B, T, D_out)
        merged = (per_head * gate_w.unsqueeze(-1)).sum(dim=2)
        out = merged * self.scaling
        if orig_ndim == 2:
            out = out.squeeze(1)
        return out


# ══════════════════════════════════════════════════════════════════════════
# 4. LoRAMoEAdapter — Dou 2023 ACL (K experts, softmax gate, balance loss)
# ══════════════════════════════════════════════════════════════════════════

class LoRAMoEAdapter(nn.Module):
    """K independent LoRA experts with a softmax gate and localized balancing
    constraint (Dou 2023 Eq. 5 + Eq. 8).

        y_delta = (α/r) · Σ_i ω_i · B_i A_i x,   ω = softmax(x W_g^T / τ)

    Balance loss = σ²(Z) / μ(Z) where Z_{n,m} is the gate-weight expectation
    matrix. We expose the balance loss via `self.last_aux_loss` for the
    trainer to aggregate — LoRAMoE's loss is CE + β · ℒ_lbc with β=0.1.

    Native paper setup attaches to FFN-only (not attention); this is enforced
    at the MODEL level (lora_moe_model.py passes FFN targets only), not here.
    """

    def __init__(self, in_features, out_features, num_experts, rank,
                 alpha=None, dropout=0.0, balance_tau=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else 2.0 * rank
        self.scaling = self.alpha / self.rank
        self.balance_tau = balance_tau

        self.A = nn.Parameter(torch.empty(num_experts, rank, in_features))
        self.B = nn.Parameter(torch.zeros(num_experts, out_features, rank))
        self.gate = nn.Linear(in_features, num_experts, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for k in range(num_experts):
            _kaiming_A(self.A[k])
        nn.init.zeros_(self.gate.weight)

        # last_aux_loss is set on every forward pass; trainer aggregates via
        # models.lora_base.sum_aux_losses().
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(self, x, routing=None):
        orig_ndim = x.ndim
        if orig_ndim == 2:
            x = x.unsqueeze(1)
        Bsz, T, D_in = x.shape
        K = self.num_experts
        x_drop = self.dropout(x)
        # Per-expert LoRA: (Bsz, T, D_in) · (K, r, D_in) → (Bsz, T, K, r)
        hidden = torch.einsum('btd,krd->btkr', x_drop, self.A)
        # (Bsz, T, K, r) · (K, D_out, r) → (Bsz, T, K, D_out)
        per_expert = torch.einsum('btkr,kor->btko', hidden, self.B)
        # Gate: (Bsz, T, D_in) → (Bsz, T, K)
        gate_w = F.softmax(self.gate(x_drop) / self.balance_tau, dim=-1)
        # Merge
        merged = (per_expert * gate_w.unsqueeze(-1)).sum(dim=2)
        out = merged * self.scaling
        # Balance loss (coefficient of variation of gate usage across tokens),
        # Dou 2023 Eq. 8.
        gate_mean = gate_w.mean(dim=(0, 1))  # (K,)
        var = gate_mean.var(unbiased=False)
        mean = gate_mean.mean()
        self.last_aux_loss = var / (mean.square() + 1e-8)
        if orig_ndim == 2:
            out = out.squeeze(1)
        return out


# ══════════════════════════════════════════════════════════════════════════
# 5. MoCLEAdapter — Gou 2024 (E task experts + 1 universal + cluster-gate)
# ══════════════════════════════════════════════════════════════════════════

class MoCLEAdapter(nn.Module):
    """E task LoRA experts + 1 always-on universal expert, with a cluster-
    conditional gate that takes a per-sample cluster embedding as input.

    Forward (Gou 2024 Eq. 3):
        y_delta = (α/r_task) · [Σ_e G_e · B_e A_e x  +  (1 - G_max) · B_u A_u x]

    where G is top-1 softmax over E task experts (input = cluster embedding +
    noise, Eq. 1 in Gou 2024), and G_max is the max gate weight (universal
    expert gets the remainder). Note τ = 0.05 default.

    The cluster embedding is looked up from a learnable embedding table C
    (shape (num_clusters, embed_dim)) — each cluster_id maps to its own
    vector. C is initialized to small random values; Gou 2024 initializes
    to the BGE-large centroid of the cluster, which we can plug in via
    `.init_cluster_embeddings(centroids)`.

    Routing expected as a dict: {'cluster_id': (B,) long tensor}.
    """

    def __init__(self, in_features, out_features, num_task_experts, rank,
                 num_clusters, cluster_embed_dim=None, alpha=None,
                 dropout=0.0, temperature=0.05, noise_std=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_task_experts = num_task_experts
        self.num_clusters = num_clusters
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else 2.0 * rank
        self.scaling = self.alpha / self.rank
        self.temperature = temperature
        # Gou 2024 Eq. 1 default noise std is 1/E, which provides exploration
        # during training without overwhelming the gate signal. Previous
        # default (0.0) was too conservative — combined with zero gate init
        # it forced all samples to route to expert 0 at step 0 (dead routing).
        self.noise_std = noise_std if noise_std is not None else 1.0 / num_task_experts

        # Cluster embedding table — one learnable vector per cluster. When
        # `.init_cluster_embeddings(BGE_centroids)` is subsequently called
        # (run_lora.py does this for MoCLE), this is overwritten with BGE-
        # large sentence embeddings of per-cluster train-task Definitions
        # (Gou 2024 Sec 3.3). Fallback init below is only the no-BGE path.
        self.cluster_embed_dim = cluster_embed_dim or in_features
        self.cluster_embeddings = nn.Embedding(num_clusters, self.cluster_embed_dim)
        nn.init.normal_(self.cluster_embeddings.weight, std=0.02)

        # Gate: cluster_embedding → E task-expert logits. Non-zero init is
        # critical: with zero gate + (possibly zero) noise, all samples get
        # identical zero logits and torch.argmax returns index 0 for every
        # sample → dead routing to expert 0 only.
        self.gate = nn.Linear(self.cluster_embed_dim, num_task_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)

        # E task experts: stacked A, B
        self.A = nn.Parameter(torch.empty(num_task_experts, rank, in_features))
        self.B = nn.Parameter(torch.zeros(num_task_experts, out_features, rank))
        # Universal expert (single LoRA)
        self.A_u = nn.Parameter(torch.empty(rank, in_features))
        self.B_u = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for e in range(num_task_experts):
            _kaiming_A(self.A[e])
        _kaiming_A(self.A_u)

    def init_cluster_embeddings(self, centroids: torch.Tensor):
        """Initialize cluster_embeddings to BGE-large centroids (Gou 2024).

        centroids: (num_clusters, cluster_embed_dim) tensor.
        """
        with torch.no_grad():
            self.cluster_embeddings.weight.copy_(centroids)

    def forward(self, x, routing=None):
        """
        x: (B, T, D_in).
        routing: {'cluster_id': (B,) long tensor}.
        """
        orig_ndim = x.ndim
        if orig_ndim == 2:
            x = x.unsqueeze(1)
        Bsz, T, D_in = x.shape
        E = self.num_task_experts

        if routing is None or routing.get('cluster_id', None) is None:
            # Fallback: uniform across task experts (debug/smoke).
            cluster_ids = torch.zeros(Bsz, dtype=torch.long, device=x.device)
        else:
            cluster_ids = routing['cluster_id']

        x_drop = self.dropout(x)
        # Task-expert LoRA: (B, T, D_in) · (E, r, D_in) → (B, T, E, r)
        hidden_task = torch.einsum('btd,erd->bter', x_drop, self.A)
        per_expert = torch.einsum('bter,eor->bteo', hidden_task, self.B)

        # Gate: cluster_embed(cluster_id) → (B, cluster_embed_dim)
        cluster_emb = self.cluster_embeddings(cluster_ids)  # (B, D_ce)
        if self.noise_std > 0 and self.training:
            cluster_emb = cluster_emb + torch.randn_like(cluster_emb) * self.noise_std
        gate_logits = self.gate(cluster_emb) / self.temperature  # (B, E)
        gate_w = F.softmax(gate_logits, dim=-1)  # (B, E)
        # top-1 gate (keep only argmax, set others to 0): Gou 2024 Eq. 1.
        max_w, max_idx = gate_w.max(dim=-1, keepdim=True)  # (B, 1)
        gate_top1 = torch.zeros_like(gate_w).scatter_(-1, max_idx, max_w)
        # (B, 1, E, 1) × (B, T, E, D_out) → sum → (B, T, D_out)
        gate_bcast = gate_top1.view(Bsz, 1, E, 1)
        task_out = (per_expert * gate_bcast).sum(dim=2)

        # Universal expert: always-on, weight = 1 - G_max
        hidden_u = F.linear(x_drop, self.A_u)
        u_out = F.linear(hidden_u, self.B_u)
        u_weight = (1.0 - max_w).view(Bsz, 1, 1)  # (B, 1, 1)
        u_contribution = u_out * u_weight

        out = (task_out + u_contribution) * self.scaling
        if orig_ndim == 2:
            out = out.squeeze(1)
        return out

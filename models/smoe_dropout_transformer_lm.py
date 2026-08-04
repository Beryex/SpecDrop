"""SMoE-Dropout Transformer LM (Chen et al., ICLR 2023).

Faithful implementation of per-token fixed-random routing with linear k-schedule
("self-slimmable" MoE) and optional stochastic expert dropout.

Reference:
    Chen et al., "Sparse MoE as the New Dropout: Scaling Dense and Self-Slimmable
    Transformers", ICLR 2023. https://arxiv.org/abs/2303.01610

Core mechanism (Section 3.2):
    1. **Modulization**: split dense FFN (hidden H) into N small experts, each with
       hidden H/N (so total capacity ≈ dense MLP).
    2. **Random routing**: a fixed randomly-initialized `Linear(D, N)` router is
       frozen at init. For each token, compute logits, pick top-k experts, softmax
       over top-k weights.
    3. **Linear k-schedule (Chen 2023 §3.2)**: k grows from k_init=1 → N across
       training using
           k(t) = k_init + (N - k_init) · (t / (T - 1))
       so the final epoch reaches k = N. At inference k = N always (all experts
       active, full capacity, "self-slimmable"). An earlier version of this file
       incorrectly documented a cosine schedule — corrected to match paper and
       `update_k` implementation below.
    4. **Expert dropout** (optional ablation, NOT in original paper): disabled by
       default (`expert_drop_prob=0`). Paper has no expert dropout.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_lm import _causal_mask, ParallelFFN


class SMoEDropoutFFN(nn.Module):
    """Per-token fixed-random routed MoE with linear k-schedule + expert dropout.

    Efficient implementation: compute all N experts in parallel via ParallelFFN
    and gather the top-k experts' outputs per token, then weighted-sum. Same
    total FLOPs as a dense FFN (param-matched) but one GPU kernel per block
    instead of N × k.
    """

    def __init__(self, hidden_dim, ffn_dim_per_expert, num_experts,
                 k_init=1, expert_drop_prob=0.0, router_seed=42, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.k_init = k_init
        self.k_current = k_init              # updated via set_k()
        self.expert_drop_prob = expert_drop_prob

        # Fixed random router — weights frozen at init.
        router = nn.Linear(hidden_dim, num_experts, bias=False)
        g = torch.Generator().manual_seed(router_seed)
        with torch.no_grad():
            router.weight.copy_(
                torch.randn(num_experts, hidden_dim, generator=g) * 0.02)
        for p in router.parameters():
            p.requires_grad_(False)
        self.router = router

        self.experts = ParallelFFN(hidden_dim, ffn_dim_per_expert,
                                    num_experts, dropout)

    def set_k(self, k):
        """Update current top-k (called externally by scheduler)."""
        self.k_current = max(1, min(int(k), self.num_experts))

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            y: (B, T, D) — sum of top-k experts weighted by softmax-over-top-k.
        """
        B, T, D = x.shape
        N = self.num_experts
        k = self.k_current if self.training else N   # inference: full capacity

        x_flat = x.reshape(B * T, D)
        with torch.no_grad():
            logits = self.router(x_flat)             # (BT, N), no grad through router
        top_vals, top_idx = logits.topk(k, dim=-1)   # (BT, k), (BT, k)
        top_w = F.softmax(top_vals, dim=-1)          # weights over selected experts

        # Per-expert stochastic dropout (training only; NOT in original paper).
        if self.training and self.expert_drop_prob > 0.0 and k > 1:
            keep = torch.bernoulli(
                torch.full((N,), 1.0 - self.expert_drop_prob, device=x.device))
            keep_for_token = keep[top_idx]           # (BT, k)
            top_w = top_w * keep_for_token
            denom = top_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            top_w = top_w / denom

        # Compute all N experts in parallel, then gather top-k per token and
        # weighted-sum. Equivalent to the for-loop dispatch but one kernel.
        all_out = self.experts(x)                                 # (B, T, N, D)
        all_out_flat = all_out.reshape(B * T, N, D)               # (BT, N, D)
        idx = top_idx.unsqueeze(-1).expand(-1, -1, D)             # (BT, k, D)
        selected = all_out_flat.gather(dim=1, index=idx)          # (BT, k, D)
        top_w = top_w.to(selected.dtype).unsqueeze(-1)            # (BT, k, 1)
        out = (selected * top_w).sum(dim=1)                       # (BT, D)
        return out.reshape(B, T, D)


class SMoEDropoutBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_experts, ffn_dim_per_expert,
                 k_init, expert_drop_prob=0.0, router_seed=42, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.moe = SMoEDropoutFFN(hidden_dim, ffn_dim_per_expert, num_experts,
                                  k_init, expert_drop_prob, router_seed, dropout)

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.drop(h)
        x = x + self.moe(self.ln2(x))
        return x


class SMoEDropoutTransformerLM(nn.Module):
    """SMoE-Dropout LM (Chen 2023) for SlimPajama.

    Exposes `update_k(epoch, total_epochs)` to be called once per epoch so each
    block's top-k follows the cosine schedule.
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads,
                 num_experts, ffn_dim_per_expert, max_seq_len,
                 k_init=1, expert_drop_prob=0.0, router_seed=42, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k_init = k_init

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SMoEDropoutBlock(hidden_dim, num_heads, num_experts, ffn_dim_per_expert,
                             k_init, expert_drop_prob, router_seed + layer_idx,
                             dropout)
            for layer_idx in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def update_k(self, epoch, total_epochs):
        """Linear schedule: k_init → N across total_epochs (Chen 2023 §3.2).

        Paper specifies k(t) progressively reaches N by the end of training.
        Divide by (total_epochs - 1) so the FINAL epoch (t = total_epochs - 1)
        has progress = 1 → k = N. Previously we divided by total_epochs, which
        left k < N at the last epoch while eval uses k = N (train/test mismatch).
        Called once per epoch at train-time; inference always uses k=N.
        """
        t = min(max(epoch, 0), total_epochs - 1)
        progress = t / max(total_epochs - 1, 1)
        k = self.k_init + (self.num_experts - self.k_init) * progress
        for block in self.blocks:
            block.moe.set_k(round(k))

    def forward(self, input_ids):
        """
        Returns:
            logits: (B, T, vocab_size)
            aux_loss: 0 (SMoE-Dropout has no auxiliary loss).
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        attn_mask = _causal_mask(T, input_ids.device)

        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, torch.tensor(0.0, device=input_ids.device)

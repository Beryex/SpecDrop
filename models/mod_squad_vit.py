"""Mod-Squad ViT (Chen et al., CVPR 2023).

Faithful implementation of the **MoE MLP** branch of Mod-Squad on a ViT backbone
(MoE attention omitted by design; see paper Section 3.2 Figure 2 — we keep Attention
shared for this benchmark).

Reference:
    Chen et al., "Mod-Squad: Designing Mixtures of Experts As Modular Multi-Task
    Learners", CVPR 2023. https://arxiv.org/abs/2212.08066

Routing (Section 3.2, Eq. 2, noisy top-k):
    logits = x @ W_g + N(0, 1) ⊙ softplus(x @ W_noise)
    top_k(logits) → keep top-k, zero the rest
    weights = softmax(top_k_logits)

MI auxiliary loss (Section 3.4, Eq. 5-6):
    P(E_i, T_j) = (1/|T_j|) Σ_{x ∈ T_j} softmax(logits(x))_i
    I(T; E) = Σ_{i,j} P(T_j, E_i) log[P(T_j, E_i) / (P(T_j) P(E_i))]
    L = L_CE − w_MI · I(T; E)     (maximize I)

For single-task (ImageNet) we use per-sample routing (analogous to per-task in
multi-task) — router operates per input token in the ViT MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import PatchEmbed, Attention, DropPath, _trunc_normal_


class ModSquadFFN(nn.Module):
    """Per-token noisy top-k MoE FFN with MI loss (Chen 2023)."""

    def __init__(self, dim, expert_hidden, num_experts, top_k=2,
                 noise_std=None, mi_weight=0.001, drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        # `noise_std` is a BOOLEAN ENABLE flag (>0 means "apply learned noise";
        # 0 disables it). The actual noise magnitude is learned per-expert
        # via the noise_gate softplus at forward time — this matches Mod-Squad
        # paper Eq. 2 where noise std is σ(W_noise·x). Default 1/N is kept
        # for backward-compat but any positive value gives the same behaviour.
        self.noise_std = (1.0 / num_experts) if noise_std is None else noise_std
        self.mi_weight = mi_weight

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.noise_gate = nn.Linear(dim, num_experts, bias=False)
        # Efficient dense-parallel expert dispatch: compute all N experts via a
        # single batched einsum (ParallelFFN), then gather top-k outputs per
        # token and weighted-sum. With the param-matched setting N×hidden ≈
        # dense MLP width, total FLOPs equal a dense ViT MLP but we fire one
        # kernel per block instead of k×N (ViT-S: top-2 × 16 × 12 = 384 tiny
        # launches per step under the old for-loop dispatch).
        from .transformer_lm import ParallelFFN
        self.experts = ParallelFFN(dim, expert_hidden, num_experts, drop)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, T, D = x.shape
        N = self.num_experts
        k = self.top_k

        x_flat = x.reshape(B * T, D)
        clean_logits = self.gate(x_flat)                     # (BT, N)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(clean_logits) * F.softplus(self.noise_gate(x_flat))
            logits = clean_logits + noise
        else:
            logits = clean_logits
        probs_full = F.softmax(logits, dim=-1)               # (BT, N)  — MI uses this

        top_vals, top_idx = logits.topk(k, dim=-1)           # (BT, k)
        top_w = F.softmax(top_vals, dim=-1)                  # softmax over top-k

        # MI loss over (sample, expert) pairs using the full softmax distribution.
        # For single-task, treat each sample as its own task. We estimate P(E|T) via
        # the per-sample routing probabilities.
        if self.training and self.mi_weight > 0:
            P_E_given_T = probs_full                         # (BT, N), row-stochastic
            P_E = P_E_given_T.mean(dim=0)                     # (N,)
            H_E = -(P_E * (P_E + 1e-12).log()).sum()
            H_E_given_T = -(P_E_given_T * (P_E_given_T + 1e-12).log()).sum(dim=-1).mean()
            mi = H_E - H_E_given_T
            self.aux_loss = -self.mi_weight * mi              # maximize MI → negate
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device)

        # Compute all N experts in parallel (single einsum), then gather the
        # top-k experts per token and weighted-sum. Non-selected experts are
        # multiplied by 0 via gather → no CE gradient to them, matching the
        # for-loop semantics bit-for-bit (up to fp precision).
        all_out = self.experts(x)                             # (B, T, N, D)
        all_out_flat = all_out.reshape(B * T, N, D)           # (BT, N, D)
        idx = top_idx.unsqueeze(-1).expand(-1, -1, D)         # (BT, k, D)
        selected = all_out_flat.gather(dim=1, index=idx)      # (BT, k, D)
        top_w = top_w.to(selected.dtype).unsqueeze(-1)        # (BT, k, 1)
        out = (selected * top_w).sum(dim=1)                   # (BT, D)
        return out.reshape(B, T, D)


class _DenseFFN(nn.Module):
    """Standard FFN: Linear → GELU → Linear."""

    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class ModSquadBlock(nn.Module):
    """Pre-norm ViT block with shared MHSA + Mod-Squad MoE MLP."""

    def __init__(self, dim, num_heads, expert_hidden, num_experts,
                 top_k=2, mi_weight=0.001, noise_std=None,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.moe = ModSquadFFN(dim, expert_hidden, num_experts,
                                top_k=top_k, noise_std=noise_std,
                                mi_weight=mi_weight, drop=drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.moe(self.norm2(x)))
        return x


class ModSquadViT(nn.Module):
    """Mod-Squad ViT (Chen 2023) with MoE MLPs, for ImageNet-1K.

    Args:
        num_experts: N (default 16 per paper Section 4).
        expert_hidden: per-expert FFN hidden dim. For param-match to dense ViT-S/16
            (MLP hidden=1536), set ≈ 1536/N so that N·expert_hidden ≈ 1536.
        top_k: experts selected per token (default 2).
        mi_weight: w_MI in L = L_CE − w_MI · I(T;E) (default 0.001).
        noise_std: Gaussian noise scale in noisy top-k gating.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=384, depth=12, num_heads=6,
                 num_experts=16, expert_hidden=None, top_k=2,
                 mi_weight=0.001, noise_std=None, qkv_bias=True,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        if expert_hidden is None:
            # Param-match: N · h ≈ 4·embed_dim (dense MLP hidden).
            expert_hidden = max(8, round(4 * embed_dim / num_experts))
        self.expert_hidden = expert_hidden

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            ModSquadBlock(embed_dim, num_heads, expert_hidden, num_experts,
                          top_k=top_k, mi_weight=mi_weight, noise_std=noise_std,
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
        """
        Returns:
            logits: (B, num_classes)
            aux_loss: aggregated negative MI across all blocks (scalar tensor).
        """
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x[:, 0])

        aux_loss = sum(blk.moe.aux_loss for blk in self.blocks)
        return logits, aux_loss

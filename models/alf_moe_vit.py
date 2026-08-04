"""Auxiliary-Loss-Free (ALF) top-k MoE ViT (Wang et al., 2024).

Additional baseline: load balancing WITHOUT any
auxiliary loss, via a per-expert bias added to the routing scores only.

Reference:
    Wang et al., "Auxiliary-Loss-Free Load Balancing Strategy for
    Mixture-of-Experts", 2024. https://arxiv.org/abs/2408.15664

This file is a deliberate clone of `models/mod_squad_vit.py` (same expert FFN
structure, same batched ParallelFFN dispatch, same block insertion points,
same drop_path / init) so that the {ModSquadViT, ALFMoEViT} pair isolates the
load-balancing mechanism:

    Mod-Squad:  noisy top-k gating (softmax weights) + MI auxiliary loss
    ALF:        sigmoid gating, bias-corrected top-k SELECTION, NO aux loss

ALF routing per token:
    s   = sigmoid(x @ W_g)                    # gate scores, (BT, N)
    idx = top_k(s + b)                        # SELECTION uses bias b
    w   = s[idx] / sum(s[idx])                # COMBINE uses s ONLY (renormalized)
    out = sum_j w_j * Expert_idx_j(x)

Bias update (training forward only, no gradient):
    load_i    = fraction of tokens in the batch that selected expert i
    b_i      += u * sign(mean_load - load_i),   u = 1e-3
    (bias rises for starved experts, falls for overloaded ones)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import PatchEmbed, Attention, DropPath, _trunc_normal_


class ALFMoEFFN(nn.Module):
    """Per-token top-k MoE FFN with auxiliary-loss-free balancing (Wang 2024)."""

    def __init__(self, dim, expert_hidden, num_experts, top_k=2,
                 bias_update_rate=1e-3, drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        # u in Wang et al. 2024: per-step magnitude of the sign update on the
        # expert-wise bias. Default 1e-3 per the paper's described mechanism.
        self.bias_update_rate = bias_update_rate

        # Gate: s = sigmoid(Linear(d, N)). bias=False mirrors ModSquadFFN's
        # gate so the pair differs only in the balancing mechanism.
        self.gate = nn.Linear(dim, num_experts, bias=False)
        # Per-expert routing bias b (N,): a BUFFER, not a Parameter — it is
        # updated by the sign rule below, never by gradient descent, and it
        # only affects expert SELECTION, never the combine weights.
        self.register_buffer('expert_bias', torch.zeros(num_experts))

        # Efficient dense-parallel expert dispatch — identical to ModSquadFFN:
        # compute all N experts via a single batched einsum (ParallelFFN),
        # gather top-k outputs per token, weighted-sum.
        from .transformer_lm import ParallelFFN
        self.experts = ParallelFFN(dim, expert_hidden, num_experts, drop)
        # No auxiliary loss by construction; keep the attribute so the ViT
        # wrapper (and trainer) interface matches ModSquadViT exactly.
        self.aux_loss = torch.tensor(0.0)
        # Monitoring only (not part of state_dict): per-expert token-selection
        # fraction from the most recent forward. Used by tests / diagnostics.
        self.last_load = None

    def forward(self, x):
        B, T, D = x.shape
        N = self.num_experts
        k = self.top_k

        x_flat = x.reshape(B * T, D)
        s = torch.sigmoid(self.gate(x_flat))                 # (BT, N)

        # Top-k SELECTION on the bias-corrected scores (s + b). The bias b is
        # detached by construction (buffer + no_grad update), so no gradient
        # ever flows through it.
        biased_scores = s + self.expert_bias                 # (BT, N)
        _, top_idx = biased_scores.topk(k, dim=-1)           # (BT, k)

        # COMBINE weights use s ONLY, renormalized over the selected k.
        # Design choice (documented per planning review): Wang et al. 2024 use
        # the bias exclusively for routing/selection; the actual mixture
        # weights come from the raw gate scores of the chosen experts. With a
        # sigmoid gate the selected scores do not sum to 1, so we renormalize
        # over the k selected experts (as in DeepSeek-V3-style sigmoid
        # gating) to keep the output scale comparable to Mod-Squad's
        # softmax-over-top-k weights.
        s_top = s.gather(dim=-1, index=top_idx)              # (BT, k)
        top_w = s_top / (s_top.sum(dim=-1, keepdim=True) + 1e-9)

        # Auxiliary-loss-free bias update — training forward only, no grad.
        # Implemented from Wang et al. 2024's described mechanism
        # (sequence-wise sign update).
        with torch.no_grad():
            one_hot = F.one_hot(top_idx, N)                  # (BT, k, N)
            load = one_hot.sum(dim=(0, 1)).float() / x_flat.shape[0]  # (N,)
            self.last_load = load.detach()
            if self.training:
                mean_load = load.mean()                      # = k / N
                self.expert_bias += (
                    self.bias_update_rate
                    * torch.sign(mean_load - load).to(self.expert_bias.dtype)
                )

        # No auxiliary loss — that is the point of this baseline.
        self.aux_loss = torch.tensor(0.0, device=x.device)

        # Dispatch identical to ModSquadFFN: all N experts in one einsum,
        # gather top-k per token, weighted-sum. Non-selected experts get
        # weight 0 via gather → no CE gradient to them.
        all_out = self.experts(x)                             # (B, T, N, D)
        all_out_flat = all_out.reshape(B * T, N, D)           # (BT, N, D)
        idx = top_idx.unsqueeze(-1).expand(-1, -1, D)         # (BT, k, D)
        selected = all_out_flat.gather(dim=1, index=idx)      # (BT, k, D)
        top_w = top_w.to(selected.dtype).unsqueeze(-1)        # (BT, k, 1)
        out = (selected * top_w).sum(dim=1)                   # (BT, D)
        return out.reshape(B, T, D)


class ALFMoEBlock(nn.Module):
    """Pre-norm ViT block with shared MHSA + ALF MoE MLP.

    Identical to ModSquadBlock except the MoE FFN (no MI loss, no noise)."""

    def __init__(self, dim, num_heads, expert_hidden, num_experts,
                 top_k=2, bias_update_rate=1e-3,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.moe = ALFMoEFFN(dim, expert_hidden, num_experts,
                             top_k=top_k, bias_update_rate=bias_update_rate,
                             drop=drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.moe(self.norm2(x)))
        return x


class ALFMoEViT(nn.Module):
    """Auxiliary-Loss-Free MoE ViT (Wang 2024) for ImageNet-1K.

    Args:
        num_experts: N (default 16, matched to the Mod-Squad baseline).
        expert_hidden: per-expert FFN hidden dim. For param-match to dense
            ViT-S/16 (MLP hidden=1536), set ≈ 1536/N so N·expert_hidden ≈ 1536.
        top_k: experts selected per token (default 2).
        bias_update_rate: u in the ALF sign update (default 1e-3).
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=384, depth=12, num_heads=6,
                 num_experts=16, expert_hidden=None, top_k=2,
                 bias_update_rate=1e-3, qkv_bias=True,
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
            ALFMoEBlock(embed_dim, num_heads, expert_hidden, num_experts,
                        top_k=top_k, bias_update_rate=bias_update_rate,
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
            aux_loss: always 0.0 (scalar tensor) — kept for trainer interface
                parity with ModSquadViT. ALF has no auxiliary loss.
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

"""Soft MoE ViT (Puigcerver et al., ICLR 2024).

Faithful implementation of the Soft MoE layer (Algorithm 1) on a ViT backbone.
Parameter-matched to dense ViT-S/16 via narrow experts.

Reference:
    Puigcerver et al., "From Sparse to Soft Mixtures of Experts", ICLR 2024.
    https://arxiv.org/abs/2308.00951

Soft MoE layer (Algorithm 1):
    Given token sequence X ∈ R^(T × D), N experts with S slots each (total NS slots).
    Φ ∈ R^(D × N × S): learned slot parameters.
        D = softmax(X Φ, axis=0)   ∈ R^(T × N × S)   (softmax OVER tokens, for each slot)
        X̃ = D^T X                   ∈ R^(N × S × D)   (per-slot mixture of tokens)
        Ỹ_{n,s} = E_n( X̃_{n,s} )    per-expert FFN applied to that expert's S slots
        C = softmax(X Φ, axis=(1,2)) ∈ R^(T × N × S)  (softmax OVER slots, for each token)
        Y = sum_{n,s} C_{t,n,s} Ỹ_{n,s}  ∈ R^(T × D)
    No top-k, no discrete routing, no load-balance loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import Block, PatchEmbed, Attention, DropPath, _trunc_normal_


class SoftMoEFFN(nn.Module):
    """Soft MoE layer (Puigcerver 2024 Algorithm 1)."""

    def __init__(self, dim, expert_hidden, num_experts, slots_per_expert=1,
                 drop=0.0, normalize=True, cls_dense_hidden=None):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize

        # Φ ∈ R^(D × N × S): slot parameters (learned).
        self.phi = nn.Parameter(torch.empty(dim, num_experts, slots_per_expert))
        _trunc_normal_(self.phi, std=0.02)

        # Puigcerver §2.3 stability fix for ViT-S and larger: L2-normalize both
        # X and Φ independently, then scale by a learnable scalar (init 1).
        if normalize:
            self.scale = nn.Parameter(torch.ones(1))

        # NOTE (Puigcerver adaptation disclosed in Appendix A): the original
        # Soft MoE paper does not explicitly specify how the CLS token is routed.
        # Our implementation puts CLS through the same dispatch/combine as patch
        # tokens. Adding a dedicated dense FFN for CLS would double the MLP param
        # budget and break the param-matched comparison against dense ViT-S.

        # N experts, each a standard FFN.
        self.experts_fc1 = nn.Parameter(
            torch.empty(num_experts, expert_hidden, dim))
        self.experts_b1 = nn.Parameter(torch.zeros(num_experts, expert_hidden))
        self.experts_fc2 = nn.Parameter(
            torch.empty(num_experts, dim, expert_hidden))
        self.experts_b2 = nn.Parameter(torch.zeros(num_experts, dim))
        _trunc_normal_(self.experts_fc1, std=0.02)
        _trunc_normal_(self.experts_fc2, std=0.02)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            y: (B, T, D)
        """
        B, T, D = x.shape
        N, S = self.num_experts, self.slots_per_expert

        # Puigcerver §2.3: normalize both X (tokens) and Φ (slots) to unit L2.
        if self.normalize:
            x_n = F.normalize(x, dim=-1)
            phi_n = F.normalize(self.phi, dim=0)
            logits = self.scale * torch.einsum('btd,dns->btns', x_n, phi_n)
        else:
            logits = torch.einsum('btd,dns->btns', x, self.phi)

        # Dispatch weights: softmax over tokens for each (N, S). All tokens
        # (including CLS) participate in dispatch; see docstring note above.
        D_weights = F.softmax(logits, dim=1)                   # (B, T, N, S)
        X_tilde = torch.einsum('btns,btd->bnsd', D_weights, x)

        hidden = torch.einsum('bnsd,nhd->bnsh', X_tilde, self.experts_fc1) \
                 + self.experts_b1.view(1, N, 1, -1)
        hidden = F.gelu(hidden)
        Y_tilde = torch.einsum('bnsh,ndh->bnsd', hidden, self.experts_fc2) \
                  + self.experts_b2.view(1, N, 1, -1)

        # Combine weights: softmax over (N, S) for each token.
        C_weights = F.softmax(logits.reshape(B, T, N * S), dim=-1).reshape(B, T, N, S)
        y = torch.einsum('btns,bnsd->btd', C_weights, Y_tilde)
        return self.drop(y)


class SoftMoEBlock(nn.Module):
    """Pre-norm ViT block with shared MHSA + Soft MoE FFN."""

    def __init__(self, dim, num_heads, expert_hidden, num_experts,
                 slots_per_expert=1, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.moe = SoftMoEFFN(dim, expert_hidden, num_experts,
                              slots_per_expert=slots_per_expert, drop=drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.moe(self.norm2(x)))
        return x


class SoftMoEViT(nn.Module):
    """Soft MoE ViT (Puigcerver 2024) for ImageNet-1K.

    Args:
        num_experts: N (paper Section 4 uses N ∈ {32, 128, …}).
        expert_hidden: per-expert FFN hidden. For param-match to dense ViT-S/16 MLP
            (H=1536), set ≈ 1536/N (ignoring small Φ and bias overhead).
        slots_per_expert: S (paper uses 1 for ImageNet classification).
        moe_start_block: blocks with index < moe_start_block are standard dense
            ViT blocks (full-width 4× MLP, models.vit.Block); blocks >= are
            SoftMoEBlocks. Default 0 (all blocks MoE) is bit-identical to the
            pre-existing behavior and state-dict compatible. Puigcerver 2024
            canonically replaces only the second half of the blocks
            (moe_start_block = depth // 2).
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=384, depth=12, num_heads=6,
                 num_experts=32, expert_hidden=None, slots_per_expert=1,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, moe_start_block=0):
        super().__init__()
        assert 0 <= moe_start_block <= depth, \
            f"moe_start_block {moe_start_block} must be in [0, depth={depth}]"
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.moe_start_block = moe_start_block

        if expert_hidden is None:
            expert_hidden = max(8, round(4 * embed_dim / num_experts))
        self.expert_hidden = expert_hidden

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # Blocks [0, moe_start_block) are standard dense ViT blocks (full-width
        # 4× MLP); blocks [moe_start_block, depth) are Soft MoE blocks. With the
        # default moe_start_block=0, this constructs exactly the same module
        # sequence (same RNG consumption, same state-dict keys) as before the
        # parameter existed.
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            if i < moe_start_block else
            SoftMoEBlock(embed_dim, num_heads, expert_hidden, num_experts,
                         slots_per_expert=slots_per_expert,
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
            aux_loss: 0 (Soft MoE has no auxiliary loss — fully soft routing).
        """
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits, torch.tensor(0.0, device=x.device)

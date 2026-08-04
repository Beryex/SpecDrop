"""COMET ViT (Shaier et al., ICLR 2025).

Faithful implementation of Conditionally-Overlapping Experts via fixed random
projection + k-winner-take-all neuron masking, applied to each ViT MLP's hidden
layer (the 4D-dim expansion). NO independent experts — "overlapping experts" are
implicit subsets of the shared backbone's own hidden neurons.

Reference:
    Shaier et al., "More Experts Than Galaxies: Conditionally-overlapping Experts
    With Biologically-Inspired Fixed Routing", ICLR 2025.
    https://arxiv.org/abs/2410.08003

Mechanism (Eq. 4-8):
    Backbone MLP (dense):
        a_1 = W_1 x_0       x_1 = f(a_1)       (fc1 + GELU)
        a_2 = W_2 x_1       x_2 = f(a_2)       (fc2 → MLP output)
    COMET routing network runs in parallel with its own FIXED-RANDOM weights V_ℓ:
        c_1 = V_1 x_0
        m_1 = k-WTA(c_1)        # binary mask, top-k largest entries of c_1
        z_1 = m_1 ⊙ g(c_1)       # routing net forward (unused beyond layer 1 here)
    Apply mask to backbone hidden:
        x_1 = m_1 ⊙ f(a_1)
        a_2 = W_2 x_1
        x_2 = f(a_2)   (no mask at final layer of MLP, per paper Eq. 3 note)

V_ℓ and any routing-network biases are fixed random and stored as buffers
(non-trainable), so trainable params = exact ViT dense baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import PatchEmbed, Attention, DropPath, _trunc_normal_


class COMETFFN(nn.Module):
    """ViT MLP with COMET k-WTA masking on the hidden layer.

    Trainable params = standard MLP (fc1 + fc2). Routing projection V1 + bias are
    frozen random buffers (not counted in trainable params).
    """

    def __init__(self, dim, hidden, p_keep=0.25, routing_seed=42, drop=0.0):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.k = max(1, int(round(p_keep * hidden)))     # number of active neurons

        # Standard backbone MLP weights (trainable).
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

        # Fixed-random routing projection V1 (same shape as fc1), stored as buffer.
        # Weights per paper Sec 4: U(-N^{-1/2}, N^{-1/2}). Paper is silent on
        # the bias; zero-init to avoid an unspecified source of randomness
        # entering the k-WTA routing decisions.
        g = torch.Generator().manual_seed(routing_seed)
        bound = dim ** -0.5
        V1_w = (torch.rand(hidden, dim, generator=g) * 2 - 1) * bound
        V1_b = torch.zeros(hidden)
        self.register_buffer('V1_w', V1_w)
        self.register_buffer('V1_b', V1_b)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            y: (B, T, D)
        """
        # Backbone fc1
        a1 = self.fc1(x)                     # (B, T, H)

        # Routing pre-activation with frozen V1
        c1 = F.linear(x, self.V1_w, self.V1_b)   # (B, T, H)

        # k-WTA mask: 1 at positions where c1 is in the top-k along H dim.
        # top-k per token (independent across B, T).
        _, topk_idx = c1.topk(self.k, dim=-1)   # (B, T, k)
        mask = torch.zeros_like(c1)
        mask.scatter_(-1, topk_idx, 1.0)

        # Apply mask to backbone hidden activation (Eq. 8).
        x1 = mask * F.gelu(a1)

        # Final MLP layer — no mask (paper Eq. 3 note: "output a_L computed before m_L").
        return self.drop(self.fc2(x1))


class COMETBlock(nn.Module):
    """Pre-norm ViT block with shared MHSA + COMET-masked MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, p_keep=0.25,
                 routing_seed=42, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = COMETFFN(dim, int(dim * mlp_ratio),
                            p_keep=p_keep, routing_seed=routing_seed, drop=drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class COMETViT(nn.Module):
    """COMET ViT (Shaier 2025) for ImageNet-1K.

    No added trainable parameters — routing V is frozen random (buffer). Trainable
    param count = exactly the dense ViT-S/16 backbone.

    Args:
        p_keep: fraction of hidden neurons kept per token (paper sweeps this).
        routing_seed: base seed for V; each layer l uses routing_seed + l.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 p_keep=0.25, routing_seed=42, qkv_bias=True,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            COMETBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio,
                       p_keep=p_keep, routing_seed=routing_seed + i,
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
            aux_loss: 0 (COMET has no auxiliary loss).
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

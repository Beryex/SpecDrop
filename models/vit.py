"""Vision Transformer for ImageNet-1K (ViT-Small baseline).

Community-standard ViT-Small/16 architecture:
  - Dosovitskiy et al. 2020 (original ViT, https://arxiv.org/abs/2010.11929)
  - Touvron et al. 2020 (DeiT training recipe + DropPath)
  - timm vit_small_patch16_224 (reference implementation)

Defaults for ViT-S/16 on ImageNet-1K:
  img_size=224, patch=16, embed_dim=384, depth=12, heads=6, mlp_ratio=4,
  qkv_bias=True, pre-norm, CLS token, learned 1D pos embedding, GELU.
  Total params: ~22.05M (1000 classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_(tensor, mean=0.0, std=1.0):
    """In-place truncated normal (timm convention: a=-2*std, b=2*std)."""
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2 * std, b=2 * std)


def _drop_path(x, drop_prob=0.0, training=False):
    """Per-sample stochastic depth (Huang et al., 2016)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0:
        mask.div_(keep_prob)
    return x * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """Image → sequence of patch embeddings via non-overlapping Conv2d."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size {img_size} must be divisible by patch_size {patch_size}"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input {H}x{W} does not match model img_size {self.img_size}"
        x = self.proj(x)                      # (B, D, H/P, W/P)
        return x.flatten(2).transpose(1, 2)   # (B, N, D)


class Attention(nn.Module):
    """Multi-head self-attention using torch.nn.functional.scaled_dot_product_attention."""

    def __init__(self, dim, num_heads=6, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)               # each (B, H, N, d)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    """Feed-forward: Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, in_dim, hidden_dim, out_dim=None, drop=0.0):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class Block(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    """Vision Transformer (generic).

    Default config is ViT-Small/16 for ImageNet. Override args for other scales.

    Args:
        img_size: square input size.
        patch_size: patch side length in pixels.
        in_channels: input channels (3 for RGB).
        num_classes: classifier output dimension.
        embed_dim: token embedding dim.
        depth: number of Transformer blocks.
        num_heads: attention heads.
        mlp_ratio: MLP hidden / embed_dim.
        qkv_bias: bias term in QKV projection.
        drop_rate: dropout on MLP/proj output and classifier input.
        attn_drop_rate: dropout on attention weights.
        drop_path_rate: max stochastic-depth rate (linearly scaled 0 → rate across depth).
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0):
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
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
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

    def forward_features(self, x):
        x = self.patch_embed(x)                         # (B, N, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)                  # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                   # CLS pooled

    def forward(self, x, branch_mask=None):
        """branch_mask is ignored (API compatibility with multi-branch models)."""
        return self.head(self.forward_features(x))


def vit_small(num_classes=1000, img_size=224, patch_size=16,
              drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
    """ViT-Small/16: embed=384, depth=12, heads=6, mlp_ratio=4 (~22M params)."""
    return ViT(
        img_size=img_size, patch_size=patch_size, in_channels=3,
        num_classes=num_classes,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
    )

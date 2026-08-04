"""Multi-branch Vision Transformer for ImageNet-1K (SpecDrop + MoE-style baselines).

Architecture (default = ViT-Small/16):
  Shared: patch_embed + cls_token + pos_embed + attention (all blocks)
          + final LayerNorm + classifier head + optional shared-expert MLP per block
  Branched: K parallel MLPs per block (narrow hidden dim), fused via batched einsum
  Merging: per-sample branch_mask (B, K) applied identically at every block,
           fixed-denominator merge (matches MultiBranchResNet110 / MultiBranchTransformerLM)

Default K=8, branch_hidden=192 → param-matched (~22.08M vs dense ViT-S/16's 22.05M).
If branch_hidden is None, it is auto-computed as round(embed_dim*mlp_ratio / K)
so K independent branches collectively approximate one dense MLP's parameter count.

When shared_expert_dim > 0, each block additionally runs a dense MLP(dim → SE_dim → dim)
on the pre-norm hidden, whose output is added to the (masked) merged branch output
before the residual — mirroring SharedExpertMoETransformerBlock in transformer_lm.py.
In that mode the caller should split its FFN budget via data.slimpajama.split_ffn_budget_for_se
so that K·branch_hidden + shared_expert_dim still param-matches the dense MLP budget.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import PatchEmbed, Attention, DropPath, MLP, _trunc_normal_


class ParallelMLP(nn.Module):
    """K parallel FFNs fused via batched einsum (no for-loop).

    Equivalent to K independent Linear(D → H) + GELU + Linear(H → D) stacks.
    """

    def __init__(self, dim, branch_hidden, num_branches, drop=0.0):
        super().__init__()
        self.num_branches = num_branches
        self.dim = dim
        self.branch_hidden = branch_hidden

        self.w1 = nn.Parameter(torch.empty(num_branches, branch_hidden, dim))
        self.b1 = nn.Parameter(torch.zeros(num_branches, branch_hidden))
        self.w2 = nn.Parameter(torch.empty(num_branches, dim, branch_hidden))
        self.b2 = nn.Parameter(torch.zeros(num_branches, dim))
        self.drop = nn.Dropout(drop)

        # Match dense ViT Linear init (trunc_normal std=0.02, zero biases)
        _trunc_normal_(self.w1, std=0.02)
        _trunc_normal_(self.w2, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (B, N, D) input token sequence (post-norm).
        Returns:
            (B, N, K, D) outputs of all K branches stacked.
        """
        # (B, N, D) · (K, H, D) → (B, N, K, H)
        hidden = torch.einsum('bnd,khd->bnkh', x, self.w1) + self.b1
        hidden = F.gelu(hidden)
        # (B, N, K, H) · (K, D, H) → (B, N, K, D)
        out = torch.einsum('bnkh,kdh->bnkd', hidden, self.w2) + self.b2
        return self.drop(out)


class MultiBranchBlock(nn.Module):
    """Pre-norm Transformer block: shared MHSA + K parallel MLPs with masked merge.

    If shared_expert_dim > 0, a dense MLP(dim, SE_dim, dim) is applied to the
    same pre-norm hidden and its output added to the merged routed output
    before the residual — mirroring the NLP SharedExpertMoETransformerBlock.
    """

    def __init__(self, dim, num_heads, num_branches, branch_hidden,
                 shared_expert_dim=0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.num_branches = num_branches
        self.shared_expert_dim = shared_expert_dim
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.parallel_mlp = ParallelMLP(dim, branch_hidden, num_branches, drop=drop)
        self.shared_expert = (
            MLP(dim, shared_expert_dim, drop=drop) if shared_expert_dim > 0 else None)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x, branch_mask=None, mask_scale=None):
        B, N, D = x.shape
        K = self.num_branches

        x = x + self.drop_path1(self.attn(self.norm1(x)))

        h = self.norm2(x)
        branch_outputs = self.parallel_mlp(h)   # (B, N, K, D)

        if branch_mask is not None:
            # Keep mask in its original dtype (fp32 from algorithm); let PyTorch
            # auto-promote stacked × mask to fp32 under low-precision AMP.
            # Explicit downcast to branch_outputs.dtype was removed after it
            # caused seed-dependent NaN in the sibling CIFAR path (see
            # models/multi_branch.py for the full rationale).
            mask = branch_mask.view(B, 1, K, 1)
            branch_outputs = branch_outputs * mask
            if mask_scale is not None:
                routed = branch_outputs.sum(dim=2) / mask_scale
            else:
                raise RuntimeError(
                    'branch_mask provided but self.mask_scale is None. '
                    'Algorithm must set model.mask_scale = algorithm.expected_mask_sum '
                    '(fixed-denominator merge). Silent per-sample normalization '
                    'was removed: it reintroduces the Jensen bias this paper eliminates.')
        else:
            routed = branch_outputs.mean(dim=2)

        if self.shared_expert is not None:
            merged = self.shared_expert(h) + routed
        else:
            merged = routed

        x = x + self.drop_path2(merged)
        return x


class MultiBranchViT(nn.Module):
    """Multi-branch Vision Transformer (default ViT-Small/16).

    Args:
        img_size: square input size.
        patch_size: patch side length in pixels.
        in_channels: input channels (3 for RGB).
        num_classes: classifier output dim.
        embed_dim: token dim (shared across branches).
        depth: number of transformer blocks.
        num_heads: attention heads per block.
        mlp_ratio: dense-equivalent MLP expansion (used to auto-compute branch_hidden).
        num_branches: K parallel MLPs per block.
        branch_hidden: MLP hidden dim per branch. If None, auto-computed as
            round(embed_dim*mlp_ratio / num_branches) for parameter matching.
        qkv_bias: include bias in QKV projection.
        drop_rate: dropout on MLP/proj output and classifier input.
        attn_drop_rate: dropout on attention weights.
        drop_path_rate: max stochastic-depth rate (linear schedule across depth).
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 num_branches=8, branch_hidden=None, shared_expert_dim=0,
                 qkv_bias=True,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.num_branches = num_branches
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.shared_expert_dim = shared_expert_dim

        if branch_hidden is None:
            # Auto-compute branch_hidden to param-match dense MLP(embed_dim,
            # embed_dim*mlp_ratio, embed_dim). Total FFN budget is
            # K × branch_hidden + shared_expert_dim, so we subtract SE first
            # and divide the remaining evenly across K branches. Pre-fix
            # (2026-04) version divided standard_hidden directly → at SE>0
            # the model over-budget by SE/standard_hidden, failing the 2%
            # sanity check for SE_ratio ≥ ~0.3x. Latent because all current
            # shell scripts pass explicit branch_hidden; fix preemptive.
            standard_hidden = int(embed_dim * mlp_ratio)
            remaining = standard_hidden - shared_expert_dim
            branch_hidden = max(8, round(remaining / num_branches))
        self.branch_hidden = branch_hidden

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            MultiBranchBlock(embed_dim, num_heads, num_branches, branch_hidden,
                             shared_expert_dim=shared_expert_dim,
                             qkv_bias=qkv_bias, drop=drop_rate,
                             attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        # Fixed-denominator merge (set externally by algorithm via mask_scale).
        self.mask_scale = None

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

    def get_stem_features(self, x):
        """Return patch-embedding conv output as (B, D, H_patches, W_patches).

        4-D format matches ResNet stems so trainer.py can apply
        adaptive_avg_pool2d + flatten to obtain (B, D) routing features.
        """
        return self.patch_embed.proj(x)

    def forward_from_stem(self, stem_features, branch_mask=None):
        """Continue from 4-D stem: add CLS + pos, run blocks, classify."""
        B = stem_features.size(0)
        x = stem_features.flatten(2).transpose(1, 2)         # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x, branch_mask=branch_mask, mask_scale=self.mask_scale)
        x = self.norm(x)
        return self.head(x[:, 0])

    def forward(self, x, branch_mask=None):
        stem_features = self.get_stem_features(x)
        return self.forward_from_stem(stem_features, branch_mask)


def multi_branch_vit_small(num_classes=1000, num_branches=46, branch_hidden=None,
                           shared_expert_dim=0,
                           img_size=224, patch_size=16,
                           drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
    """MultiBranchViT-Small/16: embed=384, depth=12, heads=6, default K=46 (BREEDS superclasses, param-matched).

    With shared_expert_dim>0, each block gains a dense MLP(384, SE_dim, 384)
    added in parallel with the K routed branches; caller should compute
    (branch_hidden, shared_expert_dim) via data.slimpajama.split_ffn_budget_for_se
    to keep the full FFN budget within ±2% of dense ViT-Small.
    """
    return MultiBranchViT(
        img_size=img_size, patch_size=patch_size, in_channels=3,
        num_classes=num_classes,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
        num_branches=num_branches, branch_hidden=branch_hidden,
        shared_expert_dim=shared_expert_dim,
        qkv_bias=True,
        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
    )

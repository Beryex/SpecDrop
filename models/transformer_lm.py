"""Transformer language models with MoE-style parallel FFN branches.

Architecture:
  MultiBranchTransformerLM: embedding -> N MoE blocks -> LN -> lm_head (tied)
    Each MoE block: LN -> causal self-attention -> LN -> K parallel FFN -> masked merge
  TransformerLM: standard dense transformer baseline (no branches)

The branch_mask (B, K) is document-level: same mask for all tokens and all layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_mask(T, device):
    """Upper-triangular causal mask (additive, -inf for masked positions)."""
    return torch.triu(
        torch.full((T, T), float('-inf'), device=device), diagonal=1)


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    """Feed-forward network: Linear -> GELU -> Linear."""

    def __init__(self, hidden_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class ParallelFFN(nn.Module):
    """K parallel FFNs computed in one batched matmul (no for-loop).

    Equivalent to K separate FFN(hidden_dim, ffn_dim) but uses stacked weights
    and torch.einsum for parallel computation in a single kernel call.
    """

    def __init__(self, hidden_dim, ffn_dim, num_branches, dropout=0.1):
        super().__init__()
        self.num_branches = num_branches
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim

        # Stacked weights: K copies of (ffn_dim, hidden_dim) and (hidden_dim, ffn_dim)
        self.w1 = nn.Parameter(torch.empty(num_branches, ffn_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_branches, ffn_dim))
        self.w2 = nn.Parameter(torch.empty(num_branches, hidden_dim, ffn_dim))
        self.b2 = nn.Parameter(torch.zeros(num_branches, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        # GPT-2 style init (std=0.02) for AMP stability — matches dense FFN
        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        nn.init.normal_(self.w2, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, K, D) — output of all K branches stacked
        """
        # x: (B, T, D), w1: (K, FF, D) → hidden: (B, T, K, FF)
        hidden = torch.einsum('btd,kfd->btkf', x, self.w1) + self.b1
        hidden = F.gelu(hidden)
        # hidden: (B, T, K, FF), w2: (K, D, FF) → out: (B, T, K, D)
        out = torch.einsum('btkf,kdf->btkd', hidden, self.w2) + self.b2
        return self.dropout(out)


class NonUniformParallelFFN(nn.Module):
    """K parallel FFNs with heterogeneous per-branch ffn_dim.

    Drop-in replacement for ParallelFFN when the per-branch hidden width
    varies (e.g. data-proportional branch allocation on imbalanced SlimPajama
    domains). Returns the same (B, T, K, D) output shape so downstream
    masked-merge code needs no changes.

    Implementation: ModuleList of K independent FFN(hidden_dim, ffn_dim_k)
    modules + a final torch.stack along the branch axis. Each branch is a
    separate nn.Linear call, so this is ~10-30% slower than ParallelFFN's
    batched einsum at K=7, hidden=384 on RTX 5090 — uniform configs should
    keep using ParallelFFN for speed (dispatched by the MoE blocks).

    Init matches ParallelFFN: GPT-2 style N(0, 0.02²) on fc1/fc2 weights.
    """

    def __init__(self, hidden_dim, ffn_dims, dropout=0.1):
        super().__init__()
        self.num_branches = len(ffn_dims)
        self.hidden_dim = hidden_dim
        self.ffn_dims = list(ffn_dims)

        self.ffns = nn.ModuleList([
            FFN(hidden_dim, ffn_dim, dropout) for ffn_dim in ffn_dims
        ])

        # GPT-2 style init (std=0.02) for AMP stability — matches ParallelFFN.
        for ffn in self.ffns:
            nn.init.normal_(ffn.fc1.weight, mean=0.0, std=0.02)
            nn.init.normal_(ffn.fc2.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, K, D) — per-branch outputs stacked along the K axis.
        """
        # Each ffn(x) returns (B, T, D). Stack along new axis 2 to get (B, T, K, D).
        outs = [ffn(x) for ffn in self.ffns]
        return torch.stack(outs, dim=2)


def _build_parallel_ffn(hidden_dim, ffn_width, num_branches, dropout):
    """Dispatch helper: scalar ffn_width → ParallelFFN (uniform, fast einsum);
    list/tuple ffn_width → NonUniformParallelFFN (non-uniform, ModuleList).

    Both return a module whose forward maps (B, T, D) → (B, T, K, D), so
    the MoE block's masked-merge code is unchanged.
    """
    if isinstance(ffn_width, (list, tuple)):
        assert len(ffn_width) == num_branches, (
            f"ffn_dims list length {len(ffn_width)} != num_branches {num_branches}")
        return NonUniformParallelFFN(hidden_dim, ffn_width, dropout)
    return ParallelFFN(hidden_dim, ffn_width, num_branches, dropout)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block (dense FFN)."""

    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim, ffn_dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Pre-norm causal self-attention
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.drop(h)
        # Pre-norm FFN
        x = x + self.ffn(self.ln2(x))
        return x


class MoETransformerBlock(nn.Module):
    """Pre-norm transformer block with K parallel FFN branches.

    Attention is shared; FFN is split into K independent branches.
    Branch outputs are combined using the (B, K) mask with fixed-denominator merge.
    """

    def __init__(self, hidden_dim, num_heads, num_branches, ffn_dim_per_branch,
                 dropout=0.1):
        super().__init__()
        self.num_branches = num_branches

        # Shared attention
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        # K parallel FFN branches. Scalar ffn_dim_per_branch → batched einsum
        # (uniform, fast). List → per-branch ModuleList (non-uniform).
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.parallel_ffn = _build_parallel_ffn(
            hidden_dim, ffn_dim_per_branch, num_branches, dropout)

    def forward(self, x, branch_mask=None, mask_scale=None, attn_mask=None):
        """
        Args:
            x: (B, T, D)
            branch_mask: (B, K) or None
            mask_scale: fixed denominator or None
            attn_mask: (T, T) causal mask
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape
        K = self.num_branches

        # Pre-norm causal self-attention
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.drop(h)

        # Pre-norm parallel FFN branches (single batched einsum call)
        h = self.ln2(x)
        branch_outputs = self.parallel_ffn(h)  # (B, T, K, D)

        # Masked merge
        if branch_mask is not None:
            # Keep mask in its original dtype (fp32 from algorithm); PyTorch
            # auto-promotes bf16 × fp32 → fp32 for the merge. Explicit
            # downcast removed — see models/multi_branch.py for rationale.
            # branch_mask: (B, K) -> (B, 1, K, 1) for broadcasting.
            mask = branch_mask.view(B, 1, K, 1)
            branch_outputs = branch_outputs * mask
            if mask_scale is not None:
                merged = branch_outputs.sum(dim=2) / mask_scale
            else:
                raise RuntimeError(
                    'branch_mask provided but self.mask_scale is None. '
                    'Algorithm must set model.mask_scale = algorithm.expected_mask_sum '
                    '(fixed-denominator merge). Silent per-sample normalization '
                    'was removed: it reintroduces the Jensen bias this paper eliminates.')
        else:
            merged = branch_outputs.mean(dim=2)  # (B, T, D)

        x = x + merged
        return x


class SandwichMoETransformerBlock(nn.Module):
    """Pre-norm transformer block with shared dense + K branches + shared dense.

    FFN structure: dense_pre → K parallel branches → masked merge → dense_post.
    The shared dense layers learn common representations; branches specialize.
    """

    def __init__(self, hidden_dim, num_heads, num_branches, ffn_dim_per_branch,
                 sandwich_dim=128, dropout=0.1):
        super().__init__()
        self.num_branches = num_branches

        # Shared attention
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        # Sandwich FFN: dense_pre → branches → dense_post.
        # Branches can be uniform (scalar ffn_dim_per_branch) or non-uniform (list).
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dense_pre = FFN(hidden_dim, sandwich_dim, dropout)
        if isinstance(ffn_dim_per_branch, (list, tuple)):
            assert len(ffn_dim_per_branch) == num_branches, (
                f"ffn_dims list length {len(ffn_dim_per_branch)} != num_branches {num_branches}")
            self.branches = nn.ModuleList([
                FFN(hidden_dim, ffn_dim_per_branch[k], dropout)
                for k in range(num_branches)
            ])
        else:
            self.branches = nn.ModuleList([
                FFN(hidden_dim, ffn_dim_per_branch, dropout)
                for _ in range(num_branches)
            ])
        self.dense_post = FFN(hidden_dim, sandwich_dim, dropout)

    def forward(self, x, branch_mask=None, mask_scale=None, attn_mask=None):
        B, T, D = x.shape
        K = self.num_branches

        # Pre-norm causal self-attention
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.drop(h)

        # Pre-norm sandwich FFN
        h = self.ln2(x)

        # Shared dense pre (common backbone)
        h = h + self.dense_pre(h)

        # Parallel branches (specialization)
        branch_outputs = torch.stack(
            [branch(h) for branch in self.branches], dim=2)  # (B, T, K, D)

        # Masked merge
        if branch_mask is not None:
            # Keep mask in fp32 for auto-promoted merge (see multi_branch.py).
            mask = branch_mask.view(B, 1, K, 1)
            branch_outputs = branch_outputs * mask
            if mask_scale is not None:
                merged = branch_outputs.sum(dim=2) / mask_scale
            else:
                raise RuntimeError(
                    'branch_mask provided but self.mask_scale is None. '
                    'Algorithm must set model.mask_scale = algorithm.expected_mask_sum '
                    '(fixed-denominator merge). Silent per-sample normalization '
                    'was removed: it reintroduces the Jensen bias this paper eliminates.')
        else:
            merged = branch_outputs.mean(dim=2)

        # Shared dense post (combination)
        merged = merged + self.dense_post(merged)

        x = x + merged
        return x


class SharedExpertMoETransformerBlock(nn.Module):
    """Pre-norm transformer block with 1 shared expert + K routed branches.

    FFN output = shared_expert(x) + masked_merge(branch_1(x), ..., branch_K(x)).
    The shared expert always processes all tokens (no routing).
    Inspired by DeepSeekMoE's shared expert design.
    """

    def __init__(self, hidden_dim, num_heads, num_branches, ffn_dim_per_branch,
                 shared_expert_dim=220, dropout=0.1):
        super().__init__()
        self.num_branches = num_branches

        # Shared attention
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        # Shared expert (always active) + K routed branches. Scalar
        # ffn_dim_per_branch → batched einsum; list → per-branch ModuleList.
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.shared_expert = FFN(hidden_dim, shared_expert_dim, dropout)
        self.parallel_ffn = _build_parallel_ffn(
            hidden_dim, ffn_dim_per_branch, num_branches, dropout)

    def forward(self, x, branch_mask=None, mask_scale=None, attn_mask=None):
        B, T, D = x.shape
        K = self.num_branches

        # Pre-norm causal self-attention
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.drop(h)

        # Pre-norm FFN: shared expert + routed branches (single einsum call)
        h = self.ln2(x)
        shared_out = self.shared_expert(h)

        branch_outputs = self.parallel_ffn(h)  # (B, T, K, D)

        if branch_mask is not None:
            # Keep mask in fp32 for auto-promoted merge (see multi_branch.py).
            mask = branch_mask.view(B, 1, K, 1)
            branch_outputs = branch_outputs * mask
            if mask_scale is not None:
                routed_out = branch_outputs.sum(dim=2) / mask_scale
            else:
                raise RuntimeError(
                    'branch_mask provided but self.mask_scale is None. '
                    'Algorithm must set model.mask_scale = algorithm.expected_mask_sum '
                    '(fixed-denominator merge). Silent per-sample normalization '
                    'was removed: it reintroduces the Jensen bias this paper eliminates.')
        else:
            routed_out = branch_outputs.mean(dim=2)

        x = x + shared_out + routed_out
        return x


# ---------------------------------------------------------------------------
# Language models
# ---------------------------------------------------------------------------

class MultiBranchTransformerLM(nn.Module):
    """Transformer LM with K parallel FFN branches per layer (MoE-style).

    The branch_mask is document-level: the same (B, K) mask applies to every
    token position and every layer.
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads,
                 num_branches, ffn_dim_per_branch, max_seq_len, dropout=0.1,
                 sandwich_dim=0, shared_expert_dim=0):
        super().__init__()
        self.num_branches = num_branches
        self.mask_scale = None  # set by algorithm's expected_mask_sum

        # Token + position embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)

        # N MoE transformer blocks (variant selection)
        if shared_expert_dim > 0:
            self.blocks = nn.ModuleList([
                SharedExpertMoETransformerBlock(hidden_dim, num_heads, num_branches,
                                                ffn_dim_per_branch, shared_expert_dim, dropout)
                for _ in range(num_layers)
            ])
        elif sandwich_dim > 0:
            self.blocks = nn.ModuleList([
                SandwichMoETransformerBlock(hidden_dim, num_heads, num_branches,
                                            ffn_dim_per_branch, sandwich_dim, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                MoETransformerBlock(hidden_dim, num_heads, num_branches,
                                    ffn_dim_per_branch, dropout)
                for _ in range(num_layers)
            ])

        # Final layer norm + LM head (weight-tied with token embedding)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_stem_features(self, input_ids):
        """Compute token + position embeddings (stem for NLP).

        Returns: (B, T, hidden_dim) embedding features.
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

    def forward_from_stem(self, stem_features, branch_mask=None):
        """Continue forward from pre-computed embedding features."""
        T = stem_features.size(1)
        attn_mask = _causal_mask(T, stem_features.device)

        x = stem_features
        for block in self.blocks:
            x = block(x, branch_mask=branch_mask, mask_scale=self.mask_scale,
                      attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def forward(self, input_ids, branch_mask=None):
        """
        Args:
            input_ids: (B, T) token indices
            branch_mask: (B, K) float mask, same for all positions and layers.
                         None -> average all branches equally (1/K).
        Returns:
            logits: (B, T, vocab_size)
        """
        stem_features = self.get_stem_features(input_ids)
        return self.forward_from_stem(stem_features, branch_mask)


class TransformerLM(nn.Module):
    """Standard dense transformer LM baseline (no branches).

    Same interface as MultiBranchTransformerLM; branch_mask is ignored.
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads,
                 ffn_dim, max_seq_len, dropout=0.1):
        super().__init__()

        # Token + position embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)

        # N standard transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm + LM head (weight-tied with token embedding)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, branch_mask=None):
        """
        Args:
            input_ids: (B, T) token indices
            branch_mask: ignored (API compatibility with multi-branch models)
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        attn_mask = _causal_mask(T, input_ids.device)

        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

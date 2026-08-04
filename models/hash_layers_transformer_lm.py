"""Hash Layers Transformer LM (Roller et al., NeurIPS 2021).

Faithful implementation of per-token hash routing: token_id → expert index via a
fixed random hash table. No learned routing parameters, no auxiliary loss.

Reference:
    Roller et al., "Hash Layers For Large Sparse Models", NeurIPS 2021.
    https://arxiv.org/abs/2106.04426

Routing (Section 3):
    expert_idx(token_id) = H[token_id mod |V|]    # fixed random table
    y_t = E_{expert_idx(id_t)}(x_t)               # one hop, no gate

The hash table H is drawn once at init from Uniform[0, N) and frozen. No training
signal flows through the routing decision (it is deterministic given token IDs).
"""

import torch
import torch.nn as nn

from .transformer_lm import _causal_mask, ParallelFFN


class HashLayerFFN(nn.Module):
    """Hash-routed MoE FFN. Each token's expert is determined by hash(token_id).

    Efficient implementation: compute all N experts in parallel via ParallelFFN
    and gather the hash-selected expert's output per token. Same total FLOPs
    as a dense FFN (param-matched: N × ffn_per_expert = dense FFN hidden) but
    one GPU kernel per block instead of N. Training signal is identical — the
    hash is deterministic, so only one expert per token has non-zero gradient.
    """

    def __init__(self, hidden_dim, ffn_dim_per_expert, num_experts,
                 vocab_size, hash_seed=42, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.vocab_size = vocab_size

        self.experts = ParallelFFN(hidden_dim, ffn_dim_per_expert,
                                    num_experts, dropout)

        # Fixed random hash table: vocab_size → expert index ∈ [0, N).
        g = torch.Generator().manual_seed(hash_seed)
        table = torch.randint(0, num_experts, (vocab_size,), generator=g)
        self.register_buffer('hash_table', table)

    def forward(self, x, token_ids):
        """
        Args:
            x: (B, T, D) pre-normed hidden states.
            token_ids: (B, T) original token indices for hash lookup.
        Returns:
            y: (B, T, D)
        """
        B, T, D = x.shape
        expert_idx = self.hash_table[token_ids]                   # (B, T)

        # Compute all N experts in parallel (single einsum per block).
        all_out = self.experts(x)                                 # (B, T, N, D)
        idx = expert_idx.view(B, T, 1, 1).expand(-1, -1, 1, D)    # (B, T, 1, D)
        return all_out.gather(dim=2, index=idx).squeeze(2)        # (B, T, D)


class HashLayerBlock(nn.Module):
    """Pre-norm block with shared MHA + Hash-routed FFN."""

    def __init__(self, hidden_dim, num_heads, num_experts, ffn_dim_per_expert,
                 vocab_size, hash_seed, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.moe = HashLayerFFN(hidden_dim, ffn_dim_per_expert, num_experts,
                                vocab_size, hash_seed, dropout)

    def forward(self, x, token_ids, attn_mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.drop(h)
        x = x + self.moe(self.ln2(x), token_ids)
        return x


class HashLayersTransformerLM(nn.Module):
    """Hash Layers LM (Roller 2021) for SlimPajama.

    Each layer uses a different fixed hash table (per-layer seed offset).

    Args:
        num_experts: N.
        ffn_dim_per_expert: narrow per-expert FFN hidden dim (param-match).
        hash_seed: base seed; layer l uses hash_seed + l.
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads,
                 num_experts, ffn_dim_per_expert, max_seq_len,
                 hash_seed=42, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            HashLayerBlock(hidden_dim, num_heads, num_experts, ffn_dim_per_expert,
                           vocab_size, hash_seed + layer_idx, dropout)
            for layer_idx in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

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

    def forward(self, input_ids):
        """
        Returns:
            logits: (B, T, vocab_size)
            aux_loss: 0 (Hash Layers has no auxiliary loss).
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        attn_mask = _causal_mask(T, input_ids.device)

        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x, token_ids=input_ids, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, torch.tensor(0.0, device=input_ids.device)

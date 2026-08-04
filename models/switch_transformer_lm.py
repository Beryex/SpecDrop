"""Switch Transformer LM (Fedus et al., JMLR 2022).

Faithful implementation of per-token top-1 expert routing with load-balance loss.
Parameter-matched to the dense TransformerLM by narrowing each expert's FFN.

Reference:
    Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models
    with Simple and Efficient Sparsity", JMLR 2022.
    https://arxiv.org/abs/2101.03961

Routing (Section 2.1):
    h_t = W_r · x_t                                # router logits
    p_t = softmax(h_t)                              # (N,) normalized routing probs
    i*   = argmax(p_t)                              # hard top-1
    y_t  = p_t[i*] · E_{i*}(x_t)                    # selected expert output, scaled by gate

Load-balance auxiliary loss (Eq. 4):
    f_i = (1/T) Σ_t 1[i* = i]                      # fraction of tokens routed to expert i
    P_i = (1/T) Σ_t p_t[i]                          # mean router prob for expert i
    L_aux = α · N · Σ_i f_i · P_i                   # load-balance, α = 0.01 typical
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_lm import _causal_mask, ParallelFFN


class SwitchFFN(nn.Module):
    """Switch per-token top-1 MoE FFN (Fedus 2022).

    Efficient implementation: compute all N experts in parallel via a single
    batched einsum (ParallelFFN) and gather the top-1 expert's output per
    token. Because we param-match (N × ffn_per_expert = dense FFN hidden),
    this has the same FLOPs as a dense FFN but fires one GPU kernel per
    block instead of N. On a 30M model with N=32 tiny experts, kernel-launch
    overhead would otherwise dominate wall time (~30× slower than dense).

    Training signal is identical to the paper: non-selected experts are
    multiplied by gate=0 and receive no gradient from the CE loss; the
    load-balance loss is computed from the full softmax distribution.
    """

    def __init__(self, hidden_dim, ffn_dim_per_expert, num_experts,
                 load_balance_weight=0.01, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight

        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = ParallelFFN(hidden_dim, ffn_dim_per_expert,
                                    num_experts, dropout)
        # Last aux loss computed in forward; aggregated by parent model.
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            y: (B, T, D) with per-token top-1 expert output.
        """
        B, T, D = x.shape
        N = self.num_experts
        x_flat = x.reshape(B * T, D)

        # Fedus 2022 §2.3 "Selective Precision": run router logits + softmax in
        # fp32 regardless of model AMP dtype. With bf16 + N=32, direct bf16
        # softmax has only ~3 bits of precision; fp32 casting avoids gate collapse.
        logits = self.router(x_flat).float()                     # (BT, N) fp32
        probs = F.softmax(logits, dim=-1)                        # (BT, N) fp32
        top1_probs, top1_idx = probs.max(dim=-1)                 # each (BT,)
        top1_probs = top1_probs.to(x.dtype)                      # cast gate back

        # Load-balance loss (eq. 4): α · N · Σ_i f_i · P_i.
        if self.training:
            one_hot = F.one_hot(top1_idx, num_classes=N).float() # (BT, N)
            f = one_hot.mean(dim=0)                              # (N,)
            P = probs.mean(dim=0)                                # (N,)
            self.aux_loss = self.load_balance_weight * N * (f * P).sum()
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device)

        # Compute all N experts in parallel (single einsum per block).
        all_out = self.experts(x)                                # (B, T, N, D)
        # Gather the top-1 expert's output per token and scale by gate.
        idx = top1_idx.view(B, T, 1, 1).expand(-1, -1, 1, D)     # (B, T, 1, D)
        out = all_out.gather(dim=2, index=idx).squeeze(2)        # (B, T, D)
        out = out * top1_probs.view(B, T, 1)
        return out


class SwitchBlock(nn.Module):
    """Pre-norm Transformer block with shared MHA + Switch FFN."""

    def __init__(self, hidden_dim, num_heads, num_experts, ffn_dim_per_expert,
                 load_balance_weight=0.01, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.moe = SwitchFFN(hidden_dim, ffn_dim_per_expert, num_experts,
                             load_balance_weight, dropout)

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.drop(h)
        x = x + self.moe(self.ln2(x))
        return x


class SwitchTransformerLM(nn.Module):
    """Switch Transformer LM (Fedus 2022) for SlimPajama.

    Args:
        vocab_size, hidden_dim, num_layers, num_heads, max_seq_len: standard LM.
        num_experts: N (number of parallel experts per MoE layer).
        ffn_dim_per_expert: per-expert FFN hidden dim. For param-match to dense
            FFN of width H_dense, set this to H_dense / N (approx).
        load_balance_weight: α in L_aux = α · N · Σ f_i P_i (default 0.01).
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads,
                 num_experts, ffn_dim_per_expert, max_seq_len,
                 load_balance_weight=0.01, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SwitchBlock(hidden_dim, num_heads, num_experts, ffn_dim_per_expert,
                        load_balance_weight, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        self._init_weights()

    def _init_weights(self):
        # Fedus et al. 2022 §2.1 specifies truncated-normal init for the router
        # to avoid occasional extreme logits that skew the load-balance loss at
        # step 0.  All other Linear / Embedding layers follow standard normal.
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
        # Router-specific re-init (truncated normal, overrides the Linear default).
        for block in self.blocks:
            nn.init.trunc_normal_(block.moe.router.weight,
                                   mean=0.0, std=0.02, a=-2 * 0.02, b=2 * 0.02)

    def forward(self, input_ids):
        """
        Returns:
            logits: (B, T, vocab_size)
            aux_loss: scalar tensor, aggregated load-balance loss across all blocks.
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        attn_mask = _causal_mask(T, input_ids.device)

        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        aux_loss = sum(block.moe.aux_loss for block in self.blocks)
        return logits, aux_loss

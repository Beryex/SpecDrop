"""DEMix Transformer LM (Gururangan et al., ACL 2022).

Faithful implementation of per-document hard domain-to-expert routing.
One expert per training domain; at inference, a domain posterior is used for
mixture-of-experts combination (Section 3.3).

Reference:
    Gururangan et al., "DEMix Layers: Disentangling Domains for Modular Language
    Modeling", ACL 2022. https://arxiv.org/abs/2108.05036

Routing (Section 3):
    Train: document's domain d → expert E_d processes all tokens of that document.
    Inference options (Section 3.3 "Inference"):
        - **Single-expert** (oracle domain): use P(y|x, E_d*) for known d*.
        - **Mixture-of-Experts**: y = Σ_d P(d|x) · E_d(x), with posteriors
            P(d|x) ∝ P(x|d) · P(d), estimated by running the LM with each expert.
    We implement both: `forward(input_ids, domain_ids)` for single-expert train/eval;
    `forward_mixture(input_ids)` for the MoE-inference mode (Section 3.3 Eq. 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_lm import _causal_mask, ParallelFFN


class DemixFFN(nn.Module):
    """Per-document domain-routed FFN. Each document uses its domain's expert only.

    Efficient implementation: compute all N domain experts in parallel via
    ParallelFFN, then gather the expert for each document (single-expert train)
    or weighted-sum across experts (mixture inference). Same total FLOPs as a
    dense FFN (param-matched) but one GPU kernel per block.
    """

    def __init__(self, hidden_dim, ffn_dim_per_expert, num_domains, dropout=0.1):
        super().__init__()
        self.num_domains = num_domains
        self.experts = ParallelFFN(hidden_dim, ffn_dim_per_expert,
                                    num_domains, dropout)

    def forward(self, x, domain_ids=None, mixture_weights=None):
        """
        Args:
            x: (B, T, D)
            domain_ids: (B,) long tensor of per-document domain indices (0..N-1).
                Required unless `mixture_weights` is provided.
            mixture_weights: (B, N) float tensor of P(d|x) posteriors. If given,
                output is Σ_d w_d · E_d(x) per document.
        """
        B, T, D = x.shape
        N = self.num_domains

        # Compute all N experts in parallel (single einsum per block).
        all_out = self.experts(x)                                  # (B, T, N, D)

        if mixture_weights is not None:
            # Mixture-of-Experts inference: Σ_d w_d · E_d(x) per document.
            w = mixture_weights.view(B, 1, N, 1).to(all_out.dtype)  # (B, 1, N, 1)
            return (all_out * w).sum(dim=2)                         # (B, T, D)

        # Single-expert (train / oracle-domain eval): all tokens in doc → one expert.
        assert domain_ids is not None, "DemixFFN needs domain_ids or mixture_weights"
        idx = domain_ids.view(B, 1, 1, 1).expand(-1, T, 1, D)       # (B, T, 1, D)
        return all_out.gather(dim=2, index=idx).squeeze(2)          # (B, T, D)


class DemixBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_domains, ffn_dim_per_expert,
                 dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.moe = DemixFFN(hidden_dim, ffn_dim_per_expert, num_domains, dropout)

    def forward(self, x, attn_mask=None, domain_ids=None, mixture_weights=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.drop(h)
        x = x + self.moe(self.ln2(x), domain_ids=domain_ids,
                          mixture_weights=mixture_weights)
        return x


class DemixTransformerLM(nn.Module):
    """DEMix LM (Gururangan 2022) for SlimPajama (N = num_domains).

    SlimPajama has 7 domains → num_domains=7; each domain gets its own FFN expert.
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads,
                 num_domains, ffn_dim_per_expert, max_seq_len, dropout=0.1):
        super().__init__()
        self.num_domains = num_domains

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DemixBlock(hidden_dim, num_heads, num_domains, ffn_dim_per_expert,
                       dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # Cached domain prior P(d) used at inference (Gururangan 2022 Section 3.3).
        # Default uniform 1/N; call `set_cached_prior(prior)` with training-set
        # domain frequencies for the faithful DEMix inference protocol.
        self.register_buffer(
            'cached_prior',
            torch.full((num_domains,), 1.0 / num_domains))

        self._init_weights()

    def set_cached_prior(self, prior):
        """Set the cached P(d) prior from training-set domain frequencies.

        Args:
            prior: (num_domains,) float tensor, need not be normalized.
        """
        assert prior.shape == (self.num_domains,)
        assert (prior >= 0).all(), "prior must be non-negative"
        total = prior.sum()
        assert total > 0, "prior must sum to > 0"
        self.cached_prior.copy_((prior / total).to(self.cached_prior))

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

    def forward(self, input_ids, domain_ids=None, mixture_weights=None):
        """Forward pass with either hard domain routing or per-layer soft mixing.

        Args:
            input_ids: (B, T).
            domain_ids: (B,) long tensor of per-document domain labels
                (hard routing; used at training and oracle-domain eval).
            mixture_weights: (B, N) soft weights per document summing to 1;
                when provided, every FFN mixes experts as Σ_d w_d · E_d(h)
                (faithful per-layer MoE, Gururangan 2022 Eq. 2).

        Exactly one of {domain_ids, mixture_weights} must be given.

        Returns:
            logits: (B, T, vocab_size)
            aux_loss: scalar 0 (DEMix has no aux loss).
        """
        assert (domain_ids is None) ^ (mixture_weights is None), \
            "Must provide exactly one of domain_ids / mixture_weights."
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        attn_mask = _causal_mask(T, input_ids.device)

        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask,
                      domain_ids=domain_ids,
                      mixture_weights=mixture_weights)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, torch.tensor(0.0, device=input_ids.device)

    @torch.no_grad()
    def forward_mixture(self, input_ids, domain_prior=None):
        """Faithful MoE inference (Gururangan 2022 Section 3.3 + Eq. 2).

        Two-stage: (1) compute per-document posterior P(d|x) via Bayes under
        each expert's hard-routed LM; (2) re-forward ONCE with `mixture_weights`
        = posterior so every FFN mixes experts per Eq. 2 (per-layer, not final
        logits).

        Args:
            input_ids: (B, T).
            domain_prior: (N,) overriding prior; if None, uses `self.cached_prior`
                (defaults to uniform unless `set_cached_prior()` was called).

        Returns:
            mixture_logits: (B, T, vocab_size) — from the single forward with
                per-layer soft FFN mixing.
            posteriors: (B, N) — P(d|x) per document.
        """
        B, T = input_ids.shape
        N = self.num_domains
        device = input_ids.device
        prior = (domain_prior if domain_prior is not None
                 else self.cached_prior).to(device)

        # Stage 1: per-expert sequence log-likelihood via hard routing.
        per_expert_logp = []
        for d in range(N):
            fake_domain = torch.full((B,), d, dtype=torch.long, device=device)
            logits_d, _ = self.forward(input_ids, domain_ids=fake_domain)
            logp_d = F.log_softmax(logits_d[:, :-1], dim=-1)
            tgt = input_ids[:, 1:]
            tok_logp = logp_d.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            per_expert_logp.append(tok_logp.sum(dim=-1))                 # (B,)

        seq_logp_all = torch.stack(per_expert_logp, dim=1)                # (B, N)
        log_prior = torch.log(prior.clamp(min=1e-12))
        posteriors = F.softmax(seq_logp_all + log_prior, dim=-1)          # (B, N)

        # Stage 2: single forward with per-layer FFN mixing using posterior.
        mixture_logits, _ = self.forward(input_ids, mixture_weights=posteriors)
        return mixture_logits, posteriors

"""Multi-branch LoRA adapters for post-training instruction fine-tuning.

Architecture: K parallel LoRA heads attached to each target linear module of a
frozen pretrained CausalLM. Each sample's cluster_id selects a soft routing
distribution (via SoftSpecDrop) over the K heads; contributions are merged with
a fixed denominator `S = p_a + (K-1) p_i` (identical semantics to the CIFAR /
ViT / NLP multi-branch models in this repo).

When `shared_expert_rank > 0`, a single always-on "shared LoRA" of rank r_SE
is added in parallel with the K routed heads — analog to the shared-expert
FFN in models/transformer_lm.py::SharedExpertMoETransformerBlock and the
shared_expert_dim in models/multi_branch_vit.py::MultiBranchBlock. Caller
splits the total-rank budget via data.slimpajama.split_lora_rank_budget_for_se
to keep K·r_expert + r_SE within ±3% of the no-SE baseline's total rank.

The actual HF CausalLM wrapping (replacing q/k/v/o/gate/up/down projections
with LoRALinear) lives in Batch 3 (training/trainer_lora.py pipeline); this
module focuses on the core ParallelLoRAAdapter math so it can be tested
standalone without downloading a 3B base model.

Backward-compat: NO dependency on any existing model. Zero impact on CIFAR /
NLP / ViT paths. All new symbols.
"""

import math

import torch
import torch.nn as nn


class ParallelLoRAAdapter(nn.Module):
    """K parallel LoRA heads (+ optional shared LoRA) fused via batched einsum.

    Mathematically equivalent to K independent LoRA(r=rank) stacks plus
    (optionally) one shared LoRA of rank r_SE, merged with the fixed-
    denominator routing rule.

    Args:
        in_features:        D_in of the wrapped linear.
        out_features:       D_out of the wrapped linear.
        num_experts:        K (number of routed LoRA heads).
        rank:               r per routed head.
        alpha:              LoRA scaling α; scaling = α/r. Default α = 2r.
        dropout:            input dropout before LoRA (LoRA's native dropout).
        shared_expert_rank: r_SE for always-on shared LoRA. 0 disables it.

    Forward:
        x: (B, T, D_in) or (B, D_in).
        branch_mask: (B, K) soft weights from SoftSpecDrop.
                     If None → uniform 1/K mean (smoke/debug).
        mask_scale set via self.mask_scale = algorithm.expected_mask_sum.
    """

    def __init__(self, in_features, out_features, num_experts, rank,
                 alpha=None, dropout=0.0, shared_expert_rank=0):
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be ≥ 1, got {num_experts}")
        if rank < 1:
            raise ValueError(f"rank must be ≥ 1, got {rank}")
        if shared_expert_rank < 0:
            raise ValueError(
                f"shared_expert_rank must be ≥ 0, got {shared_expert_rank}")

        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else 2.0 * rank
        self.scaling = self.alpha / self.rank
        self.shared_expert_rank = shared_expert_rank

        # Routed heads: A_k: (K, r, D_in), B_k: (K, D_out, r)
        self.A = nn.Parameter(torch.empty(num_experts, rank, in_features))
        self.B = nn.Parameter(torch.zeros(num_experts, out_features, rank))

        # Shared LoRA (rank r_SE), always-on if enabled.
        if shared_expert_rank > 0:
            self.A_se = nn.Parameter(
                torch.empty(shared_expert_rank, in_features))
            self.B_se = nn.Parameter(
                torch.zeros(out_features, shared_expert_rank))
            self.se_scaling = self.alpha / shared_expert_rank
        else:
            self.register_parameter('A_se', None)
            self.register_parameter('B_se', None)
            self.se_scaling = 0.0

        self.lora_dropout = (
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity())

        # Fixed-denominator merge scale: S = p_a + (K-1) p_i, set by wrapper
        # from algorithm.expected_mask_sum at training step setup time.
        self.mask_scale = None

        self._reset_parameters()

    def _reset_parameters(self):
        # Hu 2022 LoRA convention: Kaiming uniform init on A, zeros on B
        # → LoRA contribution is exactly 0 at step 0 (A·B·x = 0), so wrapping
        # a frozen Linear with this adapter is bit-identical to the frozen
        # linear alone at initialization. This is verified by
        # test_lora_off_identity.
        for k in range(self.num_experts):
            nn.init.kaiming_uniform_(self.A[k], a=math.sqrt(5))
        # B already zeros from torch.zeros() in __init__.
        if self.A_se is not None:
            nn.init.kaiming_uniform_(self.A_se, a=math.sqrt(5))
            # B_se also zeros from init.

    def forward(self, x, branch_mask=None):
        """Compute the LoRA contribution only (caller adds it to frozen_linear(x)).

        Args:
            x:           (B, T, D_in) or (B, D_in).
            branch_mask: (B, K) soft weights. None → uniform 1/K mean (only
                         used for debug / LoRA-off paths).

        Returns:
            (B, T, D_out) or (B, D_out) LoRA delta, SAME shape as x's implied
            linear output.
        """
        original_ndim = x.ndim
        if original_ndim == 2:
            # Lift 2-D input to a 3-D sequence of length 1 for einsum unity.
            x = x.unsqueeze(1)
        assert x.ndim == 3, (
            f"ParallelLoRAAdapter expects (B, T, D) or (B, D), got shape {tuple(x.shape)}")
        B, T, D_in = x.shape
        K = self.num_experts
        assert D_in == self.in_features, (
            f"LoRA input dim {D_in} != in_features {self.in_features}")

        x_drop = self.lora_dropout(x)

        # (B, T, D_in) · (K, r, D_in) → (B, T, K, r)
        hidden = torch.einsum('btd,krd->btkr', x_drop, self.A)
        # (B, T, K, r) · (K, D_out, r) → (B, T, K, D_out)
        routed = torch.einsum('btkr,kdr->btkd', hidden, self.B)

        if branch_mask is not None:
            assert branch_mask.shape[0] == B and branch_mask.shape[1] == K, (
                f"branch_mask shape {tuple(branch_mask.shape)} != ({B}, {K})")
            # Keep mask in its input dtype (fp32 from algorithm); PyTorch
            # auto-promotes routed × mask to fp32 under low-precision AMP.
            # (Same rationale as models/multi_branch_vit.py — prevents the
            # Jensen-fix dtype bug documented in the merge-dtype fix.)
            mask = branch_mask.view(B, 1, K, 1)
            routed = routed * mask
            if self.mask_scale is None:
                raise RuntimeError(
                    'branch_mask provided but self.mask_scale is None. '
                    'Set adapter.mask_scale = algorithm.expected_mask_sum '
                    '(fixed-denominator merge). Silent per-sample '
                    'normalization is forbidden: it reintroduces the Jensen '
                    'bias this paper eliminates.')
            merged = routed.sum(dim=2) / self.mask_scale
        else:
            # Debug path: uniform 1/K mean (used in smoke tests / LoRA-off).
            merged = routed.mean(dim=2)

        output = merged * self.scaling

        if self.A_se is not None:
            # Shared LoRA: (B, T, D_in) · (r_SE, D_in) → (B, T, r_SE)
            se_hidden = torch.einsum('btd,rd->btr', x_drop, self.A_se)
            # (B, T, r_SE) · (D_out, r_SE) → (B, T, D_out)
            se_out = torch.einsum('btr,dr->btd', se_hidden, self.B_se)
            output = output + se_out * self.se_scaling

        if original_ndim == 2:
            output = output.squeeze(1)
        return output

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"K={self.num_experts}, r={self.rank}, α={self.alpha}, "
                f"r_SE={self.shared_expert_rank}")


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a ParallelLoRAAdapter.

    Output = base_linear(x) + ParallelLoRAAdapter(x, branch_mask).

    The base Linear's parameters are frozen (requires_grad=False) by this
    wrapper, so they are NOT added to the optimizer's parameter list. Only the
    adapter's A / B / A_se / B_se matrices are trainable.

    The branch_mask is stored on this wrapper as `self.current_mask`; the
    wrapper forward() reads it and passes it to the adapter. The caller
    (MultiBranchLoRA / trainer_lora) is responsible for calling
    `set_mask(mask, mask_scale)` on every LoRALinear before the forward pass
    of a given batch.
    """

    def __init__(self, base_linear, num_experts, rank,
                 alpha=None, dropout=0.0, shared_expert_rank=0):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(
                f"base_linear must be nn.Linear, got {type(base_linear).__name__}")
        self.base = base_linear
        # Freeze base weights (additive fine-tuning).
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.lora = ParallelLoRAAdapter(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            num_experts=num_experts, rank=rank, alpha=alpha,
            dropout=dropout, shared_expert_rank=shared_expert_rank,
        )
        # Runtime routing state — set via set_mask() before each forward.
        self.current_mask = None

    def set_mask(self, branch_mask, mask_scale):
        """Install the current batch's routing mask + fixed denominator.

        Called once per forward pass by the parent wrapper. No gradient is
        carried through the mask; it is a discrete (B, K) tensor of per-sample
        per-expert weights from SoftSpecDrop.
        """
        self.current_mask = branch_mask
        self.lora.mask_scale = mask_scale

    def forward(self, x):
        y = self.base(x)
        y = y + self.lora(x, branch_mask=self.current_mask)
        return y

    def extra_repr(self):
        return (f"base=Linear(in={self.base.in_features}, "
                f"out={self.base.out_features}), lora_wrapped=True")


def count_lora_params(module):
    """Helper: count trainable (= LoRA) params in a LoRALinear or a container
    of them. Excludes the frozen base.

    Returns (num_params, breakdown) where breakdown is a list of
    (module_name, param_tensor.numel()) for inspectability.
    """
    breakdown = []
    total = 0
    for name, p in module.named_parameters():
        if p.requires_grad:
            breakdown.append((name, p.numel()))
            total += p.numel()
    return total, breakdown

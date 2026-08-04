"""HF CausalLM + LoRA model classes for the post-train track.

Each class wraps a frozen HF CausalLM and injects a method-specific adapter
at every target linear. Forward signature is unified:

    model(input_ids=..., attention_mask=..., labels=..., routing=<info>)
    -> namespace with .loss, .logits, .aux_loss

The `routing` kwarg is method-specific:
  - SingleLoRAModel:      ignored (adapter doesn't route).
  - MultiBranchLoRAModel: {'mask': (B, K), 'mask_scale': float}.
  - HydraLoRAModel:       ignored (gate is learned from hidden state).
  - LoRAMoEModel:         ignored (gate is learned, and adapter is FFN-only).
  - MoCLEModel:           {'cluster_id': (B,)} plus optional BGE centroid init.

The ablation chain and the main table differentiate methods by `model.type` in
YAML configs; `build_lora_model(cfg)` routes to the right class below.

Backward-compat: these classes are additive; no existing model file is modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .lora_adapters import (HydraLoRAAdapter, LoRAMoEAdapter, MoCLEAdapter,
                             MultiBranchSoftSpecDropAdapter, SingleLoRAAdapter)
from .lora_base import (DEFAULT_TARGETS_ALL, DEFAULT_TARGETS_MLP,
                         LoRAInjectedLinear, freeze_base_params,
                         inject_lora_adapters, set_routing_all,
                         sum_aux_losses)

# ── Shared base class ────────────────────────────────────────────────────────


@dataclass
class LoRAOutputs:
    loss: torch.Tensor
    logits: torch.Tensor
    aux_loss: torch.Tensor


def _load_hf_causal_lm(base_model_name: str,
                       torch_dtype=torch.bfloat16,
                       attn_implementation: Optional[str] = None):
    """Deferred-import HF loader. We load with bf16 and (optionally) Flash
    Attention 2 for training efficiency. Raises ImportError with install hint
    if transformers is absent (SpecDrop's default env may not have it until
    the LoRA track is exercised on the 5090)."""
    try:
        from transformers import AutoModelForCausalLM
        import transformers
    except ImportError as e:
        raise ImportError(
            "transformers is required for LoRA models; install with "
            "`pip install transformers>=4.48 accelerate>=0.30`.") from e

    # HF transformers 4.48 renamed `torch_dtype` kwarg → `dtype` and deprecated
    # the old name. Use `dtype` on ≥4.48 to avoid the warning; fall back to
    # the legacy kwarg on older versions for backward-compat.
    kwargs = {}
    tf_ver = tuple(int(x) for x in transformers.__version__.split('.')[:2])
    if tf_ver >= (4, 48):
        kwargs['dtype'] = torch_dtype
    else:
        kwargs['torch_dtype'] = torch_dtype
    if attn_implementation is not None:
        kwargs['attn_implementation'] = attn_implementation
    return AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)


class BaseLoRAModel(nn.Module):
    """Common scaffolding: freeze base HF model, delegate forward, expose loss.

    Subclasses must:
      - Call super().__init__(base_model_name, torch_dtype, attn_implementation).
      - Call `self._inject(adapter_factory, target_modules)` after __init__.
      - Implement `prepare_routing(cluster_id=None, **kwargs)` to turn the
        trainer's routing info into whatever the adapters expect.
    """

    def __init__(self, base_model_name: str,
                 torch_dtype=torch.bfloat16,
                 attn_implementation: Optional[str] = None):
        super().__init__()
        self.base_model_name = base_model_name
        self.base = _load_hf_causal_lm(
            base_model_name, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation)
        self.wrapped_sites: List[str] = []

    def _inject(self, adapter_factory, target_modules,
                 gradient_checkpointing: bool = True):
        self.wrapped_sites = inject_lora_adapters(
            self.base, adapter_factory=adapter_factory,
            target_modules=target_modules)
        self.num_trainable = freeze_base_params(self.base)

        # Activation-memory saver: re-compute activations in backward pass.
        # Essential for Llama-3.2-3B LoRA training on 32GB GPUs — without it,
        # batch 4 × seq 2048 alone saturates VRAM (bf16 activations for 28
        # layers × 3072 hidden × 8192 intermediate easily exceeds 30GB).
        # `use_reentrant=False` is the non-deprecated / more composable variant.
        # KV cache is only useful for inference (autoregressive generation);
        # training forward passes never read it, so keeping use_cache=True
        # just wastes memory. Always disable regardless of grad_ckpt state.
        # Historically this was nested inside the grad_ckpt branch to silence
        # HF's auto-disable warning; now the setting is principled, not
        # warning-driven.
        if hasattr(self.base, 'config'):
            self.base.config.use_cache = False

        if gradient_checkpointing:
            self.base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})
            # With frozen base + trainable LoRA adapters, the base's input
            # embeddings don't have requires_grad=True by default, so the
            # checkpoint function needs explicit enable_input_require_grads()
            # to let gradients flow back through the checkpointed segments
            # to our LoRA parameters.
            if hasattr(self.base, 'enable_input_require_grads'):
                self.base.enable_input_require_grads()

    def prepare_routing(self, cluster_id=None, **kwargs) -> Optional[Dict]:
        """Default: no routing info. Subclasses override."""
        return None

    def forward(self, input_ids, attention_mask=None, labels=None,
                cluster_id=None, **kwargs) -> LoRAOutputs:
        routing = self.prepare_routing(cluster_id=cluster_id, **kwargs)
        set_routing_all(self.base, routing)
        out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                         labels=labels)
        aux = sum_aux_losses(self.base).to(out.loss.device if out.loss is not None
                                            else out.logits.device)
        return LoRAOutputs(loss=out.loss, logits=out.logits, aux_loss=aux)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ── 1. SingleLoRAModel (Hu 2022 vanilla) ────────────────────────────────────


class SingleLoRAModel(BaseLoRAModel):
    """Hu 2022 LoRA, applied to all 7 linear modules at rank r."""

    def __init__(self, base_model_name: str, rank: int = 16, alpha: float = 32.0,
                 dropout: float = 0.05,
                 target_modules=DEFAULT_TARGETS_ALL,
                 torch_dtype=torch.bfloat16,
                 attn_implementation: Optional[str] = None):
        super().__init__(base_model_name, torch_dtype, attn_implementation)
        self.rank = rank

        def factory(base_linear, target_name, site_role):
            return SingleLoRAAdapter(
                in_features=base_linear.in_features,
                out_features=base_linear.out_features,
                rank=rank, alpha=alpha, dropout=dropout)

        self._inject(factory, target_modules)


# ── 2. MultiBranchLoRAModel (OURS + no-routing) ────────────────────────────


class MultiBranchLoRAModel(BaseLoRAModel):
    """K parallel LoRAs per target linear, with optional shared LoRA (SE).

    Routing comes from an external algorithm (SoftSpecDrop for Ours, NoDropout
    for the architecture-matched baseline). The algorithm's `get_mask` output
    is passed through `prepare_routing` each step.
    """

    def __init__(self, base_model_name: str, num_experts: int = 20,
                 rank: int = 16, alpha: float = 32.0, dropout: float = 0.05,
                 shared_expert_rank: int = 0,
                 target_modules=DEFAULT_TARGETS_ALL,
                 torch_dtype=torch.bfloat16,
                 attn_implementation: Optional[str] = None):
        super().__init__(base_model_name, torch_dtype, attn_implementation)
        self.num_experts = num_experts
        self.rank = rank
        self.shared_expert_rank = shared_expert_rank
        self._mask_scale = None  # set by trainer via set_mask_scale

        def factory(base_linear, target_name, site_role):
            return MultiBranchSoftSpecDropAdapter(
                in_features=base_linear.in_features,
                out_features=base_linear.out_features,
                num_experts=num_experts, rank=rank, alpha=alpha,
                dropout=dropout, shared_expert_rank=shared_expert_rank)

        self._inject(factory, target_modules)

    def set_mask_scale(self, mask_scale: float):
        """Install the fixed-denominator `S = p_a + (K-1) p_i` from algorithm."""
        self._mask_scale = mask_scale

    def prepare_routing(self, cluster_id=None, mask=None, **kwargs):
        # Trainer passes mask (B, K) from algorithm.get_mask(cluster_id).
        if mask is None:
            return None
        return {'mask': mask, 'mask_scale': self._mask_scale}


# ── 3. HydraLoRAModel (Tian 2024 NeurIPS) ──────────────────────────────────


class HydraLoRAModel(BaseLoRAModel):
    """Asymmetric 1A + N B with learned softmax gate per site. Routing ignored."""

    def __init__(self, base_model_name: str, num_B_heads: int = 8,
                 rank: int = 8, alpha: float = 16.0, dropout: float = 0.05,
                 target_modules=DEFAULT_TARGETS_ALL,
                 torch_dtype=torch.bfloat16,
                 attn_implementation: Optional[str] = None):
        super().__init__(base_model_name, torch_dtype, attn_implementation)
        self.num_B_heads = num_B_heads
        self.rank = rank

        def factory(base_linear, target_name, site_role):
            return HydraLoRAAdapter(
                in_features=base_linear.in_features,
                out_features=base_linear.out_features,
                num_B_heads=num_B_heads, rank=rank, alpha=alpha,
                dropout=dropout)

        self._inject(factory, target_modules)


# ── 4. LoRAMoEModel (Dou 2023 ACL — FFN-only) ──────────────────────────────


class LoRAMoEModel(BaseLoRAModel):
    """K LoRA experts + learned softmax gate + balance loss. FFN-only per
    Dou 2023 Sec 3.1 ("replace the linear layer in the FFN")."""

    def __init__(self, base_model_name: str, num_experts: int = 6,
                 rank: int = 4, alpha: float = 8.0, dropout: float = 0.05,
                 balance_weight: float = 0.1,
                 target_modules=DEFAULT_TARGETS_MLP,
                 torch_dtype=torch.bfloat16,
                 attn_implementation: Optional[str] = None):
        super().__init__(base_model_name, torch_dtype, attn_implementation)
        self.num_experts = num_experts
        self.rank = rank
        self.balance_weight = balance_weight

        def factory(base_linear, target_name, site_role):
            return LoRAMoEAdapter(
                in_features=base_linear.in_features,
                out_features=base_linear.out_features,
                num_experts=num_experts, rank=rank, alpha=alpha,
                dropout=dropout)

        self._inject(factory, target_modules)

    def forward(self, input_ids, attention_mask=None, labels=None,
                cluster_id=None, **kwargs) -> LoRAOutputs:
        out = super().forward(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels, cluster_id=cluster_id, **kwargs)
        # LoRAMoE adds β · ℒ_lbc to the main CE loss.
        if out.loss is not None:
            out.loss = out.loss + self.balance_weight * out.aux_loss
        return out


# ── 5. MoCLEModel (Gou 2024) ───────────────────────────────────────────────


class MoCLEModel(BaseLoRAModel):
    """E task + 1 universal LoRA experts, cluster-conditional top-1 gate."""

    def __init__(self, base_model_name: str, num_task_experts: int = 4,
                 num_clusters: int = 64, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.05, temperature: float = 0.05,
                 noise_std: float = 0.0,
                 cluster_embed_dim: Optional[int] = None,
                 target_modules=DEFAULT_TARGETS_ALL,
                 torch_dtype=torch.bfloat16,
                 attn_implementation: Optional[str] = None):
        super().__init__(base_model_name, torch_dtype, attn_implementation)
        self.num_task_experts = num_task_experts
        self.num_clusters = num_clusters
        self.rank = rank

        def factory(base_linear, target_name, site_role):
            return MoCLEAdapter(
                in_features=base_linear.in_features,
                out_features=base_linear.out_features,
                num_task_experts=num_task_experts, rank=rank,
                num_clusters=num_clusters,
                cluster_embed_dim=cluster_embed_dim,
                alpha=alpha, dropout=dropout,
                temperature=temperature, noise_std=noise_std)

        self._inject(factory, target_modules)

    def init_all_cluster_embeddings(self, centroids: torch.Tensor):
        """Copy BGE centroids into every MoCLEAdapter's cluster_embeddings.

        Gou 2024 recommends initializing cluster embeddings to BGE-large
        centroids of the cluster (paper Sec 3.3). centroids: (K, D_ce)."""
        for m in self.modules():
            if isinstance(m, MoCLEAdapter):
                if centroids.shape[1] != m.cluster_embed_dim:
                    raise ValueError(
                        f"centroids D={centroids.shape[1]} != "
                        f"cluster_embed_dim={m.cluster_embed_dim}")
                m.init_cluster_embeddings(centroids.to(
                    m.cluster_embeddings.weight.device))

    def prepare_routing(self, cluster_id=None, **kwargs):
        if cluster_id is None:
            return None
        return {'cluster_id': cluster_id}


# ── Factory ─────────────────────────────────────────────────────────────────


def build_lora_model(cfg) -> BaseLoRAModel:
    """Build a LoRA model from a config dict (YAML → dict).

    Required fields:
        cfg['model']['type']             ∈ {single_lora, multi_branch_lora,
                                            hydra_lora, lora_moe, mocle}
        cfg['model']['base_model_name']  HF hub id
        cfg['model']['rank'], ['alpha'], …

    Optional:
        cfg['model']['gradient_checkpointing']  default True. Set False
            to trade VRAM for ~40% speedup on compute-bound runs; at
            bs=4 seq=1024 Llama-3.2-3B this pushes VRAM usage close to
            32GB — verify no OOM before committing to a long run.

    Returns a BaseLoRAModel subclass. The returned object has base weights
    frozen; `.num_trainable` reports the LoRA-only parameter count.
    """
    mcfg = cfg['model']
    model_type = mcfg['type']
    base_name = mcfg['base_model_name']
    dtype = mcfg.get('torch_dtype', 'bfloat16')
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                 'float32': torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    attn_impl = mcfg.get('attn_implementation', None)

    if model_type == 'single_lora':
        return SingleLoRAModel(
            base_model_name=base_name,
            rank=mcfg.get('rank', 16), alpha=mcfg.get('alpha', 32.0),
            dropout=mcfg.get('dropout', 0.05),
            target_modules=tuple(mcfg.get('target_modules', DEFAULT_TARGETS_ALL)),
            torch_dtype=torch_dtype, attn_implementation=attn_impl)
    if model_type == 'multi_branch_lora':
        return MultiBranchLoRAModel(
            base_model_name=base_name,
            num_experts=mcfg.get('num_experts', 20),
            rank=mcfg.get('rank', 16), alpha=mcfg.get('alpha', 32.0),
            dropout=mcfg.get('dropout', 0.05),
            shared_expert_rank=mcfg.get('shared_expert_rank', 0),
            target_modules=tuple(mcfg.get('target_modules', DEFAULT_TARGETS_ALL)),
            torch_dtype=torch_dtype, attn_implementation=attn_impl)
    if model_type == 'hydra_lora':
        return HydraLoRAModel(
            base_model_name=base_name,
            num_B_heads=mcfg.get('num_B_heads', 8),
            rank=mcfg.get('rank', 8), alpha=mcfg.get('alpha', 16.0),
            dropout=mcfg.get('dropout', 0.05),
            target_modules=tuple(mcfg.get('target_modules', DEFAULT_TARGETS_ALL)),
            torch_dtype=torch_dtype, attn_implementation=attn_impl)
    if model_type == 'lora_moe':
        return LoRAMoEModel(
            base_model_name=base_name,
            num_experts=mcfg.get('num_experts', 6),
            rank=mcfg.get('rank', 4), alpha=mcfg.get('alpha', 8.0),
            dropout=mcfg.get('dropout', 0.05),
            balance_weight=mcfg.get('balance_weight', 0.1),
            target_modules=tuple(mcfg.get('target_modules', DEFAULT_TARGETS_MLP)),
            torch_dtype=torch_dtype, attn_implementation=attn_impl)
    if model_type == 'mocle':
        return MoCLEModel(
            base_model_name=base_name,
            num_task_experts=mcfg.get('num_task_experts', 4),
            num_clusters=mcfg.get('num_clusters', 64),
            rank=mcfg.get('rank', 8), alpha=mcfg.get('alpha', 16.0),
            dropout=mcfg.get('dropout', 0.05),
            temperature=mcfg.get('temperature', 0.05),
            noise_std=mcfg.get('noise_std', 0.0),
            cluster_embed_dim=mcfg.get('cluster_embed_dim', None),
            target_modules=tuple(mcfg.get('target_modules', DEFAULT_TARGETS_ALL)),
            torch_dtype=torch_dtype, attn_implementation=attn_impl)
    raise ValueError(f"Unknown LoRA model type: {model_type!r}")


def _apply_gradient_checkpointing_override(model: BaseLoRAModel, cfg) -> None:
    """Optional post-build override to disable gradient checkpointing.

    Default path in BaseLoRAModel._inject already enables grad_ckpt (saves
    VRAM). Cfg.model.gradient_checkpointing=false flips it off for
    compute-bound runs where VRAM budget allows (~6GB extra activations
    at bs=4 seq=1024 Llama-3.2-3B). Leaves enable_input_require_grads as-is
    (harmless for the no-ckpt path)."""
    if 'gradient_checkpointing' not in cfg.get('model', {}):
        return  # unset → keep default (True)
    want = bool(cfg['model']['gradient_checkpointing'])
    if not want and hasattr(model.base, 'gradient_checkpointing_disable'):
        model.base.gradient_checkpointing_disable()

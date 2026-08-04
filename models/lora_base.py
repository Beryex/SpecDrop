"""Shared HF CausalLM wrapping infrastructure for the LoRA post-train track.

All LoRA methods in this track (SingleLoRA, MultiBranch-LoRA (ours + no-routing),
HydraLoRA, LoRAMoE, MoCLE) share the same skeleton:

  1. Load a frozen HF CausalLM (Llama-3.2-3B base by default).
  2. Walk the model tree, locate target linear modules (q/k/v/o/gate/up/down),
     and replace each with a wrapper that adds a method-specific LoRA adapter
     in parallel with the frozen base: `y = base_linear(x) + adapter(x, ...)`.
  3. Freeze all base parameters so only the inserted adapters are trainable.

Different methods differ in:
  - Which target modules get wrapped (LoRAMoE is FFN-only; others all-linear).
  - What the adapter's routing mechanism is (fixed mask from algorithm,
    learned softmax gate, asymmetric 1A+NB, cluster-conditional, etc.).
  - Whether there are auxiliary losses (LoRAMoE balance loss, etc.).

This module provides the machinery that is IDENTICAL across methods (wrapper
class, model-tree walk, base freezing, HF forward plumbing, loss collection);
per-method adapter classes live in their own files.

Backward-compat: 0 impact on CIFAR / NLP / ViT paths. No existing file is
modified by this module's presence.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

# Default set of target module NAMES (suffix match on the parent-relative
# module name). Covers Llama / Mistral / Qwen architectures.
DEFAULT_TARGETS_ATTN = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
DEFAULT_TARGETS_MLP = ('gate_proj', 'up_proj', 'down_proj')
DEFAULT_TARGETS_ALL = DEFAULT_TARGETS_ATTN + DEFAULT_TARGETS_MLP


class LoRAInjectedLinear(nn.Module):
    """Wraps a frozen nn.Linear with a method-specific adapter.

    forward(x, *, routing=None, **kwargs) =
        base_linear(x) + adapter(x, routing=routing, **kwargs)

    The adapter is any nn.Module that accepts `(x, routing=<routing_info>)`
    and returns a tensor of the same output shape as `base_linear(x)`. The
    "routing_info" is method-specific: None (single LoRA), (B, K) soft mask
    (ours + no-routing), or cluster_id ints (MoCLE).

    State the wrapper owns:
      - `self.base`: frozen base Linear (grads off).
      - `self.adapter`: trainable LoRA block.
      - `self.current_routing`: routing info installed by the parent model
        before each forward pass. `set_routing()` is the install point.
      - `self.aux_loss`: float scalar tensor, accumulated by some adapters
        (e.g., LoRAMoE balance loss) per forward for the trainer to pull.

    Attributes:
        site_role: optional 'attn' or 'mlp' for method-specific logic that
            wants to differentiate (e.g., LoRAMoE applies FFN-only, so wrapped
            attention sites would never be created — but having the tag makes
            diagnostics cleaner).
        target_name: original name of the target linear (e.g., 'q_proj'),
            useful for MoE baselines that need per-site gating.
    """

    def __init__(self, base_linear: nn.Linear, adapter: nn.Module,
                 site_role: str = '', target_name: str = ''):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(
                f"base_linear must be nn.Linear, got {type(base_linear).__name__}")
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.adapter = adapter
        self.site_role = site_role
        self.target_name = target_name
        self.current_routing = None
        self.aux_loss = torch.tensor(0.0)

    def set_routing(self, routing) -> None:
        """Install this batch's routing info (mask tensor, cluster_ids, or None)."""
        self.current_routing = routing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        adapter_out = self.adapter(x, routing=self.current_routing)
        # Some adapters expose `.last_aux_loss` after forward for the trainer
        # to aggregate (LoRAMoE balance loss). Default: zero.
        last_aux = getattr(self.adapter, 'last_aux_loss', None)
        if last_aux is not None:
            self.aux_loss = last_aux
        return y + adapter_out


def inject_lora_adapters(
    base_model: nn.Module,
    adapter_factory: Callable[[nn.Linear, str, str], nn.Module],
    target_modules: Iterable[str] = DEFAULT_TARGETS_ALL,
) -> List[str]:
    """Walk `base_model` in-place and wrap every nn.Linear whose parent-relative
    name matches one of `target_modules`.

    `adapter_factory(base_linear, target_name, site_role)` returns an adapter
    module for the given site. site_role is 'attn' for q/k/v/o, 'mlp' for
    gate/up/down, '' otherwise.

    Returns the list of fully-qualified names that were wrapped (for logging /
    assertion in tests).
    """
    target_set = set(target_modules)
    wrapped_sites: List[str] = []

    # Discover candidate parents first, then mutate — safer than iterating
    # and mutating the module tree simultaneously.
    candidates: List[Tuple[nn.Module, str, str]] = []
    for parent_name, parent in base_model.named_modules():
        for child_name, child in parent.named_children():
            if child_name in target_set and isinstance(child, nn.Linear):
                fqn = f'{parent_name}.{child_name}' if parent_name else child_name
                candidates.append((parent, child_name, fqn))

    for parent, child_name, fqn in candidates:
        base_linear = getattr(parent, child_name)
        role = ('attn' if child_name in DEFAULT_TARGETS_ATTN
                else 'mlp' if child_name in DEFAULT_TARGETS_MLP else '')
        adapter = adapter_factory(base_linear, child_name, role)
        wrapped = LoRAInjectedLinear(
            base_linear, adapter, site_role=role, target_name=child_name)
        setattr(parent, child_name, wrapped)
        wrapped_sites.append(fqn)

    return wrapped_sites


def freeze_base_params(model: nn.Module, exclude_classes=(LoRAInjectedLinear,)):
    """Set requires_grad=False on every parameter NOT inside an adapter.

    Args:
        model: the HF model with LoRA already injected (via inject_lora_adapters).
        exclude_classes: classes whose descendant params stay trainable. By
            default, only LoRAInjectedLinear's `adapter` (not `base`) stays
            trainable because we already flipped `base` off in __init__.

    Returns the trainable parameter count for confirmation logs.
    """
    # First pass: freeze everything.
    for p in model.parameters():
        p.requires_grad_(False)
    # Second pass: unfreeze any parameter that lives inside an excluded class.
    # For LoRAInjectedLinear this restores the adapter; the base (still inside
    # the LoRAInjectedLinear module) would also become trainable — so we hit
    # the adapter specifically.
    for m in model.modules():
        if isinstance(m, LoRAInjectedLinear):
            for p in m.adapter.parameters():
                p.requires_grad_(True)
            # Keep base frozen.
            for p in m.base.parameters():
                p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable


def set_routing_all(model: nn.Module, routing) -> None:
    """Broadcast `routing` to every injected adapter's `set_routing()`.

    Called by the trainer once per forward pass with this batch's routing
    info (mask, cluster_ids, or None depending on method).
    """
    for m in model.modules():
        if isinstance(m, LoRAInjectedLinear):
            m.set_routing(routing)


def sum_aux_losses(model: nn.Module) -> torch.Tensor:
    """Sum any per-site aux losses (e.g., LoRAMoE balance term) across sites.

    Returns a scalar tensor on the same device as the first found aux tensor,
    or a CPU zero tensor if no aux exists.
    """
    total = None
    for m in model.modules():
        if isinstance(m, LoRAInjectedLinear):
            if total is None:
                total = m.aux_loss.clone()
            else:
                total = total + m.aux_loss
    if total is None:
        return torch.tensor(0.0)
    return total

"""Abstract base class for modular dropout algorithms.

To implement a new algorithm, subclass ModularDropout and implement get_mask().
Optionally override on_epoch_end() for per-epoch logic (e.g., stability freezing).

Two routing modes:
  1. Category-only (default): mask depends on category_ids alone.
     The training loop calls:
         mask = algorithm.get_mask(category_ids, training=True)
  2. Feature-based (needs_features=True): mask depends on stem features.
     The training loop calls:
         stem = model.get_stem_features(x)
         features = GAP(stem).flatten(1)   # (B, D)
         mask = algorithm.get_mask(category_ids, training=True, features=features)
         logits = model.forward_from_stem(stem, branch_mask=mask)
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class ModularDropout(ABC):
    """Base class for modular dropout strategies.

    Subclasses define how branch activation masks are generated from data categories.
    """

    def __init__(self, num_modules: int, num_categories: int):
        self.num_modules = num_modules
        self.num_categories = num_categories

    # Set True in subclasses that route from stem features (e.g., Learned, Soft MoE, COMET)
    needs_features = False

    # Set True in subclasses that ask the model to use per-sample adaptive
    # denominator (÷Σm_k) instead of fixed S. Used ONLY by the E3 ablation
    # variants (stoch_naive / rand_naive); shipped algorithms keep this False.
    use_adaptive_denom = False

    @abstractmethod
    def get_mask(self, category_ids: torch.Tensor, training: bool = True, **kwargs) -> torch.Tensor:
        """Generate branch activation mask.

        Args:
            category_ids: (B,) int tensor of superclass labels.
            training: if True, sample stochastic binary mask;
                      if False, return expected activation probabilities.
            **kwargs: optional extra args.
                features: (B, D) float tensor of pooled stem features
                          (only when needs_features=True).

        Returns:
            mask: (B, num_modules) float tensor.
        """
        ...

    @property
    def expected_mask_sum(self):
        """Expected sum of mask values per sample. Used for fixed-denominator merge.

        Override in subclasses to enable train-test expectation consistency.
        Returns None to fall back to stochastic (per-sample) normalization.
        """
        return None

    def get_auxiliary_loss(self):
        """Return auxiliary loss to add to CE loss (e.g., MI loss for Mod-Squad).

        Override in subclasses. Returns 0 by default.
        """
        return 0.0

    def state_dict(self) -> dict:
        """Return algorithm state for checkpointing. Override in subclasses with internal state."""
        return {}

    def load_state_dict(self, state: dict):
        """Restore algorithm state from checkpoint. Override in subclasses."""
        pass

    def on_epoch_end(self, branches: nn.ModuleList, epoch: int) -> dict:
        """Called at the end of each training epoch.

        Override this for per-epoch logic like stability-guided freezing.

        Args:
            branches: the model's nn.ModuleList of branch modules.
            epoch: current epoch number (0-indexed).

        Returns:
            dict with status info (e.g., which modules are frozen).
        """
        return {}

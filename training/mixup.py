"""Mixup (Zhang et al., ICLR 2018) and CutMix (Yun et al., ICCV 2019).

DeiT training recipe (Touvron et al. 2020) applies one of the two per batch
with switch probability 0.5. Label smoothing is folded into the one-hot targets.

Usage:
    mixer = MixupCutmix(mixup_alpha=0.8, cutmix_alpha=1.0,
                        switch_prob=0.5, label_smoothing=0.1, num_classes=1000)
    mixed_images, soft_targets = mixer(images, hard_labels)
    logits = model(mixed_images)
    loss = F.cross_entropy(logits, soft_targets)   # PyTorch supports soft targets
"""

import torch
import torch.nn.functional as F


def _rand_bbox(H, W, lam):
    """Random rectangle covering approximately (1 - lam) fraction of image area."""
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


class MixupCutmix:
    """Batch-level Mixup + CutMix with soft-target output.

    Args:
        mixup_alpha: Beta distribution shape for Mixup lambda (DeiT default 0.8).
        cutmix_alpha: Beta distribution shape for CutMix lambda (DeiT default 1.0).
        switch_prob: probability of choosing CutMix over Mixup per batch.
        label_smoothing: epsilon in one-hot (uniform mass ε/C on off-target classes).
        num_classes: number of classes C (for one-hot targets).
    """

    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, switch_prob=0.5,
                 label_smoothing=0.1, num_classes=1000):
        assert mixup_alpha > 0 or cutmix_alpha > 0, \
            "At least one of mixup_alpha / cutmix_alpha must be > 0"
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def _smoothed_onehot(self, y):
        """(B,) int labels → (B, C) soft one-hot with label smoothing."""
        B = y.size(0)
        C = self.num_classes
        off = self.label_smoothing / C
        on = 1.0 - self.label_smoothing + off
        oh = torch.full((B, C), off, device=y.device, dtype=torch.float)
        oh.scatter_(1, y.unsqueeze(1), on)
        return oh

    def __call__(self, images, labels):
        """Apply Mixup or CutMix batch-wise; return mixed (images, soft_targets)."""
        B = images.size(0)
        targets = self._smoothed_onehot(labels)

        use_cutmix = (torch.rand(1).item() < self.switch_prob
                      and self.cutmix_alpha > 0)

        if use_cutmix:
            lam = torch.distributions.Beta(self.cutmix_alpha,
                                            self.cutmix_alpha).sample().item()
            idx = torch.randperm(B, device=images.device)
            H, W = images.size(2), images.size(3)
            x1, y1, x2, y2 = _rand_bbox(H, W, lam)
            images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
            # Recompute lam from actual patched area (paper's adjustment).
            lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
            mixed = images
        else:
            lam = torch.distributions.Beta(self.mixup_alpha,
                                            self.mixup_alpha).sample().item() \
                  if self.mixup_alpha > 0 else 1.0
            idx = torch.randperm(B, device=images.device)
            mixed = lam * images + (1 - lam) * images[idx]

        soft_targets = lam * targets + (1 - lam) * targets[idx]
        return mixed, soft_targets

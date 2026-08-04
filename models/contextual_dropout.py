"""Contextual Dropout (Fan et al., ICLR 2021).

Paper: "Contextual Dropout: An Efficient Sample-Dependent Dropout Module"
(https://arxiv.org/abs/2103.04181)

At each layer l, a small context network h^l_φ(U^l) produces per-channel dropout
logits α^l = Φ^l_2(LeakyReLU(Φ^l_1(avgpool(U^l)))). The mask z^l is sampled from a
distribution conditioned on α^l and applied element-wise (broadcasting over spatial
dims for conv feature maps).

**Gaussian variant** (Section 2.3, reparameterizable; Algorithm 3, Appendix B):
    σ_t(α) = 1 / (1 + exp(-t · α))             scaled sigmoid, t = 0.01 (App. C)
    z^l = 1 + sqrt(σ_t(α) / (1 − σ_t(α))) · ε  reparameterized, ε ~ N(0, I)
    Inference (Section 2.5): z^l = E[z^l] = 1 (deterministic identity).

The scaled sigmoid with t = 0.01 (Appendix C, "Choice of hyper-parameters")
is a critical numerical stabilizer. With t = 1 (vanilla sigmoid), α drifting
to ±∞ pushes σ to {0, 1}, making var = σ/(1-σ) explode to >10^2 and the
54-layer activation variance compound past fp32 range. With t = 0.01, even
|α| = 100 keeps σ_t ∈ (0.27, 0.73) so var ≈ 1 throughout training — the
paper gets away without any explicit clipping precisely because of this.

Placement follows the paper (Appendix C.1 + Figure 6 for WRN): CD is placed
**inside** each residual block, after the first conv/bn/relu and before the
second conv. The identity shortcut bypasses CD entirely, so dropout only
perturbs the non-residual branch and does not compound through the 54-layer
tower. An earlier version applying CD on the block *output* saturated to
NaN within 43 steps even in fp32, because every shortcut also got multiplied
by z, and the 54-level variance product diverged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock


EPS = 1e-6
# Loose safety clip. With t = 0.01 scaled sigmoid, σ_t stays in (0.27, 0.73)
# for α ∈ (-100, 100) — we never touch these bounds in practice. They exist
# only to short-circuit log(0) / div-by-zero if α ever NaNs from upstream.
_SIGMA_CLIP = (1e-6, 1.0 - 1e-6)
# Default sigmoid scaling factor from the paper (Appendix C.1).
DEFAULT_SIGMOID_SCALE_T = 0.01


class ContextualDropout2d(nn.Module):
    """Per-channel Gaussian contextual dropout on a (B, C, H, W) feature map.

    Context network (Section 2.3):
        α = Φ_2(LeakyReLU(Φ_1(avgpool(U))))   # (B, C)
    Mask sampling (Gaussian, reparameterized):
        σ = sigmoid(α)
        var = σ / (1 - σ)                      # clip σ for numerical stability
        ε ~ N(0, 1)
        z = 1 + sqrt(var) * ε
    Inference: z = E[z] = 1.

    KL regularization (Section 2.4 ELBO): at training time the layer also
    returns the per-element KL divergence of its variational posterior
    q(z | x) = N(1, var) against a fixed prior p(z) = N(1, 1):
        KL(N(1, σ²_q) ‖ N(1, 1)) = ½ · (σ²_q − 1 − log σ²_q)
    Aggregated and multiplied by `kl_weight` in ContextualDropoutResNet.
    """

    def __init__(self, channels, reduction=8, sigmoid_scale_t=DEFAULT_SIGMOID_SCALE_T):
        super().__init__()
        self.channels = channels
        self.sigmoid_scale_t = sigmoid_scale_t
        reduced = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(reduced, channels)
        # KL term produced by last forward in training mode (zero in eval).
        self.last_kl = torch.tensor(0.0)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) conv feature map or (B, C) flat feature.
        Returns:
            same shape as x, element-wise scaled by sampled z.
        """
        if not self.training:
            # E[z] = 1 → identity. No KL contribution at test time.
            self.last_kl = torch.tensor(0.0, device=x.device)
            return x

        is_4d = (x.dim() == 4)
        if is_4d:
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)    # (B, C)
        else:
            pooled = x                                          # (B, C)

        alpha = self.fc2(self.act(self.fc1(pooled)))            # (B, C)
        # Scaled sigmoid with t = 0.01 (Appendix C): controls the encoder
        # learning rate by dampening σ's response to α. Keeps σ_t near 0.5
        # throughout training regardless of α drift, which is why the paper
        # does not need any explicit clipping on σ or the sampled z.
        sigma = torch.sigmoid(self.sigmoid_scale_t * alpha)
        sigma = sigma.clamp(_SIGMA_CLIP[0], _SIGMA_CLIP[1])     # only against NaN
        var = sigma / (1.0 - sigma)                             # (B, C)
        eps = torch.randn_like(var)
        z = 1.0 + torch.sqrt(var + EPS) * eps                   # (B, C)

        # KL(N(1, var) ‖ N(1, 1)) per (B, C); mean-reduce for a scalar.
        kl_per = 0.5 * (var - 1.0 - torch.log(var + EPS))
        self.last_kl = kl_per.mean()

        if is_4d:
            z = z.view(z.shape[0], z.shape[1], 1, 1)
        return x * z


class ContextualBasicBlock(nn.Module):
    """BasicBlock with ContextualDropout2d placed **inside** the residual path,
    after the first conv/bn/relu — matching the paper's WRN placement (Fan
    et al. 2021, Appendix C.1 + Figure 6):

        in → conv1 → bn1 → relu → [CD] → conv2 → bn2 → +identity → relu

    Critical: CD is *not* on the block output; the identity shortcut bypasses
    it entirely. This keeps the residual signal path pristine — dropout noise
    only perturbs the non-residual branch, so it does not compound through
    the 54-layer tower the way "dropout on block output" did.
    """

    def __init__(self, in_channels, out_channels, stride=1, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ctx_drop = ContextualDropout2d(out_channels, reduction=reduction)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.ctx_drop(out)                    # CD inside residual path
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


def _contextual_make_layer(in_channels, out_channels, num_blocks, stride,
                            reduction=8):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    ch = in_channels
    for s in strides:
        layers.append(ContextualBasicBlock(ch, out_channels, s, reduction))
        ch = out_channels
    return nn.Sequential(*layers)


class ContextualDropoutResNet(nn.Module):
    """ResNet-110 + Contextual Dropout at every residual block (54 blocks total).

    Args:
        num_blocks: blocks per ResNet layer group (18 = ResNet-110).
        num_classes: output classes.
        base_channels: width of conv1 (16 for CIFAR ResNet).
        reduction: context network bottleneck ratio γ (paper uses γ ∈ [8, 16]).
    """

    def __init__(self, num_blocks=18, num_classes=100, base_channels=16,
                 reduction=8, kl_weight=1e-4):
        super().__init__()
        self.kl_weight = kl_weight
        w = base_channels
        self.conv1 = nn.Conv2d(3, w, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)

        self.layer1 = _contextual_make_layer(w, w, num_blocks, stride=1,
                                              reduction=reduction)
        self.layer2 = _contextual_make_layer(w, 2 * w, num_blocks, stride=2,
                                              reduction=reduction)
        self.layer3 = _contextual_make_layer(2 * w, 4 * w, num_blocks, stride=2,
                                              reduction=reduction)
        self.fc = nn.Linear(4 * w, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, branch_mask=None, example_ids=None):
        """branch_mask / example_ids are ignored (API compatibility).

        Returns:
            (logits, kl_loss) when training (kl_loss is the aggregated KL
            regularizer over all 54 ContextualDropout2d modules, scaled by
            `self.kl_weight`). At eval, kl_loss is 0 (deterministic forward).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        logits = self.fc(out)

        # Aggregate KL contributions from every ContextualDropout2d module.
        kl_total = torch.tensor(0.0, device=x.device)
        if self.training and self.kl_weight > 0:
            for m in self.modules():
                if isinstance(m, ContextualDropout2d):
                    kl_total = kl_total + m.last_kl
            kl_total = kl_total * self.kl_weight
        return logits, kl_total


def contextual_dropout_resnet110(num_classes=100, reduction=8, kl_weight=1e-4):
    """ResNet-110 + Contextual Dropout for CIFAR-100."""
    return ContextualDropoutResNet(
        num_blocks=18, num_classes=num_classes, base_channels=16,
        reduction=reduction, kl_weight=kl_weight,
    )

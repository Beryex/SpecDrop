"""Standard ResNet for CIFAR-100 (ResNet-20/32/56)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block: two 3x3 convs with skip connection.

    Args:
        survival_prob: if < 1.0, applies stochastic depth (Huang et al., ECCV 2016).
            Training: randomly skip conv path with prob (1 - survival_prob).
            Testing: scale conv path by survival_prob.
    """

    def __init__(self, in_channels, out_channels, stride=1, survival_prob=1.0):
        super().__init__()
        self.survival_prob = survival_prob
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        if self.survival_prob < 1.0 and self.training:
            if torch.rand(1, device=identity.device) > self.survival_prob:
                # Huang et al. (2016): H_l = relu(0 + H_{l-1}) when block is dropped
                return F.relu(identity)

        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))

        if self.survival_prob < 1.0 and not self.training:
            residual = residual * self.survival_prob

        return F.relu(residual + identity)


def make_layer(in_channels, out_channels, num_blocks, stride,
               survival_probs=None):
    """Build a sequence of BasicBlocks.

    Args:
        survival_probs: optional list of per-block survival probabilities
            for stochastic depth. Length must equal num_blocks. Default None = no dropping.
    """
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    ch = in_channels
    for i, s in enumerate(strides):
        sp = survival_probs[i] if survival_probs is not None else 1.0
        layers.append(BasicBlock(ch, out_channels, s, survival_prob=sp))
        ch = out_channels
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """Standard CIFAR ResNet.

    With num_blocks=3: ResNet-20 (3 groups * 3 blocks * 2 convs + 2 = 20).
    With num_blocks=5: ResNet-32.
    """

    def __init__(self, num_blocks=3, num_classes=100, dropout_rate=0.0, base_channels=16,
                 stochastic_depth_rate=0.0):
        super().__init__()
        w = base_channels
        self.conv1 = nn.Conv2d(3, w, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)

        # Linear survival-probability schedule (Huang et al., 2016):
        # 1.0 for first block, (1 - stochastic_depth_rate) for last.
        total_blocks = num_blocks * 3
        if stochastic_depth_rate > 0 and total_blocks > 1:
            sd_probs = [1.0 - stochastic_depth_rate * (i / (total_blocks - 1))
                        for i in range(total_blocks)]
        else:
            sd_probs = [1.0] * total_blocks

        self.layer1 = make_layer(w, w, num_blocks, stride=1,
                                 survival_probs=sd_probs[:num_blocks])
        self.layer2 = make_layer(w, 2 * w, num_blocks, stride=2,
                                 survival_probs=sd_probs[num_blocks:2 * num_blocks])
        self.layer3 = make_layer(2 * w, 4 * w, num_blocks, stride=2,
                                 survival_probs=sd_probs[2 * num_blocks:])

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(4 * w, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, branch_mask=None):
        """Forward pass. branch_mask is ignored (compatibility with multi-branch API)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

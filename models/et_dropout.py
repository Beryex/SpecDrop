"""Example-Tied Dropout (Maini et al., ICML 2023) — faithful per-layer impl.

Paper: "Can Neural Network Memorization Be Localized?" (https://arxiv.org/abs/2307.09542)
Section 6.2: apply ETD after EACH layer (= each BasicBlock) of a ResNet. In every
block, a fraction (1 - α_mem) of the output channels are "generalization" channels
that are never dropped; the remaining α_mem fraction are "memorization" channels
with a FIXED per-example binary mask (pre-generated once with `mask_seed`).

Training: example i keeps generalization channels + m_i over memorization channels.
Inference: memorization channels are zeroed out entirely; only generalization
  channels contribute to the final logits (paper Section 3 protocol).

Mask storage:
  Total memorization channels summed across all 54 BasicBlocks in ResNet-110:
    layer1 (18 blocks × 16ch × α_mem=0.5) = 144
    layer2 (18 blocks × 32ch × α_mem=0.5) = 288
    layer3 (18 blocks × 64ch × α_mem=0.5) = 576
    Total = 1008 memorization channels
  Mask table: (num_examples=50000, 1008) → ~50MB as bool (stored as float buffer
  for fast indexing at forward time; 50000 × 1008 × 4B = 192MB).

Requires `example_ids` kwarg during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock


class ETDBasicBlock(nn.Module):
    """BasicBlock that applies Example-Tied Dropout to its output channels.

    The last `num_mem` output channels are "memorization" (per-example masked);
    the first `num_gen = out_channels - num_mem` are "generalization" (always on).
    A `channel_offset` tells the block where its memorization-mask slice lives in
    the parent model's global (num_examples × total_mem_channels) mask table.
    """

    def __init__(self, in_channels, out_channels, stride, alpha_mem,
                 channel_offset, mem_keep_rate):
        super().__init__()
        self.block = BasicBlock(in_channels, out_channels, stride)
        self.out_channels = out_channels
        self.num_mem = int(round(out_channels * alpha_mem))
        self.num_gen = out_channels - self.num_mem
        self.channel_offset = channel_offset   # offset into global mask table
        self._mem_keep_rate = mem_keep_rate

    def forward(self, x, example_masks=None, training=True):
        """
        Args:
            x: (B, C_in, H, W) input.
            example_masks: (B, total_mem_channels) slice of the parent's mask
                table for this batch's example_ids. May be None at eval.
            training: whether to apply training-time mask (else zero mem channels).
        """
        out = self.block(x)                       # (B, C_out, H, W)
        if self.num_mem == 0:
            return out

        if training and example_masks is not None:
            local = example_masks[:, self.channel_offset:
                                     self.channel_offset + self.num_mem]
            mask = local.view(-1, self.num_mem, 1, 1).to(out.dtype)
            out = torch.cat([
                out[:, :self.num_gen],
                out[:, self.num_gen:] * mask,
            ], dim=1)
        else:
            # Inference — zero out the memorization channels, following
            # Maini et al. 2023 §6.2 and Table 1: "At test time, we drop out
            # the neurons corresponding to p_mem, that is zero-out their
            # activations, in order to adapt the model in such a way that
            # the influence of memorization can be removed." Table 1 reports
            # this as the paper's canonical deployment protocol (CIFAR-10
            # 79.3% before → 82.7% after zero-out), so this is NOT only an
            # analysis knob. Srivastava-2014 scaling would preserve the
            # per-example memorization contribution at test time, defeating
            # ETD's purpose.
            out = torch.cat([
                out[:, :self.num_gen],
                torch.zeros_like(out[:, self.num_gen:]),
            ], dim=1)
        return out


class ExampleTiedDropoutResNet(nn.Module):
    """ResNet-110 with Example-Tied Dropout applied after every BasicBlock.

    Args:
        num_blocks: blocks per layer group (18 → ResNet-110).
        num_classes: classifier output.
        base_channels: conv1 width (16 for CIFAR).
        num_examples: size of training set (50000 for CIFAR-100 train).
        alpha_mem: fraction of each block's output channels that are
            memorization channels (paper defaults 0.5).
        mem_keep_rate: per-example Bernoulli keep probability over mem channels.
        mask_seed: RNG seed for the fixed mask table.
    """

    def __init__(self, num_blocks=18, num_classes=100, base_channels=16,
                 num_examples=50000, alpha_mem=0.5, mem_keep_rate=0.5,
                 mask_seed=42):
        super().__init__()
        assert 0.0 < alpha_mem < 1.0
        assert 0.0 < mem_keep_rate <= 1.0

        w = base_channels
        self.conv1 = nn.Conv2d(3, w, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)

        # Build 3 layer groups with ETD blocks, tracking running channel offset.
        self.layers = nn.ModuleList()
        offset = 0

        def _make_layer(in_ch, out_ch, n_blocks, stride):
            nonlocal offset
            blocks = nn.ModuleList()
            ch_in = in_ch
            for i in range(n_blocks):
                s = stride if i == 0 else 1
                block = ETDBasicBlock(ch_in, out_ch, s,
                                      alpha_mem=alpha_mem,
                                      channel_offset=offset,
                                      mem_keep_rate=mem_keep_rate)
                offset += block.num_mem
                blocks.append(block)
                ch_in = out_ch
            return blocks

        self.layer1 = _make_layer(w, w,       num_blocks, stride=1)
        self.layer2 = _make_layer(w, 2 * w,   num_blocks, stride=2)
        self.layer3 = _make_layer(2 * w, 4 * w, num_blocks, stride=2)
        self.total_mem_channels = offset

        self.num_examples = num_examples
        self.alpha_mem = alpha_mem
        self.mem_keep_rate = mem_keep_rate

        self.fc = nn.Linear(4 * w, num_classes)

        # Pre-generate fixed per-example Bernoulli masks over all memorization
        # channels in the network (shared across all blocks via global offset).
        gen = torch.Generator().manual_seed(mask_seed)
        masks = (torch.rand(num_examples, self.total_mem_channels, generator=gen)
                 < mem_keep_rate).to(torch.float32)
        # persistent=False excludes from state_dict / checkpoint. At
        # `num_examples=50000 × 1008 channels × 4B = 200MB` per ckpt, keeping
        # it persistent would bloat every checkpoint (and OOM on a 1.28M
        # ImageNet port). The mask table is deterministically reconstructed
        # from `mask_seed` whenever the model is re-instantiated.
        self.register_buffer('example_masks', masks, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, example_ids=None, branch_mask=None):
        if self.training:
            if example_ids is None:
                raise ValueError(
                    "ExampleTiedDropoutResNet requires `example_ids` during training "
                    "(dataset index of each sample, for fixed per-layer mask lookup).")
            if example_ids.max() >= self.num_examples or example_ids.min() < 0:
                raise IndexError(
                    f"example_ids out of range [0, {self.num_examples}).")
            masks = self.example_masks[example_ids]    # (B, total_mem_channels)
        else:
            masks = None

        out = F.relu(self.bn1(self.conv1(x)))
        for block in self.layer1:
            out = block(out, masks, training=self.training)
        for block in self.layer2:
            out = block(out, masks, training=self.training)
        for block in self.layer3:
            out = block(out, masks, training=self.training)

        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        logits = self.fc(out)
        # Uniform return signature: (logits, aux_loss=0) matches all other
        # faithful baselines (Switch / Mod-Squad / SoftMoE / DEMix / ContextualDropout).
        return logits, torch.tensor(0.0, device=x.device)


def example_tied_dropout_resnet110(num_classes=100, num_examples=50000,
                                    alpha_mem=0.5, mem_keep_rate=0.5,
                                    mask_seed=42):
    """ResNet-110 + Example-Tied Dropout for CIFAR-100 (per-layer channel-wise)."""
    return ExampleTiedDropoutResNet(
        num_blocks=18, num_classes=num_classes, base_channels=16,
        num_examples=num_examples, alpha_mem=alpha_mem,
        mem_keep_rate=mem_keep_rate, mask_seed=mask_seed,
    )

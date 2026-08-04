"""Multi-branch ResNet for CIFAR-100 with modular dropout support.

MultiBranchResNet: shared stem -> K parallel branches (layer2 only) -> merge -> shared head.
MultiBranchResNet110: shared conv1 -> K branches at ALL layers -> merge per layer -> shared FC.
GroupedMultiBranchResNet110: same architecture as MultiBranchResNet110 but fuses K branches
    into grouped convolutions for ~10-15x speedup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import make_layer


# ---------------------------------------------------------------------------
# Grouped-conv parallel blocks (fused K branches into one kernel per op)
# ---------------------------------------------------------------------------

class ParallelBasicBlock(nn.Module):
    """K parallel BasicBlocks fused via grouped convolution.

    Mathematically equivalent to K separate BasicBlocks processing the same
    input, but executed as a single CUDA kernel per operation.

    Input:  (B, K*in_channels, H, W)  — repeated input for K groups
    Output: (B, K*out_channels, H', W') — concatenated branch outputs
    """

    def __init__(self, K, in_channels, out_channels, stride=1, survival_prob=1.0):
        super().__init__()
        self.K = K
        self.survival_prob = survival_prob
        self.conv1 = nn.Conv2d(
            K * in_channels, K * out_channels, 3,
            stride=stride, padding=1, bias=False, groups=K)
        self.bn1 = nn.BatchNorm2d(K * out_channels)
        self.conv2 = nn.Conv2d(
            K * out_channels, K * out_channels, 3,
            stride=1, padding=1, bias=False, groups=K)
        self.bn2 = nn.BatchNorm2d(K * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(K * in_channels, K * out_channels, 1,
                          stride=stride, bias=False, groups=K),
                nn.BatchNorm2d(K * out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        if self.survival_prob < 1.0 and self.training:
            if torch.rand(1, device=identity.device) > self.survival_prob:
                return F.relu(identity)

        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))

        if self.survival_prob < 1.0 and not self.training:
            residual = residual * self.survival_prob

        return F.relu(residual + identity)


def parallel_make_layer(K, in_channels, out_channels, num_blocks, stride,
                        survival_probs=None):
    """Build a sequence of ParallelBasicBlocks (K branches fused)."""
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    ch = in_channels
    for i, s in enumerate(strides):
        sp = survival_probs[i] if survival_probs is not None else 1.0
        layers.append(ParallelBasicBlock(K, ch, out_channels, s, survival_prob=sp))
        ch = out_channels
    return nn.Sequential(*layers)


class MultiBranchResNet(nn.Module):
    """Multi-branch ResNet.

    Shared stem:  conv1 + layer1  (base_channels, 32x32)
    K branches:   each is a layer2  (base_channels->branch_channels, stride 2 -> 16x16)
    Merge:        masked average of branch outputs
    Shared head:  layer3  (branch_channels->4*base_channels, stride 2 -> 8x8) + pool + FC
    """

    def __init__(self, num_branches=4, num_blocks=3, num_classes=100,
                 branch_channels=16, base_channels=16,
                 stochastic_depth_rate=0.0, shared_expert=False):
        super().__init__()
        self.num_branches = num_branches
        self.shared_expert_flag = shared_expert
        w = base_channels

        # Compute per-block survival probs for stochastic depth
        # Total blocks: layer1(num_blocks) + branches(num_blocks) + layer3(num_blocks) = 3*num_blocks
        # Linear decay from 1.0 (first block) to (1 - stochastic_depth_rate) (last block)
        total_blocks = 3 * num_blocks
        sd_probs = None
        if stochastic_depth_rate > 0:
            sd_probs = [1.0 - stochastic_depth_rate * (i / (total_blocks - 1))
                        for i in range(total_blocks)]

        def _get_sp(layer_idx):
            """Get survival_probs slice for a layer group."""
            if sd_probs is None:
                return None
            start = layer_idx * num_blocks
            return sd_probs[start:start + num_blocks]

        # Shared stem
        self.conv1 = nn.Conv2d(3, w, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.layer1 = make_layer(w, w, num_blocks, stride=1,
                                 survival_probs=_get_sp(0))

        # K parallel branches (each equivalent to layer2, same depth index)
        branch_sp = _get_sp(1)
        self.branches = nn.ModuleList([
            make_layer(w, branch_channels, num_blocks, stride=2,
                       survival_probs=branch_sp)
            for _ in range(num_branches)
        ])

        # Optional shared expert: 1 extra branch always active for all inputs
        self.shared_expert_branch = None
        if shared_expert:
            self.shared_expert_branch = make_layer(
                w, branch_channels, num_blocks, stride=2,
                survival_probs=branch_sp)

        # Shared head
        self.layer3 = make_layer(branch_channels, 4 * w, num_blocks, stride=2,
                                 survival_probs=_get_sp(2))
        self.fc = nn.Linear(4 * w, num_classes)

        # Fixed denominator for train-test consistent merge.
        # Set externally via algorithm.expected_mask_sum; None = legacy stochastic norm.
        self.mask_scale = None
        # Opt-in per-sample adaptive denominator (÷Σm_k). Used ONLY for the
        # E3 stoch_naive / rand_naive ablations that demonstrate Jensen bias;
        # never active for shipped algorithms.
        self.use_adaptive_denom = False

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_stem_features(self, x):
        """Compute shared stem features (before branches).

        Returns: (B, base_channels, 32, 32) stem feature map.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return out

    def forward_from_stem(self, stem_features, branch_mask=None,
                           return_branch_acts=False):
        """Continue forward from pre-computed stem features.

        Args:
            stem_features: (B, base_channels, 32, 32).
            branch_mask: (B, K) per-sample branch weights or None.
            return_branch_acts: if True, also return the pre-merge per-branch
                activation tensor (B, K, C, H, W) cloned before mask × merge.
                Used by E1 specialization analysis to compute per-branch L2
                norms even when individual branch outputs are otherwise fused
                into the merge step. Default False = original API unchanged.

        Returns:
            logits, or (logits, branch_acts) if return_branch_acts=True.
        """
        B = stem_features.size(0)
        K = self.num_branches

        branch_outputs = [branch(stem_features) for branch in self.branches]
        stacked = torch.stack(branch_outputs, dim=1)

        # Capture pre-merge activations before mask × merge destroys per-branch info.
        pre_merge_acts = stacked.detach().clone() if return_branch_acts else None

        if branch_mask is not None:
            # Keep mask in its original dtype (typically fp32 from the algorithm).
            # Under fp16 AMP, PyTorch auto-promotes fp16 × fp32 → fp32 for this
            # elementwise multiply, so the sum+divide merge happens in fp32. An
            # earlier version explicitly cast mask → stacked.dtype first, forcing
            # fp16×fp16 → fp16 merge; with mask values ≠ 1 (e.g. p_inactive=0.3)
            # this lost precision and triggered seed-dependent overflow at
            # depth 54 × K=20 (ours_s456 went NaN at epoch 34 under that path,
            # while the fp32-promoted path trained cleanly for all 3 seeds).
            mask = branch_mask.view(B, K, 1, 1, 1)
            stacked = stacked * mask
            if self.mask_scale is not None:
                merged = stacked.sum(dim=1) / self.mask_scale
            elif self.use_adaptive_denom:
                # E3 ablation: per-sample stochastic denominator (÷Σm_k).
                denom = mask.sum(dim=1).clamp(min=1e-8)
                merged = stacked.sum(dim=1) / denom
            else:
                raise RuntimeError(
                    'branch_mask provided but self.mask_scale is None. '
                    'Algorithm must set model.mask_scale = algorithm.expected_mask_sum '
                    '(fixed-denominator merge). Silent per-sample normalization '
                    'was removed: it reintroduces the Jensen bias this paper eliminates.')
        else:
            merged = stacked.sum(dim=1) / K

        if self.shared_expert_branch is not None:
            merged = merged + self.shared_expert_branch(stem_features)

        out = self.layer3(merged)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(B, -1)
        out = self.fc(out)
        if return_branch_acts:
            return out, {'l_only': pre_merge_acts}
        return out

    def forward(self, x, branch_mask=None):
        """Forward pass.

        Args:
            x: (B, 3, 32, 32) input images.
            branch_mask: (B, K) float mask for branches.
                Training: binary Bernoulli samples {0, 1}.
                Inference: expected probabilities p_k(c).
                If None, all branches equally weighted (1/K each -> simple average).

        Returns:
            logits: (B, num_classes)
        """
        stem_features = self.get_stem_features(x)
        return self.forward_from_stem(stem_features, branch_mask)


class MultiBranchResNet110(nn.Module):
    """All-layer multi-branch ResNet, inspired by MoE-in-every-layer LLM design.

    Every layer group is branched into K parallel copies with per-layer merging:
        conv1 (shared) -> K×layer1 -> merge -> K×layer2 -> merge -> K×layer3 -> merge -> FC

    This mirrors modern LLM MoE practice where every FFN layer is replaced with
    MoE experts while attention (here: conv1 stem) remains shared.

    With num_blocks=18: ResNet-110 depth (3 groups × 18 blocks × 2 convs + 2 = 110).
    """

    def __init__(self, num_branches=20, num_blocks=18, num_classes=100,
                 base_channels=16, branch_channels=None,
                 shared_expert_blocks=None,
                 stochastic_depth_rate=0.0, shared_expert=False):
        """
        Args:
            num_branches: K, number of parallel branches per layer group.
            num_blocks: blocks per layer group. 18 = ResNet-110, 9 = ResNet-56, 3 = ResNet-20.
            num_classes: output classes.
            base_channels: width of conv1 output (w). Standard CIFAR ResNet uses 16.
            branch_channels: list of 3 ints [bc1, bc2, bc3] for each layer group.
                If None, auto-computed for ~parameter matching to single-branch ResNet.
            shared_expert_blocks: number of blocks for shared expert (defaults to num_blocks).
                Controls shared expert capacity: fewer blocks = smaller, more = larger.
                Same channel widths as branches (no projection needed), inspired by
                DeepSeek-MoE where shared/routed experts differ in capacity not dimension.
            stochastic_depth_rate: max drop rate for stochastic depth (0 = disabled).
            shared_expert: if True, add one always-active branch per layer group.
        """
        super().__init__()
        self.num_branches = num_branches
        self.shared_expert_flag = shared_expert
        self.shared_expert_blocks = shared_expert_blocks if shared_expert_blocks else num_blocks
        w = base_channels

        # Auto-compute branch channels to roughly match single-branch ResNet params.
        # Single-branch: layer1=w->w, layer2=w->2w, layer3=2w->4w, each 18 blocks.
        # Multi-branch: K branches × num_blocks + 1 SE × se_blocks.
        # Params per block ∝ channels². Match: (K*num_blocks + se_blocks) × bc² ≈ 18 × W²
        if branch_channels is None:
            import math
            se_b = self.shared_expert_blocks if shared_expert else 0
            effective_blocks = num_branches * num_blocks + se_b
            # Single-branch has 1 * 18 blocks at width W
            scale = math.sqrt(effective_blocks / num_blocks)
            branch_channels = [max(2, round(w / scale)),
                               max(2, round(2 * w / scale)),
                               max(2, round(4 * w / scale))]
        bc1, bc2, bc3 = branch_channels
        self.branch_channels = branch_channels

        # Stochastic depth survival probs
        total_blocks = 3 * num_blocks
        sd_probs = None
        if stochastic_depth_rate > 0:
            sd_probs = [1.0 - stochastic_depth_rate * (i / (total_blocks - 1))
                        for i in range(total_blocks)]

        def _get_sp(layer_idx):
            if sd_probs is None:
                return None
            start = layer_idx * num_blocks
            return sd_probs[start:start + num_blocks]

        # Shared stem: only conv1
        self.conv1 = nn.Conv2d(3, w, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)

        se_blocks = self.shared_expert_blocks

        # Layer1 branches: w -> bc1, stride 1
        sp1 = _get_sp(0)
        self.branches_l1 = nn.ModuleList([
            make_layer(w, bc1, num_blocks, stride=1, survival_probs=sp1)
            for _ in range(num_branches)
        ])
        self.shared_expert_l1 = None
        if shared_expert:
            self.shared_expert_l1 = make_layer(w, bc1, se_blocks, stride=1)

        # Layer2 branches: bc1 -> bc2, stride 2
        sp2 = _get_sp(1)
        self.branches_l2 = nn.ModuleList([
            make_layer(bc1, bc2, num_blocks, stride=2, survival_probs=sp2)
            for _ in range(num_branches)
        ])
        self.shared_expert_l2 = None
        if shared_expert:
            self.shared_expert_l2 = make_layer(bc1, bc2, se_blocks, stride=2)

        # Layer3 branches: bc2 -> bc3, stride 2
        sp3 = _get_sp(2)
        self.branches_l3 = nn.ModuleList([
            make_layer(bc2, bc3, num_blocks, stride=2, survival_probs=sp3)
            for _ in range(num_branches)
        ])
        self.shared_expert_l3 = None
        if shared_expert:
            self.shared_expert_l3 = make_layer(bc2, bc3, se_blocks, stride=2)

        # Expose branches for the trainer (freezing, compute tracking)
        # Flatten all layer branches into a single list for compatibility
        self.branches = nn.ModuleList(
            list(self.branches_l1) + list(self.branches_l2) + list(self.branches_l3))

        # Shared FC head
        self.fc = nn.Linear(bc3, num_classes)

        # Fixed denominator for merge (set externally by algorithm)
        self.mask_scale = None
        # Opt-in per-sample adaptive denominator (÷Σm_k) for E3 ablation only.
        self.use_adaptive_denom = False

        # Pre-allocate CUDA streams for parallel branch execution
        self._streams = None  # lazily created on first forward

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _ensure_streams(self, device):
        """Lazily create CUDA streams (only on CUDA devices)."""
        if self._streams is None and device.type == 'cuda':
            self._streams = [torch.cuda.Stream(device=device)
                             for _ in range(self.num_branches)]

    def _merge(self, branch_list, shared_expert_branch, x, branch_mask,
               return_branch_acts=False):
        """Apply branches, mask, and merge for one layer group.

        Uses CUDA streams to run K branches in parallel when on GPU.

        If return_branch_acts=True, also returns the pre-merge per-branch
        tensor (B, K, C, H, W) cloned before mask × merge.
        """
        B = x.size(0)
        K = self.num_branches

        if self._streams is not None:
            # Parallel execution via CUDA streams
            current_stream = torch.cuda.current_stream(x.device)
            outputs = [None] * K
            for k in range(K):
                self._streams[k].wait_stream(current_stream)
                with torch.cuda.stream(self._streams[k]):
                    outputs[k] = branch_list[k](x)
            # Sync all streams before stacking
            for k in range(K):
                current_stream.wait_stream(self._streams[k])
            stacked = torch.stack(outputs, dim=1)
        else:
            # CPU/MPS fallback: sequential
            outputs = [branch(x) for branch in branch_list]
            stacked = torch.stack(outputs, dim=1)  # (B, K, C, H, W)

        pre_merge_acts = stacked.detach().clone() if return_branch_acts else None

        if branch_mask is not None:
            # Keep mask in its original dtype (typically fp32 from the algorithm).
            # Under fp16 AMP, PyTorch auto-promotes fp16 × fp32 → fp32 for this
            # elementwise multiply, so the sum+divide merge happens in fp32. An
            # earlier version explicitly cast mask → stacked.dtype first, forcing
            # fp16×fp16 → fp16 merge; with mask values ≠ 1 (e.g. p_inactive=0.3)
            # this lost precision and triggered seed-dependent overflow at
            # depth 54 × K=20 (ours_s456 went NaN at epoch 34 under that path,
            # while the fp32-promoted path trained cleanly for all 3 seeds).
            mask = branch_mask.view(B, K, 1, 1, 1)
            stacked = stacked * mask
            if self.mask_scale is not None:
                merged = stacked.sum(dim=1) / self.mask_scale
            elif self.use_adaptive_denom:
                denom = mask.sum(dim=1).clamp(min=1e-8)
                merged = stacked.sum(dim=1) / denom
            else:
                raise RuntimeError(
                    'branch_mask provided but self.mask_scale is None. '
                    'Algorithm must set model.mask_scale = algorithm.expected_mask_sum '
                    '(fixed-denominator merge). Silent per-sample normalization '
                    'was removed: it reintroduces the Jensen bias this paper eliminates.')
        else:
            merged = stacked.sum(dim=1) / K

        if shared_expert_branch is not None:
            merged = merged + shared_expert_branch(x)

        if return_branch_acts:
            return merged, pre_merge_acts
        return merged

    def get_stem_features(self, x):
        """Compute shared stem features (conv1 + bn1 + relu).

        Returns: (B, base_channels, 32, 32) stem feature map.
        """
        self._ensure_streams(x.device)
        return F.relu(self.bn1(self.conv1(x)))

    def forward_from_stem(self, stem_features, branch_mask=None,
                           return_branch_acts=False):
        """Continue forward from pre-computed stem features.

        If return_branch_acts=True, also returns dict {'l1': (B,K,...),
        'l2': (B,K,...), 'l3': (B,K,...)} of pre-merge per-branch tensors.
        """
        B = stem_features.size(0)

        if not return_branch_acts:
            out = self._merge(self.branches_l1, self.shared_expert_l1, stem_features, branch_mask)
            out = self._merge(self.branches_l2, self.shared_expert_l2, out, branch_mask)
            out = self._merge(self.branches_l3, self.shared_expert_l3, out, branch_mask)
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(B, -1)
            out = self.fc(out)
            return out

        out, l1 = self._merge(self.branches_l1, self.shared_expert_l1, stem_features,
                               branch_mask, return_branch_acts=True)
        out, l2 = self._merge(self.branches_l2, self.shared_expert_l2, out,
                               branch_mask, return_branch_acts=True)
        out, l3 = self._merge(self.branches_l3, self.shared_expert_l3, out,
                               branch_mask, return_branch_acts=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(B, -1)
        out = self.fc(out)
        return out, {'l1': l1, 'l2': l2, 'l3': l3}

    def forward(self, x, branch_mask=None):
        """Forward pass.

        Args:
            x: (B, 3, 32, 32) input images.
            branch_mask: (B, K) float mask, same mask applied at every layer group.

        Returns:
            logits: (B, num_classes)
        """
        stem_features = self.get_stem_features(x)
        return self.forward_from_stem(stem_features, branch_mask)


class GroupedMultiBranchResNet110(nn.Module):
    """Grouped-conv implementation of MultiBranchResNet110.

    Mathematically equivalent to MultiBranchResNet110 but ~10-15x faster:
    K sequential branch forward passes are replaced by single grouped
    convolutions (groups=K). Input is repeated along the channel dimension,
    so each group sees the full input independently.

    BN(K*C) naturally provides per-branch normalization (each group of C
    channels has its own running stats, gamma, and beta).
    """

    def __init__(self, num_branches=20, num_blocks=18, num_classes=100,
                 base_channels=16, branch_channels=None,
                 shared_expert_blocks=None,
                 stochastic_depth_rate=0.0, shared_expert=False):
        super().__init__()
        self.num_branches = num_branches
        self.shared_expert_flag = shared_expert
        self.shared_expert_blocks = shared_expert_blocks if shared_expert_blocks else num_blocks
        self.num_blocks = num_blocks
        w = base_channels
        K = num_branches

        # Auto-compute branch channels (same logic as MultiBranchResNet110)
        if branch_channels is None:
            import math
            se_b = self.shared_expert_blocks if shared_expert else 0
            effective_blocks = num_branches * num_blocks + se_b
            scale = math.sqrt(effective_blocks / num_blocks)
            branch_channels = [max(2, round(w / scale)),
                               max(2, round(2 * w / scale)),
                               max(2, round(4 * w / scale))]
        bc1, bc2, bc3 = branch_channels
        self.branch_channels = branch_channels

        # Stochastic depth survival probs
        total_blocks = 3 * num_blocks
        sd_probs = None
        if stochastic_depth_rate > 0:
            sd_probs = [1.0 - stochastic_depth_rate * (i / (total_blocks - 1))
                        for i in range(total_blocks)]

        def _get_sp(layer_idx):
            if sd_probs is None:
                return None
            start = layer_idx * num_blocks
            return sd_probs[start:start + num_blocks]

        # Shared stem
        self.conv1 = nn.Conv2d(3, w, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)

        se_blocks = self.shared_expert_blocks

        # Layer1: K branches fused via grouped conv (w -> bc1, stride 1)
        self.layer1 = parallel_make_layer(K, w, bc1, num_blocks, stride=1,
                                          survival_probs=_get_sp(0))
        self.shared_expert_l1 = None
        if shared_expert:
            self.shared_expert_l1 = make_layer(w, bc1, se_blocks, stride=1)

        # Layer2: K branches fused (bc1 -> bc2, stride 2)
        self.layer2 = parallel_make_layer(K, bc1, bc2, num_blocks, stride=2,
                                          survival_probs=_get_sp(1))
        self.shared_expert_l2 = None
        if shared_expert:
            self.shared_expert_l2 = make_layer(bc1, bc2, se_blocks, stride=2)

        # Layer3: K branches fused (bc2 -> bc3, stride 2)
        self.layer3 = parallel_make_layer(K, bc2, bc3, num_blocks, stride=2,
                                          survival_probs=_get_sp(2))
        self.shared_expert_l3 = None
        if shared_expert:
            self.shared_expert_l3 = make_layer(bc2, bc3, se_blocks, stride=2)

        # Shared FC head
        self.fc = nn.Linear(bc3, num_classes)

        # Fixed denominator for merge (set externally by algorithm)
        self.mask_scale = None
        # Opt-in per-sample adaptive denominator (÷Σm_k) for E3 ablation only.
        self.use_adaptive_denom = False

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _merge(self, layer, shared_expert_layer, x, branch_mask,
               return_branch_acts=False):
        """Repeat input → grouped forward → reshape → mask → merge.

        If return_branch_acts=True, also returns the pre-merge per-branch
        tensor (B, K, C, H, W) cloned just after the grouped conv reshape,
        before the mask × merge step destroys per-branch info.
        """
        B = x.size(0)
        K = self.num_branches

        # Repeat input for K groups: (B, C, H, W) → (B, K*C, H, W)
        x_repeated = x.repeat(1, K, 1, 1)

        # All K branches in parallel via grouped conv
        out = layer(x_repeated)  # (B, K*C_out, H, W)

        # Reshape for masking: (B, K, C_out, H, W)
        C_out = out.size(1) // K
        stacked = out.view(B, K, C_out, out.size(2), out.size(3))

        pre_merge_acts = stacked.detach().clone() if return_branch_acts else None

        # Apply mask and merge
        if branch_mask is not None:
            # Keep mask in its original dtype (typically fp32 from the algorithm).
            # Under fp16 AMP, PyTorch auto-promotes fp16 × fp32 → fp32 for this
            # elementwise multiply, so the sum+divide merge happens in fp32. An
            # earlier version explicitly cast mask → stacked.dtype first, forcing
            # fp16×fp16 → fp16 merge; with mask values ≠ 1 (e.g. p_inactive=0.3)
            # this lost precision and triggered seed-dependent overflow at
            # depth 54 × K=20 (ours_s456 went NaN at epoch 34 under that path,
            # while the fp32-promoted path trained cleanly for all 3 seeds).
            mask = branch_mask.view(B, K, 1, 1, 1)
            stacked = stacked * mask
            if self.mask_scale is not None:
                merged = stacked.sum(dim=1) / self.mask_scale
            elif self.use_adaptive_denom:
                denom = mask.sum(dim=1).clamp(min=1e-8)
                merged = stacked.sum(dim=1) / denom
            else:
                raise RuntimeError(
                    'branch_mask provided but self.mask_scale is None. '
                    'Algorithm must set model.mask_scale = algorithm.expected_mask_sum '
                    '(fixed-denominator merge). Silent per-sample normalization '
                    'was removed: it reintroduces the Jensen bias this paper eliminates.')
        else:
            merged = stacked.sum(dim=1) / K

        # Shared expert (single branch, not grouped)
        if shared_expert_layer is not None:
            merged = merged + shared_expert_layer(x)

        if return_branch_acts:
            return merged, pre_merge_acts
        return merged

    def get_stem_features(self, x):
        """Compute shared stem features (conv1 + bn1 + relu).

        Returns: (B, base_channels, 32, 32) stem feature map.
        """
        return F.relu(self.bn1(self.conv1(x)))

    def forward_from_stem(self, stem_features, branch_mask=None,
                           return_branch_acts=False):
        """Continue forward from pre-computed stem features.

        If return_branch_acts=True, also returns dict {'l1': (B,K,...),
        'l2': (B,K,...), 'l3': (B,K,...)} of pre-merge per-branch tensors.
        """
        B = stem_features.size(0)

        if not return_branch_acts:
            out = self._merge(self.layer1, self.shared_expert_l1, stem_features, branch_mask)
            out = self._merge(self.layer2, self.shared_expert_l2, out, branch_mask)
            out = self._merge(self.layer3, self.shared_expert_l3, out, branch_mask)
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(B, -1)
            out = self.fc(out)
            return out

        out, l1 = self._merge(self.layer1, self.shared_expert_l1, stem_features,
                               branch_mask, return_branch_acts=True)
        out, l2 = self._merge(self.layer2, self.shared_expert_l2, out,
                               branch_mask, return_branch_acts=True)
        out, l3 = self._merge(self.layer3, self.shared_expert_l3, out,
                               branch_mask, return_branch_acts=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(B, -1)
        out = self.fc(out)
        return out, {'l1': l1, 'l2': l2, 'l3': l3}

    def forward(self, x, branch_mask=None):
        stem_features = self.get_stem_features(x)
        return self.forward_from_stem(stem_features, branch_mask)


# ---------------------------------------------------------------------------
# Weight conversion between for-loop and grouped implementations
# ---------------------------------------------------------------------------

def _copy_bn_params(dst_bn, src_bns):
    """Concatenate K separate BN params into one grouped BN(K*C)."""
    dst_bn.weight.data = torch.cat([bn.weight.data for bn in src_bns])
    dst_bn.bias.data = torch.cat([bn.bias.data for bn in src_bns])
    dst_bn.running_mean.data = torch.cat([bn.running_mean.data for bn in src_bns])
    dst_bn.running_var.data = torch.cat([bn.running_var.data for bn in src_bns])
    dst_bn.num_batches_tracked.data.copy_(src_bns[0].num_batches_tracked.data)


def _extract_bn_params(dst_bn, src_grouped_bn, k, C):
    """Extract branch k's BN params from grouped BN(K*C)."""
    s = k * C
    dst_bn.weight.data = src_grouped_bn.weight.data[s:s + C].clone()
    dst_bn.bias.data = src_grouped_bn.bias.data[s:s + C].clone()
    dst_bn.running_mean.data = src_grouped_bn.running_mean.data[s:s + C].clone()
    dst_bn.running_var.data = src_grouped_bn.running_var.data[s:s + C].clone()
    dst_bn.num_batches_tracked.data.copy_(src_grouped_bn.num_batches_tracked.data)


def convert_resnet110_forloop_to_grouped(forloop_model):
    """Convert MultiBranchResNet110 → GroupedMultiBranchResNet110 with identical weights.

    Both models end up on the same device as forloop_model. Conversion is done
    on CPU to avoid mixed-device issues with .data assignment.
    """
    src_device = next(forloop_model.parameters()).device
    if src_device.type != 'cpu':
        forloop_model = forloop_model.cpu()

    K = forloop_model.num_branches
    num_blocks = len(forloop_model.branches_l1[0])

    grouped = GroupedMultiBranchResNet110(
        num_branches=K,
        num_blocks=num_blocks,
        num_classes=forloop_model.fc.out_features,
        base_channels=forloop_model.conv1.out_channels,
        branch_channels=forloop_model.branch_channels,
        shared_expert_blocks=forloop_model.shared_expert_blocks,
        shared_expert=forloop_model.shared_expert_flag,
    )

    # Shared stem
    grouped.conv1.load_state_dict(forloop_model.conv1.state_dict())
    grouped.bn1.load_state_dict(forloop_model.bn1.state_dict())

    # Convert each layer group: K separate branches → 1 grouped layer
    layer_pairs = [
        (forloop_model.branches_l1, grouped.layer1),
        (forloop_model.branches_l2, grouped.layer2),
        (forloop_model.branches_l3, grouped.layer3),
    ]
    for branches, g_layer in layer_pairs:
        for bi in range(num_blocks):
            gb = g_layer[bi]  # ParallelBasicBlock

            # conv1, conv2: concat K branch weights along out_channels dim
            gb.conv1.weight.data = torch.cat(
                [branches[k][bi].conv1.weight.data for k in range(K)], dim=0)
            gb.conv2.weight.data = torch.cat(
                [branches[k][bi].conv2.weight.data for k in range(K)], dim=0)

            # bn1, bn2: concat K branch BN params
            _copy_bn_params(gb.bn1, [branches[k][bi].bn1 for k in range(K)])
            _copy_bn_params(gb.bn2, [branches[k][bi].bn2 for k in range(K)])

            # shortcut (only first block where in_ch != out_ch)
            if len(gb.shortcut) > 0:
                gb.shortcut[0].weight.data = torch.cat(
                    [branches[k][bi].shortcut[0].weight.data for k in range(K)], dim=0)
                _copy_bn_params(gb.shortcut[1],
                                [branches[k][bi].shortcut[1] for k in range(K)])

    # Shared experts (single branches, direct copy)
    for attr in ['shared_expert_l1', 'shared_expert_l2', 'shared_expert_l3']:
        src = getattr(forloop_model, attr)
        dst = getattr(grouped, attr)
        if src is not None and dst is not None:
            dst.load_state_dict(src.state_dict())

    # FC head
    grouped.fc.load_state_dict(forloop_model.fc.state_dict())
    grouped.mask_scale = forloop_model.mask_scale

    # Move both back to original device
    if src_device.type != 'cpu':
        forloop_model = forloop_model.to(src_device)
        grouped = grouped.to(src_device)

    return grouped


def convert_resnet110_grouped_to_forloop(grouped_model):
    """Convert GroupedMultiBranchResNet110 → MultiBranchResNet110 with identical weights."""
    src_device = next(grouped_model.parameters()).device
    if src_device.type != 'cpu':
        grouped_model = grouped_model.cpu()

    K = grouped_model.num_branches
    num_blocks = grouped_model.num_blocks

    forloop = MultiBranchResNet110(
        num_branches=K,
        num_blocks=num_blocks,
        num_classes=grouped_model.fc.out_features,
        base_channels=grouped_model.conv1.out_channels,
        branch_channels=grouped_model.branch_channels,
        shared_expert_blocks=grouped_model.shared_expert_blocks,
        shared_expert=grouped_model.shared_expert_flag,
    )

    # Shared stem
    forloop.conv1.load_state_dict(grouped_model.conv1.state_dict())
    forloop.bn1.load_state_dict(grouped_model.bn1.state_dict())

    # Extract K branches from each grouped layer
    layer_pairs = [
        (grouped_model.layer1, forloop.branches_l1),
        (grouped_model.layer2, forloop.branches_l2),
        (grouped_model.layer3, forloop.branches_l3),
    ]
    for g_layer, branches in layer_pairs:
        for bi in range(num_blocks):
            gb = g_layer[bi]
            C_out1 = gb.conv1.out_channels // K
            C_out2 = gb.conv2.out_channels // K

            for k in range(K):
                bb = branches[k][bi]  # BasicBlock

                # conv1, conv2: slice out branch k
                bb.conv1.weight.data = gb.conv1.weight.data[
                    k * C_out1:(k + 1) * C_out1].clone()
                bb.conv2.weight.data = gb.conv2.weight.data[
                    k * C_out2:(k + 1) * C_out2].clone()

                # bn1, bn2
                _extract_bn_params(bb.bn1, gb.bn1, k, C_out1)
                _extract_bn_params(bb.bn2, gb.bn2, k, C_out2)

                # shortcut
                if len(gb.shortcut) > 0 and len(bb.shortcut) > 0:
                    C_sc = gb.shortcut[0].out_channels // K
                    bb.shortcut[0].weight.data = gb.shortcut[0].weight.data[
                        k * C_sc:(k + 1) * C_sc].clone()
                    _extract_bn_params(bb.shortcut[1], gb.shortcut[1], k, C_sc)

    # Shared experts
    for attr in ['shared_expert_l1', 'shared_expert_l2', 'shared_expert_l3']:
        src = getattr(grouped_model, attr)
        dst = getattr(forloop, attr)
        if src is not None and dst is not None:
            dst.load_state_dict(src.state_dict())

    # FC head
    forloop.fc.load_state_dict(grouped_model.fc.state_dict())
    forloop.mask_scale = grouped_model.mask_scale

    # Move both back to original device
    if src_device.type != 'cpu':
        grouped_model = grouped_model.to(src_device)
        forloop = forloop.to(src_device)

    return forloop

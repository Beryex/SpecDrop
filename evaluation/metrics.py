"""Evaluation metrics: accuracy, per-category accuracy, specialization metrics.

This module is designed to be algorithm-agnostic. You should NOT need to modify
it when testing new algorithms -- only modify the algorithms/ module.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from data.cifar100 import NUM_SUPERCLASSES, SUPERCLASS_NAMES


def _unwrap(model):
    """Return the underlying module when wrapped by torch.compile."""
    return getattr(model, '_orig_mod', model)


def _is_faithful_self_contained(model):
    """Faithful baselines that do their own routing and may return (logits, aux)."""
    from models.et_dropout import ExampleTiedDropoutResNet
    from models.contextual_dropout import ContextualDropoutResNet
    from models.mod_squad_vit import ModSquadViT
    from models.soft_moe_vit import SoftMoEViT
    from models.comet_vit import COMETViT
    from models.alf_moe_vit import ALFMoEViT
    return isinstance(_unwrap(model), (
        ExampleTiedDropoutResNet, ContextualDropoutResNet,
        ModSquadViT, SoftMoEViT, COMETViT, ALFMoEViT,
    ))


def _unwrap_logits(output):
    """Faithful models may return (logits, aux_loss); dense models return logits."""
    if isinstance(output, tuple):
        return output[0]
    return output


def _get_mask_and_logits(model, images, coarse_labels, algorithm, device,
                          example_ids=None, shuffle_category_labels=False,
                          noise_probability=0.0,
                          num_categories=None):
    """Get logits; handles both algorithm-routed and faithful self-contained models.

    When `shuffle_category_labels=True` (label-shuffle control), the labels fed
    into the routing algorithm are randomly permuted per-batch. This breaks the
    category→branch mapping while preserving the routing mechanism — it tests
    whether the gain comes from label injection or from the branching itself.

    When `noise_probability > 0` (E5 noise robustness ablation), each sample's
    routing label is independently replaced with a uniformly random superclass
    with probability `noise_probability`. p=0 reproduces the clean baseline,
    p=1 makes routing labels fully random. Mutually exclusive with shuffle.
    """
    if _is_faithful_self_contained(model):
        from models.et_dropout import ExampleTiedDropoutResNet
        if isinstance(_unwrap(model), ExampleTiedDropoutResNet):
            return _unwrap_logits(model(images, example_ids=example_ids))
        return _unwrap_logits(model(images))

    route_labels = coarse_labels
    if shuffle_category_labels:
        route_labels = coarse_labels[torch.randperm(coarse_labels.size(0),
                                                     device=coarse_labels.device)]
    if noise_probability > 0.0:
        M = num_categories if num_categories is not None else NUM_SUPERCLASSES
        rand_labels = torch.randint(0, M, coarse_labels.shape,
                                     device=coarse_labels.device)
        flip = torch.rand(coarse_labels.shape, device=coarse_labels.device) < noise_probability
        route_labels = torch.where(flip, rand_labels, route_labels)

    use_two_phase = (algorithm is not None
                     and hasattr(model, 'num_branches')
                     and getattr(algorithm, 'needs_features', False))

    if use_two_phase:
        stem = model.get_stem_features(images)
        pooled = F.adaptive_avg_pool2d(stem, 1).flatten(1)
        mask = algorithm.get_mask(route_labels, training=False, features=pooled.float())
        return model.forward_from_stem(stem, branch_mask=mask)
    elif algorithm is not None:
        mask = algorithm.get_mask(route_labels, training=False)
        return model(images, branch_mask=mask)
    else:
        return model(images)


@torch.no_grad()
def evaluate_accuracy(model, dataloader, algorithm=None, device='cuda',
                       shuffle_category_labels=False,
                       noise_probability=0.0,
                       route_label_type='coarse'):
    """Compute overall top-1 and top-5 accuracy.

    Args:
        model: the network (ResNet or MultiBranchResNet).
        dataloader: yields (images, fine_labels, coarse_labels).
        algorithm: ModularDropout instance (None for baseline).
        device: torch device.
        noise_probability: per-sample probability of replacing the routing
            superclass label with a uniform random one (E5 ablation).
        route_label_type: 'coarse' (default, M=20) or 'fine' (M=100). When
            'fine', the routing mask is built from fine_labels — for T1.2
            granularity-alignment within-CIFAR experiment.

    Returns:
        dict with 'top1', 'top5', 'loss'.
    """
    if route_label_type not in ('coarse', 'fine'):
        raise ValueError(f'route_label_type must be coarse|fine, got {route_label_type!r}')
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total, correct1, correct5 = 0, 0, 0
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for _b in pbar:
        images, fine_labels, coarse_labels, example_ids = _b
        images = images.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)
        example_ids = example_ids.to(device)

        route_labels = fine_labels if route_label_type == 'fine' else coarse_labels
        logits = _get_mask_and_logits(model, images, route_labels, algorithm, device,
                                        example_ids=example_ids,
                                        shuffle_category_labels=shuffle_category_labels,
                                        noise_probability=noise_probability)

        loss = criterion(logits, fine_labels)
        total_loss += loss.item() * images.size(0)

        # Top-1
        _, pred = logits.max(dim=1)
        correct1 += pred.eq(fine_labels).sum().item()

        # Top-5
        _, pred5 = logits.topk(5, dim=1)
        correct5 += pred5.eq(fine_labels.unsqueeze(1)).any(dim=1).sum().item()

        total += images.size(0)
        pbar.set_postfix(top1=f"{correct1/total*100:.1f}%")

    return {
        'top1': correct1 / total * 100,
        'top5': correct5 / total * 100,
        'loss': total_loss / total,
    }


@torch.no_grad()
def evaluate_per_category(model, dataloader, algorithm=None, device='cuda',
                            route_label_type='coarse'):
    """Compute per-superclass top-1 accuracy.

    Routing mask uses route_label_type ('coarse'|'fine'). Per-category
    binning ALWAYS uses coarse_labels (NUM_SUPERCLASSES=20) so the
    reporting axis stays semantic — fine routing + coarse binning is
    the natural T1.2 view.

    Returns:
        dict mapping superclass name -> accuracy (%).
    """
    if route_label_type not in ('coarse', 'fine'):
        raise ValueError(f'route_label_type must be coarse|fine, got {route_label_type!r}')
    model.eval()
    correct = np.zeros(NUM_SUPERCLASSES)
    total = np.zeros(NUM_SUPERCLASSES)

    for batch in dataloader:
        images, fine_labels, coarse_labels, example_ids = batch
        images = images.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)
        example_ids = example_ids.to(device)

        route_labels = fine_labels if route_label_type == 'fine' else coarse_labels
        logits = _get_mask_and_logits(model, images, route_labels, algorithm, device,
                                        example_ids=example_ids)

        _, pred = logits.max(dim=1)
        correct_mask = pred.eq(fine_labels)

        for c in range(NUM_SUPERCLASSES):
            cat_mask = coarse_labels == c
            total[c] += cat_mask.sum().item()
            correct[c] += (correct_mask & cat_mask).sum().item()

    results = {}
    for c in range(NUM_SUPERCLASSES):
        acc = correct[c] / max(total[c], 1) * 100
        results[SUPERCLASS_NAMES[c]] = round(acc, 2)
    return results


@torch.no_grad()
def evaluate_specialization(model, dataloader, algorithm=None, device='cuda'):
    """Compute specialization metrics for multi-branch models.

    Metrics:
        1. Branch activation norms per category (for-loop models only):
           measures which branches respond most strongly to each category.
        2. Mutual information I(branch; category): how much knowing the
           category tells you about which branch is dominant.
           - For-loop models: dominant branch = argmax of activation norms.
           - Grouped models: dominant branch = argmax of algorithm mask weights.
        3. Pruning sensitivity KD(k, c): accuracy drop when branch k is
           removed, normalized by branch parameter count.

    Returns None if model has no branches (baseline ResNet).
    """
    if not hasattr(model, 'num_branches'):
        return None

    model.eval()
    K = model.num_branches
    M = NUM_SUPERCLASSES
    # MultiBranchResNet (single-layer branched) exposes `branches` as a flat list
    # where every branch consumes stem features → safe to iterate for activation-
    # norm MI.  MultiBranchResNet110 ALSO exposes `branches` but as the concat
    # of 3 layer groups (branches_l1 ∪ branches_l2 ∪ branches_l3), so branches
    # [K:2K] and [2K:3K] expect bc1 / bc2 input and would crash on stem features.
    # Fall back to grouped-style mask-argmax MI in that case (same as the grouped
    # implementation) to keep MI values comparable across for-loop/grouped runs.
    has_branches_list = (
        hasattr(model, 'branches') and not hasattr(model, 'branches_l1')
    )

    avg_norms = None
    dominant_counts = np.zeros((M, K))  # joint distribution P(branch=k, cat=c)
    total_samples = 0

    if has_branches_list:
        # --- For-loop models: compute activation norms + MI from branch outputs ---

        # 1. Collect branch activation norms per category
        norm_sums = np.zeros((M, K))
        norm_counts = np.zeros(M)

        for _b in dataloader:
            images, fine_labels, coarse_labels, _ex = _b
            images = images.to(device)
            coarse_labels = coarse_labels.to(device)

            # Forward through stem only
            out = model.get_stem_features(images)

            # Get each branch's output norm
            for k, branch in enumerate(model.branches):
                branch_out = branch(out)  # (B, 32, 16, 16)
                # Per-sample L2 norm
                norms = branch_out.view(branch_out.size(0), -1).norm(dim=1)  # (B,)
                for c in range(M):
                    cat_mask = (coarse_labels == c)
                    if cat_mask.any():
                        norm_sums[c, k] += norms[cat_mask].sum().item()
            for c in range(M):
                norm_counts[c] += (coarse_labels == c).sum().item()

        avg_norms = norm_sums / np.maximum(norm_counts[:, None], 1)

        # 2. MI from activation norms (dominant branch = argmax norm)
        for _b in dataloader:
            images, fine_labels, coarse_labels, _ex = _b
            images = images.to(device)
            coarse_labels = coarse_labels.to(device)

            out = model.get_stem_features(images)

            all_norms = []
            for branch in model.branches:
                branch_out = branch(out)
                norms = branch_out.view(branch_out.size(0), -1).norm(dim=1)
                all_norms.append(norms)
            all_norms = torch.stack(all_norms, dim=1)  # (B, K)
            dominant = all_norms.argmax(dim=1)  # (B,)

            for b in range(images.size(0)):
                c = coarse_labels[b].item()
                k = dominant[b].item()
                dominant_counts[c, k] += 1
                total_samples += 1

    else:
        # --- Grouped models: compute MI from algorithm mask weights ---
        # No per-branch activation norms (branches are fused in grouped conv).
        # Instead, use the algorithm's mask to determine the dominant branch
        # for each sample. This directly measures routing specialization.

        if algorithm is None:
            # No algorithm means no routing -- MI is undefined
            return None

        needs_features = getattr(algorithm, 'needs_features', False)

        for _b in dataloader:
            images, fine_labels, coarse_labels, _ex = _b
            images = images.to(device)
            coarse_labels = coarse_labels.to(device)

            features = None
            if needs_features:
                stem = model.get_stem_features(images)
                features = F.adaptive_avg_pool2d(stem, 1).flatten(1)

            mask = algorithm.get_mask(coarse_labels, training=False, features=features)
            dominant = mask.argmax(dim=1)  # (B,) -- branch with highest weight

            for b in range(images.size(0)):
                c = coarse_labels[b].item()
                k = dominant[b].item()
                dominant_counts[c, k] += 1
                total_samples += 1

    # Compute MI from dominant_counts
    joint = dominant_counts / max(total_samples, 1)  # P(k, c)
    p_branch = joint.sum(axis=0)  # P(k)
    p_cat = joint.sum(axis=1)     # P(c)

    # MI in nats (natural log). Paper-reported MI values (e.g. 0.454 for
    # SpecDrop K=20 on CIFAR) are nats. Max = log(K) = log(20) ≈ 3.00 nats.
    # Note: scripts/intra_chunk_heterogeneity.py reports entropy in bits
    # (log2); those are DIFFERENT metrics (sub-window cluster distribution
    # entropy) and each is internally consistent with its own unit.
    mi = 0.0
    for c in range(M):
        for k in range(K):
            if joint[c, k] > 0 and p_branch[k] > 0 and p_cat[c] > 0:
                mi += joint[c, k] * np.log(joint[c, k] / (p_cat[c] * p_branch[k]))

    # --- E1: alignment + usage entropy + unique branches ---
    # Alignment: fraction of samples whose dominant branch matches the
    # mode-dominant branch for that category. With round-robin A this
    # collapses to "fraction == c % K" for SpecDrop variants. For methods
    # without an explicit assignment matrix, this measures how concentrated
    # the per-category dominance is (1.0 = each category goes to one branch
    # consistently; 1/K = uniform across branches).
    cat_totals = dominant_counts.sum(axis=1)  # P(c) raw counts
    aligned = 0
    valid_cats = 0
    for c in range(M):
        if cat_totals[c] > 0:
            mode_k = int(np.argmax(dominant_counts[c]))
            aligned += int(dominant_counts[c, mode_k])
            valid_cats += int(cat_totals[c])
    alignment = aligned / max(valid_cats, 1)

    # Marginal entropy H(P(dominant_branch)) in nats (uniform → log K).
    nz = p_branch[p_branch > 0]
    usage_entropy = float(-(nz * np.log(nz)).sum())

    # # unique branches actually used as dominant for at least one sample.
    unique_branches = int((p_branch > 0).sum())

    # # unique categories each branch is mode-dominant for (alignment in count form).
    branch_mode_count = np.zeros(K)
    for c in range(M):
        if cat_totals[c] > 0:
            branch_mode_count[int(np.argmax(dominant_counts[c]))] += 1

    # --- 3. Pruning sensitivity (knowledge density) ---
    # For each branch k, measure accuracy drop per category when k is removed.
    # Works for both for-loop and grouped models (uses model(images, branch_mask=mask)).
    kd = _compute_pruning_sensitivity(model, dataloader, algorithm, device)

    result = {
        'mutual_information': float(mi),
        'alignment': float(alignment),
        'usage_entropy': float(usage_entropy),
        'unique_branches': unique_branches,
        'branch_mode_count': branch_mode_count.tolist(),
        'joint_distribution': joint.tolist(),  # (M, K) — for downstream plotting
        'pruning_sensitivity': kd,
    }
    if avg_norms is not None:
        result['avg_branch_norms'] = avg_norms.tolist()  # (M, K) list
    return result


@torch.no_grad()
def _compute_pruning_sensitivity(model, dataloader, algorithm, device):
    """Compute per-branch per-category pruning sensitivity.

    KD(k, c) = (Acc_c^full - Acc_c^{without_k}) / |theta_k|

    Works for both for-loop models (with model.branches) and grouped models.
    For grouped models, per-branch param count is estimated as total_params / K
    since individual branches are fused in grouped convolutions.

    Returns dict with 'kd_matrix' (M x K) and 'branch_param_counts'.

    Implementation note: caches per-batch (coarse_labels, fine_labels, mask,
    full-model correctness) to skip the redundant full-model forward inside
    the K-branch ablation loop. This roughly halves the wall-clock vs the
    naive (K+1)-pass version.
    """
    if not hasattr(model, 'num_branches'):
        return None

    K = model.num_branches
    M = NUM_SUPERCLASSES

    # Count params per branch
    if hasattr(model, 'branches'):
        param_counts = [
            sum(p.numel() for p in branch.parameters())
            for branch in model.branches
        ]
    else:
        # Grouped models: estimate per-branch params as total / K
        total = sum(p.numel() for p in model.parameters())
        per_branch = total // K
        param_counts = [per_branch] * K

    # Full accuracy per category (all branches active) + cache batch state
    full_acc = np.zeros(M)
    full_total = np.zeros(M)
    cache = []  # list of dicts: images, fine, coarse, base_mask, full_correct

    for _b in dataloader:
        images, fine_labels, coarse_labels, example_ids = _b
        images, fine_labels = images.to(device), fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)
        example_ids = example_ids.to(device)

        # Compute base mask once for this batch (algorithm-dependent).
        if algorithm is not None:
            features = None
            if getattr(algorithm, 'needs_features', False):
                features = F.adaptive_avg_pool2d(
                    model.get_stem_features(images), 1).flatten(1)
            base_mask = algorithm.get_mask(coarse_labels, training=False,
                                            features=features)
        else:
            base_mask = torch.ones(images.size(0), K, device=device)

        # Full forward (uses base_mask path for consistency with ablation).
        if algorithm is not None:
            if getattr(algorithm, 'needs_features', False):
                stem = model.get_stem_features(images)
                logits = model.forward_from_stem(stem, branch_mask=base_mask)
            else:
                logits = model(images, branch_mask=base_mask)
        else:
            logits = _get_mask_and_logits(model, images, coarse_labels, algorithm,
                                           device, example_ids=example_ids)
        _, pred = logits.max(dim=1)
        full_correct = pred.eq(fine_labels)

        for c in range(M):
            cat_mask = coarse_labels == c
            full_total[c] += cat_mask.sum().item()
            full_acc[c] += (full_correct & cat_mask).sum().item()

        cache.append({
            'images': images,
            'fine': fine_labels,
            'coarse': coarse_labels,
            'base_mask': base_mask,
            'full_correct': full_correct,
        })

    full_acc = full_acc / np.maximum(full_total, 1)

    # Accuracy without branch k — reuse cached masks per batch.
    kd_matrix = np.zeros((M, K))
    for k in range(K):
        ablated_acc = np.zeros(M)
        for batch in cache:
            mask = batch['base_mask'].clone()
            mask[:, k] = 0.0
            logits = model(batch['images'], branch_mask=mask)
            _, pred = logits.max(dim=1)
            correct = pred.eq(batch['fine'])
            for c in range(M):
                cat_mask = batch['coarse'] == c
                ablated_acc[c] += (correct & cat_mask).sum().item()

        ablated_acc = ablated_acc / np.maximum(full_total, 1)
        for c in range(M):
            kd_matrix[c, k] = (full_acc[c] - ablated_acc[c]) / param_counts[k]

    return {
        'kd_matrix': kd_matrix.tolist(),
        'branch_param_counts': param_counts,
        'full_acc_per_category': full_acc.tolist(),
    }

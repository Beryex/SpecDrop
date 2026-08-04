"""ImageNet-1K data loader with BREEDS hierarchy superclass mapping.

Uses HuggingFace datasets library to load ImageNet-1K.
Maps the 1000 classes to K superclasses derived from the BREEDS
curated WordNet hierarchy (Santurkar et al., ICLR 2021).

Algorithm: Starting from BREEDS level-3 (29 groups), recursively replace
any group exceeding T classes with its children, provided the expansion
produces at most C subgroups. This prevents fragmentation when a node
has one dominant child and many singletons (e.g., carnivore → 118 dogs
+ 25 individual species). Classes not in the BREEDS hierarchy (110) are
assigned to a "miscellaneous" group. Default T=60, C=10 yields K=46.

Data files: data/breeds_hierarchy/{class_hierarchy.txt, node_names.txt,
            dataset_class_info.json} from github.com/MadryLab/BREEDS-Benchmarks

Usage:
    train_loader, test_loader, num_superclasses = get_imagenet_dataloaders(
        data_dir='./data_cache/imagenet', batch_size=256)
"""

import os
import json
from collections import defaultdict, deque

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import ImageFile

# ImageNet-1k has ~50 out of 1.28M training images with truncated EOF markers
# (dataset-level quirk, also present in HF distribution). PIL otherwise emits
# "UserWarning: Truncated File Read" and can raise on strict decode; this
# flag is the PyTorch-official ImageNet example's standard workaround —
# partial-decode the image, train on what's recoverable. Impact on top1 is
# <0.01% (handful of images out of 1.28M).
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGENET_NUM_CLASSES = 1000
BREEDS_HIERARCHY_DIR = os.path.join(os.path.dirname(__file__), 'breeds_hierarchy')
NUM_SUPERCLASSES = 46   # BREEDS default (T=60, C=10). Matches K used by MB-ViT.


def compute_category_fractions(data_dir='./data_cache/imagenet',
                                num_categories=NUM_SUPERCLASSES,
                                cache_dir=None):
    """Return per-superclass fraction of ImageNet-1K fine-classes (length
    num_categories, sums to ~1.0), used by SoftSpecDrop per-category routing.

    Counted at the *class* level (how many of the 1000 fine-classes fall into
    each BREEDS superclass), not the *image* level. ImageNet-1K's per-class
    training count varies modestly (~732 to ~1300 images), so for K=46-sized
    groups class-level fractions approximate image-level fractions within
    ~2-3%; the routing mechanism is robust to this noise (β exponent
    re-amplifies the ordering). Using class-level counts avoids a ~1.28M-image
    dataset scan and keeps the helper callable without downloaded ImageNet.

    Args:
        data_dir: directory used to cache the BREEDS mapping (same convention
            as get_imagenet_dataloaders).
        num_categories: expected K; must equal the BREEDS mapping size.
        cache_dir: where to load/save the mapping cache. Defaults to data_dir.

    Returns:
        List of float of length num_categories.
    """
    mapping, num_sc, _ = _load_or_build_mapping(cache_dir or data_dir)
    if num_sc != num_categories:
        raise ValueError(
            f"num_categories={num_categories} does not match BREEDS mapping "
            f"({num_sc}). Either call with num_categories={num_sc} or rebuild "
            f"the mapping with different (T, C) parameters.")

    counts = [0] * num_categories
    for cls_idx in range(IMAGENET_NUM_CLASSES):
        sc = mapping[cls_idx]
        counts[sc] += 1
    total = sum(counts)
    if total == 0:
        raise ValueError("All superclass counts are zero — BREEDS mapping is "
                          "empty; verify data/breeds_hierarchy/ is populated.")
    return [c / total for c in counts]

# ── BREEDS-based superclass mapping ─────────────────────────────────────────


def _build_breeds_mapping(breeds_dir=BREEDS_HIERARCHY_DIR, T=60, C=10):
    """Build superclass mapping from BREEDS curated WordNet hierarchy.

    Algorithm: Starting from BREEDS level-3 (29 groups), recursively
    replace any group exceeding T classes with its children, provided
    the expansion produces at most C non-empty subgroups. This prevents
    fragmentation when a node has many singleton children (e.g.,
    carnivore has ~25 children but only 'dog' is large). Classes not
    covered by the BREEDS hierarchy are assigned to 'miscellaneous'.

    Args:
        breeds_dir: path to BREEDS hierarchy data files
        T: max classes per group before splitting (default 60)
        C: max children allowed per split (default 10)

    Returns:
        mapping: dict {class_idx: superclass_idx}
        num_superclasses: int
        superclass_names: list of str
    """
    # Load BREEDS data files
    with open(os.path.join(breeds_dir, 'dataset_class_info.json')) as f:
        class_info = json.load(f)
    wnid_to_idx = {e[1]: e[0] for e in class_info}

    wnid_to_name = {}
    with open(os.path.join(breeds_dir, 'node_names.txt')) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                wnid_to_name[parts[0]] = parts[1]

    children_map = defaultdict(list)
    with open(os.path.join(breeds_dir, 'class_hierarchy.txt')) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                children_map[parts[0]].append(parts[1])

    imagenet_leaves = set(wnid_to_idx.keys())

    # BFS to assign levels (max depth = longest path from root)
    root = 'n00001740'
    level_of = {}
    queue = deque([(root, 0)])
    while queue:
        node, lvl = queue.popleft()
        if node not in level_of or lvl > level_of[node]:
            level_of[node] = lvl
        for ch in children_map.get(node, []):
            queue.append((ch, lvl + 1))

    def get_leaves(node):
        """Get ImageNet class indices reachable from this node."""
        leaves = []
        q = deque([node])
        while q:
            cur = q.popleft()
            if cur in imagenet_leaves:
                leaves.append(wnid_to_idx[cur])
            for ch in children_map.get(cur, []):
                q.append(ch)
        return sorted(leaves)

    def expand(node):
        """Recursively split node if > T classes and ≤ C children."""
        leaves = get_leaves(node)
        name = wnid_to_name.get(node, node).split(',')[0]
        if len(leaves) <= T:
            return [(name, leaves)]
        ch = [(c, get_leaves(c)) for c in children_map.get(node, [])]
        ch = [(c, lv) for c, lv in ch if lv]
        if not ch or len(ch) > C:
            return [(name, leaves)]
        result = []
        for c, lv in ch:
            result.extend(expand(c))
        return result

    # Start with level-3 nodes, recursively split
    level3_nodes = [n for n, l in level_of.items() if l == 3]
    groups = []
    for node in level3_nodes:
        groups.extend(expand(node))

    # Add miscellaneous for uncovered classes
    covered = set()
    for _, leaves in groups:
        covered.update(leaves)
    uncovered = sorted(set(range(IMAGENET_NUM_CLASSES)) - covered)
    if uncovered:
        groups.append(('miscellaneous', uncovered))

    # Sort by group size descending, build mapping
    groups.sort(key=lambda x: -len(x[1]))
    mapping = {}
    superclass_names = []
    for sc_id, (name, leaves) in enumerate(groups):
        superclass_names.append(name)
        for leaf_idx in leaves:
            mapping[leaf_idx] = sc_id

    return mapping, len(groups), superclass_names


def _load_or_build_mapping(cache_dir, T=60, C=10):
    """Load cached BREEDS mapping or build one."""
    cache_path = os.path.join(cache_dir, f'imagenet_breeds_T{T}_C{C}.json')

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        mapping = {int(k): v for k, v in data['mapping'].items()}
        return mapping, data['num_superclasses'], data['superclass_names']

    # Build from BREEDS hierarchy
    mapping, num_sc, names = _build_breeds_mapping(T=T, C=C)

    # Cache
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump({
            'method': f'BREEDS level-3 + recursive split (T={T}, C={C}) + misc',
            'reference': 'Santurkar et al., BREEDS: Benchmarks for Subpopulation Shift, ICLR 2021',
            'T': T, 'C': C,
            'num_superclasses': num_sc,
            'superclass_names': names,
            'mapping': {str(k): v for k, v in sorted(mapping.items())},
        }, f, indent=2)

    return mapping, num_sc, names


# ── Download & Validation ───────────────────────────────────────────────────

def ensure_downloaded(data_dir='./data_cache/imagenet'):
    """Download ImageNet-1K if not already cached.

    Requires HuggingFace authentication (gated dataset).
    Run `huggingface-cli login` first if not authenticated.
    Returns True if ready.
    """
    from datasets import load_dataset
    print("Checking ImageNet-1K (~150GB)...")
    train_ds = load_dataset('ILSVRC/imagenet-1k', split='train',
                            cache_dir=data_dir)
    val_ds = load_dataset('ILSVRC/imagenet-1k', split='validation',
                          cache_dir=data_dir)
    print(f"  ImageNet-1K downloaded: {len(train_ds):,} train, {len(val_ds):,} val")
    return True


def validate_format(data_dir='./data_cache/imagenet', num_samples=5):
    """Validate ImageNet dataset format and superclass mapping.

    Checks: 'image' column (PIL), 'label' column (int 0-999),
    1000 class names, BREEDS superclass mapping covers all 1000 classes.

    Returns: (ok: bool, message: str)
    """
    from datasets import load_dataset

    for split_name in ['train', 'validation']:
        ds = load_dataset('ILSVRC/imagenet-1k', split=split_name,
                          cache_dir=data_dir)
        checked = 0
        for example in ds:
            if 'image' not in example:
                return False, f"{split_name}: missing 'image' field"
            if 'label' not in example:
                return False, f"{split_name}: missing 'label' field"

            img = example['image']
            if not hasattr(img, 'mode'):
                return False, f"{split_name}: 'image' is not a PIL Image"

            label = example['label']
            if not isinstance(label, int) or label < 0 or label >= IMAGENET_NUM_CLASSES:
                return False, f"{split_name}: label {label} out of range [0, 999]"

            checked += 1
            if checked >= num_samples:
                break

    # Check class names
    ds_full = load_dataset('ILSVRC/imagenet-1k', split='train',
                           cache_dir=data_dir)
    class_names = ds_full.features['label'].names
    if len(class_names) != IMAGENET_NUM_CLASSES:
        return False, f"Expected 1000 class names, got {len(class_names)}"

    # Check superclass mapping
    mapping, num_sc, sc_names = _load_or_build_mapping(data_dir)
    if len(mapping) != IMAGENET_NUM_CLASSES:
        return False, f"Superclass mapping covers {len(mapping)}/1000 classes"

    return True, (f"ImageNet OK: PIL images + 1000 classes + "
                  f"{num_sc} BREEDS superclasses verified")


# ── HuggingFace Dataset Wrapper ───────────────────────────────────────────────

class HFImageNetDataset(Dataset):
    """Wraps a HuggingFace ImageNet dataset for PyTorch DataLoader.

    Returns (image_tensor, fine_label, coarse_label) tuples.
    """

    def __init__(self, hf_dataset, transform, superclass_mapping):
        self.dataset = hf_dataset
        self.transform = transform
        self.mapping = superclass_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.transform(image)
        fine_label = sample['label']
        coarse_label = self.mapping[fine_label]

        # 4-tuple matches CIFAR100WithSuperclass so trainer.py / evaluation/metrics.py
        # can always unpack (img, fine, coarse, example_id).
        return image, fine_label, coarse_label, idx


def get_imagenet_dataloaders(data_dir='./data_cache/imagenet',
                              batch_size=256, num_workers=8,
                              num_superclasses=None, image_size=224,
                              augmentation='basic', max_samples=None,
                              prefetch_factor=2,
                              train_subset_frac=1.0, train_subset_seed=42):
    """Build ImageNet train/val dataloaders with BREEDS superclass labels.

    Args:
        data_dir: path to HuggingFace cache directory for ImageNet
        batch_size: batch size for dataloaders
        num_workers: number of data loading workers
        num_superclasses: ignored (kept for API compat), determined by BREEDS
        image_size: input image resolution
        augmentation: 'basic' (RandomResizedCrop+HFlip+ColorJitter) or 'deit'
            (RandAugment + RandomErasing, DeiT training recipe for ViT)
        prefetch_factor: per-worker DataLoader prefetch depth (default 2 =
            PyTorch default; bump to 4 to buffer more batches when CPU has
            headroom). Ignored when num_workers=0.
        train_subset_frac: fraction ∈ (0, 1] of the TRAIN split to keep,
            via stratified random sampling (uniform per fine-class,
            preserves all 1000 classes and K=46 BREEDS superclass
            representation). Default 1.0 = full train set. Used for
            compute-budget ablation (e.g., 0.2 = ~20% = ~256K images,
            ~5x faster than full). Val split is always kept at full
            size for honest evaluation.
        train_subset_seed: RNG seed for subsample. Fixed (default 42)
            across all ablation runs so every config sees the SAME
            subsample — otherwise subsample-choice noise would confound
            the argmax comparison.

    Returns:
        train_loader, val_loader, actual_num_superclasses
    """
    from datasets import load_dataset

    print("Loading ImageNet-1K from HuggingFace cache...")

    # Load train and validation splits
    train_ds = load_dataset('ILSVRC/imagenet-1k', split='train',
                            cache_dir=data_dir)
    val_ds = load_dataset('ILSVRC/imagenet-1k', split='validation',
                          cache_dir=data_dir)

    print(f"  Train: {len(train_ds)} images")
    print(f"  Val:   {len(val_ds)} images")

    # Build BREEDS superclass mapping
    mapping, actual_num, sc_names = _load_or_build_mapping(data_dir)
    print(f"  Superclasses: {actual_num} (BREEDS level-3 derived)")
    for i, name in enumerate(sc_names):
        count = sum(1 for v in mapping.values() if v == i)
        print(f"    {i:2d}. {name}: {count} classes")

    # Standard ImageNet transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    # DeiT training recipe (Touvron et al. 2020): RandAugment + RandomErasing.
    # Mixup / CutMix are applied at the collate-fn level and can be opted in
    # via the `augmentation='deit_mixup'` variant; 'deit' here = RandAug only.
    if augmentation == 'deit':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, value='random'),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Wrap with PyTorch Dataset
    train_dataset = HFImageNetDataset(train_ds, train_transform, mapping)
    val_dataset = HFImageNetDataset(val_ds, val_transform, mapping)

    # Stratified compute-budget subsample of the TRAIN split (default 1.0 = no-op).
    # Uniformly samples per fine-class so all 1000 classes and all 46 BREEDS
    # superclasses remain represented; fixed seed so every ablation config
    # sees the identical subsample (otherwise subsample choice would confound
    # the argmax). Val split is always full.
    if 0 < train_subset_frac < 1.0:
        import numpy as _np
        from torch.utils.data import Subset
        # HF dataset column access reads only the 'label' column from parquet
        # (no image decode), typically <2s on warm cache for 1.28M samples.
        labels_arr = _np.asarray(train_ds['label'])
        rng = _np.random.default_rng(train_subset_seed)
        per_class_indices = []
        for cls in range(IMAGENET_NUM_CLASSES):
            cls_all = _np.where(labels_arr == cls)[0]
            n_keep = max(1, int(round(len(cls_all) * train_subset_frac)))
            per_class_indices.append(rng.choice(cls_all, size=n_keep, replace=False))
        keep_indices = sorted(_np.concatenate(per_class_indices).tolist())
        orig_n = len(train_dataset)
        train_dataset = Subset(train_dataset, keep_indices)
        print(f"  [subset] Stratified train subsample: frac={train_subset_frac} "
              f"seed={train_subset_seed} → {len(keep_indices):,}/{orig_n:,} images "
              f"(~{len(keep_indices)//IMAGENET_NUM_CLASSES} per class on avg)")

    # Smoke-test knob: slice a small subset of both splits to accelerate smoke runs.
    if max_samples is not None and max_samples > 0:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, list(range(min(max_samples, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(max_samples // 4 or 1, len(val_dataset)))))
        print(f"  [smoke] ImageNet sliced to max_samples={max_samples}")

    def _worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2**32
        import numpy as _np; _np.random.seed(seed + worker_id)
        import random as _r; _r.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(torch.initial_seed())
    dl_kwargs = dict(
        batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn, generator=g)
    # prefetch_factor only valid with num_workers > 0.
    if num_workers > 0:
        dl_kwargs['prefetch_factor'] = prefetch_factor
    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)

    return train_loader, val_loader, actual_num

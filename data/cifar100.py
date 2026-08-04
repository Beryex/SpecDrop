"""CIFAR-100 data loading with superclass (coarse label) support."""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

NUM_SUPERCLASSES = 20
NUM_CLASSES = 100

SUPERCLASS_NAMES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers',
    'fruit_and_vegetables', 'household_electrical_devices',
    'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores', 'medium_mammals',
    'non-insect_invertebrates', 'people', 'reptiles',
    'small_mammals', 'trees', 'vehicles_1', 'vehicles_2',
]

# Official CIFAR-100 mapping: fine label (0-99) -> coarse/superclass label (0-19)
# Verified against cifar-100-python/train coarse_labels
FINE_TO_COARSE = [
     4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
     3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
     6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
     0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
     5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
    16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
    10,  3,  2, 12, 12, 16, 12,  1,  9, 19,
     2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
    16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
    18,  1,  2, 15,  6,  0, 17,  8, 14, 13,
]


def ensure_downloaded(data_dir='./data_cache'):
    """Download CIFAR-100 if not already cached. Returns True if ready."""
    print("Checking CIFAR-100 (~170MB)...")
    ds = torchvision.datasets.CIFAR100(data_dir, train=True, download=True)
    print(f"  CIFAR-100 downloaded: {len(ds):,} train images")
    return True


def validate_format(data_dir='./data_cache'):
    """Validate CIFAR-100 dataset format matches expectations.

    Checks: 100 fine classes, 20 superclasses, image shape (3, 32, 32).

    Returns: (ok: bool, message: str)
    """
    ds = CIFAR100WithSuperclass(data_dir, train=True,
                                transform=transforms.ToTensor(), download=True)
    if len(ds) != 50000:
        return False, f"Expected 50000 train images, got {len(ds)}"

    img, fine, coarse, idx = ds[0]
    if idx != 0:
        return False, f"Expected idx=0 for ds[0], got {idx}"
    if img.shape != (3, 32, 32):
        return False, f"Expected image shape (3,32,32), got {img.shape}"
    if not (0 <= fine < NUM_CLASSES):
        return False, f"Fine label {fine} out of range [0, {NUM_CLASSES})"
    if not (0 <= coarse < NUM_SUPERCLASSES):
        return False, f"Coarse label {coarse} out of range [0, {NUM_SUPERCLASSES})"

    ds_test = CIFAR100WithSuperclass(data_dir, train=False,
                                     transform=transforms.ToTensor(), download=True)
    if len(ds_test) != 10000:
        return False, f"Expected 10000 test images, got {len(ds_test)}"

    return True, "CIFAR-100 OK: 50K train + 10K test, 100 classes, 20 superclasses"


class CIFAR100WithSuperclass(torchvision.datasets.CIFAR100):
    """CIFAR-100 dataset that also returns the superclass (coarse) label."""

    def __init__(self, root, train=True, transform=None, download=True):
        super().__init__(root, train=train, transform=transform, download=download)
        self.coarse_targets = [FINE_TO_COARSE[t] for t in self.targets]

    def __getitem__(self, index):
        img, fine_label = super().__getitem__(index)
        coarse_label = self.coarse_targets[index]
        # Return (img, fine, coarse, index) — example index enables Example-Tied Dropout
        # (Maini 2023) which needs a fixed per-example mask lookup during training.
        return img, fine_label, coarse_label, index


def _worker_init_fn(worker_id):
    """Seed each DataLoader worker for reproducibility."""
    seed = torch.initial_seed() % 2**32
    import numpy as np; np.random.seed(seed + worker_id)
    import random; random.seed(seed + worker_id)


def get_dataloaders(data_dir='./data_cache', batch_size=128, num_workers=4, device='cpu'):
    """Get CIFAR-100 train/test dataloaders with superclass labels.

    Each batch yields (images, fine_labels, coarse_labels).
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = CIFAR100WithSuperclass(
        data_dir, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR100WithSuperclass(
        data_dir, train=False, transform=test_transform, download=True)

    pin = device not in ('mps',)
    persistent = num_workers > 0
    g = torch.Generator()
    g.manual_seed(torch.initial_seed())
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persistent,
        worker_init_fn=_worker_init_fn, generator=g)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persistent,
        worker_init_fn=_worker_init_fn, generator=g)

    return train_loader, test_loader

#!/usr/bin/env python3
"""Inspect CIFAR-100 superclass balance.

Trivially expected to be 2500/each (uniform) since CIFAR-100 is balanced
by construction. Run as sanity check and baseline for NLP comparison.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from data.cifar100 import CIFAR100WithSuperclass, SUPERCLASS_NAMES


def main():
    ds = CIFAR100WithSuperclass(root='./data_cache', train=True, download=False)
    coarse = np.array([ds[i][2] for i in range(len(ds))])
    u, c = np.unique(coarse, return_counts=True)
    total = len(ds)
    balanced = bool((c == c[0]).all())

    print(f"Total train samples: {total}")
    print(f"Superclass count range: [{c.min()}, {c.max()}]")
    print(f"Balanced: {balanced}")
    print()
    print(f"{'superclass':<30} {'count':>6} {'frac':>8}")
    print('-' * 48)
    for ui, ci in zip(u, c):
        print(f"{SUPERCLASS_NAMES[ui]:<30} {ci:>6} {ci/total:>8.3%}")

    print()
    print(f"Takeaway: CIFAR-100 is exactly balanced ({c[0]}/class × 20 = {total})")
    print(f"  → uniform K=20 branches at ffn-equivalent width is natural match.")


if __name__ == '__main__':
    main()

"""Unit tests for data.imagenet.compute_category_fractions.

Uses only the BREEDS mapping cache (no ImageNet dataset download required)
so it runs locally even on MPS hosts.

Run: python tests/test_imagenet_fractions.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.imagenet import (
    compute_category_fractions, _load_or_build_mapping,
    IMAGENET_NUM_CLASSES, NUM_SUPERCLASSES,
)


def test_num_superclasses_constant_matches_breeds():
    _, num_sc, _ = _load_or_build_mapping('./data_cache/imagenet')
    assert num_sc == NUM_SUPERCLASSES, \
        f"NUM_SUPERCLASSES={NUM_SUPERCLASSES} does not match BREEDS mapping size {num_sc}"
    print(f"  PASS num_superclasses_constant_matches_breeds ({num_sc})")


def test_fractions_sum_to_one():
    fracs = compute_category_fractions('./data_cache/imagenet')
    assert len(fracs) == NUM_SUPERCLASSES, f"len={len(fracs)}"
    total = sum(fracs)
    assert abs(total - 1.0) < 1e-6, f"fractions sum to {total}, not 1.0"
    print(f"  PASS fractions_sum_to_one (sum={total:.6f})")


def test_fractions_all_nonnegative():
    fracs = compute_category_fractions('./data_cache/imagenet')
    assert all(f >= 0 for f in fracs), "Negative fraction"
    assert all(f <= 1 for f in fracs), "Fraction > 1"
    print(f"  PASS fractions_all_nonnegative")


def test_fractions_cover_all_1000_classes():
    """Each of 1000 fine classes must be mapped to exactly one superclass,
    so the sum of per-superclass counts must equal 1000."""
    mapping, num_sc, _ = _load_or_build_mapping('./data_cache/imagenet')
    counts = [0] * num_sc
    for cls_idx in range(IMAGENET_NUM_CLASSES):
        counts[mapping[cls_idx]] += 1
    assert sum(counts) == IMAGENET_NUM_CLASSES, \
        f"mapping covers {sum(counts)}/{IMAGENET_NUM_CLASSES} classes"
    print(f"  PASS fractions_cover_all_1000_classes")


def test_fractions_rejects_wrong_k():
    try:
        compute_category_fractions('./data_cache/imagenet', num_categories=7)
    except ValueError as e:
        assert 'BREEDS' in str(e)
        print(f"  PASS fractions_rejects_wrong_k (raised: {e})")
        return
    raise AssertionError("Expected ValueError for num_categories=7")


def test_fractions_interpretable_spread():
    """Superclass with the most classes should dominate, miscellaneous should
    not be empty — spot-check mapping sensibility without overfitting."""
    fracs = compute_category_fractions('./data_cache/imagenet')
    assert max(fracs) > 0.05, \
        f"max fraction {max(fracs)} too small — mapping looks degenerate"
    assert min(fracs) >= 0.0
    nonzero = sum(1 for f in fracs if f > 0)
    assert nonzero == NUM_SUPERCLASSES, \
        f"only {nonzero}/{NUM_SUPERCLASSES} superclasses have classes — mapping broken"
    print(f"  PASS fractions_interpretable_spread "
          f"(max={max(fracs):.3f}, min={min(fracs):.3f})")


if __name__ == '__main__':
    print("=" * 60)
    print(" ImageNet compute_category_fractions tests")
    print("=" * 60)
    test_num_superclasses_constant_matches_breeds()
    test_fractions_sum_to_one()
    test_fractions_all_nonnegative()
    test_fractions_cover_all_1000_classes()
    test_fractions_rejects_wrong_k()
    test_fractions_interpretable_spread()
    print("=" * 60)
    print(" All tests passed")
    print("=" * 60)

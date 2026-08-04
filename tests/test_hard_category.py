"""Unit tests for algorithms/hard_category.py (E3)."""
import sys, os
import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from algorithms.hard_category import HardCategory
from algorithms import build_algorithm


def test_one_hot_mask_shape_and_values():
    algo = HardCategory(num_modules=20, num_categories=20)
    cat = torch.tensor([0, 5, 19, 13])
    mask = algo.get_mask(cat, training=False)
    assert mask.shape == (4, 20)
    # one-hot: row sum is 1, col sum at position cat[i] is 1
    assert (mask.sum(dim=1) == 1.0).all()
    for i, c in enumerate(cat):
        assert mask[i, c] == 1.0
        assert mask[i].nonzero().numel() == 1


def test_expected_mask_sum_is_one():
    algo = HardCategory(num_modules=20, num_categories=20)
    assert algo.expected_mask_sum == 1


def test_train_eval_identical_mask():
    """Hard one-hot is deterministic; train/eval must match exactly."""
    algo = HardCategory(num_modules=10, num_categories=10)
    cat = torch.tensor([0, 3, 7])
    m_train = algo.get_mask(cat, training=True)
    m_eval = algo.get_mask(cat, training=False)
    assert torch.equal(m_train, m_eval)


def test_k_must_equal_m():
    """HardCategory enforces K==M (one branch per category)."""
    with pytest.raises(ValueError, match='K==M'):
        HardCategory(num_modules=20, num_categories=10)
    with pytest.raises(ValueError, match='K==M'):
        HardCategory(num_modules=8, num_categories=20)


def test_build_algorithm_dispatch():
    """Ensure 'hard_category' is registered in algorithms/__init__.py."""
    cfg = {
        'algorithm': {'type': 'hard_category', 'num_categories': 20},
        'model': {'num_branches': 20},
        'training': {'epochs': 200},
    }
    algo = build_algorithm(cfg)
    assert isinstance(algo, HardCategory)
    assert algo.num_modules == 20
    assert algo.num_categories == 20


def test_mask_on_device():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available; skip device test')
    algo = HardCategory(num_modules=20, num_categories=20)
    cat = torch.tensor([0, 5, 19], device='cuda')
    mask = algo.get_mask(cat, training=False)
    assert mask.device.type == 'cuda'

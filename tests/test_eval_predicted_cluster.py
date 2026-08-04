"""Unit + smoke tests for scripts/eval_predicted_cluster.py (E1)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ─── Pure-function tests ────────────────────────────────────────────────────
def test_cifar100_fine_to_coarse_shape_and_values():
    from scripts.eval_predicted_cluster import _cifar100_fine_to_coarse
    f2c = _cifar100_fine_to_coarse()
    assert f2c.shape == (100,)
    # Known: fine class 0 (apple) → superclass 4 (fruit_and_vegetables)
    assert f2c[0].item() == 4
    # Known: fine class 99 (worm) → superclass 13 (non-insect_invertebrates)
    assert f2c[99].item() == 13
    # All entries in 0..19
    assert (f2c >= 0).all() and (f2c < 20).all()
    # All 20 superclasses represented (every fine maps to a coarse)
    assert set(f2c.tolist()) == set(range(20))


def test_imagenet_fine_to_coarse_shape_and_coverage():
    """BREEDS K=46 mapping covers all 1000 classes."""
    from scripts.eval_predicted_cluster import _imagenet_fine_to_coarse
    try:
        f2c = _imagenet_fine_to_coarse()
    except FileNotFoundError as e:
        pytest.skip(f'BREEDS hierarchy data missing: {e}')
    assert f2c.shape == (1000,)
    assert (f2c >= 0).all(), 'all fine classes must map to a non-negative coarse'
    # K=46 BREEDS expected (or 47 if miscellaneous bucket needed)
    K = f2c.max().item() + 1
    assert K in (46, 47), f'expected K∈{{46,47}}, got K={K}'


def test_breeds_supercategory_names_count_46():
    """E-CLIP-1: 46 BREEDS supercategory names available + non-empty strings."""
    from scripts.eval_predicted_cluster import _get_breeds_supercategory_names
    try:
        names = _get_breeds_supercategory_names()
    except FileNotFoundError as e:
        pytest.skip(f'BREEDS hierarchy data missing: {e}')
    assert len(names) == 46
    for n in names:
        assert isinstance(n, str) and len(n) > 0


def test_imagenet_norm_to_clip_norm_inverse_identity():
    """Algebraic check: applying CLIP norm to ImageNet-normalized tensor gives
    the same result as applying CLIP norm to the raw image directly."""
    import torch
    from scripts.eval_predicted_cluster import (
        _imagenet_norm_to_clip_norm, _IMAGENET_MEAN, _IMAGENET_STD,
        _CLIP_MEAN, _CLIP_STD,
    )
    raw = torch.rand(2, 3, 4, 4)  # raw [0, 1]
    in_mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    in_std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    cl_mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
    cl_std = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
    x_imnet = (raw - in_mean) / in_std
    expected = (raw - cl_mean) / cl_std
    got = _imagenet_norm_to_clip_norm(x_imnet)
    assert torch.allclose(got, expected, atol=1e-5)


# ─── CLI smoke (catches argparse / import drift, fa6aa8b lesson) ─────────
def test_cli_help_exits_zero():
    p = subprocess.run(
        [sys.executable, '-m', 'scripts.eval_predicted_cluster', '--help'],
        cwd=str(_REPO), capture_output=True, text=True, timeout=60,
    )
    assert p.returncode == 0, (
        f'eval_predicted_cluster --help exited {p.returncode}\n'
        f'STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}')
    for flag in ('--setting', '--seed', '--device', '--predictor_run_dir',
                  '--out_json'):
        assert flag in p.stdout, f'{flag} missing from --help'
    for s in ('cifar', 'vit', 'vit_clip'):
        assert s in p.stdout, f'{s} missing from --help choices'


# ─── Algorithm-build sanity (offline; no checkpoints needed) ────────────────
def test_build_trained_algorithm_returns_terminal_state():
    from scripts.eval_predicted_cluster import _build_trained_algorithm
    cfg = {
        'algorithm': {
            'type': 'soft_specdrop',
            'p_active': 0.7, 'p_inactive': 0.3,
            'assignment': 'round_robin', 'warmup_ratio': 1.0,
            'warmup_schedule': 'cosine', 'warmup_unit': 'epoch',
            'amplification_beta': 1.0,
        },
        'model': {'num_branches': 20},
        'training': {'epochs': 200},
    }
    algo = _build_trained_algorithm(cfg, K=20, M=20)
    # After advance, warmup progress = 1.0 (terminal)
    assert algo._warmup_progress() == pytest.approx(1.0)

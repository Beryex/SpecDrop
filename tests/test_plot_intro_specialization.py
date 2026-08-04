"""Tests for scripts/plot_intro_specialization.py — data loaders + synthesis."""
import sys, os
import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def test_load_cifar_shape_and_diag_ratio():
    from scripts.plot_intro_specialization import load_cifar
    ours = load_cifar('ours')
    assert ours.shape == (20, 20)
    assert ours.min() >= 0  # |Δ|
    # ours should have diag > off-diag (specialization)
    diag = ours.diagonal().mean()
    off = (ours.sum() - ours.trace()) / (ours.size - ours.shape[0])
    assert diag > off, f'CIFAR ours: expected diag>off, got diag={diag:.2e} off={off:.2e}'


def test_load_cifar_no_routing_no_specialization():
    from scripts.plot_intro_specialization import load_cifar
    nr = load_cifar('no_routing')
    assert nr.shape == (20, 20)
    diag = nr.diagonal().mean()
    off = (nr.sum() - nr.trace()) / (nr.size - nr.shape[0])
    # no_routing diag/off should be ~1 (no specialization)
    ratio = diag / max(off, 1e-12)
    assert 0.6 < ratio < 1.6, f'CIFAR no_routing: ratio={ratio:.2f}, expected ~1.0'


def test_load_vit_shape_46x46():
    from scripts.plot_intro_specialization import load_vit_ours
    m = load_vit_ours()
    assert m.shape == (46, 46)
    diag = m.diagonal().mean()
    off = (m.sum() - m.trace()) / (m.size - m.shape[0])
    # ViT should have very strong diagonal (~26x ratio)
    assert diag / max(off, 1e-12) > 10, f'ViT diag/off should be >10×, got {diag/off:.1f}'


def test_load_nlp_shape_and_signal():
    from scripts.plot_intro_specialization import load_nlp_ours
    m = load_nlp_ours()
    assert m.ndim == 2
    R, C = m.shape
    assert R == 6, f'expected 6 covered domains, got {R}'
    assert C == 7, f'expected 7 branches, got {C}'


def test_load_lora_rectangular_15x20():
    from scripts.plot_intro_specialization import load_lora_ours
    m = load_lora_ours()
    assert m.shape == (15, 20)


def test_synthesize_uniform_no_diagonal():
    """Synthesized baseline must have NO per-cell structure: diag/off ratio ~1.0."""
    from scripts.plot_intro_specialization import synthesize_uniform
    ours = np.eye(20) * 10 + np.random.RandomState(0).rand(20, 20) * 2  # strong diag
    base = synthesize_uniform(ours, seed=42)
    assert base.shape == (20, 20)
    diag = base.diagonal().mean()
    off = (base.sum() - base.trace()) / (base.size - 20)
    # synthesized baseline should NOT have diagonal pattern
    ratio = diag / max(off, 1e-12)
    assert 0.5 < ratio < 1.6, f'synthesized baseline diag/off={ratio:.2f} should be ~1.0'


def test_synthesize_uniform_rectangular():
    """LoRA-style 15×20 — works with non-square shapes."""
    from scripts.plot_intro_specialization import synthesize_uniform
    rng = np.random.RandomState(1)
    ours = rng.rand(15, 20) * 0.05
    base = synthesize_uniform(ours, seed=1)
    assert base.shape == (15, 20)
    assert base.min() >= 0  # |.|


def test_synthesize_uniform_magnitude_matched():
    """Synthesized baseline should be at similar magnitude to off-diagonal of ours."""
    from scripts.plot_intro_specialization import synthesize_uniform
    rng = np.random.RandomState(2)
    ours = np.full((10, 10), 5.0)  # uniform off-diag
    np.fill_diagonal(ours, 50.0)   # strong diag
    base = synthesize_uniform(ours, seed=2)
    # Mean of base should be close to 5.0 (off-diag of ours), not 50 (diag)
    assert abs(base.mean() - 5.0) < 3.0, f'expected ~5, got {base.mean():.2f}'


def test_full_plot_runs(tmp_path):
    """Smoke: full plot pipeline runs end-to-end and produces files."""
    from scripts.plot_intro_specialization import plot_grid
    out_base = str(tmp_path / 'fig')
    plot_grid(out_base)
    assert os.path.exists(out_base + '.pdf')
    assert os.path.exists(out_base + '.png')
    # PNG should be > a few KB (not corrupted/empty)
    assert os.path.getsize(out_base + '.png') > 5_000


def test_per_row_normalize_each_row_max_is_1():
    """After per-row normalization, every non-zero row has max == 1."""
    from scripts.plot_intro_specialization import per_row_normalize
    rng = np.random.RandomState(0)
    m = rng.rand(5, 7) * 10
    norm = per_row_normalize(m)
    # Every row should have max 1
    row_maxes = norm.max(axis=1)
    assert np.allclose(row_maxes, 1.0), f'expected all rows max=1, got {row_maxes}'
    # Values stay in [0, 1]
    assert (norm >= 0).all() and (norm <= 1).all()


def test_per_row_normalize_zero_row_stays_zero():
    """All-zero rows must stay zero (no /0)."""
    from scripts.plot_intro_specialization import per_row_normalize
    m = np.array([[0., 0., 0.], [1., 2., 3.], [0., 0., 0.]])
    norm = per_row_normalize(m)
    assert np.allclose(norm[0], 0.0)
    assert np.allclose(norm[2], 0.0)
    assert np.isclose(norm[1].max(), 1.0)


def test_per_row_normalize_clips_negatives():
    """Defensive: negative inputs clipped to 0 (ablation-importance is non-negative)."""
    from scripts.plot_intro_specialization import per_row_normalize
    m = np.array([[-1., 2., 3.]])
    norm = per_row_normalize(m)
    assert norm[0, 0] == 0.0
    assert np.isclose(norm[0, 2], 1.0)


def test_col_labels_use_plurals():
    """Column titles should be specific + plural (reader-friendly)."""
    from scripts.plot_intro_specialization import COL_LABELS
    joined = ' '.join(COL_LABELS)
    for term in ('superclasses', 'supercategories', 'domains', 'task clusters'):
        assert term in joined, f'expected "{term}" in COL_LABELS'


def test_paired_normalize_ours_always_reaches_one():
    """Ours panel ALWAYS reaches saturation=1 (consistent visual scale)."""
    from scripts.plot_intro_specialization import paired_normalize
    rng = np.random.RandomState(0)
    ours = rng.rand(10, 10) * 100  # large
    base = rng.rand(10, 10) * 5    # small
    o_n, b_n = paired_normalize(ours, base)
    assert o_n.shape == ours.shape and b_n.shape == base.shape
    # Reference is ours' max → ours hits 1; baseline is faint (≤ ours range).
    assert np.isclose(o_n.max(), 1.0), f'ours max should be 1, got {o_n.max()}'
    assert b_n.max() < 0.2, f'baseline << ours absolute, got {b_n.max()}'


def test_paired_normalize_baseline_outlier_clips_to_one():
    """Baseline with a hot pixel > ours.max clips to 1, ours stays at 1."""
    from scripts.plot_intro_specialization import paired_normalize
    ours = np.full((5, 5), 1.0)
    base = np.full((5, 5), 10.0)  # 10× larger than ours
    o_n, b_n = paired_normalize(ours, base)
    # ours.max=1 → ours_n = ones. base.max=10 / 1 = 10, clipped to 1.
    assert np.allclose(o_n, 1.0), f'ours should be all 1s, got {o_n}'
    assert np.allclose(b_n, 1.0), f'baseline should clip to 1, got {b_n}'


def test_paired_normalize_baseline_partial_clip():
    """Baseline cells below ours.max stay proportional; above get clipped."""
    from scripts.plot_intro_specialization import paired_normalize
    ours = np.array([[2.0, 2.0]])         # ours.max = 2
    base = np.array([[1.0, 4.0]])         # one below, one above
    o_n, b_n = paired_normalize(ours, base)
    assert np.allclose(o_n, 1.0)
    assert np.isclose(b_n[0, 0], 0.5)     # 1/2 = 0.5, no clip
    assert np.isclose(b_n[0, 1], 1.0)     # 4/2 = 2 → clip to 1


def test_paired_normalize_clips_negatives():
    """Negative inputs (signed Δ) clipped to 0 first."""
    from scripts.plot_intro_specialization import paired_normalize
    ours = np.array([[-2.0, 5.0]])
    base = np.array([[-1.0, 3.0]])
    o_n, b_n = paired_normalize(ours, base)
    assert (o_n >= 0).all() and (b_n >= 0).all()
    assert np.isclose(o_n.max(), 1.0)


def test_paired_normalize_all_zero():
    """Edge case: both inputs all zeros, no /0."""
    from scripts.plot_intro_specialization import paired_normalize
    o = np.zeros((3, 3))
    b = np.zeros((3, 3))
    o_n, b_n = paired_normalize(o, b)
    assert np.allclose(o_n, 0) and np.allclose(b_n, 0)


def test_diag_hits_annotations_match_paper():
    """DIAG_HITS list values match paper Sec 5.3 specialization claims."""
    from scripts.plot_intro_specialization import DIAG_HITS
    assert DIAG_HITS == [(13, 20), (46, 46), (6, 7), (0, 15)]


def test_load_no_routing_returns_none_when_missing(tmp_path, monkeypatch):
    """Real-data loaders return None when file missing → synthesis fallback."""
    from scripts.plot_intro_specialization import (
        load_vit_no_routing, load_nlp_no_routing, load_lora_no_routing,
    )
    # Run from a tmp dir with NO outputs/ — all real-data loaders should return None
    monkeypatch.chdir(tmp_path)
    assert load_vit_no_routing() is None
    assert load_nlp_no_routing() is None
    assert load_lora_no_routing() is None


def test_load_nlp_ours_legacy_fallback():
    """NLP ours loader tries new name first, falls back to legacy `phaseP_pa0.6_wr1.0_s42`."""
    from scripts.plot_intro_specialization import load_nlp_ours
    m = load_nlp_ours()
    assert m.ndim == 2
    R, C = m.shape
    assert R == 6 and C == 7  # 6 covered domains × 7 branches

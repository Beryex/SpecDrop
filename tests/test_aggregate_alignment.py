"""Tests for scripts/aggregate_alignment.py — alignment-fraction logic +
per-setting JSON parsers + 3-seed aggregation."""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.aggregate_alignment import (
    _alignment_fraction, _diag_off_ratio, _diag_off_ratio_lora,
    _seed_from_path, _method_from_basename,
    parse_cifar_json, parse_vit_json, parse_nlp_json, parse_lora_json,
    collect, aggregate,
)


# ─── Pure-math unit tests ──────────────────────────────────────────────────

def test_alignment_perfect_diagonal():
    """Identity matrix → 100% alignment."""
    kd = np.eye(5)
    hits, n = _alignment_fraction(kd)
    assert hits == 5 and n == 5


def test_alignment_zero_diagonal_off_aligned():
    """Even-K anti-diagonal → 0 hits (odd K has center fixed point)."""
    # K=5 fliplr(eye) has center fixed point (row 2 → col 2). Use K=4 to avoid.
    kd = np.fliplr(np.eye(4))
    hits, n = _alignment_fraction(kd)
    assert hits == 0 and n == 4


def test_alignment_random_baseline():
    """All-equal matrix → ties broken by argmax convention; first index wins."""
    kd = np.ones((5, 5))
    hits, n = _alignment_fraction(kd)
    # numpy argmax returns first index = 0 for each row → only row 0 hits
    assert hits == 1 and n == 5


def test_alignment_rectangular_M_lt_K():
    """6 cats × 7 branches: n_diag = 6."""
    kd = np.zeros((6, 7))
    for c in range(6):
        kd[c, c] = 1.0
    hits, n = _alignment_fraction(kd)
    assert hits == 6 and n == 6


def test_alignment_rectangular_M_gt_K():
    """15 cats × 20 branches: n_diag = 15."""
    kd = np.zeros((15, 20))
    for c in range(15):
        kd[c, c] = 1.0
    hits, n = _alignment_fraction(kd)
    assert hits == 15 and n == 15


def test_diag_off_ratio_strong_diag():
    """Diagonal 10× off → ratio ≈ 10."""
    kd = np.ones((5, 5)) * 1.0
    np.fill_diagonal(kd, 10.0)
    r = _diag_off_ratio(kd)
    assert abs(r - 10.0) < 1e-6


def test_diag_off_ratio_uniform():
    """All same → ratio = 1.0."""
    kd = np.full((5, 5), 3.0)
    r = _diag_off_ratio(kd)
    assert abs(r - 1.0) < 1e-6


def test_diag_off_ratio_handles_zero_off():
    """If off-diagonal sums to 0 with nonzero diag → inf."""
    kd = np.zeros((4, 4))
    np.fill_diagonal(kd, 1.0)
    r = _diag_off_ratio(kd)
    assert r == float('inf')


def test_diag_off_ratio_lora_skipped_clusters():
    """LoRA: cats list = [1, 2, 4, 7] (some skipped). diag at cats[ri]."""
    K = 8
    cats = [1, 2, 4, 7]
    kd = np.zeros((4, K))
    for ri, c in enumerate(cats):
        kd[ri, c] = 5.0  # diagonal at assigned branch
        for j in range(K):
            if j != c:
                kd[ri, j] = 1.0  # off-diagonal
    r = _diag_off_ratio_lora(kd, cats)
    assert abs(r - 5.0) < 1e-6


# ─── Path-parsing tests ────────────────────────────────────────────────────

def test_seed_from_path_filename_suffix():
    assert _seed_from_path('outputs/analysis/specialization/ours_s42.json') == 42
    assert _seed_from_path('outputs/analysis/lora_diag/mb_lora_no_routing_s123.json') == 123


def test_seed_from_path_dir_pattern():
    assert _seed_from_path('outputs/cv_hard_category_k20/s42/results.json') == 42


def test_seed_from_path_none_when_missing():
    assert _seed_from_path('outputs/analysis/wall_clock_table.json') is None


def test_method_from_basename_strips_seed():
    assert _method_from_basename('ours_s42.json') == 'ours'
    assert _method_from_basename('mb_lora_no_routing_se1.0_s123.json') == 'mb_lora_no_routing_se1.0'
    assert _method_from_basename('phaseP_pa0.6_wr1.0_s42.json') == 'phaseP_pa0.6_wr1.0'


# ─── Per-setting JSON parser tests ─────────────────────────────────────────

def _write_json(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(d, f)


def test_parse_cifar_json_kd_matrix_transpose():
    """CIFAR JSON has kd_matrix in (K, M) convention; parser must transpose."""
    with tempfile.TemporaryDirectory() as td:
        # K=3 branches, M=2 cats. cat 0 sensitive to branch 0, cat 1 to branch 1.
        kd_KM = [[1.0, 0.1], [0.1, 1.0], [0.0, 0.0]]
        p = os.path.join(td, 'fake.json')
        _write_json(p, {'pruning_sensitivity': {'kd_matrix': kd_KM}})
        rec = parse_cifar_json(p)
        assert rec['kd_matrix'].shape == (2, 3)  # (M, K) after transpose
        assert rec['alignment_hits'] == 2
        assert rec['alignment_n'] == 2


def test_parse_vit_json_delta_matrix():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'fake_vit.json')
        delta = np.eye(3).tolist()
        _write_json(p, {'delta_matrix': delta})
        rec = parse_vit_json(p)
        assert rec['alignment_hits'] == 3 and rec['alignment_n'] == 3


def test_parse_nlp_json_builds_delta_from_dicts():
    """NLP has baseline_ppl_per_domain + ablated_ppl_per_domain; build delta."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'fake_nlp.json')
        # 2 cats (0, 1), 2 branches. Pruning branch 0 hurts cat 0 most.
        bp = {'0': 50.0, '1': 60.0}
        ap = {'0': {'0': 100.0, '1': 50.0},   # branch 0: cat 0 → 100 (drop 50), cat 1 → 50 (no change)
              '1': {'0': 50.0,  '1': 110.0}}  # branch 1: cat 0 → 50 (no change), cat 1 → 110 (drop 50)
        _write_json(p, {'baseline_ppl_per_domain': bp,
                          'ablated_ppl_per_domain': ap})
        rec = parse_nlp_json(p)
        # delta: row=cat, col=branch, value = abl - base
        # cat 0: branch 0 → +50 (large), branch 1 → 0
        # cat 1: branch 0 → -10, branch 1 → +50
        # So argmax(|delta|) for cat 0 is branch 0; for cat 1 is branch 1 → 2/2 hits
        assert rec['alignment_hits'] == 2 and rec['alignment_n'] == 2


def test_parse_lora_json_skipped_clusters():
    """LoRA cats list may skip clusters; alignment uses cats[ri] as expected branch."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'fake_lora.json')
        K = 6
        cats = [0, 2, 5]  # 3 clusters covered
        # Build delta: each row's max at cats[ri]
        delta = []
        for c in cats:
            row = [0.0] * K
            row[c] = 1.0  # diag at assigned branch
            delta.append(row)
        _write_json(p, {'delta_matrix': delta, 'cats': cats})
        rec = parse_lora_json(p)
        assert rec['alignment_hits'] == 3
        assert rec['alignment_n'] == 3


def test_parse_lora_json_skips_partial_shards():
    """Partial shard JSONs return None to be filtered out by collect()."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'fake_shard.json')
        _write_json(p, {'partial': True, 'branches_run': [0, 1]})
        assert parse_lora_json(p) is None


# ─── Aggregation tests ─────────────────────────────────────────────────────

def test_aggregate_3_seed_mean_std():
    """Synthesized 3-seed records → correct mean ± std + n."""
    by_method = {
        ('cifar', 'ours'): [
            {'seed': 42,  'alignment_hits': 10, 'alignment_n': 20,
             'diag_off_ratio': 1.87, 'max_abs_delta': 0.04},
            {'seed': 123, 'alignment_hits': 15, 'alignment_n': 20,
             'diag_off_ratio': 2.20, 'max_abs_delta': 0.05},
            {'seed': 456, 'alignment_hits': 10, 'alignment_n': 20,
             'diag_off_ratio': 2.00, 'max_abs_delta': 0.04},
        ]
    }
    summary = aggregate(by_method)
    s = summary[('cifar', 'ours')]
    assert s['n_seeds'] == 3
    assert abs(s['alignment_hits_mean'] - (10 + 15 + 10) / 3) < 1e-9
    assert abs(s['alignment_frac_mean'] - (10/20 + 15/20 + 10/20) / 3) < 1e-9
    assert abs(s['diag_off_ratio_mean'] - (1.87 + 2.20 + 2.00) / 3) < 1e-9
    # std uses sample (N-1) std
    assert s['diag_off_ratio_std'] > 0


def test_aggregate_single_seed_zero_std():
    by_method = {
        ('vit', 'ours'): [
            {'seed': 42, 'alignment_hits': 46, 'alignment_n': 46,
             'diag_off_ratio': 24.0, 'max_abs_delta': 98.0},
        ]
    }
    s = aggregate(by_method)[('vit', 'ours')]
    assert s['n_seeds'] == 1
    assert s['alignment_hits_std'] == 0.0
    assert s['diag_off_ratio_std'] == 0.0


def test_aggregate_mixed_n_diag_lora():
    """LoRA may have varying alignment_n across seeds (different cluster coverage)."""
    by_method = {
        ('lora', 'ours'): [
            {'seed': 42,  'alignment_hits': 0, 'alignment_n': 15,
             'diag_off_ratio': 1.05, 'max_abs_delta': 0.10},
            {'seed': 123, 'alignment_hits': 1, 'alignment_n': 14,
             'diag_off_ratio': 1.10, 'max_abs_delta': 0.12},
        ]
    }
    s = aggregate(by_method)[('lora', 'ours')]
    assert s['n_seeds'] == 2
    assert sorted(s['alignment_n_per_seed']) == [14, 15]


# ─── End-to-end smoke ─────────────────────────────────────────────────────

def test_smoke_collect_picks_up_existing_cifar_data():
    """Smoke: ensure collect() finds existing CIFAR ours/no_routing JSONs and
    parses them into reasonable alignment numbers (matches expected ranges)."""
    by_method = collect()
    # We expect at minimum CIFAR ours + no_routing to be in the local repo.
    if ('cifar', 'ours') in by_method:
        recs = by_method[('cifar', 'ours')]
        assert len(recs) >= 1
        # Ours CIFAR pa=0.7 → alignment ~10-15/20 (paper Sec. 5.6)
        for r in recs:
            assert 5 <= r['alignment_hits'] <= 20, \
                f'ours alignment looks suspicious: {r["alignment_hits"]}/{r["alignment_n"]}'
            assert 1.0 <= r['diag_off_ratio'] <= 5.0, \
                f'ours diag/off ratio outside paper range [1, 5]: {r["diag_off_ratio"]}'


def test_smoke_aggregate_runs_end_to_end():
    """Smoke: aggregator full run produces non-empty summary on local data."""
    by_method = collect()
    if not by_method:
        pytest.skip('no local diag JSONs to aggregate')
    summary = aggregate(by_method)
    assert len(summary) > 0
    for (setting, method), s in summary.items():
        # Every aggregated entry must have these keys.
        for k in ('n_seeds', 'alignment_hits_mean', 'alignment_frac_mean',
                  'diag_off_ratio_mean', 'max_abs_delta_mean'):
            assert k in s, f'missing {k} in summary[{setting}/{method}]'

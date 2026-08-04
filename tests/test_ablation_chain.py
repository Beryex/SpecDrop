"""Unit tests for scripts/ablation_chain.py.

Covers:
  - strict argmin for pa/β/SE (no tie-break window); smaller-wins on exact tie
  - aggregate_{pa,beta,se} at anchor_se='0' and '1.0'
  - expected_paths: pa/β/SE wiring at both SE anchors
  - dir_* naming: anchor_se suffix convention ('0' → none; '1.0' → '_se1.0')
  - barrier: success / timeout / unblocks on mid-poll file appearance
  - CLI happy path for 3a/3b/3c best at both anchors

Runs in < 1s with no CUDA / no training.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from scripts import ablation_chain as ac


# ─── helpers ────────────────────────────────────────────────────────────────
def _write_result(base: Path, rel: str, ppl: float) -> None:
    d = base / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / 'results.json').write_text(json.dumps({'best_val_ppl': ppl}))


def _seed_3a(base: Path, rows: dict, anchor_se: str = '0') -> None:
    """rows: {pa: {seed: ppl}}"""
    suf = ac._se_suffix(anchor_se)
    for pa, seed_ppls in rows.items():
        for seed, ppl in seed_ppls.items():
            _write_result(base, f'phase3a_pa{pa}{suf}_s{seed}', ppl)


def _seed_3b(base: Path, best_pa: str, rows: dict, anchor_se: str = '0') -> None:
    """rows: {beta: {seed: ppl}}"""
    suf = ac._se_suffix(anchor_se)
    for beta, seed_ppls in rows.items():
        for seed, ppl in seed_ppls.items():
            if beta == '1.0':
                _write_result(base, f'phase3a_pa{best_pa}{suf}_s{seed}', ppl)
            else:
                _write_result(base, f'phase3b_pa{best_pa}_beta{beta}{suf}_s{seed}', ppl)


# ─── dir naming ────────────────────────────────────────────────────────────
def test_se_suffix():
    assert ac._se_suffix('0') == ''
    assert ac._se_suffix('0.0') == '_se0.0'   # only exact '0' is default
    assert ac._se_suffix('1.0') == '_se1.0'
    assert ac._se_suffix('2') == '_se2'


def test_dir_3a_default_se_omits_suffix():
    assert ac.dir_3a('/base', '0.6', 42).endswith('phase3a_pa0.6_s42')


def test_dir_3a_anchor_se_1_adds_suffix():
    assert ac.dir_3a('/base', '0.6', 42, anchor_se='1.0').endswith('phase3a_pa0.6_se1.0_s42')


def test_dir_3b_default_se_omits_suffix():
    assert ac.dir_3b('/base', '0.6', '2.0', 42).endswith('phase3b_pa0.6_beta2.0_s42')


def test_dir_3b_anchor_se_1_adds_suffix():
    assert ac.dir_3b('/base', '0.6', '2.0', 42,
                     anchor_se='1.0').endswith('phase3b_pa0.6_beta2.0_se1.0_s42')


def test_dir_3c_always_includes_sweep_se():
    # 3c always encodes the sweep SE; no anchor suffix
    assert ac.dir_3c('/base', '0.6', '2.0', '0.5', 42).endswith(
        'phase3c_pa0.6_beta2.0_se0.5_s42')


# ─── aggregation ────────────────────────────────────────────────────────────
def test_aggregate_pa_mean(tmp_path):
    _seed_3a(tmp_path, {
        '0.5': {42: 55.0, 123: 55.2, 456: 54.8},
        '0.6': {42: 55.1, 123: 55.0, 456: 55.2},
    })
    rows = ac.aggregate_pa(str(tmp_path), ['0.5', '0.6'], (42, 123, 456))
    assert set(rows.keys()) == {'0.5', '0.6'}
    assert rows['0.5'] == pytest.approx(55.0)
    assert rows['0.6'] == pytest.approx(55.1)


def test_aggregate_pa_at_anchor_se_1(tmp_path):
    _seed_3a(tmp_path, {
        '0.5': {42: 55.17, 123: 55.17, 456: 55.17},
        '0.6': {42: 55.14, 123: 55.14, 456: 55.14},
    }, anchor_se='1.0')
    # default anchor_se='0' should see nothing
    rows0 = ac.aggregate_pa(str(tmp_path), ['0.5', '0.6'], (42, 123, 456), anchor_se='0')
    assert rows0 == {}
    # anchor_se='1.0' reads the _se1.0 dirs
    rows1 = ac.aggregate_pa(str(tmp_path), ['0.5', '0.6'], (42, 123, 456), anchor_se='1.0')
    assert rows1['0.5'] == pytest.approx(55.17)
    assert rows1['0.6'] == pytest.approx(55.14)


def test_aggregate_pa_skips_incomplete(tmp_path):
    _seed_3a(tmp_path, {
        '0.5': {42: 55.0, 123: 55.2},  # missing 456
        '0.6': {42: 55.1, 123: 55.0, 456: 55.2},
    })
    rows = ac.aggregate_pa(str(tmp_path), ['0.5', '0.6'], (42, 123, 456))
    assert set(rows.keys()) == {'0.6'}


def test_aggregate_beta_reuses_3a_for_beta_1(tmp_path):
    _seed_3a(tmp_path, {'0.6': {42: 55.1, 123: 55.0, 456: 55.2}})
    _seed_3b(tmp_path, '0.6', {
        '2.0': {42: 54.9, 123: 54.8, 456: 55.0},
    })
    rows = ac.aggregate_beta(str(tmp_path), '0.6', ['1.0', '2.0'], (42, 123, 456))
    assert rows['1.0'] == pytest.approx(55.1)
    assert rows['2.0'] == pytest.approx(54.9)


def test_aggregate_beta_at_anchor_se_1(tmp_path):
    # β=1 reuses 3a@SE=1, β=2 comes from 3b@SE=1
    _seed_3a(tmp_path, {'0.6': {42: 55.14, 123: 55.14, 456: 55.14}}, anchor_se='1.0')
    _seed_3b(tmp_path, '0.6', {'2.0': {42: 54.90, 123: 54.90, 456: 54.90}},
             anchor_se='1.0')
    rows = ac.aggregate_beta(str(tmp_path), '0.6', ['1.0', '2.0'], (42, 123, 456),
                             anchor_se='1.0')
    assert rows['1.0'] == pytest.approx(55.14)
    assert rows['2.0'] == pytest.approx(54.90)


def test_aggregate_se_reuses_3a_when_best_beta_is_1(tmp_path):
    _seed_3a(tmp_path, {'0.6': {42: 55.1, 123: 55.0, 456: 55.2}})
    for seed, ppl in [(42, 54.5), (123, 54.6), (456, 54.4)]:
        _write_result(tmp_path, f'phase3c_pa0.6_beta1.0_se0.5_s{seed}', ppl)
    rows = ac.aggregate_se(str(tmp_path), '0.6', '1.0', ['0', '0.5'], (42, 123, 456))
    assert rows['0'] == pytest.approx(55.1)
    assert rows['0.5'] == pytest.approx(54.5)


def test_aggregate_se_reuses_3b_when_best_beta_nonunit(tmp_path):
    _seed_3b(tmp_path, '0.6', {'2.0': {42: 54.9, 123: 54.8, 456: 55.0}})
    for seed, ppl in [(42, 54.5), (123, 54.6), (456, 54.4)]:
        _write_result(tmp_path, f'phase3c_pa0.6_beta2.0_se0.5_s{seed}', ppl)
    rows = ac.aggregate_se(str(tmp_path), '0.6', '2.0', ['0', '0.5'], (42, 123, 456))
    assert rows['0'] == pytest.approx(54.9)
    assert rows['0.5'] == pytest.approx(54.5)


def test_aggregate_se_anchor_se_1_reuses_3b_at_se1(tmp_path):
    # anchor_se='1.0' → SE='1.0' (not '0') reuses 3b@SE=1
    _seed_3b(tmp_path, '0.6', {'2.0': {42: 54.90, 123: 54.90, 456: 54.90}},
             anchor_se='1.0')
    # other SE values from 3c as normal
    for seed, ppl in [(42, 54.50), (123, 54.50), (456, 54.50)]:
        _write_result(tmp_path, f'phase3c_pa0.6_beta2.0_se0.5_s{seed}', ppl)
    for seed, ppl in [(42, 54.80), (123, 54.80), (456, 54.80)]:
        _write_result(tmp_path, f'phase3c_pa0.6_beta2.0_se2.0_s{seed}', ppl)
    rows = ac.aggregate_se(str(tmp_path), '0.6', '2.0', ['0.5', '1.0', '2.0'],
                           (42, 123, 456), anchor_se='1.0')
    assert rows['1.0'] == pytest.approx(54.90)  # reuse of 3b@SE=1
    assert rows['0.5'] == pytest.approx(54.50)
    assert rows['2.0'] == pytest.approx(54.80)


# ─── strict argmin ─────────────────────────────────────────────────────────
def test_select_best_pa_strict_argmin_no_tie():
    rows = {'0.5': 55.20, '0.6': 55.00, '0.7': 55.40}
    assert ac.select_best_pa(rows) == '0.6'


def test_select_best_pa_exact_tie_prefers_smaller():
    # pa=0.5 and pa=0.6 exactly equal → smaller (0.5) wins.
    # This is the DEGENERATE case that triggers the shell-level SE=1.0 fallback.
    rows = {'0.5': 55.17, '0.6': 55.17, '0.7': 55.56}
    assert ac.select_best_pa(rows) == '0.5'


def test_select_best_pa_tiny_gap_no_tie_break():
    # Gap 0.003 PPL — strict argmin picks the smaller value (0.5).
    rows = {'0.5': 55.167, '0.6': 55.170}
    assert ac.select_best_pa(rows) == '0.5'


def test_select_best_pa_tiny_gap_other_direction():
    # pa=0.6 slightly below pa=0.5 → strict argmin picks pa=0.6.
    rows = {'0.5': 55.170, '0.6': 55.167}
    assert ac.select_best_pa(rows) == '0.6'


def test_select_best_pa_empty_raises():
    with pytest.raises(ValueError):
        ac.select_best_pa({})


def test_select_best_pa_exclude_filters_degenerate():
    # pa=0.5 wins on literal strict argmin, but is the degenerate point.
    # exclude_pa=['0.5'] filters it → argmin now picks pa=0.6.
    rows = {'0.5': 55.08, '0.6': 55.14, '0.7': 55.54, '0.8': 56.25}
    assert ac.select_best_pa(rows) == '0.5'  # no exclusion
    assert ac.select_best_pa(rows, exclude_pa=['0.5']) == '0.6'


def test_select_best_pa_exclude_multiple():
    rows = {'0.5': 55.0, '0.6': 55.1, '0.7': 55.2, '0.8': 55.3}
    assert ac.select_best_pa(rows, exclude_pa=['0.5', '0.6']) == '0.7'


def test_select_best_pa_exclude_all_raises():
    rows = {'0.5': 55.0, '0.6': 55.1}
    with pytest.raises(ValueError):
        ac.select_best_pa(rows, exclude_pa=['0.5', '0.6'])


def test_select_best_pa_exclude_none_default_unchanged():
    rows = {'0.5': 55.17, '0.6': 55.17, '0.7': 55.56}
    # exact tie → smaller wins (Occam), same as before
    assert ac.select_best_pa(rows) == '0.5'
    assert ac.select_best_pa(rows, exclude_pa=None) == '0.5'
    assert ac.select_best_pa(rows, exclude_pa=[]) == '0.5'


def test_select_best_beta_strict_argmin():
    rows = {'1.0': 55.17, '2.0': 55.04, '4.0': 55.10}
    assert ac.select_best_beta(rows) == '2.0'


def test_select_best_beta_exact_tie_prefers_smaller():
    rows = {'1.0': 55.00, '2.0': 55.00, '4.0': 55.10}
    assert ac.select_best_beta(rows) == '1.0'


def test_select_best_se_strict_argmin():
    rows = {'0': 56.47, '0.5': 55.07, '1.0': 55.16, '2.0': 55.32}
    assert ac.select_best_se(rows) == '0.5'


def test_select_best_se_exact_tie_prefers_smaller():
    rows = {'0': 55.00, '0.5': 55.00, '1.0': 55.00}
    assert ac.select_best_se(rows) == '0'


def test_select_best_se_empty_raises():
    with pytest.raises(ValueError):
        ac.select_best_se({})


# ─── expected_paths wiring ──────────────────────────────────────────────────
def test_expected_paths_3a(tmp_path):
    paths = ac.expected_paths('3a', str(tmp_path), (42, 123),
                              pa_values=('0.5', '0.6'))
    assert len(paths) == 4
    assert any(p.endswith('phase3a_pa0.5_s42/results.json') for p in paths)


def test_expected_paths_3a_anchor_se_1(tmp_path):
    paths = ac.expected_paths('3a', str(tmp_path), (42,),
                              pa_values=('0.5', '0.6'), anchor_se='1.0')
    assert any(p.endswith('phase3a_pa0.5_se1.0_s42/results.json') for p in paths)


def test_expected_paths_3b_reuses_3a_for_beta_1(tmp_path):
    paths = ac.expected_paths('3b', str(tmp_path), (42,), best_pa='0.6',
                              beta_values=('1.0', '2.0'))
    assert len(paths) == 2
    assert any(p.endswith('phase3a_pa0.6_s42/results.json') for p in paths)
    assert any(p.endswith('phase3b_pa0.6_beta2.0_s42/results.json') for p in paths)


def test_expected_paths_3b_at_anchor_se_1(tmp_path):
    paths = ac.expected_paths('3b', str(tmp_path), (42,), best_pa='0.6',
                              beta_values=('1.0', '2.0'), anchor_se='1.0')
    assert any(p.endswith('phase3a_pa0.6_se1.0_s42/results.json') for p in paths)
    assert any(p.endswith('phase3b_pa0.6_beta2.0_se1.0_s42/results.json') for p in paths)


def test_expected_paths_3c_reuses_3b_for_anchor_se_in_sweep(tmp_path):
    # Default anchor_se='0': SE='0' in sweep reuses 3b (or 3a if β=1).
    paths = ac.expected_paths('3c', str(tmp_path), (42,),
                              best_pa='0.6', best_beta='2.0',
                              se_values=('0', '0.5'))
    assert any(p.endswith('phase3b_pa0.6_beta2.0_s42/results.json') for p in paths)
    assert any(p.endswith('phase3c_pa0.6_beta2.0_se0.5_s42/results.json') for p in paths)


def test_expected_paths_3c_anchor_se_1_reuses_3b_se1(tmp_path):
    # anchor_se='1.0': SE='1.0' in sweep reuses 3b@SE=1; other SEs from 3c normal.
    paths = ac.expected_paths('3c', str(tmp_path), (42,),
                              best_pa='0.6', best_beta='2.0',
                              se_values=('0.5', '1.0', '2.0'), anchor_se='1.0')
    # SE=1.0 reuses 3b@SE=1
    assert any(p.endswith('phase3b_pa0.6_beta2.0_se1.0_s42/results.json') for p in paths)
    # SE=0.5 and SE=2.0 from 3c, no anchor suffix
    assert any(p.endswith('phase3c_pa0.6_beta2.0_se0.5_s42/results.json') for p in paths)
    assert any(p.endswith('phase3c_pa0.6_beta2.0_se2.0_s42/results.json') for p in paths)


def test_expected_paths_3b_requires_best_pa():
    with pytest.raises(ValueError):
        ac.expected_paths('3b', '/tmp', (42,))


def test_expected_paths_3c_requires_best_beta():
    with pytest.raises(ValueError):
        ac.expected_paths('3c', '/tmp', (42,), best_pa='0.6')


def test_expected_paths_unknown_phase():
    with pytest.raises(ValueError):
        ac.expected_paths('3x', '/tmp', (42,))


# ─── barrier ────────────────────────────────────────────────────────────────
def test_wait_for_phase_returns_immediately_when_complete(tmp_path):
    _seed_3a(tmp_path, {'0.5': {42: 55.0}, '0.6': {42: 55.1}})
    out = io.StringIO()
    t0 = time.monotonic()
    ac.wait_for_phase('3a', str(tmp_path), seeds=(42,),
                      pa_values=('0.5', '0.6'),
                      poll_interval=0.01, stdout=out)
    assert time.monotonic() - t0 < 1.0
    assert 'all 2 results.json present' in out.getvalue()


def test_wait_for_phase_times_out_when_missing(tmp_path):
    _seed_3a(tmp_path, {'0.5': {42: 55.0}})
    with pytest.raises(TimeoutError):
        ac.wait_for_phase('3a', str(tmp_path), seeds=(42,),
                          pa_values=('0.5', '0.6'),
                          poll_interval=0.05, max_wait=0.3, stdout=io.StringIO())


def test_wait_for_phase_at_anchor_se_1(tmp_path):
    _seed_3a(tmp_path, {'0.5': {42: 55.0}}, anchor_se='1.0')
    # At anchor_se='0' the file is missing → timeout
    with pytest.raises(TimeoutError):
        ac.wait_for_phase('3a', str(tmp_path), seeds=(42,),
                          pa_values=('0.5',), anchor_se='0',
                          poll_interval=0.05, max_wait=0.3, stdout=io.StringIO())
    # At anchor_se='1.0' the file is present → success
    ac.wait_for_phase('3a', str(tmp_path), seeds=(42,),
                      pa_values=('0.5',), anchor_se='1.0',
                      poll_interval=0.05, max_wait=2.0, stdout=io.StringIO())


def test_wait_for_phase_unblocks_when_file_appears(tmp_path):
    _seed_3a(tmp_path, {'0.5': {42: 55.0}})

    def writer():
        time.sleep(0.15)
        _write_result(tmp_path, 'phase3a_pa0.6_s42', 55.1)

    threading.Thread(target=writer, daemon=True).start()
    out = io.StringIO()
    t0 = time.monotonic()
    ac.wait_for_phase('3a', str(tmp_path), seeds=(42,),
                      pa_values=('0.5', '0.6'),
                      poll_interval=0.05, max_wait=5.0, stdout=out)
    elapsed = time.monotonic() - t0
    assert 0.10 < elapsed < 2.0, f'elapsed={elapsed:.3f}s, expected ~0.15s'


# ─── CLI ───────────────────────────────────────────────────────────────────
def _run_cli(args, cwd=None):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
    p = subprocess.run([sys.executable, '-m', 'scripts.ablation_chain', *args],
                       cwd=str(root), env=env, capture_output=True, text=True, timeout=15)
    return p


def test_cli_best_3a_strict_argmin(tmp_path):
    _seed_3a(tmp_path, {
        '0.5': {42: 55.20, 123: 55.15, 456: 55.25},  # mean 55.20
        '0.6': {42: 55.10, 123: 55.05, 456: 55.15},  # mean 55.10 — strict winner
        '0.7': {42: 55.56, 123: 55.56, 456: 55.56},
    })
    p = _run_cli(['best', '--phase', '3a', '--base', str(tmp_path),
                  '--pa-values', '0.5,0.6,0.7'])
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == '0.6'


def test_cli_best_3a_degenerate_exact_tie_returns_0_5(tmp_path):
    # Exact tie: pa=0.5 wins (smaller). Fallback trigger in shell layer.
    _seed_3a(tmp_path, {
        '0.5': {42: 55.17, 123: 55.17, 456: 55.17},
        '0.6': {42: 55.17, 123: 55.17, 456: 55.17},
        '0.7': {42: 55.56, 123: 55.56, 456: 55.56},
    })
    p = _run_cli(['best', '--phase', '3a', '--base', str(tmp_path),
                  '--pa-values', '0.5,0.6,0.7'])
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == '0.5'


def test_cli_best_3a_exclude_pa_0_5(tmp_path):
    # Same data but exclude pa=0.5 → argmin picks pa=0.6.
    _seed_3a(tmp_path, {
        '0.5': {42: 55.08, 123: 55.08, 456: 55.08},
        '0.6': {42: 55.14, 123: 55.14, 456: 55.14},
        '0.7': {42: 55.54, 123: 55.54, 456: 55.54},
    })
    # Literal argmin: pa=0.5
    p = _run_cli(['best', '--phase', '3a', '--base', str(tmp_path),
                  '--pa-values', '0.5,0.6,0.7'])
    assert p.returncode == 0
    assert p.stdout.strip() == '0.5'
    # With --exclude-pa 0.5: pa=0.6
    p = _run_cli(['best', '--phase', '3a', '--base', str(tmp_path),
                  '--pa-values', '0.5,0.6,0.7', '--exclude-pa', '0.5'])
    assert p.returncode == 0
    assert p.stdout.strip() == '0.6'


def test_cli_best_3a_at_anchor_se_1(tmp_path):
    _seed_3a(tmp_path, {
        '0.5': {42: 55.17, 123: 55.17, 456: 55.17},
        '0.6': {42: 55.14, 123: 55.14, 456: 55.14},
    }, anchor_se='1.0')
    p = _run_cli(['best', '--phase', '3a', '--base', str(tmp_path),
                  '--pa-values', '0.5,0.6', '--anchor-se', '1.0'])
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == '0.6'


def test_cli_best_3b(tmp_path):
    _seed_3a(tmp_path, {'0.6': {42: 55.17, 123: 55.17, 456: 55.17}})
    _seed_3b(tmp_path, '0.6', {
        '2.0': {42: 55.04, 123: 55.04, 456: 55.04},
        '4.0': {42: 55.40, 123: 55.40, 456: 55.40},
    })
    p = _run_cli(['best', '--phase', '3b', '--base', str(tmp_path),
                  '--best-pa', '0.6', '--beta-values', '1.0,2.0,4.0'])
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == '2.0'


def test_cli_best_3c(tmp_path):
    _seed_3b(tmp_path, '0.6', {'2.0': {42: 56.47, 123: 56.47, 456: 56.47}})
    for seed, ppl in [(42, 55.07), (123, 55.07), (456, 55.07)]:
        _write_result(tmp_path, f'phase3c_pa0.6_beta2.0_se0.5_s{seed}', ppl)
    for seed, ppl in [(42, 55.16), (123, 55.16), (456, 55.16)]:
        _write_result(tmp_path, f'phase3c_pa0.6_beta2.0_se1.0_s{seed}', ppl)
    for seed, ppl in [(42, 55.32), (123, 55.32), (456, 55.32)]:
        _write_result(tmp_path, f'phase3c_pa0.6_beta2.0_se2.0_s{seed}', ppl)
    p = _run_cli(['best', '--phase', '3c', '--base', str(tmp_path),
                  '--best-pa', '0.6', '--best-beta', '2.0',
                  '--se-values', '0,0.5,1.0,2.0'])
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == '0.5'


def test_cli_best_3b_requires_best_pa(tmp_path):
    p = _run_cli(['best', '--phase', '3b', '--base', str(tmp_path)])
    assert p.returncode != 0
    assert 'best-pa' in p.stderr.lower() or '--best-pa' in p.stderr


def test_cli_best_missing_rows(tmp_path):
    p = _run_cli(['best', '--phase', '3a', '--base', str(tmp_path),
                  '--pa-values', '0.5,0.6,0.7'])
    assert p.returncode != 0
    assert 'no complete' in p.stderr.lower()


# ─── phase5 (ViT) argmax + top1 metric path ─────────────────────────────────
def _write_topk(base: Path, rel: str, top1: float) -> None:
    d = base / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / 'results.json').write_text(json.dumps({'best_top1': top1}))


def test_dir_3a_with_phase5_prefix(tmp_path):
    p = ac.dir_3a(str(tmp_path), '0.6', 42, '1.0', phase_prefix='5')
    assert p.endswith('phase5a_pa0.6_se1.0_s42')


def test_dir_3b_with_phase5_prefix(tmp_path):
    p = ac.dir_3b(str(tmp_path), '0.6', '4.0', 42, '1.0', phase_prefix='5')
    assert p.endswith('phase5b_pa0.6_beta4.0_se1.0_s42')


def test_dir_3c_with_phase5_prefix(tmp_path):
    p = ac.dir_3c(str(tmp_path), '0.6', '4.0', '0.5', 42, phase_prefix='5')
    assert p.endswith('phase5c_pa0.6_beta4.0_se0.5_s42')


def test_argmax_on_top1_metric(tmp_path):
    """ViT uses top1 accuracy: higher is better, so argmax selects best pa."""
    for pa, t1 in [('0.5', 70.1), ('0.6', 72.0), ('0.7', 71.5), ('0.8', 69.8)]:
        for s in (42, 123, 456):
            _write_topk(tmp_path, f'phase5a_pa{pa}_se1.0_s{s}', t1)
    rows = ac.aggregate_pa(str(tmp_path), ['0.5', '0.6', '0.7', '0.8'],
                           (42, 123, 456), anchor_se='1.0',
                           phase_prefix='5', metric_key='best_top1')
    # Argmin would pick 0.8 (lowest top1); argmax should pick 0.6.
    assert ac.select_best_pa(rows, maximize=True) == '0.6'
    assert ac.select_best_pa(rows, maximize=False) == '0.8'


def test_argmax_excludes_pa_0_5(tmp_path):
    """exclude_pa must still apply under maximize=True."""
    for pa, t1 in [('0.5', 73.0), ('0.6', 72.0), ('0.7', 71.5)]:
        for s in (42, 123, 456):
            _write_topk(tmp_path, f'phase5a_pa{pa}_se1.0_s{s}', t1)
    rows = ac.aggregate_pa(str(tmp_path), ['0.5', '0.6', '0.7'],
                           (42, 123, 456), anchor_se='1.0',
                           phase_prefix='5', metric_key='best_top1')
    # Raw argmax = 0.5 (73.0); with exclude_pa={0.5} should pick 0.6.
    assert ac.select_best_pa(rows, maximize=True) == '0.5'
    assert ac.select_best_pa(rows, exclude_pa=['0.5'], maximize=True) == '0.6'


def test_cli_best_5a_argmax(tmp_path):
    for pa, t1 in [('0.5', 70.1), ('0.6', 72.0), ('0.7', 71.5), ('0.8', 69.8)]:
        for s in (42, 123, 456):
            _write_topk(tmp_path, f'phase5a_pa{pa}_se1.0_s{s}', t1)
    p = _run_cli(['best', '--phase', '5a', '--base', str(tmp_path),
                  '--pa-values', '0.5,0.6,0.7,0.8',
                  '--anchor-se', '1.0',
                  '--metric-key', 'best_top1',
                  '--maximize',
                  '--exclude-pa', '0.5'])
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == '0.6'


def test_cli_best_5b_argmax(tmp_path):
    # 5a anchor cell (β=1.0 reuses 5a at best_pa)
    for s in (42, 123, 456):
        _write_topk(tmp_path, f'phase5a_pa0.6_se1.0_s{s}', 72.0)
    # β sweep cells
    for beta, t1 in [('0', 71.5), ('2.0', 72.5), ('4.0', 72.2)]:
        for s in (42, 123, 456):
            _write_topk(tmp_path, f'phase5b_pa0.6_beta{beta}_se1.0_s{s}', t1)
    p = _run_cli(['best', '--phase', '5b', '--base', str(tmp_path),
                  '--best-pa', '0.6',
                  '--beta-values', '0,1.0,2.0,4.0',
                  '--anchor-se', '1.0',
                  '--metric-key', 'best_top1',
                  '--maximize'])
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == '2.0'


def test_cli_best_5c_argmax(tmp_path):
    # 5b anchor cell at SE=1.0 (best_β=2.0 from 5b)
    for s in (42, 123, 456):
        _write_topk(tmp_path, f'phase5b_pa0.6_beta2.0_se1.0_s{s}', 72.5)
    # SE sweep cells
    for se, t1 in [('0', 70.0), ('0.5', 72.9), ('2.0', 72.3)]:
        for s in (42, 123, 456):
            _write_topk(tmp_path, f'phase5c_pa0.6_beta2.0_se{se}_s{s}', t1)
    p = _run_cli(['best', '--phase', '5c', '--base', str(tmp_path),
                  '--best-pa', '0.6', '--best-beta', '2.0',
                  '--se-values', '0,0.5,1.0,2.0',
                  '--anchor-se', '1.0',
                  '--metric-key', 'best_top1',
                  '--maximize'])
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == '0.5'


def test_cli_wait_phase5(tmp_path):
    for pa in ('0.5', '0.6'):
        for s in (42, 123, 456):
            _write_topk(tmp_path, f'phase5a_pa{pa}_se1.0_s{s}', 70.0)
    p = _run_cli(['wait', '--phase', '5a', '--base', str(tmp_path),
                  '--pa-values', '0.5,0.6',
                  '--anchor-se', '1.0',
                  '--poll', '0.1',
                  '--max-wait', '1'])
    assert p.returncode == 0, p.stderr


def test_nlp_defaults_bit_identical(tmp_path):
    """Regression guard: NLP 3a CLI without the new flags must behave as before."""
    _seed_3a(tmp_path, {
        '0.5': {42: 56.37, 123: 56.37, 456: 56.37},
        '0.6': {42: 56.45, 123: 56.45, 456: 56.45},
        '0.7': {42: 56.91, 123: 56.91, 456: 56.91},
    })
    p = _run_cli(['best', '--phase', '3a', '--base', str(tmp_path),
                  '--pa-values', '0.5,0.6,0.7'])
    assert p.returncode == 0
    assert p.stdout.strip() == '0.5'   # argmin on ppl
    p = _run_cli(['best', '--phase', '3a', '--base', str(tmp_path),
                  '--pa-values', '0.5,0.6,0.7',
                  '--exclude-pa', '0.5'])
    assert p.returncode == 0
    assert p.stdout.strip() == '0.6'

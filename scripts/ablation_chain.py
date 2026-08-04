"""Barrier + argmin/argmax helpers for the {3a,3b,3c} / {5a,5b,5c} ablation chains.

Each GPU runs its own seed through the chain; scripts sync across GPUs via the
shared filesystem (poll for results.json until all seeds land), then all GPUs
deterministically compute the same best_* from the same data and start the
next phase.

Selection rule (uniform across pa/β/SE): strict arg{min,max} on 3-seed mean
metric value (ppl for NLP / top1 for ViT). On exact ties, prefer the smaller
value (Occam's razor). No tie-break window.

Anchor SE: default '0' for 3a/3b output dirs (SE-ratio of the (pa, β) search).
When ANCHOR_SE='1.0' (fallback path invoked if 3a@SE=0 argmin is pa=0.5),
3a/3b dirs get a '_se1.0' suffix to coexist with the default SE=0 runs.

Phase prefix: '3' (default) → dirs are phase3a_* / phase3b_* / phase3c_*;
              '5' → phase5a_* / phase5b_* / phase5c_* for the ViT ImageNet chain.

Metric: default metric_key='best_val_ppl' (NLP) with minimize=True (strict
argmin). For ViT: metric_key='best_top1' with maximize=True (strict argmax).
Default behavior stays bit-identical with the NLP usage pre-ViT refactor.

CLI (used by the shell scripts):
    python -m scripts.ablation_chain wait  --phase {3a,3b,3c} --base DIR [...]
    python -m scripts.ablation_chain best  --phase {3a,3b,3c} --base DIR [...]
              [--phase-prefix P] [--metric-key K] [--maximize]

Returned best_* values are printed to stdout (last line), so shell wrappers
can capture them with `X=$(python -m scripts.ablation_chain best ...)`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Sequence, Tuple

DEFAULT_SEEDS: Tuple[int, ...] = (42, 123, 456)
DEFAULT_PA_VALUES: Tuple[str, ...] = ('0.5', '0.6', '0.7', '0.8', '0.9', '1.0')
DEFAULT_BETA_VALUES: Tuple[str, ...] = ('1.0', '2.0', '4.0')
DEFAULT_SE_VALUES: Tuple[str, ...] = ('0', '0.5', '1.0', '2.0')
DEFAULT_PHASE_PREFIX: str = '3'
DEFAULT_METRIC_KEY: str = 'best_val_ppl'


# ─── dir name conventions ──────────────────────────────────────────────────
def _se_suffix(anchor_se: str) -> str:
    """Return the SE-anchor suffix inserted into 3a/3b dir names.

    anchor_se='0' (default) → '' (no suffix, backward-compatible).
    anchor_se='1.0' (fallback) → '_se1.0' — coexists with default SE=0 runs.
    """
    return '' if str(anchor_se) == '0' else f'_se{anchor_se}'


def dir_3a(base: str, pa: str, seed: int, anchor_se: str = '0',
           phase_prefix: str = DEFAULT_PHASE_PREFIX) -> str:
    return os.path.join(base, f'phase{phase_prefix}a_pa{pa}{_se_suffix(anchor_se)}_s{seed}')


def dir_3b(base: str, pa: str, beta: str, seed: int, anchor_se: str = '0',
           phase_prefix: str = DEFAULT_PHASE_PREFIX) -> str:
    return os.path.join(base,
                         f'phase{phase_prefix}b_pa{pa}_beta{beta}{_se_suffix(anchor_se)}_s{seed}')


def dir_3c(base: str, pa: str, beta: str, se: str, seed: int,
           phase_prefix: str = DEFAULT_PHASE_PREFIX) -> str:
    # 3c already sweeps SE; the dir always encodes se, no anchor suffix.
    return os.path.join(base,
                         f'phase{phase_prefix}c_pa{pa}_beta{beta}_se{se}_s{seed}')


def results_path(d: str) -> str:
    return os.path.join(d, 'results.json')


# ─── result loading + aggregation ──────────────────────────────────────────
def _load_metric(path: str, metric_key: str = DEFAULT_METRIC_KEY) -> float | None:
    if not os.path.exists(path):
        return None
    try:
        return float(json.load(open(path))[metric_key])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _mean_metric(paths: Iterable[str], metric_key: str = DEFAULT_METRIC_KEY) -> float | None:
    vals = [_load_metric(p, metric_key) for p in paths]
    if any(v is None for v in vals) or not vals:
        return None
    return sum(vals) / len(vals)


# Back-compat aliases (kept so external callers continue to work).
_load_ppl = _load_metric
_mean_ppl = _mean_metric


# ─── barrier ───────────────────────────────────────────────────────────────
def expected_paths(
    phase: str,
    base: str,
    seeds: Sequence[int],
    pa_values: Sequence[str] | None = None,
    beta_values: Sequence[str] | None = None,
    se_values: Sequence[str] | None = None,
    best_pa: str | None = None,
    best_beta: str | None = None,
    anchor_se: str = '0',
    phase_prefix: str = DEFAULT_PHASE_PREFIX,
) -> List[str]:
    """Return list of results.json paths that must exist to call the phase done.

    The `phase` argument is the phase letter ('3a'/'3b'/'3c' or '5a'/'5b'/'5c').
    The leading digit is ignored — phase_prefix controls the dir naming instead,
    so both prefixes share a single dispatch. Accepted values: anything ending
    in 'a', 'b', or 'c'.
    """
    paths: List[str] = []
    letter = phase[-1] if phase else ''
    if letter == 'a':
        pas = pa_values or DEFAULT_PA_VALUES
        for pa in pas:
            for s in seeds:
                paths.append(results_path(dir_3a(base, pa, s, anchor_se, phase_prefix)))
    elif letter == 'b':
        if best_pa is None:
            raise ValueError("phase=*b requires best_pa")
        betas = beta_values or DEFAULT_BETA_VALUES
        for beta in betas:
            # β=1.0 reuses the a-phase result (at the same anchor_se)
            if beta == '1.0':
                for s in seeds:
                    paths.append(results_path(dir_3a(base, best_pa, s, anchor_se, phase_prefix)))
            else:
                for s in seeds:
                    paths.append(results_path(dir_3b(base, best_pa, beta, s, anchor_se, phase_prefix)))
    elif letter == 'c':
        if best_pa is None or best_beta is None:
            raise ValueError("phase=*c requires best_pa and best_beta")
        ses = se_values or DEFAULT_SE_VALUES
        for se in ses:
            # SE=str(anchor_se) reuses the anchor cell (a-phase if best_β=1, else b-phase).
            if se == str(anchor_se):
                if best_beta == '1.0':
                    for s in seeds:
                        paths.append(results_path(dir_3a(base, best_pa, s, anchor_se, phase_prefix)))
                else:
                    for s in seeds:
                        paths.append(results_path(dir_3b(base, best_pa, best_beta, s, anchor_se, phase_prefix)))
            else:
                for s in seeds:
                    paths.append(results_path(dir_3c(base, best_pa, best_beta, se, s, phase_prefix)))
    else:
        raise ValueError(f"unknown phase {phase!r}")
    return paths


def wait_for_phase(
    phase: str,
    base: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    pa_values: Sequence[str] | None = None,
    beta_values: Sequence[str] | None = None,
    se_values: Sequence[str] | None = None,
    best_pa: str | None = None,
    best_beta: str | None = None,
    anchor_se: str = '0',
    phase_prefix: str = DEFAULT_PHASE_PREFIX,
    poll_interval: float = 60.0,
    max_wait: float | None = None,
    stdout=sys.stdout,
) -> None:
    """Block until every expected results.json exists. Raises TimeoutError if max_wait elapsed."""
    paths = expected_paths(phase, base, seeds,
                           pa_values=pa_values, beta_values=beta_values, se_values=se_values,
                           best_pa=best_pa, best_beta=best_beta, anchor_se=anchor_se,
                           phase_prefix=phase_prefix)
    t0 = time.monotonic()
    while True:
        missing = [p for p in paths if not os.path.exists(p)]
        if not missing:
            print(f'  [{phase}] all {len(paths)} results.json present', file=stdout, flush=True)
            return
        if max_wait is not None and time.monotonic() - t0 > max_wait:
            raise TimeoutError(f'{phase} barrier timed out; missing {len(missing)}/{len(paths)}')
        print(f'  [{phase}] still {len(missing)}/{len(paths)} missing, sleeping {poll_interval:g}s',
              file=stdout, flush=True)
        time.sleep(poll_interval)


# ─── argmin/argmax helpers ─────────────────────────────────────────────────
def aggregate_pa(
    base: str, pa_values: Sequence[str], seeds: Sequence[int], anchor_se: str = '0',
    phase_prefix: str = DEFAULT_PHASE_PREFIX, metric_key: str = DEFAULT_METRIC_KEY,
) -> Dict[str, float]:
    """Return {pa: 3-seed mean metric}. Skip pa values with any missing seed."""
    rows: Dict[str, float] = {}
    for pa in pa_values:
        paths = [results_path(dir_3a(base, pa, s, anchor_se, phase_prefix)) for s in seeds]
        m = _mean_metric(paths, metric_key)
        if m is not None:
            rows[pa] = m
    return rows


def aggregate_beta(
    base: str, best_pa: str, beta_values: Sequence[str], seeds: Sequence[int],
    anchor_se: str = '0',
    phase_prefix: str = DEFAULT_PHASE_PREFIX, metric_key: str = DEFAULT_METRIC_KEY,
) -> Dict[str, float]:
    rows: Dict[str, float] = {}
    for beta in beta_values:
        if beta == '1.0':
            paths = [results_path(dir_3a(base, best_pa, s, anchor_se, phase_prefix)) for s in seeds]
        else:
            paths = [results_path(dir_3b(base, best_pa, beta, s, anchor_se, phase_prefix)) for s in seeds]
        m = _mean_metric(paths, metric_key)
        if m is not None:
            rows[beta] = m
    return rows


def aggregate_se(
    base: str, best_pa: str, best_beta: str, se_values: Sequence[str], seeds: Sequence[int],
    anchor_se: str = '0',
    phase_prefix: str = DEFAULT_PHASE_PREFIX, metric_key: str = DEFAULT_METRIC_KEY,
) -> Dict[str, float]:
    rows: Dict[str, float] = {}
    for se in se_values:
        if se == str(anchor_se):
            if best_beta == '1.0':
                paths = [results_path(dir_3a(base, best_pa, s, anchor_se, phase_prefix)) for s in seeds]
            else:
                paths = [results_path(dir_3b(base, best_pa, best_beta, s, anchor_se, phase_prefix)) for s in seeds]
        else:
            paths = [results_path(dir_3c(base, best_pa, best_beta, se, s, phase_prefix)) for s in seeds]
        m = _mean_metric(paths, metric_key)
        if m is not None:
            rows[se] = m
    return rows


def _strict_arg_smaller_wins(rows: Dict[str, float], label: str,
                              maximize: bool = False) -> str:
    """Strict arg{min,max}. On exact float tie, the smaller-valued key wins (Occam).

    maximize=False (default) → arg-min (NLP PPL).
    maximize=True → arg-max (ViT top1 accuracy).
    """
    if not rows:
        raise ValueError(f"no {label} rows provided")
    best_m = max(rows.values()) if maximize else min(rows.values())
    winners = [k for k, m in rows.items() if m == best_m]
    return min(winners, key=float)


# Back-compat alias
_strict_argmin_smaller_wins = _strict_arg_smaller_wins


def select_best_pa(rows: Dict[str, float], exclude_pa: Sequence[str] | None = None,
                   maximize: bool = False) -> str:
    """Strict arg{min,max} over pa. On exact tie, smaller pa wins (Occam).

    If `exclude_pa` is provided, those pa keys are filtered before argmin. This
    is how we drop the mechanism-off degenerate point pa=0.5 (p_a = p_i → g = 0
    → gap_c = 0 for all c and all β) from the "best method config" search
    while still reporting it as a reference baseline in tables.

    No tie-break window. Defensible: no invented threshold to justify,
    and the exclude set is a structural choice (not empirical).

    maximize=False (default) → NLP argmin-on-PPL. maximize=True → ViT argmax-on-top1.
    """
    if exclude_pa:
        rows = {pa: m for pa, m in rows.items() if pa not in set(exclude_pa)}
    return _strict_arg_smaller_wins(rows, 'pa', maximize=maximize)


def select_best_beta(rows: Dict[str, float], maximize: bool = False) -> str:
    """Strict arg{min,max}. On exact tie, smaller β wins (Occam)."""
    return _strict_arg_smaller_wins(rows, 'beta', maximize=maximize)


def select_best_se(rows: Dict[str, float], maximize: bool = False) -> str:
    """Strict arg{min,max}. On exact tie, smaller SE wins (Occam)."""
    return _strict_arg_smaller_wins(rows, 'SE', maximize=maximize)


# ─── CLI ───────────────────────────────────────────────────────────────────
def _csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def _parse_seeds(s: str) -> List[int]:
    return [int(x) for x in _csv(s)]


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--phase', required=True, choices=('3a', '3b', '3c', '5a', '5b', '5c'))
    common.add_argument('--base', required=True, help='outputs/rtx5090_nlp_mini_ablation')
    common.add_argument('--seeds', default='42,123,456')
    common.add_argument('--pa-values', default=','.join(DEFAULT_PA_VALUES))
    common.add_argument('--beta-values', default=','.join(DEFAULT_BETA_VALUES))
    common.add_argument('--se-values', default=','.join(DEFAULT_SE_VALUES))
    common.add_argument('--best-pa', default=None)
    common.add_argument('--best-beta', default=None)
    common.add_argument('--anchor-se', default='0',
                        help="SE ratio of the (pa, β) search anchor. "
                             "Default '0'; '1.0' for the fallback path.")
    common.add_argument('--exclude-pa', default='',
                        help="Comma-separated pa values to exclude from argmin "
                             "(e.g. '0.5' for the mechanism-off degenerate "
                             "point). Default: no exclusion.")
    common.add_argument('--phase-prefix', default=None,
                        help="Dir-name prefix (default derived from --phase's "
                             "leading digit: '3' or '5').")
    common.add_argument('--metric-key', default=DEFAULT_METRIC_KEY,
                        help="results.json key to aggregate. Default "
                             f"{DEFAULT_METRIC_KEY!r} (NLP); use 'best_top1' for ViT.")
    common.add_argument('--maximize', action='store_true',
                        help="Select arg-max instead of arg-min (use for "
                             "accuracy metrics like ViT top1).")

    w = sub.add_parser('wait', parents=[common])
    w.add_argument('--poll', type=float, default=60.0)
    w.add_argument('--max-wait', type=float, default=None, help='seconds; default unlimited')

    _ = sub.add_parser('best', parents=[common])

    args = p.parse_args(argv)
    seeds = _parse_seeds(args.seeds)
    phase_prefix = args.phase_prefix or args.phase[0]  # '3' or '5'
    letter = args.phase[-1]

    if args.cmd == 'wait':
        wait_for_phase(
            args.phase, args.base, seeds=seeds,
            pa_values=_csv(args.pa_values),
            beta_values=_csv(args.beta_values),
            se_values=_csv(args.se_values),
            best_pa=args.best_pa, best_beta=args.best_beta,
            anchor_se=args.anchor_se,
            phase_prefix=phase_prefix,
            poll_interval=args.poll, max_wait=args.max_wait,
        )
        return 0

    if args.cmd == 'best':
        if letter == 'a':
            rows = aggregate_pa(args.base, _csv(args.pa_values), seeds,
                                anchor_se=args.anchor_se,
                                phase_prefix=phase_prefix, metric_key=args.metric_key)
            if not rows:
                print('ERROR: no complete pa rows found', file=sys.stderr)
                return 1
            exclude = _csv(args.exclude_pa) if args.exclude_pa else None
            out = select_best_pa(rows, exclude_pa=exclude, maximize=args.maximize)
        elif letter == 'b':
            if args.best_pa is None:
                print('ERROR: --best-pa required for phase *b', file=sys.stderr)
                return 1
            rows = aggregate_beta(args.base, args.best_pa, _csv(args.beta_values), seeds,
                                  anchor_se=args.anchor_se,
                                  phase_prefix=phase_prefix, metric_key=args.metric_key)
            if not rows:
                print('ERROR: no complete beta rows found', file=sys.stderr)
                return 1
            out = select_best_beta(rows, maximize=args.maximize)
        elif letter == 'c':
            if args.best_pa is None or args.best_beta is None:
                print('ERROR: --best-pa and --best-beta required for phase *c', file=sys.stderr)
                return 1
            rows = aggregate_se(args.base, args.best_pa, args.best_beta,
                                _csv(args.se_values), seeds, anchor_se=args.anchor_se,
                                phase_prefix=phase_prefix, metric_key=args.metric_key)
            if not rows:
                print('ERROR: no complete SE rows found', file=sys.stderr)
                return 1
            out = select_best_se(rows, maximize=args.maximize)
        else:
            raise AssertionError('unreachable')
        print(out)
        return 0

    return 1


if __name__ == '__main__':
    raise SystemExit(main())

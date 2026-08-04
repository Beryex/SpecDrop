#!/usr/bin/env python3
"""Pick an optimal k for k-means clustering via silhouette score.

Silhouette score formalizes "compact clusters + well-separated from each
other" exactly as intuitively described:
    s(i) = (b(i) − a(i)) / max(a(i), b(i))
where a(i) is the mean distance from point i to other points in its own
cluster, and b(i) is the mean distance from i to points in the nearest
other cluster. Mean over all points ∈ [−1, 1]; higher is better.

Uses a random subsample (default 5000 points) for tractability — exact
silhouette on 195K × 1024 points is O(N²) and impractical. Subsampled
silhouette is stable to ±0.001 at this sample size.

Inputs:
    --emb-cache   pt file from data.cluster_chunks with 'embeddings' key
    --k-values    space-separated ints (default: 5 7 10 14)
    --sample-size sample for silhouette (default 5000)
    --seed        random seed for KMeans + subsample (default 42)
    --output      optional file to write the chosen k to (single int line)

Output to stdout (and optionally a file): table of (k, silhouette, inertia)
and the argmax-silhouette k. The chosen k is also echoed to the output file
(if --output given) for downstream scripts.

CPU-only, no GPU required. Runtime: ~10s × len(k-values) on 195K embeddings.

Usage:
    python scripts/select_optimal_k.py \\
        --emb-cache data_cache/slimpajama/embeddings_train_seq512_tok100000000_bge-large-en-v1.5.pt \\
        --k-values 5 7 10 14 \\
        --output outputs/analysis/optimal_k.txt
"""
import os

# Cap BLAS thread counts BEFORE numpy / sklearn imports — OpenBLAS reads these
# at load time. Hosts with > 128 CPU cores otherwise segfault in sklearn
# KMeans via joblib × per-thread OpenBLAS hitting the 128-thread precompiled
# ceiling. setdefault respects an override from the caller's environment.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '32')
os.environ.setdefault('OMP_NUM_THREADS', '32')
os.environ.setdefault('MKL_NUM_THREADS', '32')

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_embeddings(emb_cache_path):
    print(f'[select-k] loading embeddings from {emb_cache_path}...', flush=True)
    blob = torch.load(emb_cache_path, weights_only=False)
    emb = blob['embeddings']
    if torch.is_tensor(emb):
        emb = emb.numpy()
    emb = np.asarray(emb, dtype=np.float32)
    print(f'[select-k] loaded {emb.shape[0]:,} × {emb.shape[1]}-dim '
          f'(embedder={blob.get("embedder")})', flush=True)
    return emb


def _compute_metrics(embeddings, labels, sample_size, seed):
    """Compute silhouette (subsampled), Calinski-Harabasz, Davies-Bouldin.

    Three independent internal cluster-validity metrics for cross-checking:
      - silhouette:         higher better, ∈ [−1, 1]
                             (compactness vs nearest-cluster separation)
      - calinski_harabasz:  higher better, ∈ [0, ∞)
                             (between-cluster variance / within-cluster variance)
      - davies_bouldin:     LOWER better, ∈ [0, ∞)
                             (mean per-cluster similarity to its nearest other cluster)

    CH and DB use the full data; silhouette is subsampled since it's O(N²).
    """
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                   davies_bouldin_score)
    sil = silhouette_score(embeddings, labels,
                            sample_size=sample_size, random_state=seed)
    ch = calinski_harabasz_score(embeddings, labels)
    db = davies_bouldin_score(embeddings, labels)
    return float(sil), float(ch), float(db)


def scan_k(embeddings, k_values, sample_size=5000, seed=42, n_init=10,
            mini_batch=False, mb_batch_size=1024, mb_max_iter=100,
            verbose=True):
    """Scan candidate k values and return per-k metrics (silhouette, CH, DB,
    inertia).

    mini_batch=True switches to MiniBatchKMeans, which is ~10-100x faster on
    large N (recommended when |k_values| > 10 or N > 50K). Slightly less
    tight than full KMeans but stable enough for metric-based selection.
    """
    if mini_batch:
        from sklearn.cluster import MiniBatchKMeans
        # sklearn warns on deprecated `n_init='auto'` transition; pin to explicit small n_init.
        mb_n_init = max(1, min(n_init, 3))
        def _fit(k):
            return MiniBatchKMeans(
                n_clusters=k, random_state=seed, n_init=mb_n_init,
                batch_size=mb_batch_size, max_iter=mb_max_iter,
                reassignment_ratio=0.01)
        fit_tag = f'MiniBatchKMeans(bs={mb_batch_size}, n_init={mb_n_init})'
    else:
        from sklearn.cluster import KMeans
        def _fit(k):
            return KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        fit_tag = f'KMeans(n_init={n_init})'

    print(f'[select-k] using {fit_tag}', flush=True)

    results = []
    for i, k in enumerate(k_values):
        t0 = time.time()
        km = _fit(k)
        labels = km.fit_predict(embeddings)
        fit_t = time.time() - t0

        t1 = time.time()
        sil, ch, db = _compute_metrics(embeddings, labels, sample_size, seed)
        metric_t = time.time() - t1

        sizes = np.bincount(labels, minlength=k)
        results.append({
            'k': int(k),
            'silhouette': sil,
            'calinski_harabasz': ch,
            'davies_bouldin': db,
            'inertia': float(km.inertia_),
            'fit_seconds': float(fit_t),
            'metric_seconds': float(metric_t),
            'sizes': sizes.tolist(),
        })
        if verbose:
            print(f'  [{i+1:>3}/{len(k_values)}] k={k:>3}  '
                  f'sil={sil:+.4f}  CH={ch:>8.1f}  DB={db:.4f}  '
                  f'inertia={km.inertia_:.3e}  '
                  f'fit={fit_t:>5.1f}s  metrics={metric_t:>4.1f}s',
                  flush=True)
    return results


def format_report(results, sample_size):
    """Print a 3-metric cross-check table + consensus top picks."""
    lines = [
        '',
        '=' * 88,
        f' Optimal-k scan — 3-metric cross-check (silhouette sample={sample_size})',
        '=' * 88,
        f"{'k':>4}  {'silhouette':>11}  {'CH':>10}  {'DB':>8}  {'inertia':>10}  "
        f"{'fit (s)':>8}",
        '-' * 88,
    ]
    for r in results:
        lines.append(
            f"{r['k']:>4}  {r['silhouette']:>+11.4f}  "
            f"{r['calinski_harabasz']:>10.1f}  "
            f"{r['davies_bouldin']:>8.4f}  "
            f"{r['inertia']:>10.3e}  "
            f"{r['fit_seconds']:>8.1f}")

    # Per-metric argmax (with appropriate direction).
    best_sil = max(results, key=lambda r: r['silhouette'])
    best_ch = max(results, key=lambda r: r['calinski_harabasz'])
    best_db = min(results, key=lambda r: r['davies_bouldin'])  # LOWER is better

    lines.append('')
    lines.append('Per-metric argmax:')
    lines.append(f"  silhouette  (higher better): BEST k = {best_sil['k']:>3}  "
                  f"(score = {best_sil['silhouette']:+.4f})")
    lines.append(f"  CH          (higher better): BEST k = {best_ch['k']:>3}  "
                  f"(score = {best_ch['calinski_harabasz']:.1f})")
    lines.append(f"  DB          (lower  better): BEST k = {best_db['k']:>3}  "
                  f"(score = {best_db['davies_bouldin']:.4f})")

    # Consensus check
    picks = {best_sil['k'], best_ch['k'], best_db['k']}
    if len(picks) == 1:
        k_consensus = picks.pop()
        lines.append(f"\n✓ CONSENSUS k = {k_consensus} (all 3 metrics agree)")
    elif len(picks) == 2:
        lines.append(f"\n△ PARTIAL consensus: picks = {sorted(picks)} "
                      f"(2 out of 3 metrics agree)")
    else:
        lines.append(f"\n✗ NO consensus: silhouette→{best_sil['k']}, "
                      f"CH→{best_ch['k']}, DB→{best_db['k']}")

    # Top 5 per metric for borderline cases
    top_n = 5
    sil_top = sorted(results, key=lambda r: r['silhouette'], reverse=True)[:top_n]
    ch_top = sorted(results, key=lambda r: r['calinski_harabasz'], reverse=True)[:top_n]
    db_top = sorted(results, key=lambda r: r['davies_bouldin'])[:top_n]
    lines.append(f"\nTop {top_n} per metric:")
    lines.append(f"  silhouette: {[r['k'] for r in sil_top]}")
    lines.append(f"  CH:         {[r['k'] for r in ch_top]}")
    lines.append(f"  DB:         {[r['k'] for r in db_top]}")
    lines.append('')

    return '\n'.join(lines), best_sil['k']  # keep silhouette's pick as the
                                             # "best k" written to --output
                                             # (backward compatibility)


def main():
    p = argparse.ArgumentParser(
        description='Pick optimal k for KMeans via silhouette score')
    p.add_argument('--emb-cache', required=True,
                    help='Sentence-embedding cache (.pt from cluster_chunks)')
    p.add_argument('--k-values', type=int, nargs='+', default=[5, 7, 10, 14],
                    help='Explicit candidate k values to scan. Ignored if '
                         '--k-min/--k-max are set. (default: 5 7 10 14)')
    p.add_argument('--k-min', type=int, default=None,
                    help='Range scan minimum k (requires --k-max)')
    p.add_argument('--k-max', type=int, default=None,
                    help='Range scan maximum k, inclusive')
    p.add_argument('--k-step', type=int, default=1,
                    help='Range scan step (default 1)')
    p.add_argument('--mini-batch', action='store_true',
                    help='Use MiniBatchKMeans (10-100x faster for large N; '
                         'strongly recommended when scanning >10 k values)')
    p.add_argument('--sample-size', type=int, default=5000,
                    help='Subsample size for silhouette (default 5000)')
    p.add_argument('--seed', type=int, default=42,
                    help='Random seed for KMeans + subsample (default 42)')
    p.add_argument('--n-init', type=int, default=10,
                    help='KMeans n_init (default 10; MiniBatch uses min(n_init,3))')
    p.add_argument('--output', default=None,
                    help='Write best-k (single int line) to this file. '
                         'Downstream scripts can `$(cat FILE)` to read.')
    p.add_argument('--full-output', default=None,
                    help='Optional: write full results table to this file')
    args = p.parse_args()

    if not os.path.exists(args.emb_cache):
        print(f'error: embedding cache not found: {args.emb_cache}',
              file=sys.stderr)
        sys.exit(1)

    emb = load_embeddings(args.emb_cache)

    # Range form overrides explicit --k-values when both set.
    if args.k_min is not None and args.k_max is not None:
        if args.k_min < 2:
            print(f'error: --k-min must be ≥ 2 (silhouette undefined for k=1)',
                  file=sys.stderr); sys.exit(1)
        if args.k_max < args.k_min:
            print(f'error: --k-max must be ≥ --k-min', file=sys.stderr); sys.exit(1)
        k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
        print(f'[select-k] range scan k ∈ [{args.k_min}, {args.k_max}] '
              f'step {args.k_step} → {len(k_values)} values', flush=True)
    else:
        k_values = list(args.k_values)
        print(f'[select-k] explicit scan: {k_values}', flush=True)

    results = scan_k(emb, k_values, sample_size=args.sample_size,
                      seed=args.seed, n_init=args.n_init,
                      mini_batch=args.mini_batch)

    report, best_k = format_report(results, args.sample_size)
    print(report)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        # Single int line for shell-friendliness: `K=$(cat file)`.
        with open(args.output, 'w') as f:
            f.write(f'{best_k}\n')
        print(f'[select-k] wrote best_k={best_k} to {args.output}', flush=True)

    if args.full_output:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.full_output)),
                    exist_ok=True)
        with open(args.full_output, 'w') as f:
            json.dump({
                'results': results,
                'best_k': best_k,
                'sample_size': args.sample_size,
                'seed': args.seed,
            }, f, indent=2)
        print(f'[select-k] wrote full results to {args.full_output}', flush=True)


if __name__ == '__main__':
    main()

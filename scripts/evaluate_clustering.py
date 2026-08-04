#!/usr/bin/env python3
"""Offline diagnostic for Phase W cluster-cache quality.

Runs four analyses on a cluster cache produced by `data.cluster_chunks`:

  (A) Cluster × Domain cross-tab (row-normalized).
      For each cluster, the % of its chunks that came from each SlimPajama
      source domain. Pure clusters (one row ≈ 100% in one column) mean the
      embedder recovered the original domain boundaries. Mixed rows mean
      the embedder found a semantic axis that cuts across domains (CC is
      typically spread across multiple clusters).

  (B) Domain × Cluster cross-tab (col-normalized → row-normalized after
      transpose). For each source domain, the % of its chunks that landed
      in each cluster. Cohesive domains (one row dominant) stayed together;
      fragmented domains (one row uniform) got split across clusters.

  (C) Information-theoretic metrics:
      - V-measure: harmonic mean of homogeneity + completeness (0–1, higher=
        more aligned with original domains).
      - ARI (Adjusted Rand): 0 = random, 1 = perfect match.
      - AMI (Adjusted Mutual Information): chance-corrected mutual info.
      - H(domain | cluster): avg bits of domain-uncertainty left after
        knowing cluster. Lower = cluster predicts domain well.
      - H(cluster | domain): avg bits of cluster-uncertainty left after
        knowing domain. Lower = domain predicts cluster well.

  (D) Per-cluster preview — a few sample texts (truncated) plus the top
      word-frequency tokens, for human eyeball interpretation of what each
      cluster "is about."

CPU-only, no GPU required. Safe to run in parallel with Phase W training
on the same machine (light I/O, ~30s–1min wall-clock for 100M-token cache).

Usage:
    python scripts/evaluate_clustering.py \\
        --cluster-cache data_cache/slimpajama/clusters_train_seq512_tok100000000_bge-large_k7.pt

Auto-derives:
    - token cache:  from the cluster cache's `source_cache` field (same dir)
    - text cache:   via data.cluster_chunks._derive_text_cache_path
Override either with --token-cache / --text-cache.

Output goes to stdout by default; use --output FILE to write a Markdown report.
"""
import argparse
import os
import re
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.slimpajama import DOMAIN_NAMES, NUM_DOMAINS
from data.cluster_chunks import _derive_text_cache_path


# Short-name mapping so tables stay narrow.
DOMAIN_SHORT = {
    'RedPajamaCommonCrawl':   'CC',
    'RedPajamaC4':            'C4',
    'RedPajamaGithub':        'Github',
    'RedPajamaBook':          'Book',
    'RedPajamaArXiv':         'ArXiv',
    'RedPajamaWikipedia':     'Wiki',
    'RedPajamaStackExchange': 'SE',
}
DOMAIN_SHORT_LIST = [DOMAIN_SHORT[n] for n in DOMAIN_NAMES]

# Very light word regex — adequate for English tokens with apostrophes.
_WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")

# Stopwords to skip in top-token signatures (keeps signatures content-bearing).
_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what
when where which while who whom why will with would you your yours yourself
yourselves s t d ll m o re ve also one two three can may like would get got
go going make made use used new
""".split())


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _derive_token_cache_path(cluster_cache_path, cluster_blob):
    """Derive the source token cache path from the cluster cache metadata."""
    base_dir = os.path.dirname(cluster_cache_path)
    src = cluster_blob.get('source_cache')
    if not src:
        raise ValueError(
            f"{cluster_cache_path} has no 'source_cache' metadata; specify "
            f"--token-cache explicitly.")
    return os.path.join(base_dir, src)


def _load_all(cluster_cache, token_cache, text_cache):
    print(f'[eval] cluster cache: {cluster_cache}', flush=True)
    cblob = torch.load(cluster_cache, weights_only=False)
    cluster_ids = cblob['cluster_ids']
    if torch.is_tensor(cluster_ids):
        cluster_ids = cluster_ids.numpy()
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)

    if token_cache is None:
        token_cache = _derive_token_cache_path(cluster_cache, cblob)
    if not os.path.exists(token_cache):
        raise FileNotFoundError(f'token cache not found: {token_cache}')
    print(f'[eval] token cache:   {token_cache}', flush=True)
    tblob = torch.load(token_cache, weights_only=False)
    domain_ids = tblob['domain_ids']
    if torch.is_tensor(domain_ids):
        domain_ids = domain_ids.numpy()
    domain_ids = np.asarray(domain_ids, dtype=np.int64)

    assert len(domain_ids) == len(cluster_ids), \
        f'length mismatch: domain={len(domain_ids)} cluster={len(cluster_ids)}'

    # Text cache is optional — only needed for per-cluster previews.
    texts = None
    if text_cache is None:
        text_cache = _derive_text_cache_path(token_cache)
    if text_cache and os.path.exists(text_cache):
        print(f'[eval] text cache:    {text_cache}', flush=True)
        text_blob = torch.load(text_cache, weights_only=False)
        texts = text_blob.get('texts')
        if texts is not None and len(texts) != len(cluster_ids):
            print(f'[eval] warn: text cache length {len(texts)} != cluster '
                  f'length {len(cluster_ids)}; skipping preview section')
            texts = None
    else:
        print(f'[eval] text cache:    NOT FOUND at {text_cache} — preview skipped',
              flush=True)

    meta = {
        'embedder': cblob.get('embedder', 'unknown'),
        'seed': cblob.get('seed'),
        'n_clusters': int(cblob['n_clusters']),
        'num_chunks': int(cblob.get('num_chunks', len(cluster_ids))),
    }
    return cluster_ids, domain_ids, texts, meta


# ── Analysis ─────────────────────────────────────────────────────────────────

def cross_tab(cluster_ids, domain_ids, n_clusters, n_domains=NUM_DOMAINS):
    """Raw counts, shape (C, D). ct[c, d] = # chunks in cluster c from domain d."""
    ct = np.zeros((n_clusters, n_domains), dtype=np.int64)
    np.add.at(ct, (cluster_ids, domain_ids), 1)
    return ct


def info_metrics(cluster_ids, domain_ids):
    from sklearn.metrics import (v_measure_score, adjusted_rand_score,
                                  adjusted_mutual_info_score,
                                  homogeneity_score, completeness_score)
    return {
        'v_measure':     v_measure_score(domain_ids, cluster_ids),
        'homogeneity':   homogeneity_score(domain_ids, cluster_ids),
        'completeness':  completeness_score(domain_ids, cluster_ids),
        'ari':           adjusted_rand_score(domain_ids, cluster_ids),
        'ami':           adjusted_mutual_info_score(domain_ids, cluster_ids),
    }


def conditional_entropy(joint_ct, axis=1):
    """H(X|Y) where axis specifies the conditioning variable.

    If axis=1 (sum over columns before conditioning → conditioning on rows),
    returns H(columns | rows). Implementation: H(X,Y) - H(Y).
    """
    total = joint_ct.sum()
    p = joint_ct / total
    # H(X, Y)
    nz = p[p > 0]
    h_xy = -np.sum(nz * np.log2(nz))
    # H(conditioner)
    if axis == 1:  # H(cols | rows)
        p_cond = p.sum(axis=1)
    else:  # H(rows | cols)
        p_cond = p.sum(axis=0)
    nz_c = p_cond[p_cond > 0]
    h_cond = -np.sum(nz_c * np.log2(nz_c))
    return h_xy - h_cond


def top_tokens_per_cluster(texts, cluster_ids, n_clusters, top_k=15,
                            sample_cap=500):
    """Simple word-frequency signature per cluster. Capped per cluster for
    speed (we don't need exact counts; top 15 stable at 500-text sample).
    """
    import random
    rng = random.Random(42)
    by_cluster = [[] for _ in range(n_clusters)]
    for i, c in enumerate(cluster_ids):
        by_cluster[int(c)].append(i)

    out = {}
    for c in range(n_clusters):
        idxs = by_cluster[c]
        if len(idxs) > sample_cap:
            idxs = rng.sample(idxs, sample_cap)
        cnt = Counter()
        for i in idxs:
            for tok in _WORD_RE.findall(texts[i].lower()):
                if len(tok) < 3 or tok in _STOPWORDS:
                    continue
                cnt[tok] += 1
        out[c] = cnt.most_common(top_k)
    return out


def sample_texts_per_cluster(texts, cluster_ids, n_clusters, n_samples=5,
                              trunc=200):
    import random
    rng = random.Random(42)
    by_cluster = [[] for _ in range(n_clusters)]
    for i, c in enumerate(cluster_ids):
        by_cluster[int(c)].append(i)

    out = {}
    for c in range(n_clusters):
        idxs = by_cluster[c]
        if len(idxs) > n_samples:
            idxs = rng.sample(idxs, n_samples)
        out[c] = [texts[i][:trunc].replace('\n', ' ') for i in idxs]
    return out


# ── Reporting ────────────────────────────────────────────────────────────────

def _fmt_cluster_x_domain(ct):
    """Row-normalized: each row sums to 100%. 'of cluster c, x% came from domain d'."""
    C, D = ct.shape
    total = ct.sum()
    row_tot = ct.sum(axis=1)
    col_tot = ct.sum(axis=0)
    row_pct = ct / np.maximum(row_tot[:, None], 1) * 100

    lines = []
    header = '| Cluster | Size | % of total | ' + ' | '.join(DOMAIN_SHORT_LIST[:D]) + ' |'
    sep = '|' + '---|' * (D + 3)
    lines.append(header); lines.append(sep)
    for c in range(C):
        size = row_tot[c]
        size_pct = size / total * 100
        row = ' | '.join(f'{row_pct[c, d]:>5.1f}%' for d in range(D))
        lines.append(f'| C{c} | {size:,} | {size_pct:>5.2f}% | {row} |')

    # Reference row: overall domain distribution
    domain_pct = col_tot / total * 100
    ref = ' | '.join(f'{domain_pct[d]:>5.1f}%' for d in range(D))
    lines.append(f'| *(overall)* | {total:,} | 100.00% | {ref} |')
    return '\n'.join(lines)


def _fmt_domain_x_cluster(ct):
    """Column-normalized: each column sums to 100%. 'of domain d, x% went to cluster c'."""
    C, D = ct.shape
    total = ct.sum()
    col_tot = ct.sum(axis=0)
    row_tot = ct.sum(axis=1)
    col_pct = ct / np.maximum(col_tot[None, :], 1) * 100

    lines = []
    header = '| Cluster | ' + ' | '.join(DOMAIN_SHORT_LIST[:D]) + ' |'
    sep = '|' + '---|' * (D + 1)
    lines.append(header); lines.append(sep)
    for c in range(C):
        row = ' | '.join(f'{col_pct[c, d]:>5.1f}%' for d in range(D))
        lines.append(f'| C{c} ({row_tot[c]:,}) | {row} |')
    # Reference footer: domain total sizes
    domain_cnts = ' | '.join(f'{int(col_tot[d]):>6,}' for d in range(D))
    lines.append(f'| *(total)* | {domain_cnts} |')
    return '\n'.join(lines)


def _fmt_info_metrics(m, h_dom_given_clu, h_clu_given_dom):
    lines = ['| Metric | Value | Interpretation |',
             '|---|---|---|',
             f'| V-measure | {m["v_measure"]:.4f} | 0=random, 1=perfect domain recovery |',
             f'| Homogeneity | {m["homogeneity"]:.4f} | Each cluster contains one domain |',
             f'| Completeness | {m["completeness"]:.4f} | Each domain fits one cluster |',
             f'| Adjusted Rand | {m["ari"]:.4f} | Chance-corrected agreement |',
             f'| Adjusted MI  | {m["ami"]:.4f} | Chance-corrected mutual info |',
             f'| H(domain | cluster) | {h_dom_given_clu:.4f} bits | Uncertainty about domain given cluster |',
             f'| H(cluster | domain) | {h_clu_given_dom:.4f} bits | Uncertainty about cluster given domain |']
    max_h_dom = np.log2(NUM_DOMAINS)
    lines.append(
        f'\n*(H(domain) ≤ log₂({NUM_DOMAINS}) = {max_h_dom:.3f} bits is the upper bound; '
        f'lower H(domain|cluster) = cluster predicts domain better.)*')
    return '\n'.join(lines)


def _fmt_preview(texts_per_cluster, tokens_per_cluster, n_samples_shown=3):
    lines = []
    for c in sorted(tokens_per_cluster.keys()):
        lines.append(f'\n**Cluster C{c}** — top tokens: '
                     + ', '.join(f'`{t}`({n})' for t, n in tokens_per_cluster[c]))
        if c in texts_per_cluster:
            for i, txt in enumerate(texts_per_cluster[c][:n_samples_shown]):
                lines.append(f'  - `[{i}]` {txt.strip()}')
    return '\n'.join(lines)


def build_report(cluster_cache, cluster_ids, domain_ids, texts, meta,
                  n_samples=3, top_k_tokens=15):
    n_clusters = meta['n_clusters']
    ct = cross_tab(cluster_ids, domain_ids, n_clusters)
    info = info_metrics(cluster_ids, domain_ids)
    # H(domain | cluster): conditioning on cluster (rows).
    h_dom_given_clu = conditional_entropy(ct, axis=1)
    # H(cluster | domain): conditioning on domain (cols).
    h_clu_given_dom = conditional_entropy(ct, axis=0)

    lines = [
        f'# Clustering Quality Report',
        f'',
        f'**Cluster cache**: `{os.path.basename(cluster_cache)}`  ',
        f'**Embedder**: `{meta["embedder"]}`  ',
        f'**Seed**: `{meta["seed"]}`  ',
        f'**Total chunks**: {meta["num_chunks"]:,}  ',
        f'**Clusters**: {n_clusters}  ',
        f'**Source domains**: {NUM_DOMAINS} ({", ".join(DOMAIN_SHORT_LIST)})',
        f'',
        f'## (A) Cluster × Domain cross-tab (row-normalized)',
        f'',
        f'Each row sums to 100%. *"Of all chunks assigned to cluster C, x% came from domain D."*',
        f'',
        _fmt_cluster_x_domain(ct),
        f'',
        f'## (B) Domain × Cluster cross-tab (column-normalized)',
        f'',
        f'Each column sums to 100%. *"Of all chunks in domain D, x% landed in cluster C."*',
        f'',
        _fmt_domain_x_cluster(ct),
        f'',
        f'## (C) Information-theoretic metrics',
        f'',
        _fmt_info_metrics(info, h_dom_given_clu, h_clu_given_dom),
    ]

    if texts is not None:
        print('[eval] building per-cluster preview (sampling texts + top tokens)...',
              flush=True)
        tokens_per_cluster = top_tokens_per_cluster(
            texts, cluster_ids, n_clusters, top_k=top_k_tokens)
        texts_per_cluster = sample_texts_per_cluster(
            texts, cluster_ids, n_clusters, n_samples=n_samples)
        lines.extend([
            f'',
            f'## (D) Per-cluster preview',
            f'',
            _fmt_preview(texts_per_cluster, tokens_per_cluster,
                          n_samples_shown=n_samples),
        ])

    return '\n'.join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--cluster-cache', required=True)
    p.add_argument('--token-cache', default=None,
                    help='Override; default derives from cluster cache metadata')
    p.add_argument('--text-cache', default=None,
                    help='Override; default derives from token cache filename. '
                         'Missing is OK — preview section is skipped.')
    p.add_argument('--output', default=None,
                    help='Write Markdown report to this file (default: stdout)')
    p.add_argument('--samples', type=int, default=3,
                    help='Per-cluster sample texts to include (default 3)')
    p.add_argument('--top-tokens', type=int, default=15,
                    help='Per-cluster top-token signature length (default 15)')
    args = p.parse_args()

    if not os.path.exists(args.cluster_cache):
        print(f'error: cluster cache not found: {args.cluster_cache}', file=sys.stderr)
        sys.exit(1)

    cluster_ids, domain_ids, texts, meta = _load_all(
        args.cluster_cache, args.token_cache, args.text_cache)

    report = build_report(args.cluster_cache, cluster_ids, domain_ids, texts, meta,
                           n_samples=args.samples, top_k_tokens=args.top_tokens)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(report + '\n')
        print(f'\n[eval] report written to {args.output}', flush=True)
    else:
        print('\n' + report)


if __name__ == '__main__':
    main()

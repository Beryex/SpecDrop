#!/usr/bin/env python3
"""Measure within-chunk semantic heterogeneity via sub-window embeddings.

For a random sample of 512-token chunks, re-slice into fixed-size sub-windows
(default 64 tokens × 8 windows = one chunk), embed each sub-window with the
same BGE-large model used for the chunk-level clustering, assign each
sub-window to its nearest KMeans cluster centroid, and report:

  - distribution of "number of distinct clusters per chunk" (1 = homogeneous,
    2+ = heterogeneous)
  - mean within-chunk entropy over cluster assignments
  - mean "dominant-cluster fraction" (what % of a chunk's sub-windows fall
    into its most common cluster)

This is quantitative evidence for the hypothesis that **512-token chunks span
multiple semantic topics internally** — if a substantial fraction of chunks
show ≥ 2 distinct sub-window clusters, chunk-level routing is structurally
too coarse for the data's natural semantic granularity.

Inputs (all present on 5090 from Phase W pipeline):
    --token-cache       data_cache/slimpajama/tokenized_train_seq512_tok100000000_vocab<H>.pt
    --cluster-cache     data_cache/slimpajama/clusters_train_seq512_tok100000000_bge-large_k7.pt
    --embedding-cache   data_cache/slimpajama/embeddings_train_seq512_tok100000000_bge-large-en-v1.5.pt

Output:
    JSON + stdout table with per-chunk cluster-count histogram, entropy, etc.

Usage:
    python scripts/intra_chunk_heterogeneity.py \\
        --token-cache data_cache/slimpajama/tokenized_train_seq512_tok100000000_vocab2e62aacd7f.pt \\
        --cluster-cache data_cache/slimpajama/clusters_train_seq512_tok100000000_bge-large_k7.pt \\
        --embedding-cache data_cache/slimpajama/embeddings_train_seq512_tok100000000_bge-large-en-v1.5.pt \\
        --embedder BAAI/bge-large-en-v1.5 \\
        --n-samples 300 --sub-window 64 \\
        --device cuda \\
        --output outputs/analysis/intra_chunk_heterogeneity.json
"""
import os

# Cap BLAS thread counts BEFORE numpy / sklearn imports (prevents a segfault
# on >128-thread hosts — same root cause as select_optimal_k.py fix).
os.environ.setdefault('OPENBLAS_NUM_THREADS', '32')
os.environ.setdefault('OMP_NUM_THREADS', '32')
os.environ.setdefault('MKL_NUM_THREADS', '32')

import argparse
import json
import sys
import time
from collections import Counter

import numpy as np
import torch


def _log2(x, eps=1e-12):
    return np.log(np.maximum(x, eps)) / np.log(2.0)


def compute_centroids(embeddings, cluster_ids, n_clusters):
    """Per-cluster mean = KMeans centroid (at convergence, mean of members)."""
    C = np.zeros((n_clusters, embeddings.shape[1]), dtype=np.float32)
    for c in range(n_clusters):
        mask = cluster_ids == c
        if mask.sum() == 0:
            continue
        C[c] = embeddings[mask].mean(axis=0)
    return C


def nearest_centroid(sub_embeddings, centroids):
    """L2-nearest centroid assignment. Returns (M,) cluster IDs."""
    # (M, D) - (K, D) → broadcast (M, 1, D) - (1, K, D) → (M, K)
    d = ((sub_embeddings[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return np.argmin(d, axis=1)


def detokenize_sub_windows(token_ids_row, sub_size, tokenizer):
    """Split one (512,) token row into contiguous sub-windows, decode each."""
    L = len(token_ids_row)
    n_sub = L // sub_size
    sub_tokens = [token_ids_row[i * sub_size:(i + 1) * sub_size].tolist()
                   for i in range(n_sub)]
    return tokenizer.batch_decode(sub_tokens, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--token-cache', required=True)
    ap.add_argument('--cluster-cache', required=True)
    ap.add_argument('--embedding-cache', required=True,
                     help='The (N, 1024) BGE-large embedding cache used to fit '
                          'the k=7 KMeans. Used to reconstruct centroids.')
    ap.add_argument('--embedder', default='BAAI/bge-large-en-v1.5',
                     help='Sentence-transformer model to embed sub-windows')
    ap.add_argument('--n-samples', type=int, default=300,
                     help='How many chunks to sample from the train cache')
    ap.add_argument('--sub-window', type=int, default=64,
                     help='Sub-window length in tokens (chunk=512 default → '
                          '8 sub-windows per chunk)')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', default='cuda', help="'cuda' | 'cpu' | 'mps'")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tokenizer', default='gpt2')
    ap.add_argument('--output', default='outputs/analysis/intra_chunk_heterogeneity.json')
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    # ── 1. Load token cache + sample chunks ──────────────────────────────────
    print(f'[hetero] loading token cache {args.token_cache} ...', flush=True)
    tblob = torch.load(args.token_cache, weights_only=False)
    input_ids = tblob['input_ids']
    if torch.is_tensor(input_ids):
        input_ids = input_ids.cpu().numpy()
    else:
        input_ids = np.asarray(input_ids)
    N_total, L = input_ids.shape
    print(f'[hetero] token cache {N_total:,} chunks × {L} tokens', flush=True)
    assert L % args.sub_window == 0, \
        f'chunk length {L} not divisible by sub_window {args.sub_window}'
    n_sub_per_chunk = L // args.sub_window

    # Sample chunk indices
    sample_idxs = rng.choice(N_total, size=min(args.n_samples, N_total),
                              replace=False)
    sample_idxs.sort()
    print(f'[hetero] sampling {len(sample_idxs)} chunks, {n_sub_per_chunk} '
          f'sub-windows of {args.sub_window} tokens each = '
          f'{len(sample_idxs) * n_sub_per_chunk:,} sub-embeddings to compute',
          flush=True)

    # ── 2. Load embedding cache + cluster labels, compute centroids ──────────
    print(f'[hetero] loading embedding cache {args.embedding_cache} ...', flush=True)
    eblob = torch.load(args.embedding_cache, weights_only=False)
    emb = eblob['embeddings']
    if torch.is_tensor(emb):
        emb = emb.cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32)

    print(f'[hetero] loading cluster cache {args.cluster_cache} ...', flush=True)
    cblob = torch.load(args.cluster_cache, weights_only=False)
    cluster_ids = cblob['cluster_ids']
    if torch.is_tensor(cluster_ids):
        cluster_ids = cluster_ids.cpu().numpy()
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    K = int(cblob['n_clusters'])

    assert len(cluster_ids) == emb.shape[0] == N_total, \
        'chunk-count mismatch across caches'

    print(f'[hetero] computing {K} cluster centroids from embedding+label caches',
          flush=True)
    centroids = compute_centroids(emb, cluster_ids, K)

    # Freeing embedding cache early since we only need centroids
    del emb

    # ── 3. Detokenize all sub-windows for sampled chunks ─────────────────────
    print(f'[hetero] loading GPT-2 tokenizer ({args.tokenizer})', flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f'[hetero] detokenizing sub-windows...', flush=True)
    t0 = time.time()
    all_texts = []  # flat list of texts, ordered (chunk0_sub0, chunk0_sub1, …, chunk1_sub0, ...)
    for chunk_idx in sample_idxs:
        sub_texts = detokenize_sub_windows(
            input_ids[chunk_idx], args.sub_window, tok)
        all_texts.extend(sub_texts)
    print(f'[hetero] detokenize done in {time.time()-t0:.1f}s; '
          f'{len(all_texts):,} sub-window texts', flush=True)

    # ── 4. Embed sub-windows with BGE-large ──────────────────────────────────
    print(f'[hetero] loading {args.embedder} ...', flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embedder, device=args.device)

    print(f'[hetero] encoding {len(all_texts):,} sub-windows '
          f'(batch_size={args.batch_size}, device={args.device}) ...', flush=True)
    t0 = time.time()
    sub_emb = model.encode(
        all_texts, batch_size=args.batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False)
    sub_emb = np.asarray(sub_emb, dtype=np.float32)
    print(f'[hetero] encode done in {time.time()-t0:.1f}s, shape={sub_emb.shape}',
          flush=True)

    # ── 5. Nearest-centroid assignment ───────────────────────────────────────
    print(f'[hetero] computing nearest-centroid for each sub-window ...', flush=True)
    sub_preds = nearest_centroid(sub_emb, centroids)
    sub_preds = sub_preds.reshape(len(sample_idxs), n_sub_per_chunk)  # (N, 8)

    # Also look up the CHUNK's original cluster ID for reference
    chunk_preds = cluster_ids[sample_idxs]

    # ── 6. Stats per chunk ───────────────────────────────────────────────────
    distinct_counts = []  # per chunk: # of distinct clusters across sub-windows
    dominant_frac = []    # per chunk: fraction of sub-windows in the most common cluster
    entropy = []          # per chunk: entropy of sub-window cluster distribution
    matches_chunk_pred = []  # per chunk: fraction of sub-windows matching the chunk-level cluster ID

    for i in range(len(sample_idxs)):
        preds = sub_preds[i]
        ctr = Counter(preds.tolist())
        distinct_counts.append(len(ctr))
        dom_count = max(ctr.values())
        dominant_frac.append(dom_count / n_sub_per_chunk)
        probs = np.array(list(ctr.values()), dtype=np.float64) / n_sub_per_chunk
        entropy.append(float((-probs * _log2(probs)).sum()))
        matches_chunk_pred.append(float((preds == chunk_preds[i]).sum()) / n_sub_per_chunk)

    distinct_counts = np.array(distinct_counts)
    dominant_frac = np.array(dominant_frac)
    entropy = np.array(entropy)
    matches_chunk_pred = np.array(matches_chunk_pred)

    # ── 7. Report ────────────────────────────────────────────────────────────
    print()
    print('=' * 80)
    print(' Intra-chunk heterogeneity')
    print('=' * 80)
    print(f' Sample: {len(sample_idxs)} chunks × {n_sub_per_chunk} sub-windows of '
          f'{args.sub_window} tokens each (= {len(sample_idxs)*n_sub_per_chunk:,} '
          f'BGE-large re-embeddings)')
    print()

    print(' # distinct clusters per chunk (1=homogeneous, 8=fully heterogeneous):')
    hist = np.bincount(distinct_counts, minlength=n_sub_per_chunk + 1)
    hist_pct = 100.0 * hist / len(sample_idxs)
    print(f"   {'count':>5} {'#chunks':>8} {'pct':>8}")
    cum = 0
    for c in range(1, n_sub_per_chunk + 1):
        cum += hist_pct[c]
        print(f"   {c:>5} {hist[c]:>8} {hist_pct[c]:>7.2f}%")
    print()

    het = 100.0 * (distinct_counts >= 2).sum() / len(sample_idxs)
    strong_het = 100.0 * (distinct_counts >= 3).sum() / len(sample_idxs)
    print(f' Heterogeneity summary:')
    print(f'   chunks with ≥2 distinct sub-window clusters:  {het:.1f}%')
    print(f'   chunks with ≥3 distinct sub-window clusters:  {strong_het:.1f}%')
    print(f'   mean #distinct clusters per chunk: {distinct_counts.mean():.2f}')
    print(f'   mean dominant-cluster fraction:    {dominant_frac.mean():.3f}  '
          f'(1.0 = chunk fully homogeneous)')
    print(f'   mean sub-window entropy (bits):    {entropy.mean():.3f}  '
          f'(max {np.log2(K):.2f} at uniform over K={K})')
    print(f'   mean sub-window chunk-label agreement: {matches_chunk_pred.mean():.3f}  '
          f'(1.0 = every sub-window matches its chunk label)')

    # ── 8. Persist JSON ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    payload = {
        'token_cache': os.path.basename(args.token_cache),
        'cluster_cache': os.path.basename(args.cluster_cache),
        'embedding_cache': os.path.basename(args.embedding_cache),
        'embedder': args.embedder,
        'n_clusters': K,
        'n_chunks_sampled': int(len(sample_idxs)),
        'sub_window_tokens': args.sub_window,
        'n_sub_per_chunk': n_sub_per_chunk,
        'distinct_cluster_count_hist': {int(c): int(hist[c])
                                          for c in range(1, n_sub_per_chunk + 1)},
        'distinct_cluster_count_pct': {int(c): float(hist_pct[c])
                                         for c in range(1, n_sub_per_chunk + 1)},
        'pct_heterogeneous_ge2': float(het),
        'pct_heterogeneous_ge3': float(strong_het),
        'mean_distinct_clusters': float(distinct_counts.mean()),
        'mean_dominant_frac': float(dominant_frac.mean()),
        'mean_subwin_entropy_bits': float(entropy.mean()),
        'max_entropy_bits': float(np.log2(K)),
        'mean_chunk_label_agreement': float(matches_chunk_pred.mean()),
        'seed': args.seed,
    }
    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\n[hetero] saved {args.output}', flush=True)


if __name__ == '__main__':
    main()

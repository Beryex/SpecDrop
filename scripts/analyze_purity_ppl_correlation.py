#!/usr/bin/env python3
"""Per-chunk BGE cluster purity × ΔPPL correlation (paper App. E.15).

Goal: quantitative evidence linking intra-chunk topic mixture to the
(absence of) PPL gain on SlimPajama. For every VAL chunk we compute:
  purity  — slice the 512-token chunk into 8×64-token sub-windows, BGE-embed
            each, assign to the k=7 train KMeans centroids; purity = largest
            fraction of sub-windows agreeing on one cluster (1.0 = homogeneous).
  ΔCE     — per-chunk mean token CE under matched-SE No-Routing minus under
            ours (positive = ours better on that chunk), averaged over seeds.
Then report Pearson/Spearman correlation and homogeneous-vs-mixed binned means.

Prereqs (paths passed via CLI):
  - val token cache + 100M train token cache (data/slimpajama.py tokenize)
  - train cluster + embedding caches (data/cluster_chunks.build_cluster_cache)

Usage:
    python scripts/analyze_purity_ppl_correlation.py \
        --val-token-cache data_cache/slimpajama/<val cache>.pt \
        --train-cluster-cache data_cache/slimpajama/<train clusters>.pt \
        --train-embedding-cache data_cache/slimpajama/<train embeddings>.pt

Output: outputs/analysis/purity_ppl_correlation.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

RUNS = {
    'ours':    ['outputs/rtx5090_nlp_faithful/ours_phaseP_s{seed}'],
    'matched': ['outputs/rtx5090_nlp_faithful/no_routing_se05_s{seed}'],
}
SEEDS = (42, 123, 456)


def _load_cfg(run_dir):
    rj = f'{run_dir}/results.json'
    with open(rj) as f:
        return json.load(f)['config']


def _per_chunk_ce(run_dir, input_ids, domain_ids, device, batch_size=64):
    """(N,) mean-token CE per chunk for one trained checkpoint."""
    from run_nlp import build_nlp_model
    from algorithms import build_algorithm
    from scripts._diag_helpers import advance_softspecdrop_to_terminal

    cfg = _load_cfg(run_dir)
    model = build_nlp_model(cfg).to(device)
    ckpt = torch.load(f'{run_dir}/best.pt', map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    algo = build_algorithm(cfg)
    if algo is not None:
        advance_softspecdrop_to_terminal(algo, cfg['training']['epochs'])
        if hasattr(model, 'mask_scale'):
            model.mask_scale = algo.expected_mask_sum

    N = input_ids.shape[0]
    out = torch.empty(N)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                         enabled=device.startswith('cuda')):
        for i in range(0, N, batch_size):
            x = input_ids[i:i + batch_size].to(device)
            d = domain_ids[i:i + batch_size].to(device)
            if algo is not None:
                mask = algo.get_mask(d, training=False).to(device)
                logits = model(x, branch_mask=mask)
            else:
                logits = model(x)
            ce = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1), reduction='none'
            ).view(x.size(0), -1).mean(dim=1)
            out[i:i + x.size(0)] = ce.cpu()
    del model
    torch.cuda.empty_cache()
    return out.numpy()


def _val_purity(input_ids, centroids, embedder_name, device,
                sub_window=64, batch_size=256):
    """(N,) purity + (N,) n_distinct via sub-window BGE nearest-centroid."""
    from scripts.intra_chunk_heterogeneity import (
        detokenize_sub_windows, nearest_centroid)
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    tok = AutoTokenizer.from_pretrained('gpt2')
    st = SentenceTransformer(embedder_name, device=device)

    N, L = input_ids.shape
    nsub = L // sub_window
    texts = []
    for i in range(N):
        texts.extend(detokenize_sub_windows(input_ids[i], sub_window, tok))
    emb = st.encode(texts, batch_size=batch_size, show_progress_bar=True,
                    convert_to_numpy=True, normalize_embeddings=False)
    preds = nearest_centroid(emb, centroids).reshape(N, nsub)
    purity = np.array([np.bincount(row).max() / nsub for row in preds])
    ndist = np.array([len(np.unique(row)) for row in preds])
    return purity, ndist


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--val-token-cache', required=True)
    ap.add_argument('--train-cluster-cache', required=True)
    ap.add_argument('--train-embedding-cache', required=True)
    ap.add_argument('--embedder', default='BAAI/bge-large-en-v1.5')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-chunks', type=int, default=0,
                    help='cap on val chunks (0 = all)')
    ap.add_argument('--output', default='outputs/analysis/purity_ppl_correlation')
    args = ap.parse_args()

    tblob = torch.load(args.val_token_cache, weights_only=False)
    input_ids = tblob['input_ids']
    if not torch.is_tensor(input_ids):
        input_ids = torch.as_tensor(np.asarray(input_ids))
    domain_ids = tblob['domain_ids']
    if not torch.is_tensor(domain_ids):
        domain_ids = torch.as_tensor(np.asarray(domain_ids))
    if args.max_chunks:
        input_ids = input_ids[:args.max_chunks]
        domain_ids = domain_ids[:args.max_chunks]
    N = input_ids.shape[0]
    print(f'[purity_ppl] val chunks: {N}')

    # 1. per-chunk CE per method (mean over seeds)
    ce = {}
    for name, tmpls in RUNS.items():
        per_seed = []
        for seed in SEEDS:
            run_dir = tmpls[0].format(seed=seed)
            print(f'[purity_ppl] CE pass: {run_dir}')
            per_seed.append(_per_chunk_ce(run_dir, input_ids, domain_ids, args.device))
        ce[name] = np.mean(per_seed, axis=0)
    delta_ce = ce['matched'] - ce['ours']   # >0 → ours better on that chunk

    # 2. per-chunk purity
    from scripts.intra_chunk_heterogeneity import compute_centroids
    eblob = torch.load(args.train_embedding_cache, weights_only=False)
    emb = eblob['embeddings'] if isinstance(eblob, dict) else eblob
    if torch.is_tensor(emb):
        emb = emb.cpu().numpy()
    cblob = torch.load(args.train_cluster_cache, weights_only=False)
    cids = cblob['cluster_ids'] if isinstance(cblob, dict) else cblob
    if torch.is_tensor(cids):
        cids = cids.cpu().numpy()
    K = int(cids.max()) + 1
    centroids = compute_centroids(emb, cids, K)
    purity, ndist = _val_purity(input_ids.numpy(), centroids,
                                args.embedder, args.device)

    # 3. correlation + binned summary
    from scipy import stats
    pear = stats.pearsonr(purity, delta_ce)
    spear = stats.spearmanr(purity, delta_ce)
    homog = purity >= 0.999
    binned = {
        'homogeneous_n': int(homog.sum()),
        'mixed_n': int((~homog).sum()),
        'homogeneous_delta_ce_mean': float(delta_ce[homog].mean()),
        'mixed_delta_ce_mean': float(delta_ce[~homog].mean()),
        'homogeneous_delta_ppl_equiv': float(np.exp(ce['matched'][homog].mean()) - np.exp(ce['ours'][homog].mean())),
        'mixed_delta_ppl_equiv': float(np.exp(ce['matched'][~homog].mean()) - np.exp(ce['ours'][~homog].mean())),
    }
    payload = {
        'n_chunks': N,
        'pearson_r': float(pear[0]), 'pearson_p': float(pear[1]),
        'spearman_r': float(spear[0]), 'spearman_p': float(spear[1]),
        'purity_mean': float(purity.mean()),
        'frac_homogeneous': float(homog.mean()),
        'delta_ce_mean': float(delta_ce.mean()),
        'binned': binned,
        'runs': {k: v[0] for k, v in RUNS.items()}, 'seeds': list(SEEDS),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output + '.json', 'w') as f:
        json.dump(payload, f, indent=2)
    np.savez(args.output + '_perchunk.npz', purity=purity, ndist=ndist,
             delta_ce=delta_ce, ce_ours=ce['ours'], ce_matched=ce['matched'],
             domain_ids=domain_ids.numpy())
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()

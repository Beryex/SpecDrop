#!/usr/bin/env python3
"""Inspect SlimPajama 100M-token cache per-domain chunk counts.

Reports actual empirical proportions (post-tokenize, post-dedup) of the
cached train split. Used to design data-proportional branch allocation:
  ffn_dim_per_branch[k] = round(total_ffn * frac_k)
such that large-data domains (e.g. CommonCrawl) get bigger branches and
small-data domains (e.g. ArXiv, StackExchange) get narrower branches,
with total FFN params ≈ dense equivalent.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch

from data.slimpajama import DOMAIN_TO_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default=None,
                        help='Explicit cache path. If omitted, glob-search for '
                             'tokenized_train_seq512_tok100000000_vocab*.pt under '
                             'data_cache/slimpajama/ (the hash suffix depends on '
                             'the GPT-2 tokenizer vocab and is not fixed).')
    parser.add_argument('--glob_pattern', default='data_cache/slimpajama/tokenized_train_seq512_tok*_vocab*.pt')
    parser.add_argument('--total_ffn', type=int, default=1540,
                        help='Target total FFN width across K branches (= dense-equivalent). '
                             'K=7 uniform is 7*220=1540. Dense TransformerLM uses 1536.')
    args = parser.parse_args()

    # Resolve cache path: explicit > glob-find largest (most tokens).
    if args.cache:
        cache_path = args.cache
    else:
        import glob
        candidates = glob.glob(args.glob_pattern)
        if not candidates:
            print(f"ERROR: no cache found matching {args.glob_pattern}")
            print(f"  run an NLP experiment first to populate data_cache/slimpajama/")
            sys.exit(1)
        # Pick the one with the largest token budget (prefer 500M if both 100M and 500M exist).
        def tok_from_name(p):
            m = os.path.basename(p).split('_tok')[1].split('_')[0]
            return int(m)
        candidates.sort(key=tok_from_name, reverse=True)
        cache_path = candidates[0]
        print(f"[auto] Found {len(candidates)} cache file(s):")
        for c in candidates:
            print(f"       {os.path.basename(c)}")
        print(f"[auto] Using: {cache_path}\n")

    if not os.path.exists(cache_path):
        print(f"ERROR: cache not found at {cache_path}")
        sys.exit(1)
    args.cache = cache_path

    print(f"Loading {args.cache} ...")
    blob = torch.load(args.cache, weights_only=False)
    d = blob['domain_ids'].numpy()

    id2n = {v: k for k, v in DOMAIN_TO_ID.items()}
    u, c = np.unique(d, return_counts=True)
    total = int(c.sum())

    print(f"\nTotal 512-token chunks: {total:,}")
    print(f"Max possible tokens: {total * 512:,} ≈ {total * 512 / 1e6:.1f}M")
    print()
    print(f"{'domain':<30} {'count':>10} {'frac':>8} {'ffn@total=':>14}")
    print(f"{'':<30} {'':>10} {'':>8} {args.total_ffn:>14}")
    print('-' * 72)

    rows = []
    for ui, ci in sorted(zip(u, c), key=lambda x: -x[1]):
        frac = ci / total
        ffn = int(round(args.total_ffn * frac))
        rows.append((id2n[ui], int(ui), int(ci), frac, ffn))
        print(f"{id2n[ui]:<30} {ci:>10,} {frac:>8.3%} {ffn:>14}")

    print('-' * 72)
    print(f"{'TOTAL':<30} {total:>10,} {1.0:>8.3%} {sum(r[4] for r in rows):>14}")

    # Diagnostic lines for non-uniform branch design
    max_ffn = max(r[4] for r in rows)
    min_ffn = min(r[4] for r in rows)
    print()
    print(f"Ratio largest/smallest branch: {max_ffn / max(min_ffn, 1):.1f}×")
    print(f"If min_ffn < 32: small domains effectively starved (dense FFN < 2× hidden_dim=384)")

    print()
    print("Design implication:")
    print(f"  uniform ffn = {args.total_ffn // 7} per branch (K=7)")
    print(f"  proportional: {max_ffn} for largest, {min_ffn} for smallest domain")
    print(f"  If signal requires min capacity (e.g. ffn >= hidden_dim), data-proportional may")
    print(f"  under-provision small domains; consider capped allocation: ffn_k = max(min_ffn, prop_k).")


if __name__ == '__main__':
    main()

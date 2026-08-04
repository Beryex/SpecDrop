"""Offline semantic clustering pipeline for SlimPajama tokenized chunks.

Takes existing tokenized caches (N × seq_len int tensor) and produces sibling
`.pt` files with one cluster ID per chunk. Used by Phase W experiments to
replace SlimPajama's source-level `domain_ids` with semantic cluster labels.

Pipeline with layered caching (each stage persists so downstream ablations
can skip upstream recomputation):

    Layer 0  — tokenized chunks (from data/slimpajama.py, always present)
    Layer 1  — detokenized text      [detokenized_<split>_seq<L>_tok<N>.pt]
    Layer 2  — sentence embeddings    [embeddings_<split>_..._<embedder>.pt]
    Layer 3  — cluster IDs (fit/pred) [clusters_<split>_..._<embedder>_k<K>.pt]

Re-run cost matrix (for 100M-token mini-ablation):
    change kmeans seed / k       → Layer 3 only, <1 min
    change embedder              → Layers 2+3, ~5 min
    change token cache (100M→500M) → full rebuild
    first run                    → full rebuild, ~5-25 min

Offline because clustering must be run ONCE per cache, produces deterministic
label files that many training runs read. `_run_gpu0.sh` is the designated
primary; `_run_gpu1/2.sh` poll for the cache-exists signal before training.

Usage (as a module):
    python -m data.cluster_chunks \\
        --train-cache data_cache/slimpajama/tokenized_train_seq512_tok100000000_vocab<H>.pt \\
        --train-output data_cache/slimpajama/clusters_bge-large_k7_tok100000000.pt \\
        --val-cache data_cache/slimpajama/tokenized_val_seq512_tok5000000_vocab<H>.pt \\
        --val-output data_cache/slimpajama/clusters_bge-large_k7_val5M.pt \\
        --n_clusters 7 --embedder BAAI/bge-large-en-v1.5 --seed 42

Usage (importable):
    from data.cluster_chunks import build_cluster_cache
    build_cluster_cache(
        train_cache_path=..., train_output_path=...,
        val_cache_path=..., val_output_path=...,
        n_clusters=7)
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
import re
import sys
import time

import numpy as np
import torch


DEFAULT_EMBEDDER = 'BAAI/bge-large-en-v1.5'


# ── Path derivation helpers ──────────────────────────────────────────────────

_TOK_PREFIX_RE = re.compile(
    r'^tokenized_(?P<split>[a-z]+)_seq(?P<seq>\d+)_tok(?P<N>\d+)'
    r'(?:_(?:vocab|tok)[0-9a-f]+)?\.pt$'
)


def _parse_token_cache_name(source_cache_path):
    """Extract the '<split>_seq<L>_tok<N>' middle from a tokenized cache name.

    Handles three styles:
      - new:     tokenized_train_seq512_tok100000000_vocab2e62aacd7f.pt
      - legacy:  tokenized_train_seq500000000_tok500000000_tok2e62aacd7f.pt
      - no-hash: tokenized_val_seq512_tok200000.pt

    The token-count '_tok<N>' is anchored to `\\d+` so it's never mistaken
    for the 10-char hex hash suffix (which may also contain decimal chars).
    """
    name = os.path.basename(source_cache_path)
    m = _TOK_PREFIX_RE.match(name)
    if not m:
        raise ValueError(
            f"Cannot parse tokenized cache filename {name!r}; expected "
            f"'tokenized_<split>_seq<L>_tok<N>(_vocab<H>|_tok<H>)?.pt'")
    return f"{m.group('split')}_seq{m.group('seq')}_tok{m.group('N')}"


def _derive_text_cache_path(source_cache_path, override_dir=None):
    """Sibling-of-source path for the detokenized text cache.
    Example: tokenized_train_seq512_tok100000000_vocab<H>.pt
          → detokenized_train_seq512_tok100000000.pt
    Vocab hash is stripped because decoding with GPT-2 is deterministic and
    the source_cache field inside the file pins exact provenance.
    """
    base_dir = override_dir or os.path.dirname(source_cache_path)
    middle = _parse_token_cache_name(source_cache_path)
    return os.path.join(base_dir, f'detokenized_{middle}.pt')


def _derive_embedding_cache_path(source_cache_path, embedder_short,
                                  override_dir=None):
    """Sibling-of-source path for the sentence-embedding cache.
    Example: tokenized_train_seq512_tok100000000_vocab<H>.pt
          → embeddings_train_seq512_tok100000000_bge-large-en-v1.5.pt
    """
    base_dir = override_dir or os.path.dirname(source_cache_path)
    middle = _parse_token_cache_name(source_cache_path)
    return os.path.join(base_dir,
                         f'embeddings_{middle}_{embedder_short}.pt')


def _embedder_short_name(embedder):
    """Extract a filename-safe short name for an embedder spec.
    'BAAI/bge-large-en-v1.5' → 'bge-large-en-v1.5'
    """
    if callable(embedder):
        return 'callable'
    tail = str(embedder).split('/')[-1]
    # Extra safety: replace any leftover filesystem-unfriendly chars.
    return re.sub(r'[^A-Za-z0-9._\-]', '_', tail)


# ── Low-level steps ──────────────────────────────────────────────────────────

def detokenize_chunks(input_ids, tokenizer_name='gpt2', batch_size=1024,
                      show_progress=True):
    """GPT-2 decode each (seq_len,) row back to a text string.

    Returns list[str] of length N, aligned with input_ids row order.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    if torch.is_tensor(input_ids):
        input_ids = input_ids.cpu().numpy()
    else:
        input_ids = np.asarray(input_ids)

    assert input_ids.ndim == 2, f'expected 2-D (N, seq_len), got shape {input_ids.shape}'
    N = input_ids.shape[0]

    texts = []
    t0 = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = [row.tolist() for row in input_ids[start:end]]
        decoded = tok.batch_decode(batch, skip_special_tokens=True)
        texts.extend(decoded)
        if show_progress and (start // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            rate = (start + batch_size) / max(elapsed, 1e-6)
            print(f'  [detok] {end}/{N}  ({rate:.0f} chunks/s)', flush=True)

    assert len(texts) == N, f'detokenize produced {len(texts)} texts, expected {N}'
    return texts


class _EmbedderAdapter:
    """Wraps a sentence-transformer or a user-supplied callable into a
    consistent (texts, batch_size, device) -> (N, D) np.float32 interface.

    Used to amortize the embedder's setup cost (model load / checkpoint
    download, ~1.3 GB for BGE-large) across train + val embedding calls.
    """

    def __init__(self, embedder, device='cuda'):
        self.device = device
        if callable(embedder):
            self._name = 'callable'
            self._fn = embedder
        else:
            from sentence_transformers import SentenceTransformer
            print(f'[embed] Loading {embedder}...', flush=True)
            model = SentenceTransformer(embedder, device=device)
            self._name = str(embedder)
            self._model = model
            self._fn = None

    def encode(self, texts, batch_size=32, show_progress=True):
        if self._fn is not None:
            emb = self._fn(texts, batch_size, self.device)
            emb = np.asarray(emb, dtype=np.float32)
            assert emb.ndim == 2 and emb.shape[0] == len(texts), \
                f'embedder returned shape {emb.shape}, expected ({len(texts)}, D)'
            return emb
        print(f'[embed] Encoding {len(texts):,} texts '
              f'(batch_size={batch_size}, device={self.device})...', flush=True)
        t0 = time.time()
        emb = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        print(f'[embed] Done in {time.time() - t0:.1f}s, shape={emb.shape}',
              flush=True)
        return emb.astype(np.float32)


def embed_chunks(texts, embedder=DEFAULT_EMBEDDER, batch_size=32,
                 device='cuda', show_progress=True):
    """One-shot embed: load model and encode. For single-split / tests only."""
    adapter = _EmbedderAdapter(embedder, device=device)
    return adapter.encode(texts, batch_size=batch_size,
                           show_progress=show_progress)


def fit_kmeans(embeddings, n_clusters=7, seed=42):
    """Fit a KMeans model and return (labels, km)."""
    from sklearn.cluster import KMeans
    print(f'[kmeans-fit] n_clusters={n_clusters}  seed={seed}  '
          f'input shape={embeddings.shape}', flush=True)
    t0 = time.time()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)
    print(f'[kmeans-fit] Done in {time.time() - t0:.1f}s  '
          f'inertia={km.inertia_:.2e}', flush=True)
    _print_cluster_sizes(labels, prefix='  train  ')
    return labels.astype(np.int64), km


def predict_clusters(km, embeddings, split_name='val'):
    """Predict cluster IDs for new embeddings using a fitted KMeans model."""
    print(f'[kmeans-predict] {split_name}  input shape={embeddings.shape}',
          flush=True)
    t0 = time.time()
    labels = km.predict(embeddings).astype(np.int64)
    print(f'[kmeans-predict] Done in {time.time() - t0:.1f}s', flush=True)
    _print_cluster_sizes(labels, prefix=f'  {split_name}    ')
    return labels


def _print_cluster_sizes(labels, prefix=''):
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    for u, c in zip(unique, counts):
        print(f'{prefix}cluster {u}: {c:,} chunks ({100 * c / total:.2f}%)',
              flush=True)


def cluster_embeddings(embeddings, n_clusters=7, seed=42):
    """Fit KMeans and return labels only (backwards-compat helper for tests)."""
    labels, _ = fit_kmeans(embeddings, n_clusters=n_clusters, seed=seed)
    return labels


# ── Cache I/O (atomic writes) ────────────────────────────────────────────────

def _atomic_save(path, payload):
    """Write to path+'.tmp' then os.replace, so concurrent readers polling
    for the final path never observe a partial file.
    """
    tmp_path = path + '.tmp'
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _load_token_cache(path):
    """Return (input_ids tensor, N)."""
    print(f'[cluster] Loading token cache from {path}...', flush=True)
    blob = torch.load(path, weights_only=False)
    input_ids = blob['input_ids']
    N = input_ids.shape[0]
    print(f'[cluster] {N:,} chunks × {input_ids.shape[1]} tokens', flush=True)
    return input_ids, N


def _save_text_cache(path, texts, source_cache_path):
    _atomic_save(path, {
        'texts': texts,
        'num_chunks': len(texts),
        'source_cache': os.path.basename(source_cache_path),
    })
    print(f'[cluster] Saved {len(texts):,} texts to {path}', flush=True)


def _load_text_cache(path):
    print(f'[cluster] Loading cached texts from {path}...', flush=True)
    blob = torch.load(path, weights_only=False)
    texts = blob['texts']
    print(f'[cluster] Loaded {len(texts):,} texts (source={blob.get("source_cache")})',
          flush=True)
    return texts


def _save_embedding_cache(path, embeddings, embedder_name, source_cache_path):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    _atomic_save(path, {
        'embeddings': torch.from_numpy(embeddings),
        'num_chunks': int(embeddings.shape[0]),
        'dim': int(embeddings.shape[1]),
        'embedder': embedder_name,
        'source_cache': os.path.basename(source_cache_path),
    })
    print(f'[cluster] Saved {embeddings.shape[0]:,} × {embeddings.shape[1]}-dim '
          f'embeddings to {path}', flush=True)


def _load_embedding_cache(path):
    print(f'[cluster] Loading cached embeddings from {path}...', flush=True)
    blob = torch.load(path, weights_only=False)
    emb = blob['embeddings']
    if torch.is_tensor(emb):
        emb = emb.numpy()
    emb = np.asarray(emb, dtype=np.float32)
    print(f'[cluster] Loaded {emb.shape[0]:,} × {emb.shape[1]}-dim embeddings '
          f'(embedder={blob.get("embedder")}, source={blob.get("source_cache")})',
          flush=True)
    return emb


def _save_cluster_labels(path, labels, n_clusters, embedder_name, seed,
                         source_cache_path):
    labels = np.asarray(labels, dtype=np.int64)
    _atomic_save(path, {
        'cluster_ids': torch.from_numpy(labels).long(),
        'n_clusters': int(n_clusters),
        'embedder': embedder_name,
        'seed': int(seed),
        'source_cache': os.path.basename(source_cache_path),
        'num_chunks': int(labels.shape[0]),
    })
    print(f'[cluster] Saved {labels.shape[0]:,} cluster IDs to {path}',
          flush=True)


# ── Layered flow: text ← tokens; embeddings ← text; clusters ← embeddings ───

def _get_texts_for_split(source_cache_path, text_cache_path,
                          tokenizer_name, detok_batch_size,
                          save_intermediate):
    """Load from text cache; else decode from source tokens and optionally save."""
    if os.path.exists(text_cache_path):
        texts = _load_text_cache(text_cache_path)
        N = len(texts)
        return texts, N

    input_ids, N = _load_token_cache(source_cache_path)
    texts = detokenize_chunks(input_ids, tokenizer_name=tokenizer_name,
                               batch_size=detok_batch_size)
    del input_ids
    if save_intermediate:
        _save_text_cache(text_cache_path, texts, source_cache_path)
    return texts, N


def _get_embeddings_for_split(source_cache_path, emb_cache_path, text_cache_path,
                               embedder_adapter_fn, tokenizer_name,
                               detok_batch_size, batch_size,
                               save_intermediate):
    """Load embeddings from cache; else (load or compute) text, embed, optionally save.

    Returns (embeddings (N, D) float32, N).
    """
    if os.path.exists(emb_cache_path):
        emb = _load_embedding_cache(emb_cache_path)
        return emb, int(emb.shape[0])

    texts, N = _get_texts_for_split(source_cache_path, text_cache_path,
                                      tokenizer_name, detok_batch_size,
                                      save_intermediate)
    adapter = embedder_adapter_fn()
    emb = adapter.encode(texts, batch_size=batch_size)
    del texts
    if save_intermediate:
        _save_embedding_cache(emb_cache_path, emb, adapter._name,
                                source_cache_path)
    return emb, N


# ── High-level pipeline ──────────────────────────────────────────────────────

def build_cluster_cache(train_cache_path, train_output_path,
                         val_cache_path=None, val_output_path=None,
                         n_clusters=7, embedder=DEFAULT_EMBEDDER,
                         seed=42, batch_size=32, device='cuda',
                         tokenizer_name='gpt2', detok_batch_size=1024,
                         skip_if_exists=True,
                         text_cache_dir=None, embedding_cache_dir=None,
                         save_intermediate=True):
    """End-to-end clustering. Fits KMeans on train; predicts val (if provided).

    Layered caching: text cache (skips detokenize) and embedding cache (skips
    embedder) are persisted as sibling .pt files. Re-runs reuse them when
    present. Pass save_intermediate=False to skip persisting (slower but
    saves ~1.2 GB per 100M-token run).

    Args:
        train_cache_path: input .pt with 'input_ids' from slimpajama.py.
        train_output_path: output .pt for train cluster IDs.
        val_cache_path, val_output_path: optional, to share cluster defs with val.
        n_clusters: M, number of k-means clusters.
        embedder: sentence-transformer model name (str) or callable.
        seed: KMeans random_state.
        batch_size: embedding batch size.
        device: embedding device ('cuda'/'cpu'/'mps').
        tokenizer_name: tokenizer for decoding (default 'gpt2').
        detok_batch_size: decode batch size (CPU-only).
        skip_if_exists: return early if ALL requested outputs already exist.
        text_cache_dir: override directory for text cache (default: alongside source).
        embedding_cache_dir: override directory for embedding cache (default: alongside source).
        save_intermediate: whether to persist text and embedding caches.

    Returns:
        dict with keys 'train' (always) and 'val' (if requested) mapping to
        output paths.
    """
    want_val = val_cache_path is not None
    if want_val and val_output_path is None:
        raise ValueError('val_output_path must be set when val_cache_path is given')

    # Idempotency: if all requested CLUSTER outputs exist, skip entirely.
    train_cluster_exists = os.path.exists(train_output_path)
    val_cluster_exists = (not want_val) or os.path.exists(val_output_path)
    if skip_if_exists and train_cluster_exists and val_cluster_exists:
        print(f'[cluster] All outputs exist — skipping')
        out = {'train': train_output_path}
        if want_val:
            out['val'] = val_output_path
        return out

    # Intermediate cache paths (derived from source).
    embedder_short = _embedder_short_name(embedder)
    train_text_path = _derive_text_cache_path(train_cache_path, text_cache_dir)
    train_emb_path = _derive_embedding_cache_path(
        train_cache_path, embedder_short, embedding_cache_dir)

    # Lazy embedder — only loaded when we actually need to embed (i.e., no
    # embedding cache for at least one split). If both embedding caches exist,
    # we never pay BGE-large's 1.3 GB download + model-load cost.
    _adapter_slot = [None]
    def _get_adapter():
        if _adapter_slot[0] is None:
            _adapter_slot[0] = _EmbedderAdapter(embedder, device=device)
        return _adapter_slot[0]

    # ── Train: get embeddings, fit KMeans ────────────────────────────────────
    train_emb, N_train = _get_embeddings_for_split(
        train_cache_path, train_emb_path, train_text_path,
        _get_adapter, tokenizer_name, detok_batch_size, batch_size,
        save_intermediate)

    train_labels, km = fit_kmeans(train_emb, n_clusters=n_clusters, seed=seed)
    del train_emb

    assert train_labels.shape == (N_train,), \
        f'train label shape {train_labels.shape} != ({N_train},)'
    assert train_labels.min() >= 0 and train_labels.max() < n_clusters, \
        f'train label range [{train_labels.min()}, {train_labels.max()}] ' \
        f'outside [0, {n_clusters})'

    # The embedder short name is authoritative even when embedder is a
    # callable (tests); use adapter's own name when one was actually loaded.
    embedder_name = (_adapter_slot[0]._name
                      if _adapter_slot[0] is not None
                      else (str(embedder) if not callable(embedder) else 'callable'))

    if not (skip_if_exists and train_cluster_exists):
        _save_cluster_labels(train_output_path, train_labels, n_clusters,
                               embedder_name, seed, train_cache_path)
    else:
        print(f'[cluster] Train output already exists, not overwriting: '
              f'{train_output_path}')

    out = {'train': train_output_path}

    # ── Val: predict with the fitted KMeans ──────────────────────────────────
    if want_val:
        val_text_path = _derive_text_cache_path(val_cache_path, text_cache_dir)
        val_emb_path = _derive_embedding_cache_path(
            val_cache_path, embedder_short, embedding_cache_dir)

        val_emb, N_val = _get_embeddings_for_split(
            val_cache_path, val_emb_path, val_text_path,
            _get_adapter, tokenizer_name, detok_batch_size, batch_size,
            save_intermediate)

        val_labels = predict_clusters(km, val_emb, split_name='val')
        del val_emb

        assert val_labels.shape == (N_val,), \
            f'val label shape {val_labels.shape} != ({N_val},)'
        assert val_labels.min() >= 0 and val_labels.max() < n_clusters, \
            f'val label range [{val_labels.min()}, {val_labels.max()}] ' \
            f'outside [0, {n_clusters})'

        if not (skip_if_exists and val_cluster_exists):
            _save_cluster_labels(val_output_path, val_labels, n_clusters,
                                   embedder_name, seed, val_cache_path)
        else:
            print(f'[cluster] Val output already exists, not overwriting: '
                  f'{val_output_path}')
        out['val'] = val_output_path

    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Offline semantic clustering for SlimPajama chunks')
    parser.add_argument('--train-cache', required=True,
                        help='Input tokenized train .pt (from slimpajama.py)')
    parser.add_argument('--train-output', required=True,
                        help='Output cluster-label .pt for train')
    parser.add_argument('--val-cache', default=None,
                        help='Optional: input tokenized val .pt')
    parser.add_argument('--val-output', default=None,
                        help='Optional: output cluster-label .pt for val '
                             '(required iff --val-cache)')
    parser.add_argument('--n_clusters', type=int, default=7)
    parser.add_argument('--embedder', type=str, default=DEFAULT_EMBEDDER,
                        help='Sentence-transformer model name')
    parser.add_argument('--seed', type=int, default=42,
                        help='KMeans random_state')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Embedding batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help="'cuda', 'cpu', or 'mps'")
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='HuggingFace tokenizer name for decoding')
    parser.add_argument('--detok_batch_size', type=int, default=1024)
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if outputs already exist')
    parser.add_argument('--no-save-intermediate', action='store_true',
                        help="Skip persisting text+embedding caches "
                             "(saves ~1.2 GB per 100M-token run; prevents "
                             "cheap re-clustering with different k/seed)")
    parser.add_argument('--text-cache-dir', default=None,
                        help='Override directory for text cache')
    parser.add_argument('--embedding-cache-dir', default=None,
                        help='Override directory for embedding cache')
    args = parser.parse_args()

    if not os.path.exists(args.train_cache):
        print(f'error: train cache {args.train_cache} does not exist',
              file=sys.stderr)
        sys.exit(1)
    if args.val_cache and not os.path.exists(args.val_cache):
        print(f'error: val cache {args.val_cache} does not exist',
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.train_output)),
                exist_ok=True)
    if args.val_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.val_output)),
                    exist_ok=True)

    build_cluster_cache(
        train_cache_path=args.train_cache,
        train_output_path=args.train_output,
        val_cache_path=args.val_cache,
        val_output_path=args.val_output,
        n_clusters=args.n_clusters,
        embedder=args.embedder,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        tokenizer_name=args.tokenizer,
        detok_batch_size=args.detok_batch_size,
        skip_if_exists=not args.force,
        text_cache_dir=args.text_cache_dir,
        embedding_cache_dir=args.embedding_cache_dir,
        save_intermediate=not args.no_save_intermediate,
    )


if __name__ == '__main__':
    main()

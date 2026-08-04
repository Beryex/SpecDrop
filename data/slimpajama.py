"""SlimPajama-6B data loader with domain labels for SpecDrop NLP experiments.

Uses HuggingFace datasets to load DKYoon/SlimPajama-6B and tokenizes with
GPT-2 tokenizer. Each document is tagged with one of 7 domains via
meta.redpajama_set_name, providing category labels for SpecDrop.

Usage:
    train_loader, val_loader, vocab_size = get_slimpajama_dataloaders(
        data_dir='./data_cache/slimpajama', batch_size=32, max_seq_len=1024)
"""

import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError(
        "transformers is required for SlimPajama data loading. "
        "Install with: pip install transformers"
    )

# ── Domain mapping ───────────────────────────────────────────────────────────

NUM_DOMAINS = 7

DOMAIN_NAMES = [
    'RedPajamaCommonCrawl',
    'RedPajamaC4',
    'RedPajamaGithub',
    'RedPajamaBook',
    'RedPajamaArXiv',
    'RedPajamaWikipedia',
    'RedPajamaStackExchange',
]

DOMAIN_TO_ID = {name: idx for idx, name in enumerate(DOMAIN_NAMES)}


# ── Non-uniform branch allocation helpers ────────────────────────────────────

def find_tokenize_cache(data_dir='./data_cache/slimpajama', split='train',
                        max_tokens=None, max_seq_len=512):
    """Glob-find a tokenized cache file. Cache names include a vocab hash
    suffix (SHA256[:10] of the GPT-2 vocab JSON) that we don't know ahead of
    time, so we glob for `*_vocab*.pt` and pick the best match.

    Args:
        data_dir: directory containing cache files.
        split: 'train' or 'val'.
        max_tokens: if given, match only that token budget; else prefer largest.
        max_seq_len: tokens per sequence.

    Returns:
        Full path to the selected cache file.

    Raises:
        FileNotFoundError if no matching cache exists.
    """
    import glob
    # Try both new (`_vocab<H>.pt`) and legacy (`_tok<H>.pt`) hash-suffix
    # conventions. Mirrors the compat fallback in get_slimpajama_dataloaders.
    if max_tokens is not None:
        patterns = [
            os.path.join(data_dir, f'tokenized_{split}_seq{max_seq_len}_tok{max_tokens}_vocab*.pt'),
            os.path.join(data_dir, f'tokenized_{split}_seq{max_seq_len}_tok{max_tokens}_tok*.pt'),
        ]
    else:
        patterns = [
            os.path.join(data_dir, f'tokenized_{split}_seq{max_seq_len}_tok*_vocab*.pt'),
            os.path.join(data_dir, f'tokenized_{split}_seq{max_seq_len}_tok*_tok*.pt'),
        ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        raise FileNotFoundError(
            f"No tokenized cache matching any of: {patterns!r}. "
            "Run an NLP experiment first to populate the cache.")
    # Prefer the largest token budget when multiple match (pattern without max_tokens).
    def _tok_budget(p):
        return int(os.path.basename(p).split('_tok')[1].split('_')[0])
    candidates.sort(key=_tok_budget, reverse=True)
    return candidates[0]


def split_ffn_budget_for_se(total_ffn_budget, se_ratio, num_branches=NUM_DOMAINS):
    """Split a total dense-equivalent FFN budget between K branches and a shared
    expert, given an SE ratio.

    SE_ratio is defined as SE_dim / avg_branch_width. With that definition,
    keeping the dense-equivalent total fixed solves to:
        avg_branch = total / (K + se_ratio)
        SE_dim     = total * se_ratio / (K + se_ratio)
        total_branch_ffn = avg_branch * K = total * K / (K + se_ratio)

    For K=7, total=1540:
        se_ratio=0.0 → branches=1540, SE=0
        se_ratio=0.5 → branches=1437, SE=103
        se_ratio=1.0 → branches=1348, SE=192
        se_ratio=4.0 → branches=980,  SE=560

    Returns (total_branch_ffn, se_dim), both ints.
    """
    denom = num_branches + se_ratio
    total_branch_ffn = int(round(total_ffn_budget * num_branches / denom))
    se_dim = int(round(total_ffn_budget * se_ratio / denom))
    return total_branch_ffn, se_dim


def split_lora_rank_budget_for_se(total_rank, se_ratio, num_experts):
    """LoRA-rank analog of split_ffn_budget_for_se for the MultiBranch-LoRA track.

    Same semantics: keep a fixed RANK budget across (num_experts branch LoRAs
    of rank r_expert) + (one shared LoRA of rank r_SE). SE_ratio = r_SE /
    avg_r_expert, so the algebra mirrors the FFN helper exactly:
        avg_r    = total_rank / (num_experts + se_ratio)
        r_expert = round(avg_r)
        r_SE     = round(se_ratio * avg_r)

    For K=20, total_rank=320 (= 20 × 16, our no-SE baseline):
        se_ratio=0.0 → r_expert=16,  r_SE=0     total=320
        se_ratio=0.5 → r_expert=16,  r_SE=8     total=328  (+2.5%)
        se_ratio=1.0 → r_expert=15,  r_SE=15    total=315  (-1.6%)
        se_ratio=2.0 → r_expert=15,  r_SE=29    total=329  (+2.8%)
    Integer rounding produces small (<3%) deviations around the exact fixed
    budget — this matches split_ffn_budget_for_se's behaviour and is within
    utils.sanity_check.py's ±2% budget envelope (plus the MLP-site asymmetry
    in actual LoRA parameter count, which dominates the rounding noise).

    Returns (r_expert, r_SE) as ints. r_SE=0 iff se_ratio=0.
    """
    if num_experts <= 0:
        raise ValueError(f"num_experts must be > 0, got {num_experts}")
    if se_ratio < 0:
        raise ValueError(f"se_ratio must be >= 0, got {se_ratio}")
    if se_ratio == 0:
        avg_r = total_rank / num_experts
        return int(round(avg_r)), 0
    denom = num_experts + se_ratio
    avg_r = total_rank / denom
    r_expert = int(round(avg_r))
    r_SE = int(round(se_ratio * avg_r))
    return r_expert, r_SE


def load_cluster_labels(cluster_label_path):
    """Load a cluster label file produced by `data.cluster_chunks`.

    Returns a dict with keys:
      - cluster_ids: (N,) int64 tensor of cluster assignments
      - n_clusters: int
      - embedder: str
      - seed: int
      - source_cache: str (basename of the source tokenized cache)
      - num_chunks: int

    Raises FileNotFoundError if the path doesn't exist.
    """
    if not os.path.exists(cluster_label_path):
        raise FileNotFoundError(
            f"Cluster label file not found: {cluster_label_path}. "
            "Run `python -m data.cluster_chunks --cache ... --output ...` first.")
    blob = torch.load(cluster_label_path, weights_only=False)
    required_keys = {'cluster_ids', 'n_clusters'}
    missing = required_keys - set(blob.keys())
    if missing:
        raise ValueError(
            f"Cluster label file {cluster_label_path} missing keys: {missing}")
    return blob


def _count_chunks_per_domain(cache_path, num_categories=NUM_DOMAINS,
                              cluster_label_path=None):
    """Load a tokenized cache and return (counts_list, total_chunks).

    counts_list is length num_categories in category-id order; missing ids get 0.

    When `cluster_label_path` is provided, cluster IDs from that file replace
    `domain_ids` from the tokenized cache. The cluster file must have the
    same chunk count as the tokenized cache.
    """
    if cluster_label_path is not None:
        # Use cluster IDs instead of source-level domain IDs.
        cblob = load_cluster_labels(cluster_label_path)
        ids = cblob['cluster_ids']
        # Sanity check: chunk count must match the token cache.
        tblob = torch.load(cache_path, weights_only=False)
        tok_N = tblob['input_ids'].shape[0]
        if int(len(ids)) != tok_N:
            raise ValueError(
                f"Cluster label count {len(ids)} does not match token cache "
                f"chunk count {tok_N} (source: {cache_path})")
    else:
        blob = torch.load(cache_path, weights_only=False)
        ids = blob['domain_ids']

    if hasattr(ids, 'numpy'):
        ids = ids.numpy()
    else:
        import numpy as _np
        ids = _np.asarray(ids)

    total_chunks = int(len(ids))
    if total_chunks == 0:
        raise ValueError(f"Cache {cache_path} has zero chunks.")

    counts = [0] * num_categories
    import numpy as _np
    unique, c = _np.unique(ids, return_counts=True)
    for u, ci in zip(unique, c):
        k = int(u)
        if 0 <= k < num_categories:
            counts[k] = int(ci)
    return counts, total_chunks


def compute_category_fractions(cache_path, num_categories=NUM_DOMAINS,
                                cluster_label_path=None):
    """Return per-category chunk fractions in category-id order (list of floats
    summing to ~1.0), used by per-category SoftSpecDrop routing.

    When `cluster_label_path` is None (default), fractions are computed from
    SlimPajama's source-level `domain_ids` (7 domains: CC, C4, Github, Book,
    ArXiv, Wiki, StackExchange).

    When `cluster_label_path` is provided, fractions are computed from
    semantic cluster IDs produced offline by `data.cluster_chunks`. The
    cluster count must be compatible with `num_categories` (typically they
    are set equal, e.g. M=K=7).

    Sibling to compute_proportional_ffn_dims; shares the same counting logic
    but returns only the normalized fractions.
    """
    counts, total_chunks = _count_chunks_per_domain(
        cache_path, num_categories, cluster_label_path=cluster_label_path)
    return [counts[k] / total_chunks for k in range(num_categories)]


def compute_proportional_ffn_dims(cache_path, total_ffn=1540, num_branches=NUM_DOMAINS):
    """Return per-branch FFN widths proportional to each domain's chunk count,
    in domain_id order (NOT sorted by size).

    Assumes the branch→domain mapping is the round-robin identity used by
    SoftSpecDrop, i.e. branch k handles domain k. So `ffn_dims[k]` is the
    capacity allocated to domain k.

    Note ordering subtlety: because domain_id follows DOMAIN_NAMES order
    rather than data-size order, the returned list is NOT monotonically
    decreasing. E.g. Book (id=3) has less data than ArXiv (id=4), so
    ffn_dims[3] < ffn_dims[4].

    Args:
        cache_path: path to tokenized train cache (.pt).
        total_ffn: target total FFN width across K branches (default 1540,
                    = 7*220 uniform reference, ≈ dense ffn 1536).
        num_branches: K (default NUM_DOMAINS = 7).

    Returns:
        (ffn_dims, mapping) where
          - ffn_dims: list[int] of length num_branches, in branch/domain_id order
          - mapping: list[dict] with keys branch / domain_id / domain_name /
                     count / frac / ffn_dim for logging / sanity printout.
    """
    counts, total_chunks = _count_chunks_per_domain(cache_path, num_branches)

    ffn_dims = [int(round(total_ffn * (counts[k] / total_chunks)))
                for k in range(num_branches)]

    id_to_name = {v: k for k, v in DOMAIN_TO_ID.items()}
    mapping = [
        {
            'branch': k,
            'domain_id': k,
            'domain_name': id_to_name.get(k, f'unknown_id_{k}'),
            'count': counts[k],
            'frac': counts[k] / total_chunks,
            'ffn_dim': ffn_dims[k],
        }
        for k in range(num_branches)
    ]
    return ffn_dims, mapping


# ── Dataset ──────────────────────────────────────────────────────────────────

class SlimPajamaChunkedDataset(Dataset):
    """Pre-tokenized, fixed-length chunks from SlimPajama with domain labels."""

    def __init__(self, input_ids_list, domain_ids_list):
        """
        Args:
            input_ids_list: list of 1-D int tensors, each of length max_seq_len
            domain_ids_list: list of int domain indices
        """
        self.input_ids = input_ids_list
        self.domain_ids = domain_ids_list

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.domain_ids[idx]


# ── Processing ───────────────────────────────────────────────────────────────

def _process_split(hf_dataset, tokenizer, max_seq_len, max_tokens, split_name=''):
    """Tokenize documents and chunk into fixed-length sequences.

    Args:
        hf_dataset: HuggingFace dataset split
        tokenizer: GPT-2 tokenizer
        max_seq_len: number of tokens per sequence
        max_tokens: stop after accumulating this many tokens
        split_name: 'train' or 'val' — for progress bar label

    Returns:
        input_ids_list: list of (max_seq_len,) int tensors
        domain_ids_list: list of int domain indices
    """
    input_ids_list = []
    domain_ids_list = []
    domain_counts = defaultdict(int)
    total_tokens = 0

    pbar = tqdm(total=max_tokens, unit='tok', unit_scale=True,
                desc=f'Tokenizing {split_name}', dynamic_ncols=True)
    try:
        for example in hf_dataset:
            if total_tokens >= max_tokens:
                break

            text = example.get('text', '')
            if not text or not text.strip():
                continue

            # Resolve domain
            domain_name = example.get('meta', {}).get('redpajama_set_name', '')
            domain_id = DOMAIN_TO_ID.get(domain_name, None)
            if domain_id is None:
                continue  # skip unknown domains

            # Tokenize without special tokens
            token_ids = tokenizer(
                text, add_special_tokens=False, truncation=False
            )['input_ids']

            if len(token_ids) < max_seq_len:
                continue  # skip documents shorter than one chunk

            # Split into fixed-length chunks (drop remainder)
            for start in range(0, len(token_ids) - max_seq_len + 1, max_seq_len):
                if total_tokens >= max_tokens:
                    break
                chunk = token_ids[start:start + max_seq_len]
                input_ids_list.append(torch.tensor(chunk, dtype=torch.long))
                domain_ids_list.append(domain_id)
                domain_counts[domain_id] += 1
                total_tokens += max_seq_len
                pbar.update(max_seq_len)
    finally:
        pbar.close()

    # Print statistics
    total_seqs = len(input_ids_list)
    print(f"  Total sequences: {total_seqs:,}  ({total_tokens:,} tokens)")
    for did in range(NUM_DOMAINS):
        count = domain_counts.get(did, 0)
        pct = 100.0 * count / total_seqs if total_seqs > 0 else 0.0
        print(f"    {DOMAIN_NAMES[did]}: {count:,} ({pct:.1f}%)")

    return input_ids_list, domain_ids_list


# ── Download & Validation ───────────────────────────────────────────────────

def ensure_downloaded(data_dir='./data_cache/slimpajama'):
    """Download SlimPajama-6B if not already cached. Returns True if ready."""
    from datasets import load_dataset
    print("Checking SlimPajama-6B (~24GB)...")
    # load_dataset auto-downloads and caches to data_dir
    ds = load_dataset("DKYoon/SlimPajama-6B", split="train",
                      cache_dir=data_dir)
    print(f"  SlimPajama-6B downloaded: {len(ds):,} train documents")
    return True


def validate_format(data_dir='./data_cache/slimpajama', num_samples=5):
    """Validate SlimPajama dataset format matches expectations.

    Checks: 'text' column exists, 'meta.redpajama_set_name' exists and
    maps to one of 7 known domains.

    Returns: (ok: bool, message: str)
    """
    from datasets import load_dataset

    for split_name in ['train', 'validation']:
        ds = load_dataset("DKYoon/SlimPajama-6B", split=split_name,
                          cache_dir=data_dir)
        checked = 0
        for example in ds:
            # Check 'text' field
            if 'text' not in example:
                return False, f"{split_name}: missing 'text' field"
            if not isinstance(example['text'], str):
                return False, f"{split_name}: 'text' is not a string"

            # Check 'meta.redpajama_set_name' field
            meta = example.get('meta', {})
            if not meta:
                return False, f"{split_name}: missing 'meta' field"
            domain_name = meta.get('redpajama_set_name', '')
            if domain_name not in DOMAIN_TO_ID:
                return False, (f"{split_name}: unknown domain '{domain_name}', "
                               f"expected one of {DOMAIN_NAMES}")

            checked += 1
            if checked >= num_samples:
                break

    return True, f"SlimPajama OK: 'text' + 7 domains verified ({num_samples} samples/split)"


# ── Public API ───────────────────────────────────────────────────────────────

def get_slimpajama_dataloaders(data_dir='./data_cache/slimpajama',
                               batch_size=32,
                               max_seq_len=1024,
                               num_workers=4,
                               max_train_tokens=500_000_000,
                               max_val_tokens=5_000_000,
                               cluster_label_path_train=None,
                               cluster_label_path_val=None):
    """Build SlimPajama train/val dataloaders with domain labels.

    Args:
        data_dir: HuggingFace cache directory
        batch_size: batch size for dataloaders
        max_seq_len: tokens per sequence
        num_workers: data loading workers
        max_train_tokens: cap on total training tokens
        max_val_tokens: cap on total validation tokens
        cluster_label_path_train: optional path to a cluster-label .pt
            (from data.cluster_chunks). When given, the train dataset's
            per-chunk category labels are replaced with semantic cluster IDs
            instead of SlimPajama's source-level `domain_ids`. Chunk count
            must match the tokenized train cache.
        cluster_label_path_val: same as above, for val. Both paths should
            typically be set together (train fit + val predict share cluster
            defs) since the trainer uses category IDs for both train and eval
            masks.

    Returns:
        train_loader, val_loader, vocab_size (50257 for GPT-2)
    """
    from datasets import load_dataset

    print("Loading SlimPajama-6B from local cache...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size  # 50257

    # Tokenizer version fingerprint — prevents silent cache reuse after
    # `transformers` upgrade changes GPT-2 vocab/merges. Hashes the full
    # vocab+merges, truncated to 10 hex chars (collision probability negligible).
    import hashlib, json
    _vocab_repr = json.dumps(
        tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False).encode('utf-8')
    _tokenizer_hash = hashlib.sha256(_vocab_repr).hexdigest()[:10]

    def _load_or_tokenize(split_name, max_tokens):
        """Load pre-tokenized chunks from disk, else tokenize and cache."""
        cache_path = os.path.join(
            data_dir,
            f'tokenized_{split_name}_seq{max_seq_len}_tok{max_tokens}'
            f'_vocab{_tokenizer_hash}.pt'
        )
        # Backward-compat: earlier revisions used `_tok{hash}` instead of
        # `_vocab{hash}` as the suffix. If the new-name cache is missing but
        # the old-name one exists, accept it to avoid forcing a 20-40 min
        # re-tokenize on an upgrade.
        if not os.path.exists(cache_path):
            legacy_path = os.path.join(
                data_dir,
                f'tokenized_{split_name}_seq{max_seq_len}_tok{max_tokens}'
                f'_tok{_tokenizer_hash}.pt'
            )
            if os.path.exists(legacy_path):
                print(f"  [compat] Using legacy cache {os.path.basename(legacy_path)}")
                cache_path = legacy_path
        if os.path.exists(cache_path):
            print(f"  Loading cached tokenized {split_name} from {cache_path}...")
            blob = torch.load(cache_path, weights_only=True)
            ids = list(blob['input_ids'].unbind(0))  # (N, seq_len) -> list of (seq_len,)
            domains = blob['domain_ids'].tolist()
            print(f"  Loaded {len(ids):,} sequences ({len(ids) * max_seq_len:,} tokens)")
            return ids, domains

        # Cache miss: tokenize from raw dataset
        ds = load_dataset(
            "DKYoon/SlimPajama-6B",
            split='train' if split_name == 'train' else 'validation',
            cache_dir=data_dir,
        )
        print(f"  Raw {split_name} documents: {len(ds):,}")
        ids, domains = _process_split(ds, tokenizer, max_seq_len, max_tokens, split_name=split_name)

        # Save cache
        print(f"  Saving tokenized {split_name} to {cache_path}...")
        os.makedirs(data_dir, exist_ok=True)
        torch.save({
            'input_ids': torch.stack(ids),  # (N, seq_len)
            'domain_ids': torch.tensor(domains, dtype=torch.long),
        }, cache_path)
        return ids, domains

    train_ids, train_domains = _load_or_tokenize('train', max_train_tokens)
    val_ids, val_domains = _load_or_tokenize('val', max_val_tokens)

    # Phase W: optionally replace source domain_ids with semantic cluster IDs.
    # Trainer uses these as category indices for per-category routing masks
    # at both train and eval time, so both splits need their labels swapped
    # together to keep the mask definitions consistent.
    def _apply_cluster_labels(labels_list, cluster_path, split_name):
        cblob = load_cluster_labels(cluster_path)
        cids = cblob['cluster_ids']
        if int(len(cids)) != len(labels_list):
            raise ValueError(
                f"{split_name} cluster label count {len(cids)} does not match "
                f"tokenized chunk count {len(labels_list)} "
                f"(cluster file: {cluster_path})")
        print(f"  [Phase W] {split_name}: replacing source domain_ids with "
              f"cluster IDs from {os.path.basename(cluster_path)} "
              f"(n_clusters={cblob['n_clusters']}, embedder={cblob.get('embedder')})")
        return cids.tolist()

    if cluster_label_path_train is not None:
        train_domains = _apply_cluster_labels(
            train_domains, cluster_label_path_train, 'train')
    if cluster_label_path_val is not None:
        val_domains = _apply_cluster_labels(
            val_domains, cluster_label_path_val, 'val')

    # Build datasets and loaders
    train_dataset = SlimPajamaChunkedDataset(train_ids, train_domains)
    val_dataset = SlimPajamaChunkedDataset(val_ids, val_domains)

    def _worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2**32
        import numpy as _np; _np.random.seed(seed + worker_id)
        import random as _r; _r.seed(seed + worker_id)

    persistent = num_workers > 0
    g = torch.Generator()
    g.manual_seed(torch.initial_seed())
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent,
        worker_init_fn=_worker_init_fn, generator=g)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent,
        worker_init_fn=_worker_init_fn, generator=g)

    print(f"SlimPajama ready: {len(train_dataset):,} train seqs, "
          f"{len(val_dataset):,} val seqs, vocab={vocab_size}")

    return train_loader, val_loader, vocab_size

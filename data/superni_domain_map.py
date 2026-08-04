"""Build and cache the K=20 SuperNI domain → cluster_id mapping.

Wang 2022 SuperNI's `Domains` field is the semantic-content axis (analog to
CIFAR's 20 superclasses or SlimPajama's 7 URL-sourced domains). Of the ~72
unique normalized domains in the English default split (756 train tasks,
119 test tasks), we retain the top-19 most frequent + a `miscellaneous`
bucket — same frequency-cutoff rule BREEDS (Santurkar 2021) uses on ImageNet.

Normalization: split on ` -> ` and take the root segment, so
"Social Media -> Twitter" and "Social Media" both map to "Social Media".

Mapping artefacts (deterministic, versioned):
  data_cache/lora/superni_domain_map_K20.json   (cluster_id lookup table)
  data_cache/lora/superni_task_to_cluster.json  (task_id → cluster_id)

Usage:
    from data.superni_domain_map import (build_or_load_domain_map,
                                          task_to_cluster_id)
    m = build_or_load_domain_map(tasks_dir='./data_cache/lora/natural-instructions')
    cid = task_to_cluster_id(task_record, m)

For test-time inputs without a Domains field, use
`assign_by_nearest_centroid` with a BGE-large embedding of the query text.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

NUM_CLUSTERS_K20 = 20  # Default for LoRA track; K used in ablation / main table.

MISC_LABEL = 'miscellaneous'


def normalize_domain(d: str) -> str:
    """Collapse SuperNI's hierarchical domain path to its root segment.

    "Commonsense -> Concepts and Relations -> Social Commonsense" → "Commonsense"
    "Social Media -> Twitter"                                     → "Social Media"
    "Mathematics"                                                  → "Mathematics"
    """
    if d is None:
        return ''
    head = d.split(' -> ')[0].strip()
    return head


def _iter_task_jsons(tasks_dir: str):
    """Yield (task_id, task_dict) for every task JSON in tasks_dir.

    tasks_dir is the `tasks/` folder of the allenai/natural-instructions
    repo (cloned locally for offline builds).
    """
    if not os.path.isdir(tasks_dir):
        raise FileNotFoundError(
            f"SuperNI tasks/ folder not found: {tasks_dir}. "
            "Clone https://github.com/allenai/natural-instructions and point "
            "tasks_dir at its tasks/ subdirectory, or pre-fetch JSONs to a "
            "local mirror.")
    for fname in sorted(os.listdir(tasks_dir)):
        if not fname.endswith('.json'):
            continue
        task_id = fname[:-len('.json')]
        path = os.path.join(tasks_dir, fname)
        try:
            with open(path) as f:
                yield task_id, json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: skipping {fname}: {e}")


def _load_split_list(splits_dir: str, split: str) -> List[str]:
    """Load train_tasks.txt or test_tasks.txt (one task_id per line, # comments)."""
    path = os.path.join(splits_dir, f'{split}_tasks.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"SuperNI split file not found: {path}")
    tasks = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith('#'):
                tasks.append(ln)
    return tasks


def build_domain_map(tasks_dir: str, splits_dir: str,
                      K: int = NUM_CLUSTERS_K20) -> Dict:
    """Build the K-cluster mapping from train-split domain frequencies.

    Procedure:
      1. Load `train_tasks.txt` (Wang 2022 default English split, 756 tasks).
      2. For each train task, take the first element of `Domains` (when multi-
         label; ~1.25 labels/task avg) and normalize via normalize_domain().
      3. Count normalized-domain frequencies, sort descending.
      4. Keep top (K-1) normalized domains → cluster_id 0..K-2 in frequency
         order; all remaining normalized domains → cluster_id K-1 (misc).

    Returns a dict with:
      - `K`: int
      - `top_domains`: list of (domain, train_count) for id 0..K-2
      - `misc_id`: K - 1
      - `domain_to_id`: full Dict[normalized_domain, int] including top+tail
      - `train_tasks`: list of train task_ids
      - `test_tasks`: list of test task_ids
      - `source_field`: 'Domains' (Wang 2022)
      - `normalization`: 'split on " -> " take root'
      - `cutoff_rule`: f'top-{K-1} + miscellaneous'

    Idempotent: given the same tasks_dir + splits_dir, always produces the same
    mapping (frequency sort is deterministic; ties broken by sorted() order).
    """
    if K < 2:
        raise ValueError(f"K must be ≥ 2, got {K}")

    train_ids = _load_split_list(splits_dir, 'train')
    test_ids = _load_split_list(splits_dir, 'test')
    train_set = set(train_ids)

    first_norm_domain: Dict[str, str] = {}
    for task_id, d in _iter_task_jsons(tasks_dir):
        raw_domains = d.get('Domains', []) or []
        if not raw_domains:
            first_norm_domain[task_id] = ''
            continue
        first_norm_domain[task_id] = normalize_domain(raw_domains[0])

    # Frequency count on train split only (test tasks may have domains not
    # seen in train; those fall into misc at assignment time).
    counter = Counter()
    for tid in train_ids:
        nd = first_norm_domain.get(tid, '')
        if nd:
            counter[nd] += 1

    # Deterministic tie-breaking: (-count, name) sorts by descending count, then alpha.
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    top_k_minus_1 = ranked[:K - 1]
    top_domain_set = {d for d, _ in top_k_minus_1}

    domain_to_id: Dict[str, int] = {}
    for i, (d, _) in enumerate(top_k_minus_1):
        domain_to_id[d] = i
    misc_id = K - 1
    # Everything else in training that wasn't in top (K-1) → misc.
    for d, _ in ranked[K - 1:]:
        domain_to_id[d] = misc_id

    # Also record task_id → cluster_id for train + test (test novel domains
    # automatically route to misc via get_cluster_id fallback).
    task_to_cluster: Dict[str, int] = {}
    for tid, nd in first_norm_domain.items():
        task_to_cluster[tid] = domain_to_id.get(nd, misc_id)

    return {
        'K': K,
        'top_domains': top_k_minus_1,
        'misc_id': misc_id,
        'domain_to_id': domain_to_id,
        'task_to_cluster': task_to_cluster,
        'train_tasks': train_ids,
        'test_tasks': test_ids,
        'source_field': 'Domains',
        'normalization': 'split on " -> ", take root segment',
        'cutoff_rule': f'top-{K-1} most frequent in train + "{MISC_LABEL}"',
        'dataset_reference': 'Wang et al. 2022 SuperNI (arxiv 2204.07705)',
        'cutoff_reference': ('BREEDS-analog frequency cutoff '
                             '(Santurkar et al. 2021, arxiv 2008.04859)'),
    }


def save_domain_map(mapping: Dict, output_path: str) -> None:
    """Save the mapping as JSON. Atomic write."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp = output_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(mapping, f, indent=2, sort_keys=False)
    os.replace(tmp, output_path)


def load_domain_map(path: str) -> Dict:
    """Load a previously saved mapping."""
    with open(path) as f:
        return json.load(f)


def build_or_load_domain_map(tasks_dir: str, splits_dir: str,
                              cache_dir: str = './data_cache/lora',
                              K: int = NUM_CLUSTERS_K20,
                              force_rebuild: bool = False) -> Dict:
    """Idempotent: load from cache if present, else build + cache."""
    cache_path = os.path.join(cache_dir, f'superni_domain_map_K{K}.json')
    if os.path.exists(cache_path) and not force_rebuild:
        return load_domain_map(cache_path)
    m = build_domain_map(tasks_dir, splits_dir, K=K)
    save_domain_map(m, cache_path)
    return m


def get_cluster_id(task_record: Dict, mapping: Dict) -> int:
    """Resolve cluster_id for an arbitrary task record (train / test / novel).

    Uses the task's first Domain (Wang 2022 field). If the normalized domain
    is not in `mapping['domain_to_id']` (e.g., a test-split domain never
    observed in train), returns `mapping['misc_id']`.
    """
    raw_domains = task_record.get('Domains', []) or []
    if not raw_domains:
        return mapping['misc_id']
    nd = normalize_domain(raw_domains[0])
    return mapping['domain_to_id'].get(nd, mapping['misc_id'])


def task_to_cluster_id(task_id: str, mapping: Dict) -> int:
    """Cached lookup for a task that was in the training split."""
    tc = mapping.get('task_to_cluster', {})
    if task_id in tc:
        return tc[task_id]
    return mapping['misc_id']


# ── Test-time routing for queries without a Domains label ──────────────────

def _bge_embed(texts: Sequence[str], model_name: str = 'BAAI/bge-large-en-v1.5',
                device: str = 'cpu', batch_size: int = 32):
    """Embed a list of strings with BGE-large. Imports deferred so this module
    is importable in bare torch envs (Batch 1 smoke)."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device=device)
    return m.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)


def build_bge_centroids(task_definitions: Dict[str, str], mapping: Dict,
                         device: str = 'cpu'):
    """Compute one BGE-large centroid per cluster_id, using train tasks'
    Definitions grouped by their cluster_id.

    Args:
        task_definitions: dict {task_id: definition_text} for train tasks.
        mapping:          output of build_or_load_domain_map.
        device:           'cuda' / 'cpu'.

    Returns:
        centroids: numpy array of shape (K, D_bge), L2-normalized.
                    centroids[k] is mean BGE embedding of train tasks with
                    cluster_id = k.
    """
    import numpy as np
    ids = [tid for tid in task_definitions if tid in mapping['task_to_cluster']]
    if not ids:
        raise ValueError("No train tasks overlap with mapping['task_to_cluster']")
    texts = [task_definitions[tid] for tid in ids]
    clusters = [mapping['task_to_cluster'][tid] for tid in ids]
    emb = _bge_embed(texts, device=device)

    K = mapping['K']
    D = emb.shape[1]
    centroids = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        mask = np.array([c == k for c in clusters])
        if mask.any():
            centroids[k] = emb[mask].mean(axis=0)
    # L2-normalize (consistent with BGE's normalize_embeddings=True input)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid div-by-zero for empty cluster
    centroids = centroids / norms
    return centroids


def assign_by_nearest_centroid(query_texts: Sequence[str], centroids,
                                 device: str = 'cpu'):
    """BGE-embed each query, return argmax cosine-similarity cluster_id.

    Used for held-out eval where tasks have no Wang 2022 Domains label.
    """
    import numpy as np
    emb = _bge_embed(query_texts, device=device)  # (N, D)
    # Both arrays are L2-normalized → inner product = cosine similarity.
    sims = emb @ centroids.T  # (N, K)
    return sims.argmax(axis=1)

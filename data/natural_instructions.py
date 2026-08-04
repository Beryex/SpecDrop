"""SuperNaturalInstructions v2 (Wang et al. 2022) data loader for LoRA track.

Loads the English default split (756 train + 119 test tasks) directly from the
allenai/natural-instructions GitHub repo's `tasks/` JSONs, applies Wang 2022's
Tk-Instruct prompt format (Definition + 2 Positive Examples + Input), tokenizes
with the base LM's tokenizer, and yields 4-tuple batches compatible with
training/trainer_lora.py:

    (input_ids, labels, attention_mask, cluster_id)

where `cluster_id` is derived from the K=20 top-19+misc mapping built in
data.superni_domain_map. This mirrors CIFAR's (img, fine, coarse, idx) and
ImageNet's (img, fine, coarse, idx) 4-tuple convention.

Labels are -100 for instruction/input tokens so the CE loss only applies to
the response tokens (standard instruction-tuning practice, Wang 2022 Sec 5).

Usage:
    from data.natural_instructions import get_superni_dataloaders
    train_loader, eval_loader, K = get_superni_dataloaders(
        data_root='./data_cache/lora/natural-instructions',
        tokenizer=AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B'),
        batch_size=16, max_seq_len=2048, instances_per_task=100, seed=42)

Only the allenai/natural-instructions GitHub repo's raw task JSONs are
supported (not the HF `Muennighoff/natural-instructions` flattened parquet —
that version strips the Categories / Domains / Reasoning fields we need).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .superni_domain_map import (NUM_CLUSTERS_K20, build_or_load_domain_map,
                                  get_cluster_id)

# Wang 2022 Tk-Instruct default: 2 positive examples, 0 negative, with Definition.
DEFAULT_NUM_POS_EXAMPLES = 2
DEFAULT_NUM_NEG_EXAMPLES = 0


def _tasks_dir(data_root: str) -> str:
    return os.path.join(data_root, 'tasks')


def _splits_dir(data_root: str) -> str:
    # allenai repo layout: splits/default/{train,test}_tasks.txt
    return os.path.join(data_root, 'splits', 'default')


def _read_split(data_root: str, split: str) -> List[str]:
    path = os.path.join(_splits_dir(data_root), f'{split}_tasks.txt')
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]


def _read_task(data_root: str, task_id: str) -> Dict:
    with open(os.path.join(_tasks_dir(data_root), f'{task_id}.json')) as f:
        return json.load(f)


# ── Wang 2022 Tk-Instruct prompt format ─────────────────────────────────────

def _format_prompt(task: Dict, input_text: str,
                   num_pos: int = DEFAULT_NUM_POS_EXAMPLES,
                   num_neg: int = DEFAULT_NUM_NEG_EXAMPLES) -> str:
    """Construct the prompt portion (everything BEFORE the target output).

    Wang 2022 Tk-Instruct default:
      Definition: <task.Definition>
      Positive Example 1 — Input: <ex.input>  Output: <ex.output>
      Positive Example 2 — Input: <ex.input>  Output: <ex.output>
      [Negative Examples optional — 0 by default]
      Now complete the following example —
      Input: <input_text>
      Output:

    The tokenizer then appends the target output; see `_pack_ids`.
    """
    parts = []
    definition = task.get('Definition', [])
    if isinstance(definition, list):
        definition = ' '.join(x for x in definition if x)
    parts.append(f'Definition: {definition.strip()}')

    pos = task.get('Positive Examples', []) or []
    for i, ex in enumerate(pos[:num_pos]):
        inp = ex.get('input', '').strip()
        out = ex.get('output', '').strip()
        parts.append(
            f'Positive Example {i+1} —\nInput: {inp}\nOutput: {out}')

    neg = task.get('Negative Examples', []) or []
    for i, ex in enumerate(neg[:num_neg]):
        inp = ex.get('input', '').strip()
        out = ex.get('output', '').strip()
        parts.append(
            f'Negative Example {i+1} —\nInput: {inp}\nOutput: {out}')

    parts.append(f'Now complete the following example —\nInput: {input_text.strip()}\nOutput:')
    return '\n\n'.join(parts)


# Wang 2022 Tk-Instruct canonical: max_source_length=1024, max_target_length=128.
# Separate budgets are necessary because some SuperNI test targets are full
# paragraphs or JSON blobs that can exceed any realistic max_seq_len; with a
# single combined budget, the old "truncate prompt only" branch fails the
# assertion when len(target) alone > max_seq_len (latent at seq=2048, hit
# immediately at seq=1024).
_MAX_TARGET_LEN = 128


def _pack_ids(prompt: str, target: str, tokenizer, max_seq_len: int,
              max_target_len: int = _MAX_TARGET_LEN) -> Dict:
    """Tokenize prompt+target into a single sequence, mask prompt from loss.

    Returns dict with:
        input_ids:      (L,) int64, prompt ⊕ ' ' ⊕ target ⊕ eos
        labels:         (L,) int64, -100 for prompt tokens (loss-ignored),
                        target tokens for response tokens, eos_id for final.
        attention_mask: (L,) int64, ones up to real length, zeros if padded.

    Two-stage truncation (Wang 2022 Tk-Instruct canonical):
      1. If target > max_target_len tokens, RIGHT-truncate target preserving
         the trailing EOS (critical for training signal).
      2. If prompt + target > max_seq_len, LEFT-truncate prompt (target stays
         intact). Matches HF `Trainer` instruction-tuning convention.
    """
    assert max_target_len < max_seq_len, (
        f"max_target_len={max_target_len} must be < max_seq_len={max_seq_len} "
        f"to leave room for the prompt.")

    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    target_ids = tokenizer(' ' + target.strip(), add_special_tokens=False)['input_ids']

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        # Fall back to the pad or bos id — caller should set one on the tokenizer.
        eos_id = tokenizer.pad_token_id or tokenizer.bos_token_id or 0
    target_ids = target_ids + [eos_id]

    # ── Stage 1: target-length cap (Wang 2022: max_target_length=128) ─────
    # Keep the first (max_target_len - 1) tokens + final EOS, so the training
    # signal always ends on EOS. Dropping EOS would teach the model not to
    # stop — catastrophic for generation-time decoding.
    if len(target_ids) > max_target_len:
        target_ids = target_ids[:max_target_len - 1] + [eos_id]

    # ── Stage 2: combined-length cap via LEFT-truncation of prompt ────────
    overflow = len(prompt_ids) + len(target_ids) - max_seq_len
    if overflow > 0:
        prompt_ids = prompt_ids[overflow:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + list(target_ids)
    attention_mask = [1] * len(input_ids)

    # Trust-but-verify (truncation math)
    assert len(input_ids) == len(labels) == len(attention_mask) <= max_seq_len

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }


# ── Pre-tokenize + on-disk cache (mirrors data/slimpajama.py convention) ────

def _tokenizer_vocab_hash(tokenizer) -> str:
    """Short hash of the tokenizer's vocab so we can detect tokenizer-mismatch
    between a cache and the current run. Uses the first 10 hex chars of the
    SHA-256 of the sorted vocab items — stable across sessions."""
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return 'unknown'
    canonical = json.dumps(sorted(vocab.items()), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:10]


def _cache_filename(split: str, instances_per_task: int, subset_frac: float,
                     max_seq_len: int, seed: int,
                     num_pos: int, num_neg: int, vocab_hash: str) -> str:
    # tgt{_MAX_TARGET_LEN} embedded so any future change to the target-length
    # cap invalidates old caches (old ones were single-budget, tgt-implicit).
    return (f'superni_{split}_ipt{instances_per_task}_frac{subset_frac}_'
            f'seq{max_seq_len}_tgt{_MAX_TARGET_LEN}_seed{seed}_'
            f'pos{num_pos}_neg{num_neg}_vocab{vocab_hash}.pt')


def find_superni_cache(cache_dir: str, split: str, tokenizer,
                        instances_per_task: int, subset_frac: float,
                        max_seq_len: int, seed: int,
                        num_pos: int = DEFAULT_NUM_POS_EXAMPLES,
                        num_neg: int = DEFAULT_NUM_NEG_EXAMPLES) -> Optional[str]:
    """Return the cache file path if present (matches tokenizer vocab), else None."""
    vocab_hash = _tokenizer_vocab_hash(tokenizer)
    fname = _cache_filename(split, instances_per_task, subset_frac,
                             max_seq_len, seed, num_pos, num_neg, vocab_hash)
    path = os.path.join(cache_dir, fname)
    return path if os.path.exists(path) else None


def build_superni_cache(data_root: str, split: str, tokenizer, mapping: Dict,
                         cache_dir: str,
                         instances_per_task: int = 100,
                         subset_frac: float = 1.0,
                         max_seq_len: int = 2048,
                         seed: int = 42,
                         num_pos_examples: int = DEFAULT_NUM_POS_EXAMPLES,
                         num_neg_examples: int = DEFAULT_NUM_NEG_EXAMPLES,
                         force_rebuild: bool = False,
                         verbose: bool = True) -> str:
    """Tokenize the requested split once, cache to disk, return the cache path.

    Mirrors data/slimpajama.py's cache convention: tokenize at build time,
    save a dict of variable-length lists to a .pt file, and skip rebuild if
    a compatible cache (same tokenizer + knobs) already exists.

    Args:
        data_root:          local natural-instructions clone.
        split:              'train' | 'test' (uses splits/default/{split}_tasks.txt).
        tokenizer:          HF AutoTokenizer for the base model.
        mapping:            K=20 domain map from build_or_load_domain_map.
        cache_dir:          where to write/read the .pt cache.
        instances_per_task: Wang 2022 convention 100; also capped by task size.
        subset_frac:        stratified per-task subsample; 1.0 = full.
        max_seq_len:        truncation budget.
        seed:               RNG seed for instance sampling (shared across ablation).
        num_pos_examples / num_neg_examples: Wang 2022 defaults 2 / 0.

    Returns: absolute path to the cache file.
    """
    vocab_hash = _tokenizer_vocab_hash(tokenizer)
    fname = _cache_filename(split, instances_per_task, subset_frac,
                             max_seq_len, seed, num_pos_examples,
                             num_neg_examples, vocab_hash)
    cache_path = os.path.join(cache_dir, fname)
    if os.path.exists(cache_path) and not force_rebuild:
        if verbose:
            print(f'[superni_cache] hit: {cache_path}')
        return cache_path

    os.makedirs(cache_dir, exist_ok=True)
    if verbose:
        print(f'[superni_cache] building {split} cache → {cache_path}')

    task_ids = _read_split(data_root, split)
    rng = random.Random(seed)

    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []
    attn_list: List[List[int]] = []
    cluster_ids: List[int] = []
    task_ids_out: List[str] = []
    skipped = 0

    it = tqdm(task_ids, desc=f'tokenize {split}', disable=not verbose)
    for tid in it:
        try:
            task = _read_task(data_root, tid)
        except FileNotFoundError:
            skipped += 1
            continue
        instances = task.get('Instances', []) or []
        if not instances:
            continue
        n_take = min(instances_per_task, len(instances))
        idxs = rng.sample(range(len(instances)), n_take)
        if subset_frac < 1.0:
            n_keep = max(1, int(round(len(idxs) * subset_frac)))
            idxs = idxs[:n_keep]
        cid = get_cluster_id(task, mapping)
        for i in idxs:
            instance = instances[i]
            input_text = instance.get('input', '')
            output_list = instance.get('output', [])
            if isinstance(output_list, list):
                target = output_list[0] if output_list else ''
            else:
                target = str(output_list)
            prompt = _format_prompt(task, input_text,
                                    num_pos=num_pos_examples,
                                    num_neg=num_neg_examples)
            pack = _pack_ids(prompt, target, tokenizer, max_seq_len)
            input_ids_list.append(pack['input_ids'])
            labels_list.append(pack['labels'])
            attn_list.append(pack['attention_mask'])
            cluster_ids.append(cid)
            task_ids_out.append(tid)

    blob = {
        'input_ids_list': input_ids_list,
        'labels_list': labels_list,
        'attention_mask_list': attn_list,
        'cluster_ids': torch.tensor(cluster_ids, dtype=torch.long),
        'task_ids': task_ids_out,
        'metadata': {
            'split': split,
            'instances_per_task': instances_per_task,
            'subset_frac': subset_frac,
            'max_seq_len': max_seq_len,
            'seed': seed,
            'num_pos_examples': num_pos_examples,
            'num_neg_examples': num_neg_examples,
            'vocab_hash': vocab_hash,
            'tokenizer_name': getattr(tokenizer, 'name_or_path', ''),
            'num_tasks_seen': len(task_ids) - skipped,
            'num_examples': len(cluster_ids),
            'K': mapping['K'],
        },
    }
    # Atomic write
    tmp = cache_path + '.tmp'
    torch.save(blob, tmp)
    os.replace(tmp, cache_path)
    if verbose:
        print(f'[superni_cache] wrote {len(cluster_ids):,} examples '
              f'from {blob["metadata"]["num_tasks_seen"]} tasks → {cache_path}')
    return cache_path


def load_superni_cache(cache_path: str) -> Dict:
    """Load a previously built cache. Raises FileNotFoundError if absent."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"SuperNI cache not found: {cache_path}. "
            "Run build_superni_cache(...) to produce it "
            "(see rtx5090_0d_smoke_lora.sh prep phase).")
    return torch.load(cache_path, weights_only=False)


def prebuild_all_caches(data_root: str, tokenizer, mapping: Dict,
                         cache_dir: str,
                         max_seq_len: int = 2048,
                         instances_per_task: int = 100,
                         seed: int = 42,
                         subset_fracs_train=(1.0, 0.2),
                         num_pos_examples: int = DEFAULT_NUM_POS_EXAMPLES,
                         num_neg_examples: int = DEFAULT_NUM_NEG_EXAMPLES,
                         verbose: bool = True) -> Dict[str, str]:
    """Build every cache the LoRA track will need in one shot.

    Default: train @ 100%, train @ 20% (for ablation), and test @ 100%.

    Returns dict {f'{split}_frac{subset_frac}': cache_path}.
    """
    out = {}
    for frac in subset_fracs_train:
        path = build_superni_cache(
            data_root=data_root, split='train', tokenizer=tokenizer,
            mapping=mapping, cache_dir=cache_dir,
            instances_per_task=instances_per_task, subset_frac=frac,
            max_seq_len=max_seq_len, seed=seed,
            num_pos_examples=num_pos_examples, num_neg_examples=num_neg_examples,
            verbose=verbose)
        out[f'train_frac{frac}'] = path
    # Eval split always full; subset_frac=1.0.
    out['test_frac1.0'] = build_superni_cache(
        data_root=data_root, split='test', tokenizer=tokenizer,
        mapping=mapping, cache_dir=cache_dir,
        instances_per_task=instances_per_task, subset_frac=1.0,
        max_seq_len=max_seq_len, seed=seed,
        num_pos_examples=num_pos_examples, num_neg_examples=num_neg_examples,
        verbose=verbose)
    return out


# ── Dataset classes ─────────────────────────────────────────────────────────

class SuperNIDataset(Dataset):
    """Returns a flat list of (task_id, instance_idx) → tokenized dicts.

    Args:
        data_root:          local path to cloned allenai/natural-instructions.
        task_ids:           list of task_ids to include (e.g., train or test split).
        tokenizer:          HF tokenizer (e.g., Llama-3.2-1B's).
        mapping:            output of build_or_load_domain_map (K=20).
        instances_per_task: max instances/task to sample (Wang 2022: 100).
                            Passes fewer when a task has fewer instances.
        max_seq_len:        truncation budget.
        seed:               RNG seed for instance sampling.
        subset_frac:        (0, 1] fraction of instances_per_task to keep per task
                            (for ablation 20% subset; stratified per task).
                            Default 1.0 = keep all.
        num_pos_examples:   Wang 2022 default 2.
        num_neg_examples:   Wang 2022 default 0.
    """

    def __init__(self, data_root: str, task_ids: Sequence[str], tokenizer,
                  mapping: Dict, instances_per_task: int = 100,
                  max_seq_len: int = 2048, seed: int = 42,
                  subset_frac: float = 1.0,
                  num_pos_examples: int = DEFAULT_NUM_POS_EXAMPLES,
                  num_neg_examples: int = DEFAULT_NUM_NEG_EXAMPLES,
                  cache: Optional[Dict] = None):
        """
        If `cache` is provided (a dict returned by load_superni_cache), we
        skip on-the-fly tokenization and serve pre-tokenized tensors directly.
        Otherwise fall back to the original on-the-fly path (kept for
        backward-compat + synthetic test fixtures).
        """
        super().__init__()
        if not 0 < subset_frac <= 1.0:
            raise ValueError(f"subset_frac must be in (0, 1], got {subset_frac}")
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.mapping = mapping
        self.instances_per_task = instances_per_task
        self.max_seq_len = max_seq_len
        self.subset_frac = subset_frac
        self.num_pos_examples = num_pos_examples
        self.num_neg_examples = num_neg_examples
        self._cache = cache
        self._records: List[Tuple[str, int]] = []
        self._task_cache: Dict[str, Dict] = {}

        if cache is not None:
            # Cache-backed mode: just surface lengths and rely on __getitem__
            # to pull from the cache lists. No JSON walk.
            n = len(cache['input_ids_list'])
            self._records = [('__cache__', i) for i in range(n)]
            # Sanity: verify knobs match (otherwise silent mis-alignment).
            md = cache.get('metadata', {})
            for key, got, want in (
                ('max_seq_len', md.get('max_seq_len'), max_seq_len),
                ('subset_frac', md.get('subset_frac'), subset_frac),
                ('instances_per_task', md.get('instances_per_task'), instances_per_task),
                ('seed', md.get('seed'), seed),
                ('num_pos_examples', md.get('num_pos_examples'), num_pos_examples),
                ('num_neg_examples', md.get('num_neg_examples'), num_neg_examples),
            ):
                if got is not None and got != want:
                    raise ValueError(
                        f"SuperNI cache knob mismatch: {key}={got} in cache, "
                        f"but {want} requested. Rebuild cache or align knobs.")
            return

        # Legacy on-the-fly path — walks JSONs, tokenizes per __getitem__.
        rng = random.Random(seed)
        for tid in task_ids:
            try:
                task = _read_task(data_root, tid)
            except FileNotFoundError:
                print(f"  WARN: skipping missing task {tid}")
                continue
            self._task_cache[tid] = task
            instances = task.get('Instances', []) or []
            if not instances:
                continue
            n_available = len(instances)
            n_take = min(instances_per_task, n_available)
            idxs = rng.sample(range(n_available), n_take)
            # Stratified subset: apply subset_frac per task so every task stays
            # represented (otherwise rare tasks could drop out entirely).
            if subset_frac < 1.0:
                n_keep = max(1, int(round(len(idxs) * subset_frac)))
                idxs = idxs[:n_keep]
            for i in idxs:
                self._records.append((tid, i))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict:
        # Cache-backed path: pull pre-tokenized lists + cluster_id directly.
        if self._cache is not None:
            c = self._cache
            return {
                'input_ids': c['input_ids_list'][idx],
                'labels': c['labels_list'][idx],
                'attention_mask': c['attention_mask_list'][idx],
                'cluster_id': int(c['cluster_ids'][idx]),
                'task_id': c['task_ids'][idx],
            }

        # Legacy on-the-fly path.
        task_id, inst_idx = self._records[idx]
        task = self._task_cache[task_id]
        instance = task['Instances'][inst_idx]
        input_text = instance.get('input', '')
        output_list = instance.get('output', [])
        if isinstance(output_list, list):
            target = output_list[0] if output_list else ''
        else:
            target = str(output_list)

        prompt = _format_prompt(task, input_text,
                                 num_pos=self.num_pos_examples,
                                 num_neg=self.num_neg_examples)
        pack = _pack_ids(prompt, target, self.tokenizer, self.max_seq_len)

        cluster_id = get_cluster_id(task, self.mapping)
        pack['cluster_id'] = cluster_id
        pack['task_id'] = task_id
        return pack


# ── Collate ─────────────────────────────────────────────────────────────────

def _collate(batch: List[Dict], pad_token_id: int,
              target_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Right-pad to `target_len` if given, else to longest sequence in batch.

    Fixed `target_len` (= max_seq_len) makes every batch the SAME shape which
    is the prerequisite for torch.compile(mode='reduce-overhead') to reuse
    its captured CUDA graph across batches. Cost: padding waste on short
    examples; on SuperNI with max_seq_len=1024 and Wang 2022 prompts that
    regularly fill >500 tokens, the waste is moderate and the compile
    speedup more than pays for it.
    """
    max_len = target_len if target_len is not None else max(
        len(b['input_ids']) for b in batch)
    if target_len is not None:
        # Paranoia: _pack_ids caps at max_seq_len, so no example should
        # exceed target_len. Keep the assert — cheap, off the hot path.
        longest = max(len(b['input_ids']) for b in batch)
        assert longest <= target_len, (
            f"Example length {longest} exceeds collator target_len {target_len}; "
            f"_pack_ids should have truncated earlier.")

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    cluster_ids = torch.empty((len(batch),), dtype=torch.long)
    # task_id is a string, not a tensor — collected as a plain list. Needed
    # for ROUGE eval's per-task aggregation (Wang 2022 protocol: mean of
    # per-task ROUGE-L, not mean across all decodes). Dataset's __getitem__
    # emits b['task_id']; older collators dropped it, causing eval to
    # fall back to '?' and collapse all tasks into a single bucket
    # (eval_rouge_num_tasks=1 in old results). Training path ignores this
    # field — no impact on training speed or semantics.
    task_ids: List[str] = []
    for i, b in enumerate(batch):
        L = len(b['input_ids'])
        input_ids[i, :L] = torch.tensor(b['input_ids'], dtype=torch.long)
        labels[i, :L] = torch.tensor(b['labels'], dtype=torch.long)
        attention_mask[i, :L] = torch.tensor(b['attention_mask'], dtype=torch.long)
        cluster_ids[i] = b['cluster_id']
        task_ids.append(b.get('task_id', '?'))
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'cluster_id': cluster_ids,
        'task_id': task_ids,   # list[str], len = batch size
    }


# ── Factory ─────────────────────────────────────────────────────────────────

def get_superni_dataloaders(data_root: str, tokenizer,
                              batch_size: int = 16,
                              max_seq_len: int = 2048,
                              instances_per_task_train: int = 100,
                              instances_per_task_eval: int = 100,
                              subset_frac_train: float = 1.0,
                              num_workers: int = 4,
                              seed: int = 42,
                              data_subset_seed: Optional[int] = None,
                              K: int = NUM_CLUSTERS_K20,
                              cache_dir: str = './data_cache/lora',
                              num_pos_examples: int = DEFAULT_NUM_POS_EXAMPLES,
                              num_neg_examples: int = DEFAULT_NUM_NEG_EXAMPLES,
                              use_cache: bool = True,
                              auto_build_cache: bool = True,
                              ) -> Tuple[DataLoader, DataLoader, Dict]:
    """Build train + held-out eval SuperNI dataloaders.

    Args:
        data_root:                local path to allenai/natural-instructions
        tokenizer:                HF AutoTokenizer (already loaded + pad_token set)
        subset_frac_train:        ablation subset fraction (ViT-style 0.2 = 20%)
        seed:                     LEGACY / backward-compat name for the data-
                                  subset seed; if `data_subset_seed` is None
                                  this is used. Prefer `data_subset_seed` in
                                  new code.
        data_subset_seed:         RNG seed controlling which instances are
                                  sampled per task for the stratified subset.
                                  Should be FIXED (default 42 via `seed`) and
                                  DECOUPLED from the experiment seed so 3-seed
                                  std measures training variance only — not a
                                  mixture of training variance + subset-choice
                                  variance. Matches the ViT
                                  `train_subset_seed` convention (data/imagenet.py).
        K:                        target cluster count (default 20)
        cache_dir:                where to cache the domain map JSON + tokenized
                                  train/eval tensors.
        use_cache:                if True, load pre-tokenized .pt caches produced
                                  by build_superni_cache / prebuild_all_caches
                                  instead of retokenizing every run. Defaults
                                  to True — smoke test builds them upfront.
        auto_build_cache:         if True AND use_cache AND a cache is missing,
                                  build it on-the-fly (one-time cost). Saves
                                  the user from having to remember the prep step.

    Returns:
        (train_loader, eval_loader, mapping_dict).
    """
    if tokenizer.pad_token_id is None:
        # Llama-3 family often has pad=None; map to eos for padding purposes.
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve which seed to use for subsample / cache lookup. Prefer the
    # explicit `data_subset_seed` (decoupled from experiment seed); fall
    # back to legacy `seed` kwarg if caller didn't pass the new name. This
    # keeps backward-compat for existing tests while steering new callers
    # (run_lora.py, trainer_lora._build_rouge_eval_loader) toward the
    # decoupled convention. See ViT's train_subset_seed for the precedent.
    ds_seed = seed if data_subset_seed is None else data_subset_seed

    mapping = build_or_load_domain_map(
        tasks_dir=_tasks_dir(data_root),
        splits_dir=_splits_dir(data_root),
        cache_dir=cache_dir, K=K)

    train_ids = _read_split(data_root, 'train')
    test_ids = _read_split(data_root, 'test')

    train_cache = eval_cache = None
    if use_cache:
        train_cache_path = find_superni_cache(
            cache_dir, 'train', tokenizer,
            instances_per_task_train, subset_frac_train,
            max_seq_len, ds_seed, num_pos_examples, num_neg_examples)
        eval_cache_path = find_superni_cache(
            cache_dir, 'test', tokenizer,
            instances_per_task_eval, 1.0,
            max_seq_len, ds_seed, num_pos_examples, num_neg_examples)
        if train_cache_path is None and auto_build_cache:
            train_cache_path = build_superni_cache(
                data_root=data_root, split='train', tokenizer=tokenizer,
                mapping=mapping, cache_dir=cache_dir,
                instances_per_task=instances_per_task_train,
                subset_frac=subset_frac_train,
                max_seq_len=max_seq_len, seed=ds_seed,
                num_pos_examples=num_pos_examples,
                num_neg_examples=num_neg_examples)
        if eval_cache_path is None and auto_build_cache:
            eval_cache_path = build_superni_cache(
                data_root=data_root, split='test', tokenizer=tokenizer,
                mapping=mapping, cache_dir=cache_dir,
                instances_per_task=instances_per_task_eval,
                subset_frac=1.0,
                max_seq_len=max_seq_len, seed=ds_seed,
                num_pos_examples=num_pos_examples,
                num_neg_examples=num_neg_examples)
        if train_cache_path is not None:
            train_cache = load_superni_cache(train_cache_path)
        if eval_cache_path is not None:
            eval_cache = load_superni_cache(eval_cache_path)

    train_ds = SuperNIDataset(
        data_root=data_root, task_ids=train_ids, tokenizer=tokenizer,
        mapping=mapping, instances_per_task=instances_per_task_train,
        max_seq_len=max_seq_len, seed=ds_seed, subset_frac=subset_frac_train,
        num_pos_examples=num_pos_examples, num_neg_examples=num_neg_examples,
        cache=train_cache,
    )
    eval_ds = SuperNIDataset(
        data_root=data_root, task_ids=test_ids, tokenizer=tokenizer,
        mapping=mapping, instances_per_task=instances_per_task_eval,
        max_seq_len=max_seq_len, seed=ds_seed, subset_frac=1.0,  # eval always full
        num_pos_examples=num_pos_examples, num_neg_examples=num_neg_examples,
        cache=eval_cache,
    )

    pad_id = tokenizer.pad_token_id
    # Longest-in-batch padding (default). Was briefly changed to fixed
    # target_len=max_seq_len to unlock compile(reduce-overhead) CUDA graphs
    # — measured NET LOSS (1.91 → 2.85 s/it on 5090) because SuperNI's
    # variable prompt distribution made padding overhead (avg seq ~700 →
    # 1024) outweigh CUDA-graph savings. NLP's tokenized chunks ARE fixed
    # shape naturally, so reduce-overhead works there (see
    # rtx5090_4_nlp_faithful.sh _compile_mode). SuperNI doesn't share that
    # property → we use compile(mode='default') instead (inductor fusion
    # only, no CUDA graphs, handles dynamic shapes cleanly).
    def _coll(b): return _collate(b, pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=_coll,
        persistent_workers=(num_workers > 0))
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_coll,
        persistent_workers=(num_workers > 0))

    return train_loader, eval_loader, mapping

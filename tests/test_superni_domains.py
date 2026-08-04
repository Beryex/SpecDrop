"""Unit tests for data/superni_domain_map.py + data/natural_instructions.py.

Uses SYNTHETIC task JSON fixtures (written to a tmp dir) to test the pipeline
end-to-end without network access, without a 3B base model, and without the
full allenai/natural-instructions clone. Covers:

  - normalize_domain: hierarchy collapse
  - build_domain_map: frequency sort determinism, top-(K-1) + misc semantics
  - All-train-tasks mapped (no silent drop-outs)
  - Test-set novel domains fall to misc
  - Repeated build → identical map (determinism)
  - get_cluster_id + task_to_cluster_id resolve consistently
  - Prompt format matches Wang 2022 Tk-Instruct default (Definition + 2 pos + input)
  - _pack_ids masks instruction tokens to -100 (loss-only-on-target)

HF tokenizer dependence is avoided by using a trivial byte-level char
tokenizer at the pack/collate layer for testing.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from data.superni_domain_map import (build_or_load_domain_map,
                                       build_domain_map, get_cluster_id,
                                       normalize_domain, task_to_cluster_id)


# ── Fixtures: write synthetic SuperNI JSONs + splits ────────────────────────

def _task_json(domains, categories=None, reasoning=None,
                instances_n=3, definition='Test def.'):
    return {
        'Contributors': ['test'],
        'Source': ['synthetic'],
        'URL': [],
        'Categories': categories or ['Test Cat'],
        'Reasoning': reasoning or [],
        'Definition': [definition],
        'Input_language': ['English'],
        'Output_language': ['English'],
        'Instruction_language': ['English'],
        'Domains': list(domains),
        'Positive Examples': [
            {'input': 'p1 in', 'output': 'p1 out', 'explanation': ''},
            {'input': 'p2 in', 'output': 'p2 out', 'explanation': ''},
        ],
        'Negative Examples': [],
        'Instances': [
            {'id': f'inst_{i}', 'input': f'instance input {i}', 'output': [f'out {i}']}
            for i in range(instances_n)
        ],
        'Instance License': ['Other'],
    }


@pytest.fixture
def synth_superni(tmp_path):
    """Make a fake allenai/natural-instructions layout under tmp_path.

    Tasks: 30 train, 5 test. Train frequency distribution tilted so
    Mathematics (10), Wikipedia (6), News (4), Code (3), Dialogue (3),
    Medicine (2), Narrative (1), Sports (1) = 30 total. Top-5 = Math,
    Wiki, News, Code, Dialogue.
    """
    tasks_dir = tmp_path / 'tasks'
    splits_dir = tmp_path / 'splits' / 'default'
    tasks_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    train_spec = (
        [('Mathematics', 10), ('Wikipedia', 6), ('News', 4), ('Code', 3),
         ('Dialogue', 3), ('Medicine', 2), ('Narrative', 1), ('Sports', 1)])
    test_spec = (
        [('Wikipedia', 2), ('Government and Politics', 1),  # novel in test
         ('English Exams', 1), ('News', 1)])

    def _w(dom_items, prefix):
        ids = []
        idx = 0
        for domain, n in dom_items:
            for _ in range(n):
                tid = f'{prefix}_{idx:03d}'
                # Half the time, add a hierarchical form to test normalization
                d = (f'{domain} -> Subtopic' if idx % 2 == 0 else domain)
                obj = _task_json([d])
                (tasks_dir / f'{tid}.json').write_text(json.dumps(obj))
                ids.append(tid)
                idx += 1
        return ids

    train_ids = _w(train_spec, 'trn')
    test_ids = _w(test_spec, 'tst')

    (splits_dir / 'train_tasks.txt').write_text('\n'.join(train_ids) + '\n')
    (splits_dir / 'test_tasks.txt').write_text('\n'.join(test_ids) + '\n')

    return {
        'root': str(tmp_path),
        'tasks_dir': str(tasks_dir),
        'splits_dir': str(splits_dir),
        'train_ids': train_ids,
        'test_ids': test_ids,
    }


# ── normalize_domain ─────────────────────────────────────────────────────────

def test_normalize_splits_on_arrow():
    assert normalize_domain('Social Media -> Twitter') == 'Social Media'
    assert normalize_domain('A -> B -> C -> D') == 'A'
    assert normalize_domain('Mathematics') == 'Mathematics'
    assert normalize_domain('') == ''
    assert normalize_domain(None) == ''


def test_normalize_trims_whitespace():
    assert normalize_domain('  Mathematics  ') == 'Mathematics'
    assert normalize_domain('Social Media -> Twitter ') == 'Social Media'


# ── build_domain_map ─────────────────────────────────────────────────────────

def test_build_top6_plus_misc(synth_superni):
    """K=7 → top-6 domains by frequency + 'miscellaneous'."""
    m = build_domain_map(synth_superni['tasks_dir'],
                          synth_superni['splits_dir'], K=7)
    assert m['K'] == 7
    assert m['misc_id'] == 6

    # Top 6 (by train frequency): Math(10), Wiki(6), News(4), Code(3),
    # Dialogue(3), Medicine(2). 3-way tie at 3 (Code/Dialogue) broken alphabetically.
    top_names = [d for d, _ in m['top_domains']]
    assert top_names[:2] == ['Mathematics', 'Wikipedia']  # clear winners
    assert set(top_names[:6]) == {
        'Mathematics', 'Wikipedia', 'News', 'Code', 'Dialogue', 'Medicine'}
    # Tail (Narrative, Sports) → misc
    assert m['domain_to_id']['Narrative'] == 6
    assert m['domain_to_id']['Sports'] == 6


def test_build_determinism(synth_superni):
    """Repeated build produces byte-identical maps (modulo dict ordering)."""
    m1 = build_domain_map(synth_superni['tasks_dir'],
                           synth_superni['splits_dir'], K=5)
    m2 = build_domain_map(synth_superni['tasks_dir'],
                           synth_superni['splits_dir'], K=5)
    assert m1['top_domains'] == m2['top_domains']
    assert m1['domain_to_id'] == m2['domain_to_id']
    assert m1['task_to_cluster'] == m2['task_to_cluster']


def test_all_train_tasks_have_cluster(synth_superni):
    """Every train task_id present in split → has a cluster assignment."""
    m = build_domain_map(synth_superni['tasks_dir'],
                          synth_superni['splits_dir'], K=10)
    for tid in synth_superni['train_ids']:
        assert tid in m['task_to_cluster']
        assert 0 <= m['task_to_cluster'][tid] < 10


def test_test_novel_domains_fall_to_misc(synth_superni):
    """Test-split domains not seen in train map to misc via get_cluster_id."""
    m = build_domain_map(synth_superni['tasks_dir'],
                          synth_superni['splits_dir'], K=6)
    # K=6 → top-5 + misc. Test novel: "Government and Politics", "English Exams".
    novel_task = {
        'Domains': ['Government and Politics'],
    }
    assert get_cluster_id(novel_task, m) == m['misc_id']

    novel2 = {'Domains': ['English Exams']}
    assert get_cluster_id(novel2, m) == m['misc_id']


def test_empty_domains_falls_to_misc(synth_superni):
    m = build_domain_map(synth_superni['tasks_dir'],
                          synth_superni['splits_dir'], K=5)
    task = {'Domains': []}
    assert get_cluster_id(task, m) == m['misc_id']
    task2 = {}  # no Domains field at all
    assert get_cluster_id(task2, m) == m['misc_id']


def test_build_or_load_caches(synth_superni, tmp_path):
    """build_or_load_domain_map writes then loads from cache idempotently."""
    cache_dir = tmp_path / 'cache'
    m1 = build_or_load_domain_map(
        tasks_dir=synth_superni['tasks_dir'],
        splits_dir=synth_superni['splits_dir'],
        cache_dir=str(cache_dir), K=8)
    cached_file = cache_dir / 'superni_domain_map_K8.json'
    assert cached_file.exists()
    m2 = build_or_load_domain_map(
        tasks_dir=synth_superni['tasks_dir'],
        splits_dir=synth_superni['splits_dir'],
        cache_dir=str(cache_dir), K=8)
    # json round-trip turns tuples into lists; compare after normalizing both.
    def _norm(md):
        return [list(kv) for kv in md['top_domains']]
    assert _norm(m1) == _norm(m2)
    assert m1['domain_to_id'] == m2['domain_to_id']


def test_hierarchy_normalization_in_build(synth_superni):
    """Half of the fixture domains are in 'X -> Subtopic' form; verify they
    land in the same bucket as the bare 'X' versions."""
    m = build_domain_map(synth_superni['tasks_dir'],
                          synth_superni['splits_dir'], K=10)
    # 10 Mathematics tasks total (mix of raw + hierarchical) → all same bucket.
    math_counts = [v for k, v in m['domain_to_id'].items()
                   if k.startswith('Mathematics')]
    assert len(math_counts) == 1  # single normalized key


def test_K_too_small_raises(synth_superni):
    with pytest.raises(ValueError):
        build_domain_map(synth_superni['tasks_dir'],
                          synth_superni['splits_dir'], K=1)


# ── task_to_cluster_id + get_cluster_id integration ────────────────────────

def test_task_to_cluster_id(synth_superni):
    m = build_domain_map(synth_superni['tasks_dir'],
                          synth_superni['splits_dir'], K=5)
    for tid in synth_superni['train_ids']:
        cid = task_to_cluster_id(tid, m)
        assert 0 <= cid < 5


def test_task_to_cluster_id_unknown_falls_to_misc(synth_superni):
    m = build_domain_map(synth_superni['tasks_dir'],
                          synth_superni['splits_dir'], K=5)
    assert task_to_cluster_id('some_unseen_task_id', m) == m['misc_id']


# ── natural_instructions loader: Wang 2022 prompt + pack_ids mask ───────────

class _ByteTokenizer:
    """Minimal char-level tokenizer (no HF deps) for unit testing _pack_ids."""
    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.pad_token = '\x00'
        self.eos_token = '\x00'

    def __call__(self, text, add_special_tokens=False):
        ids = [ord(c) % 256 for c in text]
        return {'input_ids': ids}


def test_pack_ids_masks_prompt_to_loss_ignore():
    from data.natural_instructions import _pack_ids
    tok = _ByteTokenizer()
    prompt = 'Definition: X\n'
    target = 'hello'
    pack = _pack_ids(prompt, target, tok, max_seq_len=1024)
    # All prompt positions should have label -100
    for label in pack['labels'][:len(prompt)]:
        assert label == -100
    # Target positions should match target token ids (plus trailing eos)
    target_region = pack['labels'][len(prompt):]
    expected_target_ids = tok(' ' + target)['input_ids'] + [tok.eos_token_id]
    assert target_region == expected_target_ids


def test_pack_ids_truncates_prompt_not_target():
    from data.natural_instructions import _pack_ids
    tok = _ByteTokenizer()
    prompt = 'P' * 100
    target = 'T' * 50
    # Explicit max_target_len < max_seq_len so stage-1 target cap is a no-op
    # (target=51 tokens including EOS fits under cap=60) and the test isolates
    # stage-2 prompt LEFT-truncation behavior.
    pack = _pack_ids(prompt, target, tok, max_seq_len=80, max_target_len=60)
    assert len(pack['input_ids']) == 80
    # Target region at end must still be intact (+ eos)
    target_ids = tok(' ' + target)['input_ids'] + [tok.eos_token_id]
    assert pack['labels'][-len(target_ids):] == target_ids


def test_collate_pads_to_fixed_target_len_for_cuda_graph_reuse():
    """When collator is given `target_len`, every batch has EXACTLY that
    sequence length regardless of actual content. Prerequisite for
    torch.compile(mode='reduce-overhead') CUDA-graph reuse."""
    from data.natural_instructions import _collate
    import torch
    # Heterogeneous-length batch simulating real SuperNI variability
    batch = [
        {'input_ids': [1, 2, 3], 'labels': [-100, -100, 5],
         'attention_mask': [1, 1, 1], 'cluster_id': 0},
        {'input_ids': [7, 8, 9, 10, 11], 'labels': [-100, -100, -100, 12, 13],
         'attention_mask': [1, 1, 1, 1, 1], 'cluster_id': 3},
    ]
    out = _collate(batch, pad_token_id=0, target_len=1024)
    assert out['input_ids'].shape == (2, 1024)
    assert out['labels'].shape == (2, 1024)
    assert out['attention_mask'].shape == (2, 1024)
    # Pad region must be correctly marked (pad_id in inputs, -100 in labels,
    # 0 in attn_mask) so the model ignores it.
    assert out['input_ids'][0, 3:].eq(0).all()
    assert out['labels'][0, 3:].eq(-100).all()
    assert out['attention_mask'][0, 3:].eq(0).all()
    # Real tokens preserved
    assert out['input_ids'][1, :5].tolist() == [7, 8, 9, 10, 11]
    # Backward-compat: no target_len → longest-in-batch behavior
    out2 = _collate(batch, pad_token_id=0)
    assert out2['input_ids'].shape == (2, 5)


def test_collate_preserves_task_id_for_rouge_per_task_aggregation():
    """Regression: _collate must include `task_id` in the output batch dict.

    Dataset's __getitem__ emits task_id (e.g. 'task001_quoref_question_generation');
    ROUGE eval (SuperNIEvaluator.evaluate) calls `batch.get('task_id', ['?']*B)`
    and buckets per-example ROUGE-L by task_id for the Wang 2022 protocol
    (mean over per-task means, reported as 119 tasks). Pre-fix (before
    2026-04-24 evening), _collate dropped this key → every decode fell into
    the '?' fallback bucket → eval_rouge_num_tasks=1 (should be 119). The
    reported ROUGE-L number was numerically correct because per-task
    instance counts are uniform (10 each), but per-task breakdown was
    lost and metadata was misleading.
    """
    from data.natural_instructions import _collate
    batch = [
        {'input_ids': [1, 2, 3], 'labels': [-100, -100, 5],
         'attention_mask': [1, 1, 1], 'cluster_id': 0,
         'task_id': 'task001_quoref_question_generation'},
        {'input_ids': [7, 8], 'labels': [-100, 12],
         'attention_mask': [1, 1], 'cluster_id': 3,
         'task_id': 'task035_winogrande'},
        {'input_ids': [4], 'labels': [9],
         'attention_mask': [1], 'cluster_id': 5,
         # Emulate a dataset that forgot task_id — fallback must not crash.
         },
    ]
    out = _collate(batch, pad_token_id=0)
    assert 'task_id' in out, \
        "_collate dropped task_id — ROUGE per-task aggregation will break"
    assert isinstance(out['task_id'], list)
    assert len(out['task_id']) == 3
    assert out['task_id'][0] == 'task001_quoref_question_generation'
    assert out['task_id'][1] == 'task035_winogrande'
    assert out['task_id'][2] == '?'  # fallback for missing key


def test_collate_rejects_overlong_example():
    """Assert catches callers that bypass _pack_ids truncation."""
    from data.natural_instructions import _collate
    import pytest
    batch = [
        {'input_ids': list(range(1025)),
         'labels': [-100] * 1025,
         'attention_mask': [1] * 1025, 'cluster_id': 0},
    ]
    with pytest.raises(AssertionError, match='exceeds collator target_len'):
        _collate(batch, pad_token_id=0, target_len=1024)


def test_pack_ids_truncates_target_when_too_long():
    """Regression: long targets (> max_target_len) must be RIGHT-truncated
    with EOS preserved. Latent bug at max_seq_len=2048; hit immediately at
    max_seq_len=1024 by SuperNI test targets that are paragraph-length."""
    from data.natural_instructions import _pack_ids
    tok = _ByteTokenizer()
    prompt = 'P' * 10
    target = 'T' * 2000  # 2001 tokens incl. EOS — far exceeds max_target_len
    pack = _pack_ids(prompt, target, tok, max_seq_len=1024, max_target_len=128)
    assert len(pack['input_ids']) <= 1024
    # Target region (non -100 labels) capped at max_target_len
    target_region = [l for l in pack['labels'] if l != -100]
    assert len(target_region) == 128
    # EOS must still be the final token (training signal for stop)
    assert pack['input_ids'][-1] == tok.eos_token_id
    assert target_region[-1] == tok.eos_token_id


def test_format_prompt_wang_2022_structure():
    from data.natural_instructions import _format_prompt
    task = {
        'Definition': ['Do the thing.'],
        'Positive Examples': [
            {'input': 'in1', 'output': 'out1'},
            {'input': 'in2', 'output': 'out2'},
        ],
        'Negative Examples': [],
    }
    prompt = _format_prompt(task, input_text='real input', num_pos=2, num_neg=0)
    assert 'Definition: Do the thing.' in prompt
    assert 'Positive Example 1' in prompt
    assert 'Positive Example 2' in prompt
    assert 'Input: in1' in prompt
    assert 'Output: out1' in prompt
    assert 'Input: real input' in prompt
    assert prompt.rstrip().endswith('Output:')


def test_format_prompt_no_negative_by_default():
    from data.natural_instructions import _format_prompt
    task = {
        'Definition': ['def'],
        'Positive Examples': [{'input': 'a', 'output': 'b'}],
        'Negative Examples': [{'input': 'c', 'output': 'd'}],
    }
    prompt = _format_prompt(task, input_text='x', num_pos=1, num_neg=0)
    assert 'Negative' not in prompt  # default num_neg=0


# ── Dataset full flow with synthetic fixture ────────────────────────────────

def test_superni_dataset_yields_4tuple_with_clusterid(synth_superni):
    """End-to-end: construct SuperNIDataset on synth fixture, iterate a few."""
    from data.natural_instructions import SuperNIDataset
    m = build_or_load_domain_map(
        tasks_dir=synth_superni['tasks_dir'],
        splits_dir=synth_superni['splits_dir'],
        cache_dir=os.path.join(synth_superni['root'], 'cache'), K=5)
    tok = _ByteTokenizer()
    ds = SuperNIDataset(
        data_root=synth_superni['root'],
        task_ids=synth_superni['train_ids'][:10],
        tokenizer=tok, mapping=m,
        instances_per_task=2, max_seq_len=256, seed=42)
    assert len(ds) > 0
    sample = ds[0]
    assert 'input_ids' in sample
    assert 'labels' in sample
    assert 'attention_mask' in sample
    assert 'cluster_id' in sample
    assert 0 <= sample['cluster_id'] < 5


def test_superni_cache_build_load_roundtrip(synth_superni, tmp_path):
    """build_superni_cache writes a .pt; load_superni_cache reads it back;
    SuperNIDataset with cache= yields same records as on-the-fly path."""
    from data.natural_instructions import (build_superni_cache,
                                             load_superni_cache,
                                             find_superni_cache, SuperNIDataset)
    from data.superni_domain_map import build_or_load_domain_map

    cache_dir = tmp_path / 'cache'
    m = build_or_load_domain_map(
        tasks_dir=synth_superni['tasks_dir'],
        splits_dir=synth_superni['splits_dir'],
        cache_dir=str(cache_dir), K=5)
    tok = _ByteTokenizer()

    # Build cache.
    cache_path = build_superni_cache(
        data_root=synth_superni['root'], split='train',
        tokenizer=tok, mapping=m, cache_dir=str(cache_dir),
        instances_per_task=2, subset_frac=1.0, max_seq_len=256, seed=42,
        verbose=False)
    assert os.path.exists(cache_path)

    # find_superni_cache should locate it.
    found = find_superni_cache(
        str(cache_dir), 'train', tok,
        instances_per_task=2, subset_frac=1.0, max_seq_len=256, seed=42)
    assert found == cache_path

    # Load + assert non-empty.
    blob = load_superni_cache(cache_path)
    assert len(blob['input_ids_list']) > 0
    assert blob['cluster_ids'].shape[0] == len(blob['input_ids_list'])
    assert blob['metadata']['split'] == 'train'
    assert blob['metadata']['subset_frac'] == 1.0
    assert blob['metadata']['max_seq_len'] == 256

    # Cache-backed dataset has same len as on-the-fly one, same first item.
    ds_cached = SuperNIDataset(
        data_root=synth_superni['root'], task_ids=[], tokenizer=tok,
        mapping=m, instances_per_task=2, max_seq_len=256, seed=42,
        subset_frac=1.0, cache=blob)
    ds_otf = SuperNIDataset(
        data_root=synth_superni['root'],
        task_ids=synth_superni['train_ids'],
        tokenizer=tok, mapping=m, instances_per_task=2,
        max_seq_len=256, seed=42, subset_frac=1.0)

    assert len(ds_cached) == len(ds_otf)
    # Content should match item-by-item (modulo order within a task; tokenize
    # pass uses the same rng seed and walks tasks in the same order).
    for i in range(len(ds_cached)):
        c = ds_cached[i]
        o = ds_otf[i]
        assert c['input_ids'] == o['input_ids'], f'mismatch at idx {i}'
        assert c['labels'] == o['labels']
        assert c['cluster_id'] == o['cluster_id']


def test_superni_cache_hit_skips_rebuild(synth_superni, tmp_path):
    """Second call to build_superni_cache with same knobs returns cached path
    WITHOUT re-tokenizing (file mtime unchanged)."""
    from data.natural_instructions import build_superni_cache
    from data.superni_domain_map import build_or_load_domain_map

    cache_dir = tmp_path / 'cache'
    m = build_or_load_domain_map(
        tasks_dir=synth_superni['tasks_dir'],
        splits_dir=synth_superni['splits_dir'],
        cache_dir=str(cache_dir), K=5)
    tok = _ByteTokenizer()

    p1 = build_superni_cache(
        data_root=synth_superni['root'], split='train',
        tokenizer=tok, mapping=m, cache_dir=str(cache_dir),
        instances_per_task=2, subset_frac=1.0, max_seq_len=256, seed=42,
        verbose=False)
    mtime1 = os.path.getmtime(p1)
    # Second build should be a cache hit.
    p2 = build_superni_cache(
        data_root=synth_superni['root'], split='train',
        tokenizer=tok, mapping=m, cache_dir=str(cache_dir),
        instances_per_task=2, subset_frac=1.0, max_seq_len=256, seed=42,
        verbose=False)
    assert p1 == p2
    assert os.path.getmtime(p2) == mtime1  # not rebuilt


def test_superni_cache_knob_mismatch_raises(synth_superni, tmp_path):
    """SuperNIDataset should refuse a cache whose knobs don't match the
    requested (subset_frac, seed, etc.) — prevents silent dataset shift."""
    from data.natural_instructions import (build_superni_cache,
                                             load_superni_cache, SuperNIDataset)
    from data.superni_domain_map import build_or_load_domain_map

    cache_dir = tmp_path / 'cache'
    m = build_or_load_domain_map(
        tasks_dir=synth_superni['tasks_dir'],
        splits_dir=synth_superni['splits_dir'],
        cache_dir=str(cache_dir), K=5)
    tok = _ByteTokenizer()
    p = build_superni_cache(
        data_root=synth_superni['root'], split='train',
        tokenizer=tok, mapping=m, cache_dir=str(cache_dir),
        instances_per_task=2, subset_frac=1.0, max_seq_len=256, seed=42,
        verbose=False)
    blob = load_superni_cache(p)
    # Mismatch: request subset_frac=0.5 but cache has 1.0 → should raise.
    with pytest.raises(ValueError, match=r'subset_frac'):
        SuperNIDataset(
            data_root=synth_superni['root'], task_ids=[], tokenizer=tok,
            mapping=m, instances_per_task=2, max_seq_len=256, seed=42,
            subset_frac=0.5, cache=blob)


def test_prebuild_all_caches_makes_three(synth_superni, tmp_path):
    """prebuild_all_caches writes one cache per (split, subset_frac)."""
    from data.natural_instructions import prebuild_all_caches
    from data.superni_domain_map import build_or_load_domain_map

    cache_dir = tmp_path / 'cache'
    m = build_or_load_domain_map(
        tasks_dir=synth_superni['tasks_dir'],
        splits_dir=synth_superni['splits_dir'],
        cache_dir=str(cache_dir), K=5)
    tok = _ByteTokenizer()
    out = prebuild_all_caches(
        data_root=synth_superni['root'],
        tokenizer=tok, mapping=m, cache_dir=str(cache_dir),
        max_seq_len=256, instances_per_task=2, seed=42,
        subset_fracs_train=(1.0, 0.2), verbose=False)
    # Expect: train_frac1.0, train_frac0.2, test_frac1.0
    assert set(out.keys()) == {'train_frac1.0', 'train_frac0.2', 'test_frac1.0'}
    for p in out.values():
        assert os.path.exists(p)


def test_superni_subset_frac_stratified(synth_superni):
    """subset_frac=0.5 should roughly halve per-task instances (min 1)."""
    from data.natural_instructions import SuperNIDataset
    m = build_or_load_domain_map(
        tasks_dir=synth_superni['tasks_dir'],
        splits_dir=synth_superni['splits_dir'],
        cache_dir=os.path.join(synth_superni['root'], 'cache'), K=5)
    tok = _ByteTokenizer()
    ds_full = SuperNIDataset(
        data_root=synth_superni['root'],
        task_ids=synth_superni['train_ids'][:10],
        tokenizer=tok, mapping=m,
        instances_per_task=2, max_seq_len=256, seed=42,
        subset_frac=1.0)
    ds_half = SuperNIDataset(
        data_root=synth_superni['root'],
        task_ids=synth_superni['train_ids'][:10],
        tokenizer=tok, mapping=m,
        instances_per_task=2, max_seq_len=256, seed=42,
        subset_frac=0.5)
    # Each task had 2 instances, subset_frac=0.5 → 1 per task (ceil of 2×0.5=1)
    assert len(ds_half) <= len(ds_full)
    assert len(ds_half) >= 10  # 10 tasks × at least 1 instance = 10

    with pytest.raises(ValueError):
        SuperNIDataset(
            data_root=synth_superni['root'],
            task_ids=synth_superni['train_ids'][:2],
            tokenizer=tok, mapping=m,
            instances_per_task=2, max_seq_len=256, seed=42,
            subset_frac=0.0)

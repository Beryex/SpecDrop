#!/usr/bin/env python3
"""Back-fill ROUGE-L + exact-match on LoRA cells that finished training before
the autocast fix. Scans an output dir, finds `results.json` with
`eval_rouge_l == None`, reloads each cell's best.pt + config, runs the
post-hoc SuperNI eval (Wang 2022 protocol), and patches the JSON in place.

Usage:
    # Single cell:
    python scripts/rerun_rouge.py \
        outputs/rtx5090_lora_ablation/phase8a_pa0.5_se1.0_s42

    # Batch over every cell under a parent dir (auto-skips cells with
    # eval_rouge_l already set):
    python scripts/rerun_rouge.py outputs/rtx5090_lora_ablation

The cell dir must contain:
    best.pt          — LoRA-only state dict, saved by LoRATrainer
    _tmp*.yaml       — the full cfg dict written at run-start (any file
                       matching _tmp*.yaml is accepted; first one picked)
    results.json     — results dict (will be patched in place)

If `_tmp.yaml` is missing, we reconstruct cfg from the command-line log
file if available — but for now we require the yaml.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Optional

# Add repo root to sys.path so `from models.lora_models import ...` works
# when the script is invoked from anywhere. scripts/ is one level below root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import yaml


def _load_cfg(cell_dir: str) -> Optional[dict]:
    """Load the _tmp*.yaml that run_lora.py writes at start. Returns None
    if not found."""
    candidates = sorted(glob.glob(os.path.join(cell_dir, '_tmp*.yaml')))
    if not candidates:
        # Fallback: also accept any .yaml in the cell dir.
        candidates = sorted(glob.glob(os.path.join(cell_dir, '*.yaml')))
    if not candidates:
        return None
    with open(candidates[0]) as f:
        return yaml.safe_load(f)


def _load_best_pt(cell_dir: str) -> Optional[dict]:
    path = os.path.join(cell_dir, 'best.pt')
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu', weights_only=False)


def rerun_rouge_for_cell(cell_dir: str, device: str = 'cuda',
                          instances_per_task: int = 10,
                          max_new_tokens: int = 128,
                          force: bool = False) -> bool:
    """Rerun ROUGE-L eval for one completed cell. Returns True on success.

    Patches results.json in place with:
        eval_rouge_l, eval_exact_match, eval_rouge_num_tasks,
        eval_rouge_instances_per_task, eval_rouge_elapsed_min
    """
    results_path = os.path.join(cell_dir, 'results.json')
    if not os.path.exists(results_path):
        print(f'[skip] {cell_dir}: no results.json')
        return False
    with open(results_path) as f:
        results = json.load(f)
    if results.get('eval_rouge_l') is not None and not force:
        print(f'[skip] {cell_dir}: eval_rouge_l already set '
              f'({results["eval_rouge_l"]:.4f})')
        return True

    cfg = _load_cfg(cell_dir)
    if cfg is None:
        print(f'[ERR]  {cell_dir}: no _tmp*.yaml found, cannot reconstruct cfg')
        return False
    ckpt = _load_best_pt(cell_dir)
    if ckpt is None:
        print(f'[ERR]  {cell_dir}: no best.pt, cannot reload weights')
        return False

    print(f'[run]  {cell_dir} ({cfg["model"]["type"]})')
    t0 = time.time()

    # Build a fresh model + tokenizer, load LoRA weights from best.pt.
    from transformers import AutoTokenizer
    from models.lora_models import (build_lora_model, MoCLEModel,
                                     _apply_gradient_checkpointing_override)
    from data.natural_instructions import get_superni_dataloaders

    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['base_model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_lora_model(cfg)
    _apply_gradient_checkpointing_override(model, cfg)
    # Load LoRA-only state (trainer saves only trainable params).
    lora_state = ckpt.get('lora_state') or ckpt.get('state_dict') or ckpt
    model_base = model  # pierce any future compile wrap (same pattern as trainer)
    missing, unexpected = model_base.load_state_dict(lora_state, strict=False)
    non_lora_missing = [k for k in missing if 'adapter' in k or 'lora' in k.lower()]
    if non_lora_missing:
        print(f'[warn] {len(non_lora_missing)} LoRA keys missing from best.pt')
    model.to(device).eval()

    # MoCLE needs BGE centroids re-built before eval.
    if isinstance(model, MoCLEModel):
        from data.natural_instructions import _read_task, _read_split
        from data.superni_domain_map import build_bge_centroids
        # Need the mapping — will be re-built below by the eval loader,
        # but MoCLE init uses train tasks only, so build separately.
        data_root = cfg['data']['data_root']
        tds = {}
        for tid in _read_split(data_root, 'train'):
            try: t = _read_task(data_root, tid)
            except FileNotFoundError: continue
            df = t.get('Definition', [])
            if isinstance(df, list): df = ' '.join(x for x in df if x)
            tds[tid] = (df or '').strip()
        # Match original training's ipt/frac so cache hits (same fix as the
        # main tl/el construction below).
        dcfg_m = cfg['data']
        _, _, mapping = get_superni_dataloaders(
            data_root=data_root, tokenizer=tokenizer,
            batch_size=1, max_seq_len=cfg['training']['max_seq_len'],
            instances_per_task_train=dcfg_m.get('instances_per_task_train', 100),
            instances_per_task_eval=dcfg_m.get('instances_per_task_eval', 100),
            subset_frac_train=dcfg_m.get('subset_frac_train', 1.0),
            num_workers=0,
            data_subset_seed=dcfg_m.get('data_subset_seed', 42),
            K=dcfg_m.get('num_clusters', 20),
            cache_dir=dcfg_m.get('cluster_cache_dir', './data_cache/lora'),
        )
        centroids = build_bge_centroids(tds, mapping, device=device)
        if not isinstance(centroids, torch.Tensor):
            centroids = torch.from_numpy(centroids)
        model.init_all_cluster_embeddings(centroids.to(torch.float32))

    # Re-use trainer's eval helpers. Build a throw-away trainer just to get
    # the routing_fn / mask_scale_fn / rouge_eval plumbing.
    from training.trainer_lora import LoRATrainer
    dummy_cfg = dict(cfg); dummy_cfg.setdefault('output_dir', cell_dir)
    from data.natural_instructions import get_superni_dataloaders as _gsd
    # Use the SAME ipt/frac values as the original training run so the cache
    # filename matches what's already on disk — instant cache hit. Previous
    # version passed ipt=1 which built a separate tiny cache (~17s tokenize
    # waste on first invocation).
    dcfg = cfg['data']
    tl, el, _ = _gsd(
        data_root=dcfg['data_root'],
        tokenizer=tokenizer,
        batch_size=cfg['training']['batch_size_per_device'],
        max_seq_len=cfg['training']['max_seq_len'],
        instances_per_task_train=dcfg.get('instances_per_task_train', 100),
        instances_per_task_eval=dcfg.get('instances_per_task_eval', 100),
        subset_frac_train=dcfg.get('subset_frac_train', 1.0),
        num_workers=0,
        data_subset_seed=dcfg.get('data_subset_seed', 42),
        K=dcfg.get('num_clusters', 20),
        cache_dir=dcfg.get('cluster_cache_dir', './data_cache/lora'),
    )
    # Build the algorithm the same way run_lora.py does.
    from scripts._diag_helpers import advance_softspecdrop_to_terminal, require_keys
    require_keys(cfg, ('data', 'model', 'training'),
                  f'cfg in {run_dir}')
    require_keys(cfg['data'], ('num_clusters',), f'cfg["data"] in {run_dir}')
    require_keys(cfg['model'], ('num_experts',), f'cfg["model"] in {run_dir}')
    K = cfg['data']['num_clusters']
    algorithm = None
    acfg = cfg.get('algorithm', {}) or {}
    if acfg.get('type') == 'soft_specdrop':
        from algorithms.soft_specdrop import SoftSpecDrop
        tcfg = cfg['training']
        require_keys(acfg, ('p_active', 'p_inactive', 'assignment',
                             'warmup_ratio', 'warmup_schedule', 'warmup_unit',
                             'amplification_beta'),
                      f'cfg["algorithm"] in {run_dir}')
        require_keys(tcfg, ('epochs',), f'cfg["training"] in {run_dir}')
        algorithm = SoftSpecDrop(
            num_modules=cfg['model']['num_experts'],
            num_categories=K,
            p_active=acfg['p_active'],
            p_inactive=acfg['p_inactive'],
            assignment=acfg['assignment'],
            warmup_ratio=acfg['warmup_ratio'],
            total_epochs=tcfg['epochs'],
            assignment_seed=acfg.get('assignment_seed', 42),
            frac_per_category=acfg.get('frac_per_category', None),
            amplification_beta=acfg['amplification_beta'],
            warmup_schedule=acfg['warmup_schedule'],
            warmup_unit=acfg['warmup_unit'],
        )
        advance_softspecdrop_to_terminal(algorithm, tcfg['epochs'])
    elif acfg.get('type') == 'no_dropout':
        from algorithms.no_dropout import NoDropout
        algorithm = NoDropout(
            num_modules=cfg['model']['num_experts'],
            num_categories=K)

    trainer = LoRATrainer(
        cfg=dummy_cfg, model=model, algorithm=algorithm,
        train_loader=tl, eval_loader=el, device=device, use_wandb=False)
    rouge = trainer.run_rouge_eval(
        instances_per_task=instances_per_task,
        max_new_tokens=max_new_tokens)
    elapsed = (time.time() - t0) / 60

    # Patch results.json in place.
    results['eval_rouge_l'] = rouge['rougeL_mean']
    results['eval_exact_match'] = rouge['exact_match_mean']
    results['eval_rouge_num_tasks'] = rouge['num_tasks']
    results['eval_rouge_instances_per_task'] = instances_per_task
    results['eval_rouge_elapsed_min'] = round(elapsed, 2)
    results.pop('eval_rouge_error', None)  # clear any stale error entry
    tmp = results_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, results_path)
    print(f'[OK]   {cell_dir}: ROUGE-L={rouge["rougeL_mean"]:.4f} '
          f'EM={rouge["exact_match_mean"]:.4f} ({elapsed:.1f} min)')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target', help='Either a single cell dir or a parent dir')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--instances_per_task', type=int, default=10)
    ap.add_argument('--max_new_tokens', type=int, default=128)
    ap.add_argument('--force', action='store_true',
                     help='Rerun even if eval_rouge_l is already set.')
    args = ap.parse_args()

    target = args.target.rstrip('/')
    if os.path.isdir(target) and os.path.exists(
            os.path.join(target, 'results.json')):
        # Single cell
        ok = rerun_rouge_for_cell(target, device=args.device,
                                    instances_per_task=args.instances_per_task,
                                    max_new_tokens=args.max_new_tokens,
                                    force=args.force)
        sys.exit(0 if ok else 1)

    # Batch: all cells under target
    cells = sorted([os.path.dirname(p) for p in
                    glob.glob(os.path.join(target, '*', 'results.json'))])
    if not cells:
        print(f'No cells with results.json found under {target}')
        sys.exit(1)
    print(f'[batch] {len(cells)} candidate cells under {target}')
    n_ok = n_skip = n_err = 0
    for cell in cells:
        try:
            if rerun_rouge_for_cell(
                    cell, device=args.device,
                    instances_per_task=args.instances_per_task,
                    max_new_tokens=args.max_new_tokens, force=args.force):
                n_ok += 1
            else:
                n_skip += 1
        except Exception as e:
            print(f'[ERR]  {cell}: {e}')
            n_err += 1
    print(f'[batch done] ok={n_ok} skip={n_skip} err={n_err}')


if __name__ == '__main__':
    main()

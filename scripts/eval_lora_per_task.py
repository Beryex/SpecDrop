#!/usr/bin/env python3
"""Per-task ROUGE-L breakdown for LoRA top-3 methods (paper figure F).

For each of the 119 SuperNI held-out tasks, run greedy decoding under each
method's NATIVE routing (no algorithm swap), score per-task ROUGE-L + EM,
save full per-task dict so we can sort by `Δ = ours − HydraLoRA` and
generate a fine-grained validation of the granularity-alignment thesis:
  - Right tail (ours wins): cluster-aligned tasks (per-domain specialization)
  - Left tail (HydraLoRA wins): cross-cluster reasoning tasks

This is the LoRA analogue of paper's "Soft MoE wins broadly, ours wins on
its assigned categories" pattern, validated at fine task grain.

Paper writer should plot 119-task scatter with sorted-Δ on x-axis.

Re-uses trainer's `run_rouge_eval` machinery — already returns per_task dict
post-`dbfaac6` (`_collate` preserves task_id). Cache is hit on re-run.

Usage:
    python scripts/eval_lora_per_task.py \\
        --run_dir outputs/rtx5090_lora_faithful/ours_s42 \\
        --method ours

Output (always written to outputs/eval_lora_per_task/<method>_s<seed>.json):
    {"method": "ours", "seed": 42, "run_dir": "...",
     "rougeL_mean": 0.49, "exact_match_mean": 0.31, "num_tasks": 119,
     "instances_per_task": 10,
     "per_task": {<task_id>: {"rougeL": ..., "exact_match": ..., "n": 10}, ...}}

~30-100 min/run on RTX 5090 at ipt=10 depending on max_new_tokens. Default
ipt=10 matches training-time eval and gives directionally-correct task ranks
(the mean across 119 tasks is the more variance-sensitive quantity, not the
within-task means).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch


def _load_cfg(run_dir: str) -> dict:
    """Same 3-tier fallback as scripts/eval_uniform_mask.py."""
    import glob
    import yaml
    cands = sorted(glob.glob(f'{run_dir}/_tmp*.yaml'))
    if cands:
        with open(cands[0]) as f:
            return yaml.safe_load(f)
    rj = f'{run_dir}/results.json'
    if os.path.exists(rj):
        with open(rj) as f:
            r = json.load(f)
        if 'config' in r:
            return r['config']
    raise FileNotFoundError(
        f'no recoverable config for {run_dir}: tried _tmp*.yaml + results.json::config')


def _seed_from_run_dir(run_dir: str) -> int:
    m = re.search(r'_s(\d+)/?$', run_dir.rstrip('/'))
    if not m:
        raise ValueError(f"can't parse seed from {run_dir}; expected suffix `_s<int>`")
    return int(m.group(1))


def _build_algorithm_from_cfg(cfg, K, M):
    """Mirror run_lora.py::_build_algorithm. Returns None / NoDropout / SoftSpecDrop
    matching whatever this method was trained with.

    Required cfg keys (acfg = cfg['algorithm']) when type=='soft_specdrop':
      p_active, p_inactive, assignment, warmup_ratio, warmup_schedule,
      warmup_unit, amplification_beta. Silent defaults would silently produce
      wrong-mask reproduction.
    """
    from scripts._diag_helpers import (advance_softspecdrop_to_terminal,
                                         require_keys)
    acfg = cfg.get('algorithm', {}) or {}
    atype = acfg.get('type', 'none')
    if atype == 'none':
        return None
    if atype == 'no_dropout':
        from algorithms.no_dropout import NoDropout
        return NoDropout(num_modules=K, num_categories=M)
    if atype == 'soft_specdrop':
        from algorithms.soft_specdrop import SoftSpecDrop
        require_keys(acfg, ('p_active', 'p_inactive', 'assignment',
                             'warmup_ratio', 'warmup_schedule', 'warmup_unit',
                             'amplification_beta'),
                      'cfg["algorithm"] for soft_specdrop')
        require_keys(cfg.get('training', {}), ('epochs',),
                      'cfg["training"]')
        algo = SoftSpecDrop(
            num_modules=K, num_categories=M,
            p_active=acfg['p_active'],
            p_inactive=acfg['p_inactive'],
            assignment=acfg['assignment'],
            warmup_ratio=acfg['warmup_ratio'],
            total_epochs=cfg['training']['epochs'],
            assignment_seed=acfg.get('assignment_seed', 42),
            frac_per_category=acfg.get('frac_per_category', None),
            amplification_beta=acfg['amplification_beta'],
            warmup_schedule=acfg['warmup_schedule'],
            warmup_unit=acfg['warmup_unit'],
        )
        advance_softspecdrop_to_terminal(algo, cfg['training']['epochs'])
        return algo
    raise ValueError(f'unknown algorithm.type={atype!r}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run_dir', required=True,
                    help='e.g. outputs/rtx5090_lora_faithful/ours_s42')
    ap.add_argument('--method', required=True,
                    help='label (e.g. ours / hydra_lora / mb_lora_no_routing). '
                         'Used in output filename + payload.')
    ap.add_argument('--device', default=None)
    ap.add_argument('--instances_per_task', type=int, default=10,
                    help='ipt for ROUGE eval (default 10 = matches training).')
    ap.add_argument('--max_new_tokens', type=int, default=128,
                    help='gen budget per instance.')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    seed = _seed_from_run_dir(args.run_dir)
    print(f'[per_task] method={args.method} seed={seed} run_dir={args.run_dir}')
    print(f'[per_task] device={device}  ipt={args.instances_per_task}  '
          f'max_new_tokens={args.max_new_tokens}')

    cfg = _load_cfg(args.run_dir)
    K = cfg['model'].get('num_experts', 1)
    M = cfg['data'].get('num_clusters', 20)

    # Build model + load LoRA-only state (same pattern as eval_uniform_mask.py).
    from models.lora_models import build_lora_model, _apply_gradient_checkpointing_override
    model = build_lora_model(cfg)
    _apply_gradient_checkpointing_override(model, cfg)
    model = model.to(device)
    ckpt = torch.load(f'{args.run_dir}/best.pt', map_location=device, weights_only=False)
    lora_state = ckpt.get('lora_state') or ckpt.get('state_dict') or ckpt
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    print(f'[per_task] loaded best.pt — missing={len(missing)}  unexpected={len(unexpected)}')
    model.eval()

    algorithm = _build_algorithm_from_cfg(cfg, K, M)
    print(f'[per_task] algorithm={type(algorithm).__name__ if algorithm else "None"}')

    # Build trainer in eval-only mode (passes eval_loader as both args).
    from training.trainer_lora import LoRATrainer
    from data.natural_instructions import get_superni_dataloaders
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg['model']['base_model_name'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dcfg = cfg['data']
    _, eval_loader, _ = get_superni_dataloaders(
        data_root=dcfg['data_root'], tokenizer=tok,
        batch_size=cfg['training'].get('batch_size_per_device', 8),
        max_seq_len=cfg['training'].get('max_seq_len', 1024),
        instances_per_task_train=dcfg.get('instances_per_task_train', 100),
        instances_per_task_eval=dcfg.get('instances_per_task_eval', 100),
        subset_frac_train=dcfg.get('subset_frac_train', 1.0),
        num_workers=0,
        data_subset_seed=dcfg.get('data_subset_seed', 42),
        K=dcfg.get('num_clusters', 20),
        cache_dir=dcfg.get('cluster_cache_dir', './data_cache/lora'),
    )
    dummy_cfg = dict(cfg)
    # Override ipt for ROUGE eval so trainer.run_rouge_eval picks up our value.
    dummy_cfg.setdefault('output_dir', args.run_dir)
    tcfg_override = dict(dummy_cfg.get('training', {}))
    tcfg_override['rouge_eval_instances_per_task'] = int(args.instances_per_task)
    tcfg_override['rouge_eval_max_new_tokens'] = int(args.max_new_tokens)
    dummy_cfg['training'] = tcfg_override

    trainer = LoRATrainer(
        cfg=dummy_cfg, model=model, algorithm=algorithm,
        train_loader=eval_loader, eval_loader=eval_loader,
        device=device, use_wandb=False)

    print(f'[per_task] running ROUGE eval ...')
    result = trainer.run_rouge_eval(
        instances_per_task=int(args.instances_per_task),
        max_new_tokens=int(args.max_new_tokens))

    payload = {
        'method': args.method,
        'seed': seed,
        'run_dir': args.run_dir,
        'rougeL_mean': result['rougeL_mean'],
        'exact_match_mean': result['exact_match_mean'],
        'num_tasks': result['num_tasks'],
        'instances_per_task': int(args.instances_per_task),
        'max_new_tokens': int(args.max_new_tokens),
        'per_task': result['per_task'],
        'config_summary': {
            'model_type': cfg['model'].get('type'),
            'algorithm_type': (cfg.get('algorithm', {}) or {}).get('type', 'none'),
            'num_experts': K,
            'rank': cfg['model'].get('rank'),
            'shared_expert_rank': cfg['model'].get('shared_expert_rank', 0),
        },
    }

    out_json = (args.out_json or
                f'outputs/eval_lora_per_task/{args.method}_s{seed}.json')
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[per_task] wrote {out_json}')
    print(f'[per_task] mean ROUGE-L={result["rougeL_mean"]:.4f}  '
          f'EM={result["exact_match_mean"]:.4f}  '
          f'tasks={result["num_tasks"]}')


if __name__ == '__main__':
    main()

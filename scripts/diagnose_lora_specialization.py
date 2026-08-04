#!/usr/bin/env python3
"""Per-cluster × per-branch ROUGE-L drop when each branch is zero-ablated.

Mirror of `diagnose_nlp_specialization.py` for LoRA SuperNI (K=20 task clusters).
Output: 20×20 Δ matrix → row = cluster_id, col = ablated branch idx.
Diagonal argmin (branch k hurts cluster k most) is the specialization signal.

Usage:
    python scripts/diagnose_lora_specialization.py \
        --run_dir outputs/rtx5090_lora_faithful/ours_s42

Single-seed enough — output is a heatmap (figure), not a statistical claim.
~2h on 1 RTX 5090 (21 ROUGE-eval passes × ~6 min each at ipt=10).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import yaml


def _load_cfg(run_dir):
    import glob
    cands = sorted(glob.glob(f'{run_dir}/_tmp*.yaml')) or sorted(glob.glob(f'{run_dir}/*.yaml'))
    if not cands:
        raise FileNotFoundError(f'no yaml in {run_dir}')
    with open(cands[0]) as f:
        return yaml.safe_load(f)


def _is_specdrop_family(algorithm):
    if algorithm is None:
        return False
    return type(algorithm).__name__ in ('SoftSpecDrop', 'NoDropout', 'HardCategory')


@torch.no_grad()
def eval_per_cluster(trainer, ablate_branch, K):
    """Run ROUGE-L eval with `ablate_branch` (or None) zeroed.
    Returns {cluster_id: {'rouge_l': float, 'em': float, 'n': int}}.

    Routing-family dispatch:
      - SpecDrop family (algorithm = SoftSpecDrop/NoDropout/HardCategory):
        monkey-patch algorithm.get_mask to zero column k.
      - Learned routers (HydraLoRA / LoRAMoE / MoCLE — no external algorithm):
        install ablation hooks via scripts/_ablation_hooks.install_ablation.

    Re-uses trainer's existing run_rouge_eval machinery + Wang 2022 protocol.
    """
    use_mask = _is_specdrop_family(trainer.algorithm)
    if use_mask:
        algo = trainer.algorithm
        orig_get_mask = algo.get_mask

        def _patched_get_mask(category_ids, training=True, **kw):
            m = orig_get_mask(category_ids, training=training, **kw).clone()
            if ablate_branch is not None:
                m[:, ablate_branch] = 0.0
            return m
        algo.get_mask = _patched_get_mask
    else:
        # Learned-router path: hook into the model directly.
        if ablate_branch is not None:
            from scripts._ablation_hooks import install_ablation
            install_ablation(trainer.model, k=ablate_branch)

    try:
        # We need PER-CLUSTER ROUGE breakdown. SuperNIEvaluator returns a
        # per-task breakdown; map task_id → cluster_id via mapping in eval
        # loader's dataset (each example carries cluster_id).
        # Simplest path: run eval, get per-task scores, group by cluster_id
        # (need the cluster_id-per-task mapping from the dataset).
        result = trainer.run_rouge_eval(
            instances_per_task=int(trainer.cfg['training'].get('rouge_eval_instances_per_task', 10)),
            max_new_tokens=int(trainer.cfg['training'].get('rouge_eval_max_new_tokens', 128)))
        # SuperNIEvaluator.evaluate() returns per-task stats under 'per_task'
        # as {tid: {'rougeL': ..., 'exact_match': ..., 'n': ...}}. Group by
        # task → cluster: pull mapping from trainer's eval_loader dataset.
        per_task = result.get('per_task', {})

        # Map task_id → cluster_id
        ds = trainer.eval_loader.dataset
        if hasattr(ds, 'task_to_cluster'):
            task_to_cluster = ds.task_to_cluster
        elif hasattr(ds, 'mapping') and 'task_to_cluster' in ds.mapping:
            task_to_cluster = ds.mapping['task_to_cluster']
        else:
            task_to_cluster = {}
            for item in getattr(ds, 'items', []):
                task_to_cluster[item.get('task_id', '?')] = item.get('cluster_id', 0)

        per_cluster = defaultdict(lambda: {'rouge_l_sum': 0.0, 'em_sum': 0.0, 'n': 0})
        for task_id, stats in per_task.items():
            c = task_to_cluster.get(task_id, 0)
            per_cluster[c]['rouge_l_sum'] += float(stats['rougeL'])
            per_cluster[c]['em_sum'] += float(stats.get('exact_match', 0.0))
            per_cluster[c]['n'] += 1
        return {
            int(c): {
                'rouge_l': v['rouge_l_sum'] / max(1, v['n']),
                'em': v['em_sum'] / max(1, v['n']),
                'n_tasks': v['n'],
            }
            for c, v in per_cluster.items()
        }
    finally:
        if use_mask:
            algo.get_mask = orig_get_mask
        elif ablate_branch is not None:
            from scripts._ablation_hooks import uninstall_ablation
            uninstall_ablation(trainer.model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--out_json', default=None)
    parser.add_argument('--branch_subset', type=str, default=None,
                        help='Comma-separated branch indices to ablate, '
                             'e.g. "0,1,2,3,4,5,6". Default = all 0..K-1. '
                             'Use to parallelize across GPUs; merge shard '
                             'JSONs with scripts/merge_lora_diag.py.')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline (no-ablation) eval. Use on shards '
                             'where another GPU is computing baseline. The '
                             'shard JSON will lack baseline_per_cat; merger '
                             'fills it from the baseline-running shard.')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Diag-LoRA] run={args.run_dir}  device={device}')

    cfg = _load_cfg(args.run_dir)
    # K determination: SpecDrop family + lora_moe use num_experts; HydraLoRA
    # uses num_B_heads; MoCLE uses num_task_experts. Try in order.
    K = (cfg['model'].get('num_experts')
         or cfg['model'].get('num_B_heads')
         or cfg['model'].get('num_task_experts')
         or 20)
    M = cfg['data'].get('num_clusters', 20)

    # Build model + load LoRA-only state
    from models.lora_models import build_lora_model, _apply_gradient_checkpointing_override
    model = build_lora_model(cfg)
    _apply_gradient_checkpointing_override(model, cfg)
    model = model.to(device)
    ckpt = torch.load(f'{args.run_dir}/best.pt', map_location=device, weights_only=False)
    lora_state = ckpt.get('lora_state') or ckpt.get('state_dict') or ckpt
    model.load_state_dict(lora_state, strict=False)
    model.eval()

    # Build algorithm. For ours (`soft_specdrop`) reproduce the trained mask
    # state at terminal warmup. For `no_dropout` baseline (mb_lora_no_routing
    # checkpoints), skip warmup advance entirely — uniform 1/K mask has no
    # warmup state to advance; build NoDropout directly.
    from scripts._diag_helpers import (advance_softspecdrop_to_terminal,
                                         require_keys)
    require_keys(cfg, ('algorithm', 'training'), f'cfg in {args.run_dir}')
    acfg = cfg['algorithm']
    algo_type = acfg.get('type', 'soft_specdrop')
    if algo_type == 'no_dropout':
        from algorithms.no_dropout import NoDropout
        algorithm = NoDropout(num_modules=K, num_categories=M)
        # NoDropout has no warmup state — get_mask returns uniform 1/K
        # regardless of step/epoch. Branch ablation still measures the
        # contribution of each branch under the uniform-routing baseline.
    elif algo_type == 'soft_specdrop':
        from algorithms.soft_specdrop import SoftSpecDrop
        require_keys(acfg, ('p_active', 'p_inactive', 'assignment', 'warmup_ratio',
                             'warmup_schedule', 'warmup_unit',
                             'amplification_beta'),
                      f'cfg["algorithm"] in {args.run_dir}')
        require_keys(cfg['training'], ('epochs',),
                      f'cfg["training"] in {args.run_dir}')
        algorithm = SoftSpecDrop(
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
        advance_softspecdrop_to_terminal(algorithm, cfg['training']['epochs'])
    elif algo_type in ('none', None):
        # Learned-router methods (HydraLoRA / LoRAMoE / MoCLE). Routing is
        # baked into the model's forward; no external algorithm. eval_per_cluster
        # uses scripts/_ablation_hooks.install_ablation to zero expert k.
        algorithm = None
    else:
        raise ValueError(
            f'cfg["algorithm"]["type"]={algo_type!r} not supported by '
            f'diagnose_lora_specialization.py — expected "soft_specdrop", '
            f'"no_dropout", or "none" (learned routers)')

    # Build trainer (eval-only)
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
    dummy_cfg = dict(cfg); dummy_cfg.setdefault('output_dir', args.run_dir)
    trainer = LoRATrainer(
        cfg=dummy_cfg, model=model, algorithm=algorithm,
        train_loader=eval_loader, eval_loader=eval_loader,
        device=device, use_wandb=False)

    # Resolve branch subset (default = all branches 0..K-1).
    if args.branch_subset is not None:
        branches_to_run = [int(b.strip()) for b in args.branch_subset.split(',')
                            if b.strip()]
        if any(b < 0 or b >= K for b in branches_to_run):
            raise ValueError(
                f'--branch_subset {branches_to_run} contains out-of-range '
                f'index (K={K}; valid range 0..{K-1}).')
    else:
        branches_to_run = list(range(K))
    is_shard = (args.branch_subset is not None) or args.skip_baseline

    if not args.skip_baseline:
        print(f'[Baseline (no ablation)]', flush=True)
        baseline = eval_per_cluster(trainer, ablate_branch=None, K=K)
        print('   baseline rouge_L per cluster: '
              + str({c: round(v['rouge_l'], 4) for c, v in baseline.items()}))
    else:
        baseline = None
        print('[Baseline (no ablation)] SKIPPED — relying on merger to fill '
              'from another shard.', flush=True)

    print(f'[Ablating {len(branches_to_run)} branches: {branches_to_run}]',
          flush=True)
    ablated = {}
    for k in branches_to_run:
        print(f'   branch {k}/{K-1}...', flush=True)
        ablated[k] = eval_per_cluster(trainer, ablate_branch=k, K=K)

    if not is_shard:
        # Full run: aggregate stats inline (backward-compat path).
        cats = sorted(baseline.keys())
        Mc = len(cats)
        delta = np.zeros((Mc, K))
        for ri, c in enumerate(cats):
            for ci in range(K):
                base_r = baseline[c]['rouge_l']
                abl_r = ablated[ci].get(c, baseline[c])['rouge_l']
                delta[ri, ci] = abl_r - base_r

        diag_hits = 0
        for ri, c in enumerate(cats):
            argmin_branch = int(np.argmin(delta[ri, :]))
            if argmin_branch == c:
                diag_hits += 1
        print()
        print('=' * 80)
        print(f'  Per-cluster × per-branch ROUGE-L drop when branch k zero-ablated')
        print('=' * 80)
        print(f'Diagonal hits: {diag_hits}/{Mc}  (random: {1/K:.0%}; full spec: 100%)')
        print(f'Max |Δ|: {np.max(np.abs(delta)):.4f} ROUGE-L')
        print(f'Mean |Δ|: {np.mean(np.abs(delta)):.4f}')
    else:
        # Shard run: write partial JSON; merger does aggregation.
        cats = sorted(baseline.keys()) if baseline is not None else []
        Mc = len(cats)
        print()
        print(f'[Shard mode] {len(branches_to_run)} branches done, '
              f'baseline={"computed" if baseline is not None else "skipped"}. '
              f'Run scripts/merge_lora_diag.py on all shard JSONs to produce '
              f'final aggregated output.')

    out_json = args.out_json or f'outputs/analysis/lora_diag/{os.path.basename(args.run_dir)}.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    config_summary = {
        'algorithm': 'soft_specdrop',
        'p_active': acfg.get('p_active'),
        'amplification_beta': acfg.get('amplification_beta'),
        'shared_expert_rank': cfg['model'].get('shared_expert_rank', 0),
        'num_experts': K,
    }
    if is_shard:
        # Partial — only the bits this shard computed; merger fills the rest.
        payload = {
            'partial': True,
            'branches_run': sorted(branches_to_run),
            'run_dir': args.run_dir,
            'K': K,
            'config_summary': config_summary,
            'baseline_per_cat': (
                {int(c): baseline[c] for c in baseline}
                if baseline is not None else None),
            'ablated_per_cat': {
                int(k): {int(c): ablated[k][c] for c in ablated[k]}
                for k in ablated},
        }
        with open(out_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'\n[Diag-LoRA] wrote SHARD {out_json} '
              f'(branches={sorted(branches_to_run)}, '
              f'baseline={"included" if baseline is not None else "skipped"})')
        return
    payload = {
        'run_dir': args.run_dir,
        'config_summary': config_summary,
        'cats': cats,
        'baseline_per_cat': {int(c): baseline[c] for c in cats},
        'ablated_per_cat': {
            int(k): {int(c): ablated[k].get(c, baseline[c]) for c in cats}
            for k in ablated
        },
        'delta_matrix': delta.tolist(),
        'diag_hits': diag_hits, 'n_cats': Mc,
        'max_abs_delta': float(np.max(np.abs(delta))),
        'mean_abs_delta': float(np.mean(np.abs(delta))),
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\n[Diag-LoRA] wrote {out_json}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Entry point for LoRA post-training experiments on SuperNaturalInstructions.

Parallel to run.py (CV) and run_nlp.py (SlimPajama LM). Reads a YAML config,
builds a LoRA model (models.lora_models.build_lora_model), optional routing
algorithm (SoftSpecDrop / NoDropout for MultiBranch-LoRA), SuperNI loaders
(data.natural_instructions), and fires up the LoRATrainer.

Usage:
    python run_lora.py --config configs/lora/ours.yaml --device cuda
    python run_lora.py --config configs/lora/single_lora_r16.yaml --wandb

Backward-compat: entirely separate from run.py; does not import CV/NLP code.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings

import numpy as np
import torch

from utils.config import load_config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_algorithm(cfg, num_clusters):
    """Build a routing algorithm only for MultiBranch-LoRA methods.

    Returns None for Single-LoRA / HydraLoRA / LoRAMoE / MoCLE (they handle
    their own routing internally).
    """
    acfg = cfg.get('algorithm', {}) or {}
    atype = acfg.get('type', 'none')
    if atype == 'none':
        return None

    num_experts = cfg['model'].get('num_experts', 20)

    if atype == 'no_dropout':
        from algorithms.no_dropout import NoDropout
        return NoDropout(num_modules=num_experts, num_categories=num_clusters)

    if atype == 'soft_specdrop':
        from algorithms.soft_specdrop import SoftSpecDrop
        tcfg = cfg['training']
        return SoftSpecDrop(
            num_modules=num_experts, num_categories=num_clusters,
            p_active=acfg.get('p_active', 0.6),
            p_inactive=acfg.get('p_inactive', 0.4),
            assignment=acfg.get('assignment', 'round_robin'),
            warmup_ratio=acfg.get('warmup_ratio', 1.0),
            total_epochs=tcfg.get('epochs', 3),
            assignment_seed=acfg.get('assignment_seed', cfg.get('seed', 42)),
            frac_per_category=acfg.get('frac_per_category', None),
            amplification_beta=acfg.get('amplification_beta', 1.0),
            warmup_schedule=acfg.get('warmup_schedule', 'cosine'),
            # LoRA instruction tuning has PER-STEP LR — match pa warmup to it.
            warmup_unit=acfg.get('warmup_unit', 'step'),
        )

    raise ValueError(
        f"Unknown algorithm.type={atype!r} for LoRA track. Supported: "
        f"'none', 'no_dropout', 'soft_specdrop'.")


def main():
    parser = argparse.ArgumentParser(description='SpecDrop LoRA post-train')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg['output_dir'] = args.output_dir
    if args.seed is not None:
        cfg['seed'] = args.seed
    if 'output_dir' not in cfg:
        cfg['output_dir'] = './outputs/lora_default'
    os.makedirs(cfg['output_dir'], exist_ok=True)

    set_seed(cfg.get('seed', 42))

    # ── 1. Tokenizer + data ────────────────────────────────────────────
    dcfg = cfg['data']
    data_root = dcfg['data_root']
    base_name = cfg['model']['base_model_name']

    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "Install LoRA deps: pip install transformers>=4.40 accelerate>=0.30 "
            "peft>=0.10 rouge-score")

    print(f"[run_lora] loading tokenizer: {base_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from data.natural_instructions import get_superni_dataloaders
    # data_subset_seed is FIXED at 42 (not cfg.seed) by design: we want
    # all 3 experiment seeds to see the IDENTICAL tokenised data so the
    # 3-seed std measures training variance only (model init, shuffle
    # order, dropout, GPU non-determinism), not subset-choice variance.
    # Same convention as ViT's train_subset_seed (data/imagenet.py) and
    # hash_seed / routing_seed across the CIFAR baselines. Also means all
    # seeds share ONE pre-tokenised cache file on disk → no per-seed
    # re-tokenisation.
    train_loader, eval_loader, mapping = get_superni_dataloaders(
        data_root=data_root,
        tokenizer=tokenizer,
        batch_size=cfg['training'].get('batch_size_per_device', 16),
        max_seq_len=cfg['training'].get('max_seq_len', 2048),
        instances_per_task_train=dcfg.get('instances_per_task_train', 100),
        instances_per_task_eval=dcfg.get('instances_per_task_eval', 100),
        subset_frac_train=dcfg.get('subset_frac_train', 1.0),
        num_workers=dcfg.get('num_workers', 4),
        data_subset_seed=dcfg.get('data_subset_seed', 42),
        K=dcfg.get('num_clusters', 20),
        cache_dir=dcfg.get('cluster_cache_dir', './data_cache/lora'),
    )
    print(f"[run_lora] train: {len(train_loader.dataset)} examples / "
          f"{len(mapping['train_tasks'])} tasks")
    print(f"[run_lora] eval:  {len(eval_loader.dataset)} examples / "
          f"{len(mapping['test_tasks'])} tasks")
    K = mapping['K']

    # ── 2. Populate per-category fractions (for SoftSpecDrop β amplification) ──
    acfg = cfg.get('algorithm', {}) or {}
    if acfg.get('type') == 'soft_specdrop' and acfg.get('frac_per_category') is None:
        # Count tasks per cluster_id over train split.
        from collections import Counter
        counter = Counter(mapping['task_to_cluster'][tid]
                          for tid in mapping['train_tasks'])
        fracs = [counter.get(c, 0) for c in range(K)]
        total = sum(fracs)
        fracs = [f / total if total > 0 else 1.0 / K for f in fracs]
        acfg['frac_per_category'] = fracs
        print(f"[run_lora] frac_per_category (computed): {[f'{f:.3f}' for f in fracs]}")

    # ── 3. Model ──────────────────────────────────────────────────────
    from models.lora_models import (build_lora_model, MoCLEModel,
                                     _apply_gradient_checkpointing_override)
    print(f"[run_lora] building model: {cfg['model']['type']}")
    model = build_lora_model(cfg)
    # Respect cfg.model.gradient_checkpointing override if present (default
    # stays True from BaseLoRAModel._inject). Set false for compute-bound
    # speedups when VRAM allows.
    _apply_gradient_checkpointing_override(model, cfg)
    print(f"[run_lora] LoRA trainable params: {model.num_trainable:,}")

    # Param-budget sanity check (±3% envelope for LoRA integer rounding).
    # Skip single_lora_r16 — PEFT reference point, NOT capacity-matched.
    from utils.sanity_check import check_param_budget
    mtype = cfg['model']['type']
    if not (mtype == 'single_lora' and cfg['model'].get('rank', 0) < 100):
        check_param_budget(cfg, model)

    # ── 3b. MoCLE cluster-embedding init from BGE-large centroids ──────
    # Gou 2024 Sec 3.3: the cluster_embeddings table (keyed by cluster_id)
    # is initialised to the BGE centroid of that cluster's training-task
    # instructions. Without this init the gate has no task-semantic prior
    # and the top-1 softmax at step 0 collapses to expert 0 for all
    # samples (random-noise tiebreak, amplified by temperature τ=0.05).
    if isinstance(model, MoCLEModel):
        from data.natural_instructions import _read_task, _read_split
        from data.superni_domain_map import build_bge_centroids
        print("[run_lora] MoCLE detected; building BGE-large centroids "
              "for cluster_embeddings init (Gou 2024 Sec 3.3) ...")
        task_definitions = {}
        for tid in _read_split(data_root, 'train'):
            try:
                task = _read_task(data_root, tid)
            except FileNotFoundError:
                continue
            definition = task.get('Definition', [])
            if isinstance(definition, list):
                definition = ' '.join(x for x in definition if x)
            task_definitions[tid] = (definition or '').strip()
        centroids = build_bge_centroids(
            task_definitions, mapping, device=args.device)
        import torch as _torch
        import numpy as _np
        if isinstance(centroids, _np.ndarray):
            centroids = _torch.from_numpy(centroids)
        centroids = centroids.to(dtype=_torch.float32)
        model.init_all_cluster_embeddings(centroids)
        print(f"[run_lora] MoCLE: wrote BGE centroids → "
              f"cluster_embeddings (K={centroids.shape[0]} × D={centroids.shape[1]}); "
              f"gate init N(0, 0.02²), noise_std=1/E per Gou 2024 defaults")

    # ── 3c. torch.compile (opt-in via cfg.training.torch_compile) ──────
    # Default: ON with mode='default' + dynamic=True. This is the right
    # combo for SuperNI's variable-length batches (longest-in-batch pad):
    # inductor performs kernel fusion + constant folding, dynamo handles
    # dynamic shapes without recompile thrash, and we don't pay the
    # padding-to-max cost that CUDA-graph modes would require.
    #
    # Why not 'reduce-overhead' (which NLP 500M uses): that mode's gain
    # is CUDA-graph reuse, which REQUIRES static shapes. SlimPajama's
    # token packing is naturally fixed-shape so it works there; SuperNI
    # is naturally variable-shape, and we measured a NET LOSS after
    # adding fixed padding (1.91 → 2.85 s/it on 5090). Staying with
    # 'default' + dynamic=True is the principled choice for this track.
    #
    # HF Llama + LoRA + grad_ckpt is a fragile combo for compile;
    # if it blows up, set `torch_compile: false` in the YAML or try
    # `compile_mode: 'reduce-overhead'` with fixed-length padding (see
    # _collate target_len kwarg). trainer_lora.py pierces the
    # OptimizedModule wrapper via `_inner_model` so isinstance checks +
    # checkpoint save keys stay clean regardless.
    # Default: compile OFF. Measured on 5090 over three configurations
    # (reduce-overhead+dynamic / reduce-overhead+fixed_pad / default+dynamic)
    # all came in slower than eager expectation (~1.9-2.8 s/it observed vs
    # seq=1024 eager baseline ~1.0-1.2 s/it). HF Llama + LoRA adapters
    # + gradient checkpointing + variable-length SuperNI batches is a combo
    # for which torch.compile's fusion benefit doesn't outrun its overhead.
    # (NLP 500M uses reduce-overhead because its token packing gives fixed
    # shapes — SuperNI doesn't share that property.)
    # Opt-in via `cfg.training.torch_compile: true` if you want to retry.
    _compile_mode = None
    _compile_dynamic = None
    if cfg['training'].get('torch_compile', False):
        _compile_mode = cfg['training'].get('compile_mode', 'default')
        _compile_dynamic = bool(cfg['training'].get('compile_dynamic', True))
        print(f"[run_lora] torch.compile: mode={_compile_mode!r}, "
              f"dynamic={_compile_dynamic}")
        model = torch.compile(model, mode=_compile_mode, dynamic=_compile_dynamic)
    else:
        print("[run_lora] torch.compile: DISABLED (default for LoRA; measured "
              "slower than eager under all tested modes — variable-length "
              "SuperNI + grad_ckpt doesn't amortize compile overhead)")

    # ── 4. Algorithm ───────────────────────────────────────────────────
    algorithm = _build_algorithm(cfg, num_clusters=K)
    _atype = (cfg.get('algorithm', {}) or {}).get('type', 'none')
    if algorithm is not None:
        print(f"[run_lora] Algorithm: {_atype} ({type(algorithm).__name__})")
    else:
        _mt = cfg['model']['type']
        if _mt in ('single_lora', 'lora_moe', 'hydra_lora', 'mocle'):
            print(f"[run_lora] Algorithm: none (routing native to {_mt})")
        else:
            print(f"[run_lora] Algorithm: none (dense/baseline)")

    # ── 4b. Startup settings log (mirror run.py / run_nlp.py convention) ──
    # Every run prints its complete experiment config so post-hoc log grep
    # reveals exactly what was set for each cell. Questions like "was AMP
    # on?" / "was compile on?" / "what was pa?" must be answerable from
    # the first ~20 lines of the training log alone.
    _acfg, _mcfg, _tcfg = (cfg.get('algorithm', {}) or {}), cfg['model'], cfg['training']
    _parts = []
    if 'p_active' in _acfg:
        _parts.append(f"pa={_acfg['p_active']}")
    if 'p_inactive' in _acfg:
        _parts.append(f"pi={_acfg['p_inactive']}")
    if 'amplification_beta' in _acfg:
        _parts.append(f"β={_acfg['amplification_beta']}")
    if 'warmup_ratio' in _acfg:
        _parts.append(f"wr={_acfg['warmup_ratio']}")
    if 'warmup_schedule' in _acfg:
        _parts.append(f"schedule={_acfg['warmup_schedule']}")
    if 'warmup_unit' in _acfg:
        _parts.append(f"unit={_acfg['warmup_unit']}")
    if 'assignment' in _acfg:
        _parts.append(f"assignment={_acfg['assignment']}")
    if 'num_experts' in _mcfg:
        _parts.append(f"K={_mcfg['num_experts']}")
    if 'rank' in _mcfg:
        _parts.append(f"r={_mcfg['rank']}")
    if _mcfg.get('shared_expert_rank', 0):
        _parts.append(f"r_SE={_mcfg['shared_expert_rank']}")
    if _parts:
        print(f"[run_lora] Ablation hyperparams: {' | '.join(_parts)}")

    # Runtime knobs — single-line snapshot of "what's on/off" for this run.
    # grad_ckpt state is read from the HF base (set by build_lora_model);
    # pierces torch.compile wrapper via `_orig_mod` if present.
    _base_for_gc = getattr(model, '_orig_mod', model)
    _gc = getattr(getattr(_base_for_gc, 'base', _base_for_gc),
                  'is_gradient_checkpointing', False)
    _compile_str = (
        f"mode={_compile_mode!r}, dynamic={_compile_dynamic}"
        if _compile_mode is not None else "DISABLED"
    )
    print(
        f"[run_lora] Runtime: AMP={_tcfg.get('amp_dtype', 'bf16')} "
        f"| grad_ckpt={_gc} (use_reentrant=False) "
        f"| torch.compile={_compile_str} "
        f"| deterministic_algos=True (warn_only) "
        f"| attn={_mcfg.get('attn_implementation', 'sdpa')} "
        f"| torch_dtype={_mcfg.get('torch_dtype', 'bfloat16')}"
    )
    _eff_bs = int(_tcfg.get('batch_size_per_device', 1)) * int(_tcfg.get('grad_accum_steps', 1))
    print(
        f"[run_lora] Training: epochs={_tcfg.get('epochs')} "
        f"| bs/device={_tcfg.get('batch_size_per_device')} "
        f"| grad_accum={_tcfg.get('grad_accum_steps', 1)} "
        f"| effective_batch={_eff_bs} "
        f"| max_seq_len={_tcfg.get('max_seq_len')} "
        f"| lr={_tcfg.get('lr')} (warmup_ratio={_tcfg.get('warmup_ratio_lr', 0.03)}) "
        f"| wd={_tcfg.get('weight_decay', 0.0)} "
        f"| max_grad_norm={_tcfg.get('max_grad_norm', 1.0)}"
    )

    # ── 5. Trainer ────────────────────────────────────────────────────
    from training.trainer_lora import LoRATrainer
    trainer = LoRATrainer(
        cfg=cfg, model=model, algorithm=algorithm,
        train_loader=train_loader, eval_loader=eval_loader,
        device=args.device, use_wandb=args.wandb)

    # ── 6. Train ──────────────────────────────────────────────────────
    results = trainer.train()
    if results.get('selection_metric') == 'eval_rouge_l':
        print(f"[run_lora] done. best epoch={results.get('best_epoch')} "
              f"ROUGE-L={results.get('eval_rouge_l'):.4f} "
              f"EM={results.get('eval_exact_match'):.4f} "
              f"(eval_loss at best epoch = {results['best_eval_loss']:.4f})")
    else:
        print(f"[run_lora] done. best_eval_loss = {results['best_eval_loss']:.4f} "
              f"(ROUGE-L unavailable; selection fell back to eval_loss)")
    print(f"[run_lora] results → {cfg['output_dir']}/results.json")


if __name__ == '__main__':
    main()

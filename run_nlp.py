#!/usr/bin/env python3
"""Entry point for NLP (language modeling) experiments.

Usage:
    python run_nlp.py --config configs/nlp/soft_specdrop.yaml
    python run_nlp.py --config configs/nlp/dense.yaml --epochs 5  # quick test
"""

import argparse
import json
import os
import random
import numpy as np
import torch

from utils.config import load_config
from algorithms import build_algorithm
from training.trainer_nlp import NLPTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_nlp_model(cfg):
    """Build NLP model from config."""
    from models.transformer_lm import TransformerLM, MultiBranchTransformerLM
    from models.switch_transformer_lm import SwitchTransformerLM
    from models.hash_layers_transformer_lm import HashLayersTransformerLM
    from models.smoe_dropout_transformer_lm import SMoEDropoutTransformerLM
    from models.demix_transformer_lm import DemixTransformerLM

    mcfg = cfg['model']
    model_type = mcfg['type']

    if model_type == 'transformer_lm':
        return TransformerLM(
            vocab_size=mcfg.get('vocab_size', 50257),
            hidden_dim=mcfg.get('hidden_dim', 384),
            num_layers=mcfg.get('num_layers', 6),
            num_heads=mcfg.get('num_heads', 6),
            ffn_dim=mcfg.get('ffn_dim', 1536),
            max_seq_len=mcfg.get('max_seq_len', 512),
            dropout=mcfg.get('dropout', 0.1),
        )
    elif model_type == 'multi_branch_transformer_lm':
        # Per-branch FFN width: accept either scalar ffn_dim_per_branch
        # (uniform, existing behavior) or list ffn_dims_per_branch (non-
        # uniform, data-proportional allocation). If both are set, prefer
        # the list and warn; if neither, default to uniform 220.
        if 'ffn_dims_per_branch' in mcfg:
            if 'ffn_dim_per_branch' in mcfg:
                print("  [warn] both 'ffn_dims_per_branch' (list) and "
                      "'ffn_dim_per_branch' (scalar) set; using the list.")
            ffn_widths = mcfg['ffn_dims_per_branch']
        else:
            ffn_widths = mcfg.get('ffn_dim_per_branch', 220)
        return MultiBranchTransformerLM(
            vocab_size=mcfg.get('vocab_size', 50257),
            hidden_dim=mcfg.get('hidden_dim', 384),
            num_layers=mcfg.get('num_layers', 6),
            num_heads=mcfg.get('num_heads', 6),
            num_branches=mcfg.get('num_branches', 7),
            ffn_dim_per_branch=ffn_widths,  # scalar or list
            max_seq_len=mcfg.get('max_seq_len', 512),
            dropout=mcfg.get('dropout', 0.1),
            sandwich_dim=mcfg.get('sandwich_dim', 0),
            shared_expert_dim=mcfg.get('shared_expert_dim', 0),
        )
    elif model_type == 'switch_transformer_lm':
        return SwitchTransformerLM(
            vocab_size=mcfg.get('vocab_size', 50257),
            hidden_dim=mcfg.get('hidden_dim', 384),
            num_layers=mcfg.get('num_layers', 6),
            num_heads=mcfg.get('num_heads', 6),
            # Paper-canonical N=32 with param-matched narrow experts (32×48=1536).
            num_experts=mcfg.get('num_experts', 32),
            ffn_dim_per_expert=mcfg.get('ffn_dim_per_expert', 48),
            max_seq_len=mcfg.get('max_seq_len', 512),
            load_balance_weight=mcfg.get('load_balance_weight', 0.01),
            dropout=mcfg.get('dropout', 0.1),
        )
    elif model_type == 'hash_layers_transformer_lm':
        return HashLayersTransformerLM(
            vocab_size=mcfg.get('vocab_size', 50257),
            hidden_dim=mcfg.get('hidden_dim', 384),
            num_layers=mcfg.get('num_layers', 6),
            num_heads=mcfg.get('num_heads', 6),
            # Paper-canonical N=8, param-matched (8×192=1536).
            num_experts=mcfg.get('num_experts', 8),
            ffn_dim_per_expert=mcfg.get('ffn_dim_per_expert', 192),
            max_seq_len=mcfg.get('max_seq_len', 512),
            hash_seed=mcfg.get('hash_seed', 42),
            dropout=mcfg.get('dropout', 0.1),
        )
    elif model_type == 'smoe_dropout_transformer_lm':
        return SMoEDropoutTransformerLM(
            vocab_size=mcfg.get('vocab_size', 50257),
            hidden_dim=mcfg.get('hidden_dim', 384),
            num_layers=mcfg.get('num_layers', 6),
            num_heads=mcfg.get('num_heads', 6),
            # Paper Fig 5 tests N ∈ {4, 8, 16}; N=16 is the upper end, param-matched.
            num_experts=mcfg.get('num_experts', 16),
            ffn_dim_per_expert=mcfg.get('ffn_dim_per_expert', 96),
            max_seq_len=mcfg.get('max_seq_len', 512),
            k_init=mcfg.get('k_init', 1),
            # Paper (Chen 2023) has NO per-expert Bernoulli dropout.
            expert_drop_prob=mcfg.get('expert_drop_prob', 0.0),
            router_seed=mcfg.get('router_seed', 42),
            dropout=mcfg.get('dropout', 0.1),
        )
    elif model_type == 'demix_transformer_lm':
        return DemixTransformerLM(
            vocab_size=mcfg.get('vocab_size', 50257),
            hidden_dim=mcfg.get('hidden_dim', 384),
            num_layers=mcfg.get('num_layers', 6),
            num_heads=mcfg.get('num_heads', 6),
            num_domains=mcfg.get('num_domains', 7),
            ffn_dim_per_expert=mcfg.get('ffn_dim_per_expert', 220),
            max_seq_len=mcfg.get('max_seq_len', 512),
            dropout=mcfg.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown NLP model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='SpecDrop NLP experiments')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_train_tokens', type=int, default=None)
    parser.add_argument('--max_val_tokens', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., outputs/exp/latest.pt)')
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='Enable Weights & Biases logging (default: on)')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='SpecDrop',
                        help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='W&B run name (defaults to experiment_name)')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=None,
                        help='W&B tags for this run')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override cfg.seed for reproducibility sweeps. '
                             'When None (default), uses cfg.seed (existing '
                             'temp-YAML convention preserved).')
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        # CLI override: write to BOTH sections so whichever the loader reads
        # picks it up (historical bug: loader reads `data.batch_size` but
        # YAMLs write `training.batch_size`; see line 192 for the fix).
        cfg.setdefault('data', {})['batch_size'] = args.batch_size
        cfg.setdefault('training', {})['batch_size'] = args.batch_size
    if args.output_dir is not None:
        cfg['output_dir'] = args.output_dir
    if args.max_train_tokens is not None:
        cfg['data']['max_train_tokens'] = args.max_train_tokens
    if args.max_val_tokens is not None:
        cfg['data']['max_val_tokens'] = args.max_val_tokens

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Seed: CLI override > cfg.seed > default 42.
    seed = args.seed if args.seed is not None else cfg.get('seed', 42)
    cfg['seed'] = seed   # propagate so downstream cfg-driven code agrees
    set_seed(seed)

    print(f"=== {cfg.get('experiment_name', 'nlp_experiment')} ===")
    print(f"Device: {device}")

    # Data
    from data.slimpajama import get_slimpajama_dataloaders, NUM_DOMAINS
    dcfg = cfg.get('data', {})
    tcfg = cfg.get('training', {})
    # batch_size: prefer cfg['training'] (YAML convention), fall back to
    # cfg['data'] (legacy CLI override path), then default 32.
    # ⚠ Historical bug (pre-2026-04-24): loader previously read only
    # `data.batch_size`, so every NLP YAML's `training.batch_size: 64` was
    # silently ignored and training ran at default 32. All paper-era NLP
    # numbers are from bs=32 training; the paper setup section discloses
    # this. Fixed here to respect training.batch_size for any future run.
    _bs = tcfg.get('batch_size', dcfg.get('batch_size', 32))
    train_loader, val_loader, vocab_size = get_slimpajama_dataloaders(
        data_dir=dcfg.get('data_dir', './data_cache/slimpajama'),
        batch_size=_bs,
        max_seq_len=cfg['model'].get('max_seq_len', 512),
        num_workers=dcfg.get('num_workers', 4),
        max_train_tokens=dcfg.get('max_train_tokens', 500_000_000),
        max_val_tokens=dcfg.get('max_val_tokens', 5_000_000),
        cluster_label_path_train=dcfg.get('cluster_label_path_train'),
        cluster_label_path_val=dcfg.get('cluster_label_path_val'),
    )

    # Override vocab_size from data
    cfg['model']['vocab_size'] = vocab_size

    # Override num_categories for algorithm. Default to SlimPajama's 7 source
    # domains. If Phase W-style cluster labels are supplied, read the true
    # cluster count from the cluster cache (supports K ≠ 7, e.g. optimal-k
    # experiments scanning K ∈ {5, 10, 14}).
    if 'algorithm' in cfg:
        num_cats = NUM_DOMAINS
        cluster_path = dcfg.get('cluster_label_path_train')
        if cluster_path:
            from data.slimpajama import load_cluster_labels
            cblob = load_cluster_labels(cluster_path)
            num_cats = int(cblob['n_clusters'])
            print(f"  [cluster] num_categories ← cluster cache n_clusters = {num_cats}")
        cfg['algorithm']['_num_categories'] = num_cats

    # Model
    model = build_nlp_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['type']} | Params: {total_params:,}")

    # Sanity check: param budget vs baseline
    from utils.sanity_check import check_param_budget
    check_param_budget(cfg, model)

    # Algorithm
    algorithm = build_algorithm(cfg)
    if algorithm is not None:
        print(f"Algorithm: {cfg['algorithm']['type']}")
        if hasattr(model, 'mask_scale') and algorithm.expected_mask_sum is not None:
            model.mask_scale = algorithm.expected_mask_sum
            print(f"  Fixed mask scale: {model.mask_scale:.4f}")
        # Per-category routing table (only when soft_specdrop received fracs).
        if cfg.get('algorithm', {}).get('frac_per_category') is not None \
                and hasattr(algorithm, '_p_active_target'):
            from data.slimpajama import DOMAIN_NAMES
            fracs = cfg['algorithm']['frac_per_category']
            pa_t = algorithm._p_active_target.tolist()
            pi_t = algorithm._p_inactive_target.tolist()
            beta = cfg['algorithm'].get('amplification_beta', 1.0)
            # When cluster labels override source domain_ids (Phase W), the
            # category index no longer maps to a SlimPajama domain name.
            use_cluster = dcfg.get('cluster_label_path_train') is not None
            label_col = 'cluster' if use_cluster else 'domain'
            names = ([f'cluster_{c}' for c in range(len(fracs))]
                     if use_cluster else DOMAIN_NAMES[:len(fracs)])
            print(f"  Per-category routing (β={beta}):")
            print(f"    {label_col:<26} {'frac':>7} {'p_a^c':>8} {'p_i^c':>8} {'Σ':>7}")
            K = cfg['model'].get('num_branches', len(fracs))
            for c, name in enumerate(names):
                s_row = pa_t[c] + (K - 1) * pi_t[c]
                print(f"    {name:<26} {fracs[c]:>7.4f} "
                      f"{pa_t[c]:>8.4f} {pi_t[c]:>8.4f} {s_row:>7.4f}")
    else:
        native_moe = {'switch_transformer_lm', 'hash_layers_transformer_lm',
                      'smoe_dropout_transformer_lm', 'demix_transformer_lm'}
        if cfg['model']['type'] in native_moe:
            print(f"Algorithm: none (routing native to {cfg['model']['type']})")
        else:
            print("Algorithm: none (dense baseline)")

    # Ablation-axis hyperparams (print for post-hoc double-check of each run)
    _acfg, _mcfg = cfg.get('algorithm', {}), cfg.get('model', {})
    _parts = []
    if 'p_active' in _acfg:
        _parts.append(f"pa={_acfg['p_active']}")
    if 'p_inactive' in _acfg:
        _parts.append(f"pi={_acfg['p_inactive']}")
    if 'warmup_ratio' in _acfg:
        _parts.append(f"wr={_acfg['warmup_ratio']}")
    if 'assignment' in _acfg:
        _parts.append(f"assignment={_acfg['assignment']}")
    _ffn = _mcfg.get('ffn_dim_per_branch')
    _se = _mcfg.get('shared_expert_dim', 0) or 0
    if _ffn is not None:
        _parts.append(f"ffn_per_branch={_ffn}")
    if _se or 'shared_expert_dim' in _mcfg:
        if _ffn and _ffn > 0:
            _parts.append(f"SE_dim={_se} (SE_ratio={_se / _ffn:.2f}x)")
        else:
            _parts.append(f"SE_dim={_se}")
    if _parts:
        print(f"Ablation hyperparams: {' | '.join(_parts)}")

    # Wandb. Initialize with a 180s timeout (the 90s default has bitten us
    # with transient network hiccups), and degrade to no-wandb if init still
    # fails rather than dropping the whole training run. Pipeline uses `|
    # tee` so the python exit code is masked from the outer shell, which
    # means silent wandb failures used to waste a full 1.2h GPU hour.
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        algo_type = cfg.get('algorithm', {}).get('type', 'none')
        run_name = args.wandb_name or cfg.get('experiment_name', 'nlp_experiment')
        tags = args.wandb_tags or [algo_type, cfg['model']['type'], 'slimpajama']
        try:
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    **cfg,
                    'device': device,
                    'total_params': total_params,
                },
                tags=tags,
                settings=wandb.Settings(init_timeout=180),
            )
            # Drop log='gradients' — wandb.watch calls torch.histc() internally,
            # which is non-deterministic on CUDA under our `use_deterministic_algorithms`
            # setting, spamming UserWarnings every 100 steps. Param-only logging keeps
            # run metadata without triggering the warning. Per-step train/val metrics
            # still flow through trainer_nlp's explicit wandb.log() calls.
            wandb.watch(model, log='parameters', log_freq=100)
        except Exception as e:
            print(f"[warn] wandb.init failed ({type(e).__name__}: {e}); "
                  f"continuing without wandb.")
            use_wandb = False

    # Train
    trainer = NLPTrainer(cfg, model, algorithm, train_loader, val_loader, device,
                         use_wandb=use_wandb)
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
    results = trainer.train()

    print(f"\n=== Summary ===")
    print(f"Best Val PPL: {results['best_val_ppl']:.2f}")

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

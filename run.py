#!/usr/bin/env python3
"""Main entry point: load config, build model/algorithm, train, evaluate.

Usage:
    python run.py --config configs/cv/baseline.yaml
    python run.py --config configs/cv/specdrop.yaml
    python run.py --config configs/cv/specdrop.yaml --epochs 50   # override epochs
"""

import argparse
import os
import sys
import json
import random
import warnings
import numpy as np
import torch

# Suppress PIL warnings from corrupt EXIF metadata in ImageNet
warnings.filterwarnings("ignore", message="Metadata Warning")
warnings.filterwarnings("ignore", message="Corrupt EXIF data")

from utils.config import load_config
from data import get_dataloaders
from models import build_model
from algorithms import build_algorithm
from training import Trainer
from evaluation.metrics import evaluate_specialization


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Enforce deterministic algorithms globally; may raise error for
    # ops without deterministic implementation (set warn_only=True to
    # log warnings instead of crashing).
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    parser = argparse.ArgumentParser(description='SpecDrop experiments')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/mps/cpu)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation (load from checkpoint)')
    parser.add_argument('--eval_specialization', action='store_true',
                        help='Run specialization metrics (slower)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., outputs/exp/latest.pt)')
    parser.add_argument('--no-compile', dest='no_compile', action='store_true',
                        help='Disable torch.compile (P0 diagnostic for baseline issue)')
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='Enable Weights & Biases logging (default: on)')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='SpecDrop',
                        help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='W&B run name (defaults to experiment_name)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override cfg.seed for reproducibility sweeps. '
                             'When None (default), uses cfg.seed (existing '
                             'temp-YAML convention preserved).')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=None,
                        help='W&B tags for this run')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        cfg['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['data']['batch_size'] = args.batch_size
    if args.output_dir is not None:
        cfg['output_dir'] = args.output_dir
    if args.no_compile:
        cfg['_no_compile'] = True

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

    print(f"=== {cfg.get('experiment_name', 'experiment')} ===")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(cfg, indent=2)}")

    # Data
    dcfg = cfg.get('data', {})
    dataset = dcfg.get('dataset', 'cifar100')

    if dataset == 'imagenet':
        from data.imagenet import get_imagenet_dataloaders
        tcfg_bs = cfg.get('training', {}).get('batch_size', 256)
        train_loader, test_loader, num_superclasses = get_imagenet_dataloaders(
            data_dir=dcfg.get('data_dir', './data_cache/imagenet'),
            batch_size=dcfg.get('batch_size', tcfg_bs),
            num_workers=dcfg.get('num_workers', 8),
            num_superclasses=dcfg.get('num_superclasses', 50),
            image_size=dcfg.get('image_size', 224),
            augmentation=dcfg.get('augmentation', 'basic'),
            max_samples=dcfg.get('max_samples', None),
            prefetch_factor=dcfg.get('prefetch_factor', 2),
            train_subset_frac=dcfg.get('train_subset_frac', 1.0),
            train_subset_seed=dcfg.get('train_subset_seed', 42),
        )
        # Override num_categories in algorithm config
        if 'algorithm' in cfg:
            cfg['algorithm']['_num_categories'] = num_superclasses
        print(f"ImageNet-1K: {num_superclasses} superclasses")
    else:
        train_loader, test_loader = get_dataloaders(
            data_dir=dcfg.get('data_dir', './data_cache'),
            batch_size=dcfg.get('batch_size', 128),
            num_workers=dcfg.get('num_workers', 4),
            device=device,
        )

    # Model
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['type']} | Params: {total_params:,} (trainable: {trainable_params:,})")

    # Sanity check: param budget vs baseline
    from utils.sanity_check import check_param_budget
    check_param_budget(cfg, model)

    if hasattr(model, 'branches'):
        for k, branch in enumerate(model.branches):
            bp = sum(p.numel() for p in branch.parameters())
            print(f"  Branch {k}: {bp:,} params")
    elif hasattr(model, 'num_branches'):
        print(f"  Branches: K={model.num_branches} (grouped conv, per-branch params not itemized)")

    # Algorithm
    algorithm = build_algorithm(cfg)
    if algorithm is not None:
        print(f"Algorithm: {cfg['algorithm']['type']}")
        if hasattr(algorithm, 'get_assignment_summary'):
            summary = algorithm.get_assignment_summary()
            print(f"  Assignment: {summary}")
        # Wire fixed-denominator merge for train-test consistency
        if hasattr(model, 'mask_scale') and algorithm.expected_mask_sum is not None:
            model.mask_scale = algorithm.expected_mask_sum
            print(f"  Fixed mask scale: {model.mask_scale:.4f}")
        # E3 ablation hook: algorithm signals per-sample adaptive denominator.
        if (hasattr(model, 'use_adaptive_denom')
                and getattr(algorithm, 'use_adaptive_denom', False)):
            model.use_adaptive_denom = True
            print("  Per-sample adaptive denominator (÷Σm_k): ENABLED (E3 ablation)")
    else:
        print("Algorithm: none (baseline)")

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
    if 'shared_expert' in _mcfg:
        _parts.append(f"shared_expert={_mcfg['shared_expert']}")
    if 'branch_channels' in _mcfg:
        _parts.append(f"branch_channels={_mcfg['branch_channels']}")
    if _parts:
        print(f"Ablation hyperparams: {' | '.join(_parts)}")

    # Eval-only mode
    if args.eval_only:
        ckpt_path = os.path.join(cfg.get('output_dir', './outputs/default'), 'best.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            print(f"No checkpoint found at {ckpt_path}")
            return

        model = model.to(device)
        from evaluation.metrics import evaluate_accuracy, evaluate_per_category
        test_metrics = evaluate_accuracy(model, test_loader, algorithm, device)
        per_cat = evaluate_per_category(model, test_loader, algorithm, device)
        print(f"Test Top1: {test_metrics['top1']:.2f}% | Top5: {test_metrics['top5']:.2f}%")
        print(f"Per-category: {json.dumps(per_cat, indent=2)}")

        if args.eval_specialization and hasattr(model, 'num_branches'):
            print("\nComputing specialization metrics (this may take a while)...")
            spec = evaluate_specialization(model, test_loader, algorithm, device)
            if spec:
                print(f"Mutual Information I(branch; category): {spec['mutual_information']:.4f}")
                spec_path = os.path.join(cfg.get('output_dir', './outputs/default'), 'specialization.json')
                with open(spec_path, 'w') as f:
                    json.dump(spec, f, indent=2)
                print(f"Specialization metrics saved to {spec_path}")
        return

    # Wandb. See run_nlp.py for the timeout + graceful-degradation rationale.
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        algo_type = cfg.get('algorithm', {}).get('type', 'none')
        run_name = args.wandb_name or cfg.get('experiment_name', 'experiment')
        tags = args.wandb_tags or [algo_type, cfg['model']['type'], dataset]
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
        except Exception as e:
            print(f"[warn] wandb.init failed ({type(e).__name__}: {e}); "
                  f"continuing without wandb.")
            use_wandb = False

    # Train
    trainer = Trainer(cfg, model, algorithm, train_loader, test_loader, device,
                      use_wandb=use_wandb)
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
    results = trainer.train()

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Best Top-1 Accuracy: {results['best_top1']:.2f}%")
    print(f"Total Training Time: {results['compute']['total_training_time_sec']:.1f}s")
    print(f"Avg Epoch Time: {results['compute']['avg_epoch_time_sec']:.1f}s")

    # Optionally run specialization metrics
    if args.eval_specialization and hasattr(model, 'num_branches'):
        print("\nComputing specialization metrics...")
        spec = evaluate_specialization(model, test_loader, algorithm, device)
        if spec:
            print(f"Mutual Information I(branch; category): {spec['mutual_information']:.4f}")
            results['specialization'] = spec
            if use_wandb:
                wandb.summary['mutual_information'] = spec['mutual_information']
            # Re-save with specialization metrics
            results_path = os.path.join(cfg.get('output_dir', './outputs/default'), 'results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

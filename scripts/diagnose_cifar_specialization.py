"""CIFAR per-superclass × per-branch zero-ablation pruning sensitivity.

Mirrors `diagnose_{vit,nlp,lora}_specialization.py`'s `--run_dir/--out_json`
interface so `rtx5090_15_alignment.sh` can dispatch all 4 settings uniformly.

Output JSON schema (kept compatible with `analyze_e2_heatmap.py` cache):
{
  'run_dir': str,
  'config_summary': {...},
  'pruning_sensitivity': {
      'kd_matrix': List[List[float]] of shape (K, M)  # K branches × M cats
      'branch_param_counts': List[int]
  },
  'mutual_information': float,
  'usage_entropy': float,
  'unique_branches': int,
  'joint_distribution': List[List[float]],
  'alignment': float,            # diag-argmax / min(M, K) of pruning_sensitivity
  'diag_argmax_hits': int,
  'n_categories': int,
}

Usage:
  python scripts/diagnose_cifar_specialization.py \
      --run_dir outputs/rtx5090_cifar100_faithful/ours_s42 \
      --out_json outputs/analysis/specialization/ours_s42.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts._diag_helpers import advance_softspecdrop_to_terminal


def _load_results(run_dir):
    p = os.path.join(run_dir, 'results.json')
    if not os.path.exists(p):
        raise FileNotFoundError(f'no results.json in {run_dir}')
    with open(p) as f:
        return json.load(f)


def _alignment_from_kd(kd_matrix):
    """diag-argmax-based alignment fraction.

    kd_matrix shape: (K, M). For each category c in 0..min(M,K)-1,
    check whether argmax over branches of |kd[:, c]| equals c. Return
    (hits, n_diag) where n_diag = min(M, K).
    """
    if kd_matrix.size == 0:
        return 0, 0
    kd_abs = np.abs(np.asarray(kd_matrix, dtype=np.float64))
    K, M = kd_abs.shape
    n_diag = min(K, M)
    hits = sum(int(kd_abs[:, c].argmax() == c) for c in range(n_diag))
    return hits, n_diag


def _diag_off_ratio(kd_matrix):
    """Mean(|diag|) / Mean(|off|) — magnitude of diagonal vs off-diagonal."""
    kd = np.abs(np.asarray(kd_matrix, dtype=np.float64))
    K, M = kd.shape
    n = min(K, M)
    if n == 0:
        return float('nan')
    diag_v = np.array([kd[c, c] for c in range(n)])
    off_v = np.array([kd[k, c] for k in range(K) for c in range(M)
                       if k < n and c < n and k != c])
    if off_v.mean() <= 0:
        return float('inf') if diag_v.mean() > 0 else float('nan')
    return float(diag_v.mean() / off_v.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run_dir', required=True,
                     help='trained run dir containing best.pt + results.json')
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size', type=int, default=128)
    args = ap.parse_args()

    if os.path.exists(args.out_json):
        with open(args.out_json) as f:
            cached = json.load(f)
        if 'pruning_sensitivity' in cached and 'kd_matrix' in cached['pruning_sensitivity']:
            print(f'[diag-cifar] cached: {args.out_json} (skipping)')
            return

    from models import build_model
    from algorithms import build_algorithm
    from data.cifar100 import get_dataloaders
    from evaluation.metrics import evaluate_specialization, _compute_pruning_sensitivity

    cfg = _load_results(args.run_dir)['config']
    ckpt_path = os.path.join(args.run_dir, 'best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'no best.pt in {args.run_dir}')

    print(f'[diag-cifar] {args.run_dir}')
    print(f'  algorithm: {cfg.get("algorithm", {}).get("type")}  '
          f'K={cfg.get("model", {}).get("num_branches", "?")}')

    model = build_model(cfg).to(args.device)
    algorithm = build_algorithm(cfg)
    advance_softspecdrop_to_terminal(algorithm,
                                       cfg.get('training', {}).get('epochs', 0))
    if (algorithm is not None and hasattr(model, 'mask_scale')
            and algorithm.expected_mask_sum is not None):
        model.mask_scale = algorithm.expected_mask_sum

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v
          for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(sd)
    model.eval()

    dcfg = cfg.get('data', {})
    _, test_loader = get_dataloaders(
        data_dir=dcfg.get('data_dir', './data_cache'),
        batch_size=args.batch_size,
        num_workers=dcfg.get('num_workers', 0),
        device=args.device,
    )

    spec = evaluate_specialization(model, test_loader, algorithm, args.device)
    if spec is None:
        raise RuntimeError('evaluate_specialization returned None — model has no branches')

    kd = spec['pruning_sensitivity']
    kd_matrix = np.array(kd['kd_matrix'])
    hits, n_diag = _alignment_from_kd(kd_matrix)
    ratio = _diag_off_ratio(kd_matrix)

    out = {
        'run_dir': args.run_dir,
        'config_summary': {
            'algorithm': cfg.get('algorithm', {}).get('type'),
            'num_branches': cfg.get('model', {}).get('num_branches'),
            'num_categories': cfg.get('algorithm', {}).get('num_categories'),
        },
        'pruning_sensitivity': kd,
        'mutual_information': spec.get('mutual_information'),
        'usage_entropy': spec.get('usage_entropy'),
        'unique_branches': spec.get('unique_branches'),
        'joint_distribution': spec.get('joint_distribution'),
        'alignment_diag_argmax_hits': hits,
        'alignment_n_diag': n_diag,
        'alignment_fraction': hits / n_diag if n_diag > 0 else None,
        'diag_off_ratio': ratio,
        'max_abs_delta': float(np.abs(kd_matrix).max()),
    }
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print(f'[diag-cifar] alignment={hits}/{n_diag}  diag/off={ratio:.2f}x  '
          f'max|delta|={np.abs(kd_matrix).max():.4f}')
    print(f'[diag-cifar] wrote {args.out_json}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Logit-masking control (information-matched control, paper App. E.3) — does restricting the output
space to the current superclass explain SpecDrop's gain?

For each (setting, method, seed): load the trained checkpoint, run the val
split once, and report top-1 under BOTH decodings from the same logits:
  (a) unmasked — standard argmax over all fine classes (sanity: must
      reproduce the paper number for this checkpoint);
  (b) masked  — argmax restricted to fine classes belonging to the sample's
      coarse category (logits of out-of-category classes set to -inf).
The mask is applied symmetrically to every method, so (b)-(a) isolates the
benefit of output-space reduction from the benefit of routing.

Methods:
  cifar: dense (resnet110) / no_routing / ours   [ours ckpt lives in the
         ablation SE-0x cells — same runs as the main table]
  vit:   dense (vit_small) / no_routing_se / ours_vit

Usage:
    python scripts/eval_logit_mask.py --setting cifar --method ours --seed 42
    python scripts/eval_logit_mask.py --setting vit --method dense --seed 123

Output: outputs/eval_logit_mask/<setting>_<method>_s<seed>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch


# (run_dir template, forward mode). Forward modes:
#   dense    — model(x), no branch mask
#   uniform  — NoDropout 1/K mask (mechanism-OFF multibranch)
#   specdrop — trained SoftSpecDrop terminal mask, category-conditioned
RUNS = {
    'cifar': {
        'dense':      ('outputs/rtx5090_cifar100_faithful/resnet110_s{seed}', 'dense'),
        'no_routing': ('outputs/rtx5090_cifar100_faithful/no_routing_s{seed}', 'uniform'),
        'ours':       ('outputs/rtx5090_ablation/ablation_se_ratio_0x_pa0.7_wr1.0_s{seed}', 'specdrop'),
    },
    'vit': {
        'dense':         ('outputs/rtx5090_imagenet_vit_faithful/vit_small_s{seed}', 'dense'),
        'no_routing_se': ('outputs/rtx5090_imagenet_vit_faithful/mbvit_no_routing_se_s{seed}', 'uniform'),
        'ours':          ('outputs/rtx5090_imagenet_vit_faithful/ours_vit_s{seed}', 'specdrop'),
    },
}


def _load_cfg(run_dir: str) -> dict:
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
    raise FileNotFoundError(f'no recoverable config for {run_dir}')


def _load_model(run_dir: str, cfg: dict, device: str):
    from models import build_model
    model = build_model(cfg).to(device)
    ckpt = torch.load(f'{run_dir}/best.pt', map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model


def _coarse_to_fine_allowed(setting: str, num_fine: int, num_coarse: int) -> torch.Tensor:
    """(num_coarse, num_fine) bool matrix: allowed[c, f] = fine class f
    belongs to coarse category c."""
    if setting == 'cifar':
        from data.cifar100 import FINE_TO_COARSE
        f2c = list(FINE_TO_COARSE)
    else:
        with open('data_cache/imagenet/imagenet_breeds_T60_C10.json') as f:
            d = json.load(f)
        mapping = d['mapping']
        f2c = [mapping[str(i)] for i in range(num_fine)]
    allowed = torch.zeros(num_coarse, num_fine, dtype=torch.bool)
    for f, c in enumerate(f2c):
        allowed[c, f] = True
    assert allowed.any(dim=1).all(), 'every coarse category must own >=1 fine class'
    return allowed


def _build_algorithm_for(mode: str, cfg: dict, run_dir: str):
    if mode == 'dense':
        return None
    if mode == 'uniform':
        from algorithms.no_dropout import NoDropout
        K = cfg['model']['num_branches']
        M = cfg['data'].get('num_categories', K)
        return NoDropout(num_modules=K, num_categories=M)
    from algorithms import build_algorithm
    from scripts._diag_helpers import advance_softspecdrop_to_terminal
    acfg = dict(cfg.get('algorithm', {}))
    if acfg.get('type', 'none') == 'none':
        raise ValueError(f'{run_dir}: expected a SoftSpecDrop config, got algorithm.type=none')
    algo = build_algorithm(cfg)
    advance_softspecdrop_to_terminal(algo, cfg['training']['epochs'])
    return algo


def _val_loader(setting: str, cfg: dict, device: str):
    if setting == 'cifar':
        from data.cifar100 import get_dataloaders
        _, loader = get_dataloaders(
            data_dir=cfg['data'].get('data_dir', './data_cache'),
            batch_size=cfg['training'].get('batch_size', 128),
            num_workers=4, device=device)
        return loader
    # Build the val split directly from the cached validation arrow shards.
    # (get_imagenet_dataloaders calls load_dataset(), which memory-maps ALL
    # recorded splits — this machine only carries the val shards.)
    import glob
    from datasets import Dataset as HFDS, concatenate_datasets
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from data.imagenet import HFImageNetDataset, _load_or_build_mapping
    data_dir = cfg['data'].get('data_dir', './data_cache/imagenet')
    arrows = sorted(glob.glob(
        f'{data_dir}/ILSVRC___imagenet-1k/*/*/*/imagenet-1k-validation-*.arrow'))
    if not arrows:
        raise FileNotFoundError(f'no validation arrow shards under {data_dir}')
    val_hf = concatenate_datasets([HFDS.from_file(p) for p in arrows])
    mapping, _, _ = _load_or_build_mapping(data_dir)
    val_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = HFImageNetDataset(val_hf, val_transform, mapping)
    return DataLoader(ds, batch_size=cfg['training'].get('batch_size', 256),
                      shuffle=False, num_workers=8, pin_memory=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--setting', choices=('cifar', 'vit'), required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--run_dir', default=None,
                    help='override the checkpoint dir (e.g. a retrained cell)')
    args = ap.parse_args()

    tmpl, mode = RUNS[args.setting][args.method]
    run_dir = args.run_dir or tmpl.format(seed=args.seed)
    out_dir = 'outputs/eval_logit_mask'
    out_path = f'{out_dir}/{args.setting}_{args.method}_s{args.seed}.json'
    if os.path.exists(out_path):
        print(f'[logit_mask] {out_path} exists, skipping')
        return

    cfg = _load_cfg(run_dir)
    src_top1 = None
    rj = f'{run_dir}/results.json'
    if os.path.exists(rj):
        with open(rj) as f:
            src_top1 = json.load(f).get('best_top1')

    model = _load_model(run_dir, cfg, args.device)
    algo = _build_algorithm_for(mode, cfg, run_dir)
    if algo is not None and hasattr(model, 'mask_scale'):
        model.mask_scale = algo.expected_mask_sum

    loader = _val_loader(args.setting, cfg, args.device)
    num_fine = cfg['model'].get('num_classes', 100 if args.setting == 'cifar' else 1000)
    num_coarse = 20 if args.setting == 'cifar' else 46
    allowed = _coarse_to_fine_allowed(args.setting, num_fine, num_coarse).to(args.device)

    correct_u = correct_m = total = 0
    with torch.no_grad():
        for batch in loader:
            x, y, c = batch[0], batch[1], batch[2]
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            c = c.to(args.device, non_blocking=True)
            if algo is None:
                logits = model(x)
            else:
                mask = algo.get_mask(c, training=False).to(args.device)
                logits = model(x, branch_mask=mask)
            pred_u = logits.argmax(dim=-1)
            masked = logits.masked_fill(~allowed[c], float('-inf'))
            pred_m = masked.argmax(dim=-1)
            correct_u += (pred_u == y).sum().item()
            correct_m += (pred_m == y).sum().item()
            total += y.numel()

    top1_u = 100.0 * correct_u / max(1, total)
    top1_m = 100.0 * correct_m / max(1, total)
    payload = {
        'setting': args.setting, 'method': args.method, 'seed': args.seed,
        'src_run_dir': run_dir, 'src_metric': src_top1,
        'top1_unmasked': top1_u, 'top1_masked': top1_m,
        'mask_gain': top1_m - top1_u, 'n_eval': total,
        'metric_name': 'top1_acc',
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[logit_mask][{args.setting} {args.method} s{args.seed}] '
          f'unmasked={top1_u:.2f} (src={src_top1}) masked={top1_m:.2f} '
          f'gain={top1_m - top1_u:+.2f} -> {out_path}')


if __name__ == '__main__':
    main()

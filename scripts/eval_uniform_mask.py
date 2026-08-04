#!/usr/bin/env python3
"""Category-free inference robustness — eval ours' best.pt with UNIFORM mask
(NoDropout algorithm) instead of category-conditional SoftSpecDrop mask.

Quantifies the cost of ours' cluster_id-at-inference requirement (paper
disclosure C2). For each setting, loads the trained ours checkpoint, swaps
the algorithm to NoDropout (mask = 1/K ∀ branch, S = K, output = ensemble
mean), runs the standard eval pipeline, and writes the resulting metric
to `uniform_mask_results.json` alongside the original `results.json`.

Math note (why NoDropout is the right swap):
  SoftSpecDrop at training:    output = Σ_k m_k(c) h_k / S(c)
                               where m_k(c) ∈ {pa, pi}, S = pa + (K-1)pi
  NoDropout at eval:           output = Σ_k 1·h_k / K
                                      = (1/K) Σ_k h_k
  Same ensemble mean as SoftSpecDrop's degenerate point pa=pi (S = K · pa).
  This isolates "what if the category mechanism is OFF" — exactly the
  natural question about cluster_id-at-inference.

Usage:
    python scripts/eval_uniform_mask.py --setting cifar --seed 42
    python scripts/eval_uniform_mask.py --setting vit   --seed 42
    python scripts/eval_uniform_mask.py --setting nlp   --seed 42
    python scripts/eval_uniform_mask.py --setting lora  --seed 42

Output (always written to outputs/eval_uniform_mask/<setting>_s<seed>.json):
    {"setting": "cifar", "seed": 42,
     "src_run_dir": "outputs/...", "src_metric": 79.23,
     "uniform_mask_metric": 7X.XX, "delta": -X.XX,
     "metric_name": "top1_acc"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make repo root importable so `from data import ...` works regardless of CWD
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import yaml


def _load_cfg(run_dir: str) -> dict:
    """Recover config from a run dir, trying 3 sources in priority order:

    1. `<run_dir>/_tmp.yaml` — LoRA convention (cell-local yaml; never
       overwritten by sibling cells).
    2. `<run_dir>/results.json::config` — CIFAR/ViT/NLP convention (the
       trainer's `finalize_results()` embeds the cfg dict in results.json).
       Robust to whatever the parent's `_tmp_s{seed}.yaml` contains since
       that file is OVERWRITTEN by every method's run.
    3. `<parent>/_tmp_s{seed}.yaml` — last resort. Only safe when this
       was the most-recent method to run (which is rarely true after a
       full main-table chain). Disabled by default; set EVAL_UNIFORM_MASK_
       ALLOW_PARENT_YAML=1 to re-enable for debugging.
    """
    import glob
    # 1. Cell-local _tmp.yaml (LoRA)
    cands = sorted(glob.glob(f'{run_dir}/_tmp*.yaml'))
    if cands:
        with open(cands[0]) as f:
            return yaml.safe_load(f)
    # 2. results.json::config (CIFAR/ViT/NLP)
    rj = f'{run_dir}/results.json'
    if os.path.exists(rj):
        with open(rj) as f:
            r = json.load(f)
        if 'config' in r:
            return r['config']
    # 3. Parent _tmp_s{seed}.yaml (debug only — likely stale)
    if os.environ.get('EVAL_UNIFORM_MASK_ALLOW_PARENT_YAML') == '1':
        seed_match = run_dir.rstrip('/').rsplit('_s', 1)
        if len(seed_match) == 2:
            seed = seed_match[1]
            parent = os.path.dirname(run_dir)
            parent_yaml = f'{parent}/_tmp_s{seed}.yaml'
            if os.path.exists(parent_yaml):
                with open(parent_yaml) as f:
                    return yaml.safe_load(f)
    raise FileNotFoundError(
        f'no recoverable config for {run_dir}: tried '
        f'(1) _tmp*.yaml, (2) results.json::config, (3) parent yaml (disabled)')


def _load_src_metric(run_dir: str, metric_key: str) -> float | None:
    """Read the original ours metric from results.json."""
    p = f'{run_dir}/results.json'
    if not os.path.exists(p):
        return None
    with open(p) as f:
        r = json.load(f)
    return r.get(metric_key)


def _write_out(setting: str, seed: int, payload: dict) -> str:
    out_dir = 'outputs/eval_uniform_mask'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/{setting}_s{seed}.json'
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[uniform_mask] wrote {out_path}')
    return out_path


# ─── CIFAR ──────────────────────────────────────────────────────────────────
def eval_cifar(seed: int, device: str) -> dict:
    """ours_cifar @ K=20 → eval top1 with NoDropout."""
    run_dir = f'outputs/rtx5090_cifar100_faithful/ours_s{seed}'
    cfg = _load_cfg(run_dir)
    src_top1 = _load_src_metric(run_dir, 'best_top1') or _load_src_metric(run_dir, 'eval_top1')
    if src_top1 is not None and src_top1 <= 1.0:
        src_top1 *= 100

    # Build model + load checkpoint
    from models import build_model
    model = build_model(cfg).to(device)
    ckpt = torch.load(f'{run_dir}/best.pt', map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    # Swap algorithm to NoDropout (uniform 1/K mask)
    from algorithms.no_dropout import NoDropout
    from scripts._diag_helpers import require_keys
    require_keys(cfg['model'], ('num_branches',), f'cfg["model"] in {run_dir}')
    require_keys(cfg['data'], ('num_categories',), f'cfg["data"] in {run_dir}')
    K = cfg['model']['num_branches']
    M = cfg['data']['num_categories']
    algo = NoDropout(num_modules=K, num_categories=M)

    # Build val loader (CIFAR get_dataloaders returns 2-tuple, no aug arg)
    from data.cifar100 import get_dataloaders
    dcfg = cfg['data']
    _, val_loader = get_dataloaders(
        data_dir=dcfg.get('data_dir', './data_cache'),
        batch_size=cfg['training'].get('batch_size', 128),
        num_workers=0,
        device=device,
    )

    # Plain eval loop (avoid Trainer dependencies)
    correct = 0; total = 0
    with torch.no_grad():
        for batch in val_loader:
            # CIFAR loader yields (image, fine_label, coarse_label, example_index)
            # (per data/cifar100.py:87 — 4-tuple supports Example-Tied Dropout).
            if len(batch) >= 3:
                x, y, c = batch[0], batch[1], batch[2]
            else:
                x, y = batch[:2]; c = torch.zeros(x.size(0), dtype=torch.long)
            x = x.to(device); y = y.to(device); c = c.to(device)
            mask = algo.get_mask(c, training=False).to(device)
            # Set scale on model if it supports it
            if hasattr(model, 'set_mask_scale'):
                model.set_mask_scale(algo.expected_mask_sum)
            elif hasattr(model, 'mask_scale'):
                model.mask_scale = algo.expected_mask_sum
            try:
                logits = model(x, branch_mask=mask)
            except TypeError:
                logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item(); total += y.numel()
    top1 = 100.0 * correct / max(1, total)
    delta = top1 - (src_top1 or 0.0)

    print(f'[uniform_mask][cifar s{seed}] src={src_top1:.2f}  uniform={top1:.2f}  Δ={delta:+.2f}')
    return {
        'setting': 'cifar', 'seed': seed,
        'src_run_dir': run_dir, 'src_metric': src_top1,
        'uniform_mask_metric': top1, 'delta': delta,
        'metric_name': 'top1_acc',
    }


# ─── ViT ImageNet ───────────────────────────────────────────────────────────
def eval_vit(seed: int, device: str) -> dict:
    """ours_vit @ K=46 BREEDS → eval top1 with NoDropout."""
    run_dir = f'outputs/rtx5090_imagenet_vit_faithful/ours_vit_s{seed}'
    cfg = _load_cfg(run_dir)
    src_top1 = _load_src_metric(run_dir, 'best_top1')
    if src_top1 is not None and src_top1 <= 1.0:
        src_top1 *= 100

    from models import build_model
    model = build_model(cfg).to(device)
    ckpt = torch.load(f'{run_dir}/best.pt', map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    from algorithms.no_dropout import NoDropout
    from scripts._diag_helpers import require_keys
    require_keys(cfg['model'], ('num_branches',), f'cfg["model"] in {run_dir}')
    K = cfg['model']['num_branches']
    algo = NoDropout(num_modules=K, num_categories=K)

    from data.imagenet import get_imagenet_dataloaders
    dcfg = cfg['data']
    _, val_loader, _ = get_imagenet_dataloaders(
        data_dir=dcfg.get('data_dir', './data_cache/imagenet'),
        batch_size=cfg['training'].get('batch_size', 256),
        num_workers=8,
        augmentation='basic',
        prefetch_factor=2,
    )

    correct = 0; total = 0
    with torch.no_grad():
        for batch in val_loader:
            # ViT ImageNet loader: same 4-tuple convention as CIFAR (img, fine,
            # coarse=BREEDS-supercat, example_idx).
            if len(batch) >= 3:
                x, y, c = batch[0], batch[1], batch[2]
            else:
                x, y = batch[:2]; c = torch.zeros(x.size(0), dtype=torch.long)
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            mask = algo.get_mask(c, training=False).to(device)
            if hasattr(model, 'set_mask_scale'):
                model.set_mask_scale(algo.expected_mask_sum)
            elif hasattr(model, 'mask_scale'):
                model.mask_scale = algo.expected_mask_sum
            try:
                logits = model(x, branch_mask=mask)
            except TypeError:
                logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item(); total += y.numel()
    top1 = 100.0 * correct / max(1, total)
    delta = top1 - (src_top1 or 0.0)

    print(f'[uniform_mask][vit s{seed}] src={src_top1:.2f}  uniform={top1:.2f}  Δ={delta:+.2f}')
    return {
        'setting': 'vit', 'seed': seed,
        'src_run_dir': run_dir, 'src_metric': src_top1,
        'uniform_mask_metric': top1, 'delta': delta,
        'metric_name': 'top1_acc',
    }


# ─── NLP SlimPajama ─────────────────────────────────────────────────────────
def eval_nlp(seed: int, device: str) -> dict:
    """ours_nlp @ K=7 + SE → eval val PPL with NoDropout."""
    import math
    import torch.nn.functional as F

    run_dir = f'outputs/rtx5090_nlp_faithful/ours_phaseP_s{seed}'
    cfg = _load_cfg(run_dir)
    src_ppl = _load_src_metric(run_dir, 'best_val_ppl') or _load_src_metric(run_dir, 'eval_ppl')

    from run_nlp import build_nlp_model
    model = build_nlp_model(cfg).to(device)
    ckpt = torch.load(f'{run_dir}/best.pt', map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    from algorithms.no_dropout import NoDropout
    from scripts._diag_helpers import require_keys
    require_keys(cfg['model'], ('num_branches',), f'cfg["model"] in {run_dir}')
    K = cfg['model']['num_branches']
    algo = NoDropout(num_modules=K, num_categories=K)
    if hasattr(model, 'mask_scale'):
        model.mask_scale = algo.expected_mask_sum

    from data.slimpajama import get_slimpajama_dataloaders
    dcfg = cfg['data']
    # Only pass kwargs that are actually present in cfg, let defaults handle the rest.
    kwargs = {
        'data_dir': dcfg.get('data_dir', './data_cache/slimpajama'),
        'max_seq_len': dcfg.get('max_seq_len', 1024),
        'batch_size': cfg['training'].get('batch_size', 32),
        'num_workers': 0,
    }
    for k in ('max_train_tokens', 'cluster_label_path_train', 'cluster_label_path_val'):
        if dcfg.get(k) is not None:
            kwargs[k] = dcfg[k]
    _, val_loader, _ = get_slimpajama_dataloaders(**kwargs)

    nll_sum = 0.0; tok_cnt = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, domain_ids = batch
            input_ids = input_ids.to(device); domain_ids = domain_ids.to(device)
            mask = algo.get_mask(domain_ids, training=False).to(device)
            logits = model(input_ids, branch_mask=mask)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1), reduction='sum')
            nll_sum += float(loss); tok_cnt += shift_labels.numel()
    ppl = math.exp(nll_sum / max(1, tok_cnt))
    delta = ppl - (src_ppl or 0.0)

    print(f'[uniform_mask][nlp s{seed}] src={src_ppl:.3f}  uniform={ppl:.3f}  Δ={delta:+.3f}')
    return {
        'setting': 'nlp', 'seed': seed,
        'src_run_dir': run_dir, 'src_metric': src_ppl,
        'uniform_mask_metric': ppl, 'delta': delta,
        'metric_name': 'val_ppl',
    }


# ─── LoRA SuperNI ───────────────────────────────────────────────────────────
def eval_lora(seed: int, device: str) -> dict:
    """ours_lora @ K=20 → eval ROUGE-L with NoDropout (uniform 1/K mask)."""
    run_dir = f'outputs/rtx5090_lora_faithful/ours_s{seed}'
    cfg = _load_cfg(run_dir)
    src_rouge = _load_src_metric(run_dir, 'eval_rouge_l')

    # Build model + load LoRA-only state
    from models.lora_models import build_lora_model, _apply_gradient_checkpointing_override
    model = build_lora_model(cfg)
    _apply_gradient_checkpointing_override(model, cfg)
    model = model.to(device)
    ckpt = torch.load(f'{run_dir}/best.pt', map_location=device, weights_only=False)
    lora_state = ckpt.get('lora_state') or ckpt.get('state_dict') or ckpt
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    model.eval()

    # Swap to NoDropout — produces uniform mask 1/K, ignores cluster_id
    from algorithms.no_dropout import NoDropout
    from scripts._diag_helpers import require_keys
    require_keys(cfg['model'], ('num_experts',), f'cfg["model"] in {run_dir}')
    require_keys(cfg['data'], ('num_clusters',), f'cfg["data"] in {run_dir}')
    K = cfg['model']['num_experts']
    M = cfg['data']['num_clusters']
    algo = NoDropout(num_modules=K, num_categories=M)

    # Build trainer + run ROUGE eval (reuses canonical Wang 2022 protocol).
    # Trainer's `_make_routing_fn()` reads `self.algorithm.get_mask` — since
    # our NoDropout returns uniform mask, generation will use uniform routing.
    from training.trainer_lora import LoRATrainer
    from data.natural_instructions import get_superni_dataloaders
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg['model']['base_model_name'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dcfg = cfg['data']
    _, eval_loader, _ = get_superni_dataloaders(
        data_root=dcfg['data_root'],
        tokenizer=tok,
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
    dummy_cfg = dict(cfg); dummy_cfg.setdefault('output_dir', run_dir)
    trainer = LoRATrainer(
        cfg=dummy_cfg, model=model, algorithm=algo,
        train_loader=eval_loader, eval_loader=eval_loader, device=device, use_wandb=False,
    )
    rouge = trainer.run_rouge_eval(
        instances_per_task=int(cfg['training'].get('rouge_eval_instances_per_task', 10)),
        max_new_tokens=int(cfg['training'].get('rouge_eval_max_new_tokens', 128)))
    rouge_l = rouge['rougeL_mean']
    em = rouge['exact_match_mean']
    delta_rouge = rouge_l - (src_rouge or 0.0)

    print(f'[uniform_mask][lora s{seed}] src_rouge={src_rouge:.4f}  '
          f'uniform_rouge={rouge_l:.4f}  Δrouge={delta_rouge:+.4f}  EM={em:.4f}')
    return {
        'setting': 'lora', 'seed': seed,
        'src_run_dir': run_dir, 'src_metric': src_rouge,
        'uniform_mask_metric': rouge_l, 'delta': delta_rouge,
        'metric_name': 'rouge_l_f1',
        'uniform_mask_em': em,
        'src_em': _load_src_metric(run_dir, 'eval_exact_match'),
    }


# ─── CLI ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--setting', required=True, choices=['cifar', 'vit', 'nlp', 'lora'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    fns = {'cifar': eval_cifar, 'vit': eval_vit, 'nlp': eval_nlp, 'lora': eval_lora}
    payload = fns[args.setting](args.seed, device)
    _write_out(args.setting, args.seed, payload)


if __name__ == '__main__':
    main()

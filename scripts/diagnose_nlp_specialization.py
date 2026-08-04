#!/usr/bin/env python3
"""Per-domain × per-branch PPL increase when each branch is zero-ablated.

For a trained MultiBranchTransformerLM checkpoint, evaluates validation
PPL per domain with (a) no ablation (baseline) and (b) each branch k
zeroed in the routing mask. Reports the Δ matrix.

Interpretation:
- Diagonal argmax (branch k hurts domain k most) → latent specialization
- Uniform Δ across branches → branches interchangeable, no specialization
- Off-diagonal structure → non-trivial learned grouping (interesting)

Usage:
  python scripts/diagnose_nlp_specialization.py \
      --run_dir outputs/rtx5090_nlp_mini_ablation/phaseA_pa0.7_pi0.3_s42
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F

from run_nlp import build_nlp_model
from algorithms import build_algorithm
from data.slimpajama import get_slimpajama_dataloaders, DOMAIN_TO_ID


def load(run_dir, device):
    with open(f'{run_dir}/results.json') as f:
        results = json.load(f)
    cfg = results['config']

    model = build_nlp_model(cfg).to(device)
    algorithm = build_algorithm(cfg)

    # Advance algorithm to end-of-training. Sets BOTH current_epoch (for
    # warmup_unit='epoch') AND current_step (for 'step', e.g. NLP 500M
    # ours_phaseP). Setting only current_epoch is a silent no-op under step
    # mode → mask stays at uniform-start → ablation analysis becomes
    # artifact (cf. 2026-05-02 LoRA spec bug). See scripts/_diag_helpers.py.
    from scripts._diag_helpers import advance_softspecdrop_to_terminal
    advance_softspecdrop_to_terminal(algorithm,
                                       cfg['training'].get('epochs', 10))

    if (algorithm is not None and hasattr(model, 'mask_scale')
            and algorithm.expected_mask_sum is not None):
        model.mask_scale = algorithm.expected_mask_sum

    ckpt_path = f'{run_dir}/best.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(sd)
    model.eval()
    return cfg, model, algorithm


def _is_specdrop_family(algorithm):
    if algorithm is None:
        return False
    return type(algorithm).__name__ in ('SoftSpecDrop', 'NoDropout', 'HardCategory')


@torch.no_grad()
def eval_per_domain(model, loader, algorithm, device, ablate_branch, max_batches):
    """Return {domain_id: PPL} on up to max_batches of val data.

    Routing-family dispatch:
      - SpecDrop family: zero branch_mask column k (existing path).
      - Native MoE / learned routers (Switch / Hash / Demix / SMoE-Dropout):
        install ablation hooks via scripts/_ablation_hooks.install_ablation;
        forward without branch_mask kwarg.
    """
    use_mask = _is_specdrop_family(algorithm)
    if not use_mask and ablate_branch is not None:
        from scripts._ablation_hooks import install_ablation
        install_ablation(model, k=ablate_branch)
    try:
        nll_sum = defaultdict(float)
        tok_cnt = defaultdict(int)
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            input_ids, domain_ids = batch
            input_ids = input_ids.to(device)
            domain_ids = domain_ids.to(device)

            if use_mask:
                mask = algorithm.get_mask(domain_ids, training=False).clone()
                if ablate_branch is not None:
                    mask[:, ablate_branch] = 0.0
                logits = model(input_ids, branch_mask=mask)
            else:
                # Demix forward asserts exactly one of (domain_ids,
                # mixture_weights). Pass per-sample domain_ids so each
                # sample routes to its native expert; branch-k ablation
                # is applied via the ParallelFFN forward hook.
                from models.demix_transformer_lm import DemixTransformerLM
                if isinstance(model, DemixTransformerLM):
                    logits = model(input_ids, domain_ids=domain_ids)
                else:
                    logits = model(input_ids)
            # Switch / Hash Layers / Demix / SMoE-Dropout return
            # (logits, aux_loss); SpecDrop-family transformer LM returns
            # a single tensor. Unwrap.
            if isinstance(logits, tuple):
                logits = logits[0]

            # LM loss: shift by 1, per-token CE
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss_per_token = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='none',
            ).view(shift_labels.size(0), shift_labels.size(1))

            # Aggregate per-domain
            for b in range(input_ids.size(0)):
                d = domain_ids[b].item()
                nll_sum[d] += loss_per_token[b].sum().item()
                tok_cnt[d] += loss_per_token[b].numel()

        return {d: float(np.exp(nll_sum[d] / tok_cnt[d])) for d in nll_sum}
    finally:
        if not use_mask and ablate_branch is not None:
            from scripts._ablation_hooks import uninstall_ablation
            uninstall_ablation(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True,
                        help='Path to a trained MultiBranchTransformerLM run directory')
    parser.add_argument('--max_batches', type=int, default=50,
                        help='Val batches per forward pass (K+1 passes total)')
    parser.add_argument('--device', default=None)
    parser.add_argument('--out_json', default=None,
                        help='Where to dump the full Δ matrix as JSON (default: '
                             'outputs/analysis/nlp_diag/<run_name>.json)')
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f'[Diag] run={args.run_dir}  device={device}  max_batches={args.max_batches}')
    cfg, model, algorithm = load(args.run_dir, device)
    # K determination: SpecDrop family uses model.num_branches; native MoE
    # (switch / hash_layers / demix / smoe_dropout) exposes K via FFN
    # num_experts — auto-detect via scripts._ablation_hooks.model_branch_count.
    K = getattr(model, 'num_branches', None)
    if K is None:
        from scripts._ablation_hooks import model_branch_count
        K = model_branch_count(model)
    if not K:
        raise RuntimeError(f'cannot determine K for {args.run_dir}')
    print(f'[Diag] K={K}  algorithm={cfg.get("algorithm", {}).get("type", "none")}  '
          f'pa={cfg.get("algorithm", {}).get("p_active", "?")}  '
          f'wr={cfg.get("algorithm", {}).get("warmup_ratio", "?")}  '
          f'SE_dim={cfg.get("model", {}).get("shared_expert_dim", 0)}')

    # Val loader with same tokens + seq as training (reuses tokenize cache).
    # Pass cluster-label paths through for Phase W / Z runs, else the loader
    # would revert to source-level domain_ids and report mislabeled Δ rows.
    dcfg = cfg['data']
    cluster_train = dcfg.get('cluster_label_path_train')
    cluster_val = dcfg.get('cluster_label_path_val')
    # data.max_train_tokens isn't always set in the YAML — historically
    # `training.max_tokens` was used (silently dead code, fixed in 88c4e1a).
    # The loader's default is 500M, matching paper-canonical regime; honor
    # any value present in cfg, else default. Both legacy paper-locked NLP
    # cells (no_routing_se05) and Phase P ours follow this default.
    _, val_loader, _ = get_slimpajama_dataloaders(
        data_dir=dcfg['data_dir'],
        max_seq_len=dcfg['max_seq_len'],
        batch_size=cfg['training']['batch_size'],
        num_workers=0,
        max_train_tokens=dcfg.get('max_train_tokens', 500_000_000),
        cluster_label_path_train=cluster_train,
        cluster_label_path_val=cluster_val,
    )
    using_clusters = cluster_train is not None

    print(f'\n[1/2] Baseline (no ablation)...', flush=True)
    baseline = eval_per_domain(model, val_loader, algorithm, device,
                                ablate_branch=None, max_batches=args.max_batches)

    print(f'[2/2] Ablating each of {K} branches...', flush=True)
    ablated = {}
    for k in range(K):
        print(f'   branch {k}...', flush=True)
        ablated[k] = eval_per_domain(model, val_loader, algorithm, device,
                                      ablate_branch=k, max_batches=args.max_batches)

    # Present matrix. For Phase W / Z runs, category IDs are cluster IDs
    # (0..M-1), not SlimPajama source domains — label accordingly.
    if using_clusters:
        id2n = {c: f'cluster_{c}' for c in range(K)}
        row_label = 'cluster'
    else:
        id2n = {v: k for k, v in DOMAIN_TO_ID.items()}
        row_label = 'domain'
    print()
    print('=' * 90)
    print(f' Per-{row_label} × per-branch PPL increase when branch k is zero-ablated')
    print('=' * 90)
    header_cols = " ".join([f'Δb{k}'.rjust(7) for k in range(K)])
    print(f"{row_label:<30} {'base':>7} {header_cols}  argmax")
    print('-' * 90)

    diag_hits = 0
    n_domains = 0
    for d in sorted(baseline.keys()):
        base = baseline[d]
        deltas = [ablated[k].get(d, float('nan')) - base for k in range(K)]
        worst_k = int(np.nanargmax(deltas))
        marker = '*' if worst_k == d else ' '
        if worst_k == d:
            diag_hits += 1
        n_domains += 1
        delta_str = " ".join([f'{x:>+7.2f}' for x in deltas])
        dn = id2n.get(d, f'dom{d}')
        print(f"{dn:<30} {base:>7.2f} {delta_str}  b{worst_k}{marker}")
    print('-' * 90)
    print(f'Diagonal hits (argmax Δ on assigned branch): {diag_hits}/{n_domains}  '
          f'(random chance: {1/K:.0%}; full specialization: 100%)')

    # Overall max delta magnitude (signal strength)
    all_deltas = []
    for d in baseline:
        for k in range(K):
            all_deltas.append(abs(ablated[k].get(d, baseline[d]) - baseline[d]))
    max_abs_delta = float(max(all_deltas))
    mean_abs_delta = float(np.mean(all_deltas))
    print(f'Max |Δ| across matrix: {max_abs_delta:.2f} PPL')
    print(f'Mean |Δ| across matrix: {mean_abs_delta:.2f} PPL')
    print(f'  If max|Δ| < 0.5 → branches interchangeable (no specialization signal)')

    # Persist full matrix to JSON for rsync-back convenience.
    out_json = args.out_json
    if out_json is None:
        run_name = os.path.basename(os.path.normpath(args.run_dir))
        out_json = f'outputs/analysis/nlp_diag/{run_name}.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {
        'run_dir': args.run_dir,
        'config_summary': {
            'algorithm': cfg.get('algorithm', {}).get('type'),
            'p_active': cfg.get('algorithm', {}).get('p_active'),
            'warmup_ratio': cfg.get('algorithm', {}).get('warmup_ratio'),
            'shared_expert_dim': cfg.get('model', {}).get('shared_expert_dim', 0),
            'ffn_dim_per_branch': cfg.get('model', {}).get('ffn_dim_per_branch'),
            'num_branches': K,
        },
        'max_batches': args.max_batches,
        'using_clusters': using_clusters,
        'category_id_to_name': (
            {int(c): f'cluster_{c}' for c in range(K)} if using_clusters
            else {int(v): k for k, v in DOMAIN_TO_ID.items()}
        ),
        'domain_id_to_name': {int(v): k for k, v in DOMAIN_TO_ID.items()},
        'baseline_ppl_per_domain': {int(d): float(v) for d, v in baseline.items()},
        'ablated_ppl_per_domain': {
            int(k): {int(d): float(v) for d, v in ablated[k].items()}
            for k in ablated
        },
        'diag_hits': diag_hits,
        'n_domains': n_domains,
        'max_abs_delta': max_abs_delta,
        'mean_abs_delta': mean_abs_delta,
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\n[Diag] wrote {out_json}')


if __name__ == '__main__':
    main()

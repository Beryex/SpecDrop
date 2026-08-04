#!/usr/bin/env python3
"""Analytic FLOPs for CIFAR / NLP / LoRA main-table methods (paper App. F.3).
Companion to scripts/profile_vit_flops.py (ViT numbers already in
outputs/analysis/vit_flops.md).

CIFAR + NLP: fvcore MAC trace on models rebuilt from each run's stored config
(input: one 32x32x3 image / one 512-token sequence). LoRA: closed-form adapter
MACs on top of the shared frozen Llama-3.2-1B base (all methods share the
identical base forward; only adapter add-on differs).

Usage: python scripts/compute_flops_tables.py
Output: outputs/analysis/flops_all_settings.{md,json} (MACs; x2 = FLOPs).
"""
from __future__ import annotations

import glob
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import yaml
from fvcore.nn import FlopCountAnalysis

CIFAR_RUNS = [
    ('Dense ResNet-110',        'outputs/rtx5090_cifar100_faithful/resnet110_s42'),
    ('Stochastic Depth',        'outputs/rtx5090_cifar100_faithful/stoch_depth_s42'),
    ('Example-Tied Dropout',    'outputs/rtx5090_cifar100_faithful/et_dropout_s42'),
    ('Contextual Dropout',      'outputs/rtx5090_cifar100_faithful/ctx_dropout_s42'),
    ('No-Routing (K=20)',       'outputs/rtx5090_cifar100_faithful/no_routing_s42'),
    ('Soft SpecDrop (ours)',    'outputs/rtx5090_ablation/ablation_se_ratio_0x_pa0.7_wr1.0_s42'),
]
NLP_RUNS = [
    ('Dense Transformer',       'outputs/rtx5090_nlp_faithful/dense_s42'),
    ('Switch (N=32)',           'outputs/rtx5090_nlp_faithful/switch_s42'),
    ('Hash Layers (N=8)',       'outputs/rtx5090_nlp_faithful/hash_layers_s42'),
    ('SMoE-Dropout (N=16)',     'outputs/rtx5090_nlp_faithful/smoe_dropout_s42'),
    ('Demix (K=7)',             'outputs/rtx5090_nlp_faithful/demix_s42'),
    ('No-Routing (K=7)',        'outputs/rtx5090_nlp_faithful/no_routing_s42'),
    ('No-Routing + SE (matched)', 'outputs/rtx5090_nlp_faithful/no_routing_se05_s42'),
    ('Soft SpecDrop (ours)',    'outputs/rtx5090_nlp_faithful/ours_phaseP_s42'),
]


def _cfg(run_dir):
    cands = sorted(glob.glob(f'{run_dir}/_tmp*.yaml'))
    if cands:
        with open(cands[0]) as f:
            return yaml.safe_load(f)
    with open(f'{run_dir}/results.json') as f:
        return json.load(f)['config']


def _macs(model, inputs):
    fa = FlopCountAnalysis(model, inputs)
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)
    return fa.total()


def trace_cifar():
    from models import build_model
    rows = []
    for label, run in CIFAR_RUNS:
        cfg = _cfg(run)
        model = build_model(cfg).eval()
        x = torch.randn(1, 3, 32, 32)
        K = cfg['model'].get('num_branches')
        try:
            m = _macs(model, (x,))
        except Exception:
            mask = torch.full((1, K), 1.0 / K)
            m = _macs(model, (x, mask))
        n = sum(p.numel() for p in model.parameters())
        rows.append((label, n, m))
        print(f'[cifar] {label:28s} params={n/1e6:.3f}M  MACs={m/1e6:.1f}M')
    return rows


def trace_nlp():
    from run_nlp import build_nlp_model
    rows = []
    for label, run in NLP_RUNS:
        cfg = _cfg(run)
        model = build_nlp_model(cfg).eval()
        x = torch.randint(100, 50000, (1, 512))
        K = cfg['model'].get('num_branches')
        attempts = [(x,)]
        if K:
            attempts.append((x, torch.full((1, K), 1.0 / K)))
        attempts.append((x, torch.zeros(1, dtype=torch.long)))  # demix domain_ids
        m = None
        for inp in attempts:
            try:
                m = _macs(model, inp)
                break
            except Exception:
                continue
        if m is None:
            raise RuntimeError(f'no forward signature worked for {label}')
        n = sum(p.numel() for p in model.parameters())
        rows.append((label, n, m))
        print(f'[nlp] {label:28s} params={n/1e6:.3f}M  MACs/seq={m/1e9:.2f}G')
    return rows


def lora_analytic():
    """Adapter MACs per token, closed form. All methods share the frozen
    Llama-3.2-1B base forward (~1.24B params ≈ 1.24 GMACs/token, weight-
    dominated regime); adapters add K_active x r x (d_in + d_out) per
    attached linear. Dims: hidden 2048, GQA kv 512, ffn 8192, 16 layers."""
    D, KV, F_, L = 2048, 512, 8192, 16
    # (d_in, d_out) of the 7 attached linears per layer
    seven = [(D, D), (D, KV), (D, KV), (D, D),          # q k v o
             (D, F_), (D, F_), (F_, D)]                 # gate up down
    ffn3 = [(D, F_), (D, F_), (F_, D)]
    base_macs = 1.236e9  # ~= base param count (weight-dominated per token)

    def adapter(linears, r, k_active):
        return L * k_active * r * sum(di + do for di, do in linears)

    rows = [
        ('Single LoRA r=320',        adapter(seven, 320, 1)),
        ('MoCLE (E=4+1, r=63)',      adapter(seven, 63, 2)),          # top-1 + universal
        ('LoRAMoE (K=6, r=76, FFN)', adapter(ffn3, 76, 6)),           # soft, all experts
        ('HydraLoRA (N=8, r=67)',    adapter(seven, 67, 8)),          # shared A, 8 B-heads
        ('No-Routing (K=20, r=16)',  adapter(seven, 16, 20)),
        ('Soft SpecDrop (ours, K=20 r=15 + SE15)', adapter(seven, 15, 21)),
    ]
    out = []
    for label, am in rows:
        out.append((label, am, base_macs))
        print(f'[lora] {label:40s} adapter={am/1e6:.1f}M MACs/tok '
              f'(+{100*am/base_macs:.1f}% of base {base_macs/1e9:.2f}G)')
    return out


def main():
    cifar, nlp, lora = trace_cifar(), trace_nlp(), lora_analytic()
    os.makedirs('outputs/analysis', exist_ok=True)
    md = ['# FLOPs (MACs) — CIFAR / NLP / LoRA main tables',
          '', 'ViT numbers: see vit_flops.md. MACs x2 = FLOPs.',
          '', '## CIFAR-100 (per 32x32 image)',
          '', '| Method | Params | MACs (M) |', '|---|---|---|']
    md += [f'| {l} | {n/1e6:.3f}M | {m/1e6:.1f} |' for l, n, m in cifar]
    md += ['', '## SlimPajama 30M (per 512-token sequence)',
           '', '| Method | Params | MACs (G/seq) |', '|---|---|---|']
    md += [f'| {l} | {n/1e6:.3f}M | {m/1e9:.2f} |' for l, n, m in nlp]
    md += ['', '## SuperNI LoRA (adapter add-on per token; frozen Llama-3.2-1B '
           'base ~1.24 GMACs/token shared by all methods)',
           '', '| Method | Adapter MACs/token (M) | % of base |', '|---|---|---|']
    md += [f'| {l} | {a/1e6:.1f} | {100*a/b:.1f}% |' for l, a, b in lora]
    with open('outputs/analysis/flops_all_settings.md', 'w') as f:
        f.write('\n'.join(md) + '\n')
    with open('outputs/analysis/flops_all_settings.json', 'w') as f:
        json.dump({'cifar': cifar, 'nlp': nlp,
                   'lora_adapter_macs': [(l, a) for l, a, _ in lora]}, f, indent=2)
    print('wrote outputs/analysis/flops_all_settings.{md,json}')


if __name__ == '__main__':
    main()

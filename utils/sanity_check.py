"""Pre-experiment sanity checks: verify param counts match baselines.

Usage:
    from utils.sanity_check import check_param_budget
    check_param_budget(cfg)  # raises AssertionError if >2% deviation
                              # (lora entry overrides to 3% — see tolerance field)

Called automatically by run.py and run_nlp.py before training starts.
"""

import torch


# Reference param counts (computed once, hardcoded for speed).
# `trainable_only=True` switches the check to only count requires_grad params
# — needed for LoRA where the frozen 3B base dominates total param count but
# the thing we actually need to budget-match is the 225M LoRA adapters.
# `tolerance` overrides the module-level default per-dataset.
REFERENCE_PARAMS = {
    'cifar100': {
        'name': 'ResNet-110',
        'params': 1_736_564,  # ResNet(base_channels=16, num_blocks=18, num_classes=100)
    },
    'imagenet': {
        'name': 'ViT-Small/16',
        'params': 22_050_664,  # vit_small(num_classes=1000), matches timm vit_small_patch16_224
    },
    'slimpajama': {
        'name': 'Dense Transformer (FFN=1536)',
        'params': 30_142_848,  # TransformerLM(hidden=384, layers=6, heads=6, ffn=1536)
    },
    'lora': {
        # Ours at K=20 × r=16 on Llama-3.2-1B (GQA: num_heads=32 × head_dim=64
        # on q/o_proj but num_kv_heads=8 × head_dim=64 on k/v_proj — so
        # k/v_proj are 2048→512, NOT 2048→2048). Per-rank per-layer (7 sites):
        #   q/o:        r × (2048 + 2048) × 2 =   8192r
        #   k/v (GQA):  r × (2048 +  512) × 2 =   5120r
        #   gate/up:    r × (2048 + 8192) × 2 =  20480r
        #   down:       r × (8192 + 2048) × 1 =  10240r
        #   total/layer:                         44032r
        # × 16 layers = 704,512·r. K=20 × r=16 → 225,443,840 exactly.
        # Baselines (LoRAMoE FFN-only K=6, HydraLoRA N=8 asymmetric, MoCLE
        # E=4+uni, Single r=320) have their rank set so total ≈ 225M within
        # the 3% tolerance that covers split_lora_rank_budget_for_se rounding
        # + asymmetric gate-param additives.
        # NOTE: switched from Llama-3.2-3B (486M budget) to 1B (225M budget)
        # on 2026-04-24 for wall-time compression. Same method, smaller base,
        # ~2.5× faster training. Param-count ratios between baselines preserved.
        'name': 'MB-LoRA K=20 r=16 Llama-3.2-1B (all 7 linears, GQA-aware)',
        'params': 225_443_840,
        'trainable_only': True,
        'tolerance': 0.03,
    },
}

TOLERANCE = 0.02  # default 2% max deviation (overridable per dataset entry)


def check_param_budget(cfg, model):
    """Verify model param count is within tolerance of the reference baseline.

    Args:
        cfg: experiment config dict
        model: built PyTorch model

    Raises:
        AssertionError if param count deviates beyond the dataset's tolerance.
    """
    # Opt-out for experiments at non-default scale (e.g., T2.3 GPT-2 125M
    # SlimPajama scale-up where REFERENCE_PARAMS['slimpajama']=30M doesn't
    # apply). Logs param count for the experiment record but does not crash.
    if cfg.get('skip_param_check', False):
        n = sum(p.numel() for p in model.parameters())
        print(f'  Param check: SKIPPED via cfg.skip_param_check '
              f'(model has {n:,} params)')
        return
    # Determine dataset. Prefer explicit `data.dataset` key; fall back to data_dir
    # substring match only if the caller forgot to set it (deprecation path).
    dataset = cfg.get('data', {}).get('dataset', '')
    if not dataset:
        if 'slimpajama' in str(cfg.get('data', {}).get('data_dir', '')):
            dataset = 'slimpajama'
            print("  [sanity_check] WARNING: 'data.dataset' key missing; inferred "
                  "'slimpajama' from data_dir. Set cfg['data']['dataset'] explicitly.")
        # Infer 'lora' from a distinctive LoRA-model type (they all share the
        # 'multi_branch_lora' / 'single_lora' / 'hydra_lora' / 'lora_moe' /
        # 'mocle' types defined in models.lora_models.build_lora_model).
        elif str(cfg.get('model', {}).get('type', '')) in (
                'multi_branch_lora', 'single_lora', 'hydra_lora',
                'lora_moe', 'mocle'):
            dataset = 'lora'

    if dataset not in REFERENCE_PARAMS:
        return  # unknown dataset, skip check

    ref = REFERENCE_PARAMS[dataset]
    if ref.get('trainable_only', False):
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        count_kind = 'trainable'
    else:
        model_params = sum(p.numel() for p in model.parameters())
        count_kind = 'total'
    ratio = model_params / ref['params']
    deviation = abs(ratio - 1.0)
    # Per-cfg override (top-level key) takes precedence over the dataset-default
    # so individual experiments can opt into a wider
    # tolerance without affecting all sibling configs.
    tolerance = cfg.get('param_budget_tolerance',
                          ref.get('tolerance', TOLERANCE))

    if deviation > tolerance:
        raise AssertionError(
            f"SANITY CHECK FAILED: model has {model_params:,} {count_kind} params "
            f"({ratio:.3f}x {ref['name']} = {ref['params']:,}), "
            f"deviation {deviation:.1%} exceeds {tolerance:.0%} tolerance. "
            f"Check branch_channels / shared_expert / rank config."
        )

    print(f"  Param check ({count_kind}): {model_params:,} = "
          f"{ratio:.3f}x {ref['name']} — OK")


def check_all_configs_in_script(configs, dataset='cifar100'):
    """Batch check multiple configs (for experiment scripts).

    Args:
        configs: list of (name, cfg_dict) tuples
        dataset: 'cifar100', 'imagenet', or 'slimpajama'
    """
    from models import build_model

    ref = REFERENCE_PARAMS[dataset]
    print(f"Sanity check: all configs vs {ref['name']} ({ref['params']:,} params)")
    print(f"{'Name':>40} {'Params':>12} {'Ratio':>8} {'Status':>8}")
    print("-" * 72)

    all_pass = True
    for name, cfg in configs:
        model = build_model(cfg)
        model_params = sum(p.numel() for p in model.parameters())
        ratio = model_params / ref['params']
        deviation = abs(ratio - 1.0)

        if deviation > TOLERANCE:
            status = "FAIL"
            all_pass = False
        else:
            status = "OK"

        print(f"{name:>40} {model_params:>12,} {ratio:>8.3f} {status:>8}")
        del model

    if not all_pass:
        raise AssertionError(
            f"SANITY CHECK FAILED: one or more configs exceed {TOLERANCE:.0%} param tolerance. "
            "Fix configs before running experiments."
        )
    print("All configs passed param budget check.\n")

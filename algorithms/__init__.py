"""Algorithm factory.

After the 2026-04-12 pivot to 3-setting native comparison, only two modular-dropout
algorithms remain in the main repo:

  - `no_dropout`  — architectural reference (MultiBranch, equal weights per branch)
  - `soft_specdrop` (OURS) — our method: category-conditioned soft mask with cosine
                              warmup and fixed-denominator merge

All adapted MoE-routing algorithms (hash_routing / learned_routing / soft_moe_routing /
mod_squad / comet_routing / smoe_dropout / specdrop / random_dropout) have been moved
to `archive/algorithms/` since their faithful re-implementations now live as dedicated
models under `models/{switch,hash_layers,smoe_dropout,demix}_transformer_lm.py` and
`models/{mod_squad,soft_moe,comet}_vit.py`.
"""

from .base import ModularDropout
from .no_dropout import NoDropout
from .soft_specdrop import SoftSpecDrop
from .stochastic_specdrop import StochasticSpecDrop
from .random_dropout import RandomDropout
from .hard_category import HardCategory


def _get_feature_dim(cfg):
    """Compute stem feature dimension for algorithms that route from features."""
    mcfg = cfg.get('model', {})
    model_type = mcfg.get('type', '')

    if model_type in ('multi_branch_vit_small', 'multi_branch_vit'):
        return mcfg.get('embed_dim', 384)
    elif model_type == 'multi_branch_transformer_lm':
        return mcfg.get('hidden_dim', 384)
    else:
        # CIFAR models: stem output channels = base_channels
        return mcfg.get('base_channels', 16)


def build_algorithm(cfg):
    """Build a modular-dropout algorithm from config.

    Returns None for `algorithm.type == 'none'` (used by dense / self-contained
    faithful models whose routing is baked into the model itself).
    """
    acfg = cfg.get('algorithm', {})
    algo_type = acfg.get('type', 'none')

    if algo_type == 'none':
        return None

    num_branches = cfg['model'].get('num_branches', 4)
    # `_num_categories` is the runtime-injected value (run.py sets it for
    # ImageNet from BREEDS K=46; run_nlp.py from SlimPajama domain count).
    # `num_categories` is a YAML-direct override used by experiments that
    # want a different M from the default 20 (e.g., T1.2 fine-label CIFAR
    # M=100). Runtime injection takes precedence to preserve existing flow.
    num_categories = acfg.get('_num_categories',
                                acfg.get('num_categories', 20))

    if algo_type == 'no_dropout':
        return NoDropout(num_modules=num_branches, num_categories=num_categories)

    elif algo_type == 'soft_specdrop':
        tcfg = cfg.get('training', {})
        total_epochs = tcfg.get('epochs', 200)
        return SoftSpecDrop(
            num_modules=num_branches,
            num_categories=num_categories,
            p_active=acfg.get('p_active', 0.9),
            p_inactive=acfg.get('p_inactive', 0.1),
            assignment=acfg.get('assignment', 'round_robin'),
            warmup_ratio=acfg.get('warmup_ratio', 0.0),
            total_epochs=total_epochs,
            # E4 (random-permutation A ablation): per-seed permutation. Default
            # falls back to experiment seed so each seed gets a different A.
            assignment_seed=acfg.get('assignment_seed', cfg.get('seed', 42)),
            # Per-category routing extension. None → bit-identical to the
            # original scalar (p_a, p_i) behavior. Populated by run_nlp.py
            # from the data cache when enabled.
            frac_per_category=acfg.get('frac_per_category', None),
            amplification_beta=acfg.get('amplification_beta', 1.0),
            warmup_schedule=acfg.get('warmup_schedule', 'cosine'),
            warmup_unit=acfg.get('warmup_unit', 'epoch'),
        )

    elif algo_type == 'stochastic_specdrop':
        tcfg = cfg.get('training', {})
        total_epochs = tcfg.get('epochs', 200)
        return StochasticSpecDrop(
            num_modules=num_branches,
            num_categories=num_categories,
            p_active=acfg.get('p_active', 0.9),
            p_inactive=acfg.get('p_inactive', 0.1),
            assignment=acfg.get('assignment', 'round_robin'),
            warmup_ratio=acfg.get('warmup_ratio', 0.0),
            total_epochs=total_epochs,
            denom_mode=acfg.get('denom_mode', 'fixed'),
            assignment_seed=acfg.get('assignment_seed', cfg.get('seed', 42)),
        )

    elif algo_type == 'random_dropout':
        return RandomDropout(
            num_modules=num_branches,
            num_categories=num_categories,
            drop_prob=acfg.get('drop_prob', 0.5),
            denom_mode=acfg.get('denom_mode', 'fixed'),
        )

    elif algo_type == 'hard_category':
        return HardCategory(num_modules=num_branches,
                             num_categories=num_categories)

    else:
        raise ValueError(
            f"Unknown algorithm type: {algo_type!r}. Available: "
            f"{{'none', 'no_dropout', 'soft_specdrop', 'stochastic_specdrop', "
            f"'random_dropout', 'hard_category'}}. Faithful MoE baselines are "
            f"separate models under models/. See archive/algorithms/ for the "
            f"adapted (non-native) variants if needed for Appendix A ablations."
        )

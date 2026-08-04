"""Model factory."""

from .resnet import ResNet
from .multi_branch import (MultiBranchResNet, MultiBranchResNet110,
                           GroupedMultiBranchResNet110)
from .vit import ViT, vit_small
from .multi_branch_vit import MultiBranchViT, multi_branch_vit_small
from .et_dropout import ExampleTiedDropoutResNet, example_tied_dropout_resnet110
from .contextual_dropout import (ContextualDropoutResNet,
                                 contextual_dropout_resnet110)
from .switch_transformer_lm import SwitchTransformerLM
from .hash_layers_transformer_lm import HashLayersTransformerLM
from .smoe_dropout_transformer_lm import SMoEDropoutTransformerLM
from .demix_transformer_lm import DemixTransformerLM
from .mod_squad_vit import ModSquadViT
from .alf_moe_vit import ALFMoEViT
from .soft_moe_vit import SoftMoEViT
from .comet_vit import COMETViT


def build_model(cfg):
    """Build model from config dict."""
    mcfg = cfg['model']
    model_type = mcfg['type']
    num_blocks = mcfg.get('num_blocks', 3)
    num_classes = mcfg.get('num_classes', 100)
    base_channels = mcfg.get('base_channels', 16)

    if model_type == 'resnet':
        return ResNet(
            num_blocks=num_blocks,
            num_classes=num_classes,
            dropout_rate=mcfg.get('dropout_rate', 0.0),
            base_channels=base_channels,
            stochastic_depth_rate=mcfg.get('stochastic_depth_rate', 0.0),
        )
    elif model_type == 'multi_branch':
        return MultiBranchResNet(
            num_branches=mcfg.get('num_branches', 4),
            num_blocks=num_blocks,
            num_classes=num_classes,
            branch_channels=mcfg.get('branch_channels', 16),
            base_channels=base_channels,
            stochastic_depth_rate=mcfg.get('stochastic_depth_rate', 0.0),
            shared_expert=mcfg.get('shared_expert', False),
        )
    elif model_type in ('multi_branch_resnet110', 'grouped_multi_branch_resnet110'):
        # multi_branch_resnet110 now uses grouped conv by default (~4-5x faster).
        # Use 'multi_branch_resnet110_forloop' to get the original for-loop version.
        bc = mcfg.get('branch_channels', None)
        return GroupedMultiBranchResNet110(
            num_branches=mcfg.get('num_branches', 20),
            num_blocks=num_blocks,
            num_classes=num_classes,
            base_channels=base_channels,
            branch_channels=bc,
            shared_expert_blocks=mcfg.get('shared_expert_blocks', None),
            stochastic_depth_rate=mcfg.get('stochastic_depth_rate', 0.0),
            shared_expert=mcfg.get('shared_expert', False),
        )
    elif model_type == 'multi_branch_resnet110_forloop':
        bc = mcfg.get('branch_channels', None)
        return MultiBranchResNet110(
            num_branches=mcfg.get('num_branches', 20),
            num_blocks=num_blocks,
            num_classes=num_classes,
            base_channels=base_channels,
            branch_channels=bc,
            shared_expert_blocks=mcfg.get('shared_expert_blocks', None),
            stochastic_depth_rate=mcfg.get('stochastic_depth_rate', 0.0),
            shared_expert=mcfg.get('shared_expert', False),
        )
    elif model_type == 'vit_small':
        return vit_small(
            num_classes=num_classes,
            img_size=mcfg.get('img_size', 224),
            patch_size=mcfg.get('patch_size', 16),
            drop_rate=mcfg.get('drop_rate', 0.0),
            attn_drop_rate=mcfg.get('attn_drop_rate', 0.0),
            drop_path_rate=mcfg.get('drop_path_rate', 0.0),
        )
    elif model_type == 'vit':
        return ViT(
            img_size=mcfg.get('img_size', 224),
            patch_size=mcfg.get('patch_size', 16),
            in_channels=mcfg.get('in_channels', 3),
            num_classes=num_classes,
            embed_dim=mcfg.get('embed_dim', 384),
            depth=mcfg.get('depth', 12),
            num_heads=mcfg.get('num_heads', 6),
            mlp_ratio=mcfg.get('mlp_ratio', 4.0),
            qkv_bias=mcfg.get('qkv_bias', True),
            drop_rate=mcfg.get('drop_rate', 0.0),
            attn_drop_rate=mcfg.get('attn_drop_rate', 0.0),
            drop_path_rate=mcfg.get('drop_path_rate', 0.0),
        )
    elif model_type == 'multi_branch_vit_small':
        return multi_branch_vit_small(
            num_classes=num_classes,
            num_branches=mcfg.get('num_branches', 46),
            branch_hidden=mcfg.get('branch_hidden', None),
            shared_expert_dim=mcfg.get('shared_expert_dim', 0),
            img_size=mcfg.get('img_size', 224),
            patch_size=mcfg.get('patch_size', 16),
            drop_rate=mcfg.get('drop_rate', 0.0),
            attn_drop_rate=mcfg.get('attn_drop_rate', 0.0),
            drop_path_rate=mcfg.get('drop_path_rate', 0.0),
        )
    elif model_type == 'multi_branch_vit':
        return MultiBranchViT(
            img_size=mcfg.get('img_size', 224),
            patch_size=mcfg.get('patch_size', 16),
            in_channels=mcfg.get('in_channels', 3),
            num_classes=num_classes,
            embed_dim=mcfg.get('embed_dim', 384),
            depth=mcfg.get('depth', 12),
            num_heads=mcfg.get('num_heads', 6),
            mlp_ratio=mcfg.get('mlp_ratio', 4.0),
            num_branches=mcfg.get('num_branches', 46),
            branch_hidden=mcfg.get('branch_hidden', None),
            shared_expert_dim=mcfg.get('shared_expert_dim', 0),
            qkv_bias=mcfg.get('qkv_bias', True),
            drop_rate=mcfg.get('drop_rate', 0.0),
            attn_drop_rate=mcfg.get('attn_drop_rate', 0.0),
            drop_path_rate=mcfg.get('drop_path_rate', 0.0),
        )
    elif model_type == 'example_tied_dropout_resnet110':
        return example_tied_dropout_resnet110(
            num_classes=num_classes,
            num_examples=mcfg.get('num_examples', 50000),
            alpha_mem=mcfg.get('alpha_mem', 0.5),
            mem_keep_rate=mcfg.get('mem_keep_rate', 0.5),
            # Routing-structure seed DECOUPLED from experiment seed — fixed default
            # so 3-seed mean±std measures training variance only, not mask variance.
            mask_seed=mcfg.get('mask_seed', 42),
        )
    elif model_type == 'contextual_dropout_resnet110':
        return contextual_dropout_resnet110(
            num_classes=num_classes,
            reduction=mcfg.get('reduction', 8),
            kl_weight=mcfg.get('kl_weight', 1e-4),
        )
    elif model_type == 'switch_transformer_lm':
        return SwitchTransformerLM(
            vocab_size=mcfg['vocab_size'],
            hidden_dim=mcfg.get('hidden_dim', 384),
            num_layers=mcfg.get('num_layers', 6),
            num_heads=mcfg.get('num_heads', 6),
            # Paper-canonical N=32, param-matched (32×48=1536).
            num_experts=mcfg.get('num_experts', 32),
            ffn_dim_per_expert=mcfg.get('ffn_dim_per_expert', 48),
            max_seq_len=mcfg.get('max_seq_len', 512),
            load_balance_weight=mcfg.get('load_balance_weight', 0.01),
            dropout=mcfg.get('dropout', 0.1),
        )
    elif model_type == 'hash_layers_transformer_lm':
        return HashLayersTransformerLM(
            vocab_size=mcfg['vocab_size'],
            hidden_dim=mcfg.get('hidden_dim', 384),
            num_layers=mcfg.get('num_layers', 6),
            num_heads=mcfg.get('num_heads', 6),
            # Paper-canonical N=8, param-matched to 1536 dense FFN (8×192).
            num_experts=mcfg.get('num_experts', 8),
            ffn_dim_per_expert=mcfg.get('ffn_dim_per_expert', 192),
            max_seq_len=mcfg.get('max_seq_len', 512),
            # Routing-structure seed DECOUPLED from experiment seed.
            hash_seed=mcfg.get('hash_seed', 42),
            dropout=mcfg.get('dropout', 0.1),
        )
    elif model_type == 'smoe_dropout_transformer_lm':
        return SMoEDropoutTransformerLM(
            vocab_size=mcfg['vocab_size'],
            hidden_dim=mcfg.get('hidden_dim', 384),
            num_layers=mcfg.get('num_layers', 6),
            num_heads=mcfg.get('num_heads', 6),
            num_experts=mcfg.get('num_experts', 16),
            ffn_dim_per_expert=mcfg.get('ffn_dim_per_expert', 96),
            max_seq_len=mcfg.get('max_seq_len', 512),
            k_init=mcfg.get('k_init', 1),
            # Paper (Chen 2023) has NO per-expert Bernoulli dropout; the "dropout"
            # in SMoE-Dropout refers to the k-schedule routing sparsity itself.
            expert_drop_prob=mcfg.get('expert_drop_prob', 0.0),
            # Routing-structure seed DECOUPLED from experiment seed.
            router_seed=mcfg.get('router_seed', 42),
            dropout=mcfg.get('dropout', 0.1),
        )
    elif model_type == 'demix_transformer_lm':
        return DemixTransformerLM(
            vocab_size=mcfg['vocab_size'],
            hidden_dim=mcfg.get('hidden_dim', 384),
            num_layers=mcfg.get('num_layers', 6),
            num_heads=mcfg.get('num_heads', 6),
            num_domains=mcfg.get('num_domains', 7),
            ffn_dim_per_expert=mcfg.get('ffn_dim_per_expert', 220),
            max_seq_len=mcfg.get('max_seq_len', 512),
            dropout=mcfg.get('dropout', 0.1),
        )
    elif model_type == 'mod_squad_vit':
        return ModSquadViT(
            img_size=mcfg.get('img_size', 224),
            patch_size=mcfg.get('patch_size', 16),
            num_classes=num_classes,
            embed_dim=mcfg.get('embed_dim', 384),
            depth=mcfg.get('depth', 12),
            num_heads=mcfg.get('num_heads', 6),
            num_experts=mcfg.get('num_experts', 16),
            expert_hidden=mcfg.get('expert_hidden', None),
            top_k=mcfg.get('top_k', 2),
            mi_weight=mcfg.get('mi_weight', 0.001),
            # Default None → class __init__ falls back to 1/N (Shazeer 2017
            # standard). Explicit float overrides (e.g. 1.0 for an ablation).
            noise_std=mcfg.get('noise_std', None),
            qkv_bias=mcfg.get('qkv_bias', True),
            drop_rate=mcfg.get('drop_rate', 0.0),
            attn_drop_rate=mcfg.get('attn_drop_rate', 0.0),
            drop_path_rate=mcfg.get('drop_path_rate', 0.0),
        )
    elif model_type == 'alf_moe_vit':
        # Auxiliary-Loss-Free load balancing (Wang et al. 2024) — additional
        # baseline. Clone of mod_squad_vit minus MI loss / noise
        # gating, plus per-expert selection-bias sign update.
        return ALFMoEViT(
            img_size=mcfg.get('img_size', 224),
            patch_size=mcfg.get('patch_size', 16),
            num_classes=num_classes,
            embed_dim=mcfg.get('embed_dim', 384),
            depth=mcfg.get('depth', 12),
            num_heads=mcfg.get('num_heads', 6),
            num_experts=mcfg.get('num_experts', 16),
            expert_hidden=mcfg.get('expert_hidden', None),
            top_k=mcfg.get('top_k', 2),
            bias_update_rate=mcfg.get('bias_update_rate', 1e-3),
            qkv_bias=mcfg.get('qkv_bias', True),
            drop_rate=mcfg.get('drop_rate', 0.0),
            attn_drop_rate=mcfg.get('attn_drop_rate', 0.0),
            drop_path_rate=mcfg.get('drop_path_rate', 0.0),
        )
    elif model_type == 'soft_moe_vit':
        return SoftMoEViT(
            img_size=mcfg.get('img_size', 224),
            patch_size=mcfg.get('patch_size', 16),
            num_classes=num_classes,
            embed_dim=mcfg.get('embed_dim', 384),
            depth=mcfg.get('depth', 12),
            num_heads=mcfg.get('num_heads', 6),
            num_experts=mcfg.get('num_experts', 32),
            expert_hidden=mcfg.get('expert_hidden', None),
            slots_per_expert=mcfg.get('slots_per_expert', 1),
            moe_start_block=mcfg.get('moe_start_block', 0),
            qkv_bias=mcfg.get('qkv_bias', True),
            drop_rate=mcfg.get('drop_rate', 0.0),
            attn_drop_rate=mcfg.get('attn_drop_rate', 0.0),
            drop_path_rate=mcfg.get('drop_path_rate', 0.0),
        )
    elif model_type == 'comet_vit':
        return COMETViT(
            img_size=mcfg.get('img_size', 224),
            patch_size=mcfg.get('patch_size', 16),
            num_classes=num_classes,
            embed_dim=mcfg.get('embed_dim', 384),
            depth=mcfg.get('depth', 12),
            num_heads=mcfg.get('num_heads', 6),
            mlp_ratio=mcfg.get('mlp_ratio', 4.0),
            # Paper's k-WTA is designed for small keep fractions (0.1–0.25);
            # 0.5 is outside the tested regime.
            p_keep=mcfg.get('p_keep', 0.25),
            # Routing-structure seed DECOUPLED from experiment seed.
            routing_seed=mcfg.get('routing_seed', 42),
            qkv_bias=mcfg.get('qkv_bias', True),
            drop_rate=mcfg.get('drop_rate', 0.0),
            attn_drop_rate=mcfg.get('attn_drop_rate', 0.0),
            drop_path_rate=mcfg.get('drop_path_rate', 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

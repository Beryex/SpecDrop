"""Training loop for all experiment configurations.

This module handles the train/eval loop. It is algorithm-agnostic:
the algorithm provides masks, the model uses them, and the trainer
just orchestrates the process.
"""

import os
import json
import time
import random
import logging
import warnings
import numpy as np
import torch

# Suppress torch.compile internal warnings (pow_by_natural, graph breaks)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*pow_by_natural.*")
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import torch.nn.functional as torchF
from evaluation.metrics import evaluate_accuracy, evaluate_per_category
from evaluation.compute import ComputeTracker


def _is_multi_branch(model):
    """Check if model is a multi-branch architecture (for-loop or grouped)."""
    return hasattr(model, 'num_branches')


def _unwrap(model):
    """Return the underlying module when wrapped by torch.compile.
    torch.compile produces an OptimizedModule with the real module at
    `._orig_mod`; isinstance checks on the wrapper return False."""
    return getattr(model, '_orig_mod', model)


def _is_faithful_self_contained(model):
    """Faithful baselines that do their own routing internally and return
    (logits, aux_loss). No external algorithm is used."""
    from models.et_dropout import ExampleTiedDropoutResNet
    from models.contextual_dropout import ContextualDropoutResNet
    from models.mod_squad_vit import ModSquadViT
    from models.soft_moe_vit import SoftMoEViT
    from models.comet_vit import COMETViT
    from models.alf_moe_vit import ALFMoEViT
    return isinstance(_unwrap(model), (
        ExampleTiedDropoutResNet, ContextualDropoutResNet,
        ModSquadViT, SoftMoEViT, COMETViT, ALFMoEViT,
    ))


def _faithful_forward(model, images, example_ids):
    """Call a faithful self-contained model and normalize its return to (logits, aux)."""
    from models.et_dropout import ExampleTiedDropoutResNet
    kwargs = {}
    if isinstance(_unwrap(model), ExampleTiedDropoutResNet):
        kwargs['example_ids'] = example_ids
    output = model(images, **kwargs)
    if isinstance(output, tuple):
        logits, aux = output
    else:
        logits = output
        aux = torch.tensor(0.0, device=images.device)
    return logits, aux


class Trainer:
    """Unified trainer for baseline ResNet and multi-branch models.

    Usage:
        trainer = Trainer(cfg, model, algorithm, train_loader, test_loader, device)
        results = trainer.train()
    """

    def __init__(self, cfg, model, algorithm, train_loader, test_loader, device,
                 use_wandb=False):
        self.cfg = cfg
        self.model = model.to(device)
        self.algorithm = algorithm
        # Move algorithm's nn.Module components (e.g., router Linear) to device
        if algorithm is not None:
            for v in vars(algorithm).values():
                if isinstance(v, torch.nn.Module):
                    v.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.use_wandb = use_wandb
        self.global_step = 0

        tcfg = cfg['training']
        self.epochs = tcfg.get('epochs', 200)
        self.lr = tcfg.get('lr', 0.1)
        # NaN-loss budget: abort the run if too many NaN-skip events. Lets
        # the batch dispatcher move on to the next config instead of burning
        # GPU on a permanently-broken trajectory (e.g., compile + bf16 +
        # masked-merge edge cases). Configurable via training cfg.
        self.nan_abort_consecutive = tcfg.get('nan_abort_consecutive', 50)
        self.nan_abort_total = tcfg.get('nan_abort_total', 200)
        self._nan_consecutive = 0
        self._nan_total = 0
        self.grad_accum_steps = tcfg.get('grad_accum_steps', 1)
        # Smoke-test knob: cap training at `max_steps` mini-batches per epoch.
        self.max_steps = tcfg.get('max_steps', None)
        # Label-shuffle control: shuffle the category labels that are fed to
        # the routing algorithm, BOTH at training and eval. Tests whether the
        # gain comes from the category→branch mapping (drops to no-routing if
        # label is the carrier) or from the routing mechanism itself (stays
        # near OURS if mechanism carries the gain).
        self.shuffle_category_labels = tcfg.get('shuffle_category_labels', False)
        if self.shuffle_category_labels:
            print("  [Q3 ablation] Category labels will be RANDOMLY SHUFFLED "
                  "before routing (breaks category→branch mapping).")
        # Fine-label control: switch routing-input from coarse_labels (default M=20
        # CIFAR-100 superclasses) to fine_labels (M=100). With K=20 round_robin
        # this gives 5 fine labels per branch — direct test of granularity-
        # alignment claim within a single dataset (n+1 cross-modality datapoint).
        self.route_label_type = tcfg.get('route_label_type', 'coarse')
        if self.route_label_type not in ('coarse', 'fine'):
            raise ValueError(
                f'route_label_type must be "coarse" or "fine", got '
                f'{self.route_label_type!r}')
        if self.route_label_type == 'fine':
            print("  [T1.2 fine-label routing] using fine_labels (M=100) as "
                  "category_ids for the routing algorithm.")

        # Gather (name, param) pairs from model + algorithm. Using NAMES (not just
        # shapes) so that 3-D tensors like `pos_embed` (1, N+1, D) and `cls_token`
        # (1, 1, D) go to the no-wd group — a plain ndim<=1 filter misses them.
        named_params = list(model.named_parameters())
        if algorithm is not None:
            for k, v in vars(algorithm).items():
                if isinstance(v, nn.Parameter) and v.requires_grad:
                    named_params.append((f'algorithm.{k}', v))
                elif isinstance(v, nn.Module):
                    named_params.extend(
                        (f'algorithm.{k}.{n}', p)
                        for n, p in v.named_parameters() if p.requires_grad)
            algo_total = sum(p.numel() for n, p in named_params if n.startswith('algorithm.'))
            if algo_total > 0:
                print(f"  Algorithm trainable params: {algo_total:,}")

        params = [p for _, p in named_params]

        # Optimizer: SGD (default, for ResNet-110 CIFAR) or AdamW (for ViT ImageNet).
        opt_name = tcfg.get('optimizer', 'sgd').lower()

        # DeiT/ViT no-wd convention: biases, BN/LN gains, positional embeddings,
        # CLS tokens. Name-based filter catches `pos_embed` / `cls_token` (ndim=3
        # would otherwise put them in the decay group).
        def _split_wd_by_name(named):
            NO_WD_NAME_PATTERNS = ('pos_embed', 'cls_token')
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                is_nd1 = p.ndim <= 1
                is_bias = name.endswith('.bias')
                is_nowd_named = any(pat in name for pat in NO_WD_NAME_PATTERNS)
                if is_nd1 or is_bias or is_nowd_named:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        if opt_name == 'adamw':
            betas = tuple(tcfg.get('betas', [0.9, 0.999]))
            wd = tcfg.get('weight_decay', 0.05)
            decay_p, no_decay_p = _split_wd_by_name(named_params)
            self.optimizer = torch.optim.AdamW(
                [{'params': decay_p, 'weight_decay': wd},
                 {'params': no_decay_p, 'weight_decay': 0.0}],
                lr=self.lr,
                betas=betas,
            )
            print(f"  Optimizer: AdamW lr={self.lr} wd={wd} betas={betas} "
                  f"({len(decay_p)} decay params, {len(no_decay_p)} no-wd "
                  f"params [BN/LN/bias/pos_embed/cls_token])")
        else:
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.lr,
                momentum=tcfg.get('momentum', 0.9),
                weight_decay=tcfg.get('weight_decay', 5e-4),
            )

        # Gradient clipping (safety net, default off for SGD CIFAR to preserve
        # ablation reproducibility; default 1.0 for AdamW ViT per DeiT recipe).
        self.max_grad_norm = tcfg.get(
            'max_grad_norm',
            1.0 if opt_name == 'adamw' else 0.0)
        if self.max_grad_norm > 0:
            print(f"  Gradient clipping: max_grad_norm = {self.max_grad_norm}")

        # LR scheduler (with optional warmup)
        schedule = tcfg.get('lr_schedule', 'cosine')
        warmup_epochs = tcfg.get('warmup_epochs', 0)
        # Clamp warmup to leave at least 1 epoch for cosine
        warmup_epochs = min(warmup_epochs, max(self.epochs - 1, 0))
        if schedule == 'cosine':
            if warmup_epochs > 0:
                warmup_sched = LinearLR(
                    self.optimizer, start_factor=1e-3,
                    total_iters=warmup_epochs)
                cosine_sched = CosineAnnealingLR(
                    self.optimizer, T_max=self.epochs - warmup_epochs)
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[warmup_epochs])
                print(f"  LR schedule: linear warmup ({warmup_epochs}ep) + cosine ({self.epochs - warmup_epochs}ep)")
            else:
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(self.epochs, 1))
        else:
            self.scheduler = None

        label_smoothing = tcfg.get('label_smoothing', 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if label_smoothing > 0:
            print(f"  Label smoothing: {label_smoothing}")

        # Mixup / CutMix (DeiT training recipe). Active iff either alpha > 0.
        mixup_alpha = tcfg.get('mixup_alpha', 0.0)
        cutmix_alpha = tcfg.get('cutmix_alpha', 0.0)
        if mixup_alpha > 0 or cutmix_alpha > 0:
            from .mixup import MixupCutmix
            self.mixer = MixupCutmix(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                switch_prob=tcfg.get('mixup_switch_prob', 0.5),
                label_smoothing=label_smoothing,
                num_classes=cfg['model'].get('num_classes', 100),
            )
            print(f"  Mixup α={mixup_alpha}, CutMix α={cutmix_alpha}, "
                  f"switch_prob={tcfg.get('mixup_switch_prob', 0.5)}")
        else:
            self.mixer = None

        # AMP mixed precision. Default fp16 (ResNet CIFAR); configurable to bf16
        # (preferred for ViT ImageNet — fp16 can overflow in attention softmax,
        # bf16 has fp32 range so GradScaler is unnecessary) or fp32 (forces
        # AMP off entirely — needed e.g. for Contextual Dropout where 54-layer
        # Gaussian reparameterization × fp16 compounds to 10^16 variance).
        self.use_amp = (device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'))
        amp_name = tcfg.get('amp_dtype', 'fp16').lower()
        if amp_name in ('fp32', 'float32', 'none', 'off'):
            self.use_amp = False
            self.amp_dtype = torch.float32
            self.scaler = GradScaler('cuda', enabled=False)
        elif amp_name in ('bf16', 'bfloat16'):
            self.amp_dtype = torch.bfloat16
            self.scaler = GradScaler('cuda', enabled=False)   # bf16 has fp32 range
        else:
            self.amp_dtype = torch.float16
            self.scaler = GradScaler('cuda', enabled=self.use_amp)
        if self.use_amp:
            torch.set_float32_matmul_precision('high')
            print(f"  AMP mixed precision: {amp_name} "
                  f"(scaler={'enabled' if self.scaler.is_enabled() else 'disabled'})")
        else:
            print(f"  AMP mixed precision: disabled ({amp_name} forces full fp32)")

        # Compute faithful-model flag BEFORE compile: torch.compile wraps the
        # model in OptimizedModule, and isinstance against the raw class would
        # return False post-compile, incorrectly routing ETD/ContextualDropout
        # through the generic branch_mask path and dropping the example_ids
        # kwarg / (logits, kl) unpack.
        self.is_faithful = _is_faithful_self_contained(self.model)

        # torch.compile for kernel fusion speedup. Determinism is ensured
        # by torch.use_deterministic_algorithms(True) set in run.py.
        # Use 'default' mode for stochastic depth (random branching breaks CUDA Graphs).
        no_compile = cfg.get('_no_compile', False) or tcfg.get('_no_compile', False)
        if no_compile:
            print("  torch.compile: disabled (--no-compile flag)")
        elif self.use_amp:
            has_stoch_depth = cfg.get('model', {}).get('stochastic_depth_rate', 0) > 0
            override = tcfg.get('_compile_mode', None)
            compile_mode = override if override else (
                'default' if has_stoch_depth else 'reduce-overhead')
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
                print(f"  torch.compile: enabled ({compile_mode})")
            except Exception:
                print("  torch.compile: skipped (unsupported)")

        # Output directory
        self.output_dir = cfg.get('output_dir', './outputs/default')
        os.makedirs(self.output_dir, exist_ok=True)

        # Compute tracker
        branch_params = None
        if hasattr(model, 'branches'):
            branch_params = [
                sum(p.numel() for p in branch.parameters())
                for branch in model.branches
            ]
        self.compute_tracker = ComputeTracker(total_branch_params=branch_params)

        # Best accuracy tracking
        self.best_acc = 0.0
        self.start_epoch = 0

    def _log_wandb(self, metrics, step=None):
        """Log metrics to wandb if enabled."""
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def resume_from_checkpoint(self, path):
        """Load checkpoint and prepare to resume training."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.best_acc = ckpt.get('best_acc', 0.0)
        self.global_step = ckpt.get('global_step', 0)
        self.start_epoch = ckpt['epoch'] + 1
        if self.scheduler is not None and 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        if self.algorithm is not None and 'algorithm_state' in ckpt:
            self.algorithm.load_state_dict(ckpt['algorithm_state'])
            if hasattr(self.model, 'mask_scale') and self.algorithm.expected_mask_sum is not None:
                self.model.mask_scale = self.algorithm.expected_mask_sum
            if (hasattr(self.model, 'use_adaptive_denom')
                    and getattr(self.algorithm, 'use_adaptive_denom', False)):
                self.model.use_adaptive_denom = True
        # Restore RNG states for reproducibility
        if 'rng_states' in ckpt:
            rng = ckpt['rng_states']
            torch.set_rng_state(rng['torch'].cpu().to(torch.uint8))
            np.random.set_state(rng['numpy'])
            random.setstate(rng['python'])
            if torch.cuda.is_available() and 'cuda' in rng:
                torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in rng['cuda']])
        print(f"  Resumed from {path} (epoch {ckpt['epoch']}, best_acc={self.best_acc:.2f}%)")

    def train(self):
        """Run full training loop. Returns final results dict."""
        all_epoch_results = []
        train_start = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            self.compute_tracker.start_epoch()

            # Train one epoch
            train_metrics = self._train_epoch(epoch)

            # Evaluate BEFORE on_epoch_end, so eval uses current epoch's
            # mask_scale/warmup (not next epoch's)
            test_metrics = evaluate_accuracy(
                self.model, self.test_loader, self.algorithm, self.device,
                shuffle_category_labels=self.shuffle_category_labels,
                route_label_type=self.route_label_type)

            # Check for freezing (SpecDrop) and epoch-end callbacks
            freeze_info = {}
            frozen_flags = None
            if self.algorithm is not None and _is_multi_branch(self.model):
                branches = getattr(self.model, 'branches', None)
                freeze_info = self.algorithm.on_epoch_end(branches, epoch)
                frozen_flags = freeze_info.get('frozen', None)
                # Update mask_scale for algorithms with dynamic schedules (e.g., SMoE-Dropout)
                if hasattr(self.model, 'mask_scale') and self.algorithm.expected_mask_sum is not None:
                    self.model.mask_scale = self.algorithm.expected_mask_sum

            self.compute_tracker.end_epoch(frozen_flags=frozen_flags)

            # Update LR
            if self.scheduler is not None:
                self.scheduler.step()

            # Track best
            if test_metrics['top1'] > self.best_acc:
                self.best_acc = test_metrics['top1']
                self._save_checkpoint(epoch, is_best=True)

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            epoch_time = self.compute_tracker.epoch_times[-1]
            epoch_result = {
                'epoch': epoch,
                'train_loss': round(train_metrics['loss'], 4),
                'train_acc': round(train_metrics['acc'], 2),
                'test_top1': round(test_metrics['top1'], 2),
                'test_top5': round(test_metrics['top5'], 2),
                'test_loss': round(test_metrics['loss'], 4),
                'lr': round(lr, 6),
                'frozen_modules': freeze_info.get('frozen', []),
                'epoch_time': epoch_time,
            }
            all_epoch_results.append(epoch_result)

            # Log epoch-level metrics to wandb
            num_frozen = len(freeze_info.get('frozen', []))
            self._log_wandb({
                'epoch': epoch,
                'train/loss_epoch': train_metrics['loss'],
                'train/acc_epoch': train_metrics['acc'],
                'test/top1': test_metrics['top1'],
                'test/top5': test_metrics['top5'],
                'test/loss': test_metrics['loss'],
                'best_top1': self.best_acc,
                'lr': lr,
                'epoch_time_s': epoch_time,
                'frozen_modules': num_frozen,
            })

            newly_frozen = freeze_info.get('newly_frozen', [])
            freeze_str = f" | frozen: {frozen_flags}" if frozen_flags else ""
            new_freeze_str = f" | NEW frozen: {newly_frozen}" if newly_frozen else ""
            print(f"Epoch {epoch:3d}/{self.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.2f}% | "
                  f"Test Top1: {test_metrics['top1']:.2f}% Top5: {test_metrics['top5']:.2f}% | "
                  f"LR: {lr:.5f} | "
                  f"Time: {epoch_time:.1f}s"
                  f"{freeze_str}{new_freeze_str}")

            # Save latest checkpoint for resume (overwrite each epoch)
            self._save_checkpoint(epoch, is_best=False)

        total_time = time.time() - train_start

        # Final evaluation
        print("\n=== Final Evaluation ===")
        final_test = evaluate_accuracy(
            self.model, self.test_loader, self.algorithm, self.device,
            shuffle_category_labels=self.shuffle_category_labels,
            route_label_type=self.route_label_type)
        per_cat = evaluate_per_category(
            self.model, self.test_loader, self.algorithm, self.device,
            route_label_type=self.route_label_type)
        compute_summary = self.compute_tracker.get_summary()

        # Log per-category accuracy to wandb
        if self.use_wandb:
            import wandb
            for cat_id, acc in per_cat.items():
                self._log_wandb({f'test/acc_cat{cat_id}': acc})
            wandb.summary['best_top1'] = self.best_acc
            wandb.summary['total_time_h'] = round(total_time / 3600, 2)
            wandb.summary['total_params'] = sum(
                p.numel() for p in self.model.parameters())

        results = {
            'config': self.cfg,
            'best_top1': round(self.best_acc, 2),
            'final_test': final_test,
            'per_category_accuracy': per_cat,
            'compute': compute_summary,
            'epoch_history': all_epoch_results,
        }

        # Save results
        results_path = os.path.join(self.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

        # Training completed successfully: drop latest.pt (it exists only for
        # mid-run resume; best.pt + results.json are the persistent artifacts).
        latest_path = os.path.join(self.output_dir, 'latest.pt')
        if os.path.exists(latest_path):
            try:
                os.remove(latest_path)
                print(f"Removed redundant latest.pt ({latest_path})")
            except OSError as e:
                print(f"[warn] could not remove {latest_path}: {e}")

        return results

    def _train_epoch(self, epoch):
        """Train for one epoch. Returns dict with 'loss' and 'acc'."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        accum = self.grad_accum_steps

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        faithful = self.is_faithful
        for step_idx, batch in enumerate(pbar):
            # Batches are 4-tuples: (images, fine, coarse, example_id).
            # non_blocking=True + pin_memory=True lets host-to-device transfer
            # overlap with the prior kernel (async DMA). Without non_blocking,
            # pin_memory's benefit is lost — the 4 .to() calls serialize.
            images, fine_labels, coarse_labels, example_ids = batch
            images = images.to(self.device, non_blocking=True)
            fine_labels = fine_labels.to(self.device, non_blocking=True)
            coarse_labels = coarse_labels.to(self.device, non_blocking=True)
            example_ids = example_ids.to(self.device, non_blocking=True)

            # Apply Mixup / CutMix if active. ETD needs example_ids keyed to the
            # ORIGINAL (un-permuted) sample — we pass example_ids before mixing so
            # the mask lookup is stable across augmentations.
            if self.mixer is not None:
                mixed_images, soft_targets = self.mixer(images, fine_labels)
            else:
                mixed_images, soft_targets = images, None

            ce_for_log = None
            aux_for_log = 0.0
            if faithful:
                with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    logits, model_aux = _faithful_forward(
                        self.model, mixed_images, example_ids)
                    ce = torchF.cross_entropy(logits, soft_targets) \
                         if soft_targets is not None \
                         else self.criterion(logits, fine_labels)
                    loss = ce + model_aux
                    # Detach before float() — model_aux may still have requires_grad
                    # (e.g. ContextualDropout's KL), which triggers a UserWarning.
                    ce_for_log, aux_for_log = ce, float(model_aux.detach())
                    if accum > 1:
                        loss = loss / accum
            else:
                # Existing path: optional external algorithm provides branch masks.
                mask = None
                use_two_phase = (self.algorithm is not None
                                 and _is_multi_branch(self.model)
                                 and getattr(self.algorithm, 'needs_features', False))

                # T1.2: optionally swap to fine_labels (M=100). Default 'coarse'
                # preserves prior behavior across all 4 main settings.
                route_labels = (fine_labels if self.route_label_type == 'fine'
                                else coarse_labels)
                # Q3 ablation: break the category→branch mapping by randomly
                # permuting the labels we feed to the router. Keeps the routing
                # mechanism intact but removes meaningful category information.
                if self.shuffle_category_labels:
                    route_labels = route_labels[torch.randperm(route_labels.size(0),
                                                                 device=route_labels.device)]

                if use_two_phase:
                    with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                        stem = self.model.get_stem_features(images)
                        pooled = torchF.adaptive_avg_pool2d(stem, 1).flatten(1)
                    mask = self.algorithm.get_mask(route_labels, training=True,
                                                    features=pooled.float())
                elif self.algorithm is not None and _is_multi_branch(self.model):
                    mask = self.algorithm.get_mask(route_labels, training=True)

                with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    if use_two_phase:
                        logits = self.model.forward_from_stem(stem, branch_mask=mask)
                    elif mask is not None:
                        logits = self.model(mixed_images, branch_mask=mask)
                    else:
                        logits = self.model(mixed_images)
                    ce = (torchF.cross_entropy(logits, soft_targets)
                          if soft_targets is not None
                          else self.criterion(logits, fine_labels))
                    algo_aux = 0.0
                    if self.algorithm is not None:
                        algo_aux = self.algorithm.get_auxiliary_loss()
                    loss = ce + (algo_aux if algo_aux != 0.0 else 0)
                    ce_for_log = ce
                    if algo_aux != 0.0:
                        aux_for_log = float(algo_aux.detach()) \
                            if torch.is_tensor(algo_aux) else float(algo_aux)
                    else:
                        aux_for_log = 0.0
                    if accum > 1:
                        loss = loss / accum

            # Aux-loss sanity print once per run on BOTH paths (faithful +
            # algorithm-based). Flags mis-scaled aux that would silently dominate.
            if self.global_step == 0 and epoch == 0 and step_idx == 0 and ce_for_log is not None:
                _ce = ce_for_log.item()
                _ratio = abs(aux_for_log) / max(_ce, 1e-9)
                print(f"  [aux-loss sanity] CE={_ce:.4f} aux={aux_for_log:+.5f} "
                      f"|aux|/CE={_ratio:.3%}")

            # NaN-skip with abort budget: occasional fp16/bf16 spikes can
            # recover, but a permanently broken trajectory (e.g. compile + AMP
            # edge cases) would otherwise burn the full epoch budget. Skip the
            # backward step on isolated NaN; raise RuntimeError once the
            # consecutive or cumulative budget is exceeded so the dispatcher
            # script moves on to the next run.
            if not torch.isfinite(loss).item():
                self._nan_consecutive += 1
                self._nan_total += 1
                print(f"\n  [DEBUG] NaN/Inf loss at step {self.global_step} "
                      f"(epoch {epoch}, step {step_idx}): loss={loss.item()} "
                      f"— skipping backward (consec={self._nan_consecutive}, "
                      f"total={self._nan_total})")
                self.optimizer.zero_grad(set_to_none=True)
                if self._nan_consecutive >= self.nan_abort_consecutive:
                    raise RuntimeError(
                        f"Aborting run: {self._nan_consecutive} consecutive "
                        f"NaN/Inf losses (budget={self.nan_abort_consecutive}). "
                        f"Trajectory is unrecoverable; investigate offline.")
                if self._nan_total >= self.nan_abort_total:
                    raise RuntimeError(
                        f"Aborting run: {self._nan_total} cumulative NaN/Inf "
                        f"losses (budget={self.nan_abort_total}). Trajectory "
                        f"is too unstable; investigate offline.")
                continue
            else:
                self._nan_consecutive = 0  # reset on any healthy step

            # Backward (accumulate gradients)
            self.scaler.scale(loss).backward()

            # Optimizer step every accum mini-batches (or at end of epoch)
            if (step_idx + 1) % accum == 0 or (step_idx + 1) == len(self.train_loader):
                if self.max_grad_norm > 0:
                    # Unscale before clipping when using GradScaler (fp16 path).
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    all_p = [p for g in self.optimizer.param_groups for p in g['params']]
                    torch.nn.utils.clip_grad_norm_(all_p, self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Unscale loss for logging
            actual_loss = loss.item() * accum if accum > 1 else loss.item()
            total_loss += actual_loss * images.size(0)
            _, pred = logits.max(dim=1)
            correct += pred.eq(fine_labels).sum().item()
            total += images.size(0)

            # Per-step wandb logging (every 50 optimizer steps)
            if self.global_step % 50 == 0 and (step_idx + 1) % accum == 0:
                self._log_wandb({
                    'train/loss': actual_loss,
                    'train/acc': correct / total * 100,
                    'global_step': self.global_step,
                }, step=self.global_step)

            pbar.set_postfix(loss=f"{actual_loss:.3f}", acc=f"{correct/total*100:.1f}%")

            if self.max_steps is not None and (step_idx + 1) >= self.max_steps:
                break

        return {
            'loss': total_loss / total,
            'acc': correct / total * 100,
        }

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with full state for resume."""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'global_step': self.global_step,
            'scaler_state_dict': self.scaler.state_dict(),
            'rng_states': {
                'torch': torch.get_rng_state(),
                'numpy': np.random.get_state(),
                'python': random.getstate(),
            },
        }
        if self.scheduler is not None:
            ckpt['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.algorithm is not None:
            ckpt['algorithm_state'] = self.algorithm.state_dict()
        if torch.cuda.is_available():
            ckpt['rng_states']['cuda'] = torch.cuda.get_rng_state_all()
        name = 'best.pt' if is_best else 'latest.pt'
        path = os.path.join(self.output_dir, name)
        torch.save(ckpt, path)

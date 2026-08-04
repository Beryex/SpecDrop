"""Training loop for language modeling experiments.

Handles causal LM training with per-domain routing.
Algorithm-agnostic: algorithms produce masks from domain_ids,
the model uses them in its MoE FFN layers.
"""

import os
import json
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm


def _unwrap(model):
    """Return the underlying module when wrapped by torch.compile."""
    return getattr(model, '_orig_mod', model)


def _is_faithful_lm(model):
    """Faithful LM baselines do their own routing internally and return (logits, aux)."""
    from models.switch_transformer_lm import SwitchTransformerLM
    from models.hash_layers_transformer_lm import HashLayersTransformerLM
    from models.smoe_dropout_transformer_lm import SMoEDropoutTransformerLM
    from models.demix_transformer_lm import DemixTransformerLM
    return isinstance(_unwrap(model), (
        SwitchTransformerLM, HashLayersTransformerLM,
        SMoEDropoutTransformerLM, DemixTransformerLM,
    ))


def _faithful_lm_forward(model, inputs, domain_ids):
    """Call a faithful LM and normalize return to (logits, aux)."""
    from models.demix_transformer_lm import DemixTransformerLM
    if isinstance(_unwrap(model), DemixTransformerLM):
        output = model(inputs, domain_ids=domain_ids)
    else:
        output = model(inputs)
    if isinstance(output, tuple):
        logits, aux = output
    else:
        logits = output
        aux = torch.tensor(0.0, device=inputs.device)
    return logits, aux


class NLPTrainer:
    """Trainer for multi-branch transformer LM experiments.

    Usage:
        trainer = NLPTrainer(cfg, model, algorithm, train_loader, val_loader, device)
        results = trainer.train()
    """

    def __init__(self, cfg, model, algorithm, train_loader, val_loader, device,
                 use_wandb=False):
        self.cfg = cfg
        self.model = model.to(device)
        self.algorithm = algorithm
        # Move algorithm's nn.Module components (e.g., router Linear) to device
        if algorithm is not None:
            for v in vars(algorithm).values():
                if isinstance(v, nn.Module):
                    v.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb

        tcfg = cfg['training']
        self.epochs = tcfg.get('epochs', 10)
        # Label-shuffle control: shuffle the domain_ids fed to the routing algorithm.
        # Tests whether NLP gain comes from the domain-label signal vs the
        # routing mechanism.
        self.shuffle_category_labels = tcfg.get('shuffle_category_labels', False)
        if self.shuffle_category_labels:
            print("  [Q3 ablation] Domain IDs will be RANDOMLY SHUFFLED before routing.")
        self.lr = tcfg.get('lr', 3e-4)
        self.max_grad_norm = tcfg.get('max_grad_norm', 1.0)
        # NaN-loss budget (mirrors CV trainer): abort run after too many
        # NaN-skip events so the dispatcher moves on to the next config
        # instead of burning the full epoch budget on a broken trajectory.
        self.nan_abort_consecutive = tcfg.get('nan_abort_consecutive', 50)
        self.nan_abort_total = tcfg.get('nan_abort_total', 200)
        self._nan_consecutive = 0
        self._nan_total = 0

        # Collect all trainable parameters (model + algorithm)
        params = list(model.parameters())
        if algorithm is not None:
            algo_params = []
            for v in vars(algorithm).values():
                if isinstance(v, nn.Parameter) and v.requires_grad:
                    algo_params.append(v)
                elif isinstance(v, nn.Module):
                    algo_params.extend(p for p in v.parameters() if p.requires_grad)
            if algo_params:
                params.extend(algo_params)
                total_algo = sum(p.numel() for p in algo_params)
                print(f"  Algorithm has {len(algo_params)} trainable parameter(s) ({total_algo:,} params)")

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=(tcfg.get('beta1', 0.9), tcfg.get('beta2', 0.95)),
            weight_decay=tcfg.get('weight_decay', 0.01),
        )

        # LR schedule: linear warmup + cosine decay
        total_steps = len(train_loader) * self.epochs
        warmup_steps = tcfg.get('warmup_steps', min(1000, total_steps // 10))
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.global_step = 0

        # When the algorithm supports per-step pa warmup (warmup_unit='step'
        # on SoftSpecDrop), hand over the total-step budget so it can scale
        # its cosine/linear ramp denominator correctly. No-op for 'epoch' unit.
        if self.algorithm is not None and hasattr(self.algorithm, 'set_total_steps'):
            self.algorithm.set_total_steps(total_steps)

        self.output_dir = cfg.get('output_dir', './outputs/nlp_default')
        os.makedirs(self.output_dir, exist_ok=True)

        # AMP mixed precision (bf16). torch.compile is opt-in via config: dense
        # Transformer can use compile, but multi-branch fixed-mask paths were
        # historically unstable under compile + bf16 (NaN logits from step 1).
        # After the merge-dtype fix this may be retestable; we keep
        # the default as DISABLED so existing runs are bit-identical.
        self.use_amp = (device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'))
        self.amp_dtype = torch.bfloat16
        if self.use_amp:
            torch.set_float32_matmul_precision('high')
        # GradScaler is only needed for fp16 (bf16 doesn't need loss scaling)
        self.scaler = GradScaler('cuda', enabled=False)
        if self.use_amp:
            print(f"  AMP mixed precision: enabled (bfloat16)")

        # torch.compile dispatch (opt-in). Honors the same flags as the CV
        # trainer: `_no_compile: true` forces eager; `_compile_mode: <mode>`
        # selects 'default' / 'reduce-overhead' / 'max-autotune'. If neither is
        # set we default to disabled (preserves prior numerical behavior).
        no_compile = cfg.get('_no_compile', False) or tcfg.get('_no_compile', False)
        compile_mode = tcfg.get('_compile_mode', None)
        if no_compile or compile_mode is None:
            print("  torch.compile: disabled (default for NLP; set _compile_mode to enable)")
        elif self.use_amp:
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
                print(f"  torch.compile: enabled ({compile_mode})")
            except Exception as e:
                print(f"  torch.compile: skipped (unsupported: {e})")

        self.best_val_ppl = float('inf')
        self.start_epoch = 0

    def _get_lr(self):
        """Linear warmup + cosine decay."""
        if self.global_step < self.warmup_steps:
            return self.lr * self.global_step / max(self.warmup_steps, 1)
        progress = (self.global_step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _update_lr(self):
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

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
        self.best_val_ppl = ckpt.get('best_val_ppl', float('inf'))
        self.global_step = ckpt.get('global_step', 0)
        self.start_epoch = ckpt['epoch'] + 1
        if 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        if self.algorithm is not None and 'algorithm_state' in ckpt:
            self.algorithm.load_state_dict(ckpt['algorithm_state'])
            if hasattr(self.model, 'mask_scale') and self.algorithm.expected_mask_sum is not None:
                self.model.mask_scale = self.algorithm.expected_mask_sum
        if 'rng_states' in ckpt:
            rng = ckpt['rng_states']
            torch.set_rng_state(rng['torch'])
            np.random.set_state(rng['numpy'])
            random.setstate(rng['python'])
            if torch.cuda.is_available() and 'cuda' in rng:
                torch.cuda.set_rng_state_all(rng['cuda'])
        print(f"  Resumed from {path} (epoch {ckpt['epoch']}, best_ppl={self.best_val_ppl:.2f})")

    @torch.no_grad()
    def _init_demix_cached_prior(self):
        """Estimate P(d) from training-set domain frequencies and store on the model.

        Faithful to Gururangan 2022 Section 3.3 ("cached prior"). Scans the first
        epoch's worth of batches once; each batch yields `domain_ids` (B,) over
        the 7 SlimPajama domains.
        """
        counts = torch.zeros(self.model.num_domains, device=self.device)
        for _, domain_ids in self.train_loader:
            did = domain_ids.to(self.device).long()
            counts.scatter_add_(0, did, torch.ones_like(did, dtype=counts.dtype))
        if counts.sum() == 0:
            print("  [DEMix] WARNING: no domain_ids seen, keeping uniform cached_prior")
            return
        prior = counts / counts.sum()
        self.model.set_cached_prior(prior)
        print(f"  [DEMix] Cached P(d) from train: {[round(p.item(), 3) for p in prior]}")

    def train(self):
        """Run full training loop. Returns results dict."""
        all_epoch_results = []
        train_start = time.time()

        # DEMix: pre-compute cached domain prior from training-set domain frequencies
        # (Gururangan 2022 Section 3.3). A single scan of train_loader batches suffices.
        from models.demix_transformer_lm import DemixTransformerLM
        if isinstance(self.model, DemixTransformerLM):
            self._init_demix_cached_prior()

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            train_metrics = self._train_epoch(epoch)

            # Eval BEFORE on_epoch_end so eval uses the same schedule state the
            # epoch was trained with (matches CV trainer convention). Algorithms'
            # on_epoch_end(E) advances state to epoch E+1; calling it before eval
            # would evaluate with next-epoch mask and mis-match training.
            val_metrics = self._eval(self.val_loader)

            if self.algorithm is not None and hasattr(self.model, 'num_branches'):
                self.algorithm.on_epoch_end(None, epoch)
                if hasattr(self.model, 'mask_scale') and self.algorithm.expected_mask_sum is not None:
                    self.model.mask_scale = self.algorithm.expected_mask_sum
            epoch_time = time.time() - epoch_start

            if val_metrics['ppl'] < self.best_val_ppl:
                self.best_val_ppl = val_metrics['ppl']
                self._save_checkpoint(epoch, is_best=True)

            epoch_result = {
                'epoch': epoch,
                'train_loss': round(train_metrics['loss'], 4),
                'train_ppl': round(train_metrics['ppl'], 2),
                'val_loss': round(val_metrics['loss'], 4),
                'val_ppl': round(val_metrics['ppl'], 2),
                'lr': round(train_metrics['lr'], 7),
                'epoch_time_s': round(epoch_time, 1),
            }
            all_epoch_results.append(epoch_result)

            # Log epoch-level metrics to wandb
            self._log_wandb({
                'epoch': epoch,
                'train/loss_epoch': train_metrics['loss'],
                'train/ppl_epoch': train_metrics['ppl'],
                'val/loss': val_metrics['loss'],
                'val/ppl': val_metrics['ppl'],
                'best_val_ppl': self.best_val_ppl,
                'epoch_time_s': epoch_time,
            })

            print(f"Epoch {epoch}/{self.epochs} | "
                  f"Train PPL: {train_metrics['ppl']:.2f} | "
                  f"Val PPL: {val_metrics['ppl']:.2f} | "
                  f"LR: {train_metrics['lr']:.6f} | "
                  f"Time: {epoch_time:.0f}s")

            # Save latest checkpoint for resume (overwrite each epoch)
            self._save_checkpoint(epoch, is_best=False)

        total_time = time.time() - train_start

        # Final per-domain evaluation
        print("\n=== Final Per-Domain Evaluation ===")
        val_metrics = self._eval(self.val_loader, per_domain=True)
        print(f"Overall Val PPL: {val_metrics['ppl']:.2f}")
        if 'domain_ppl' in val_metrics:
            domain_wandb = {}
            for dname, dppl in val_metrics['domain_ppl'].items():
                print(f"  {dname}: {dppl:.2f}")
                # Clean domain name for wandb key
                short = dname.replace('RedPajama', '')
                domain_wandb[f'val/ppl_{short}'] = dppl
            self._log_wandb(domain_wandb)

        # Log summary metrics to wandb
        if self.use_wandb:
            import wandb
            wandb.summary['best_val_ppl'] = self.best_val_ppl
            wandb.summary['total_time_h'] = round(total_time / 3600, 2)
            wandb.summary['total_params'] = sum(
                p.numel() for p in self.model.parameters())
            if 'domain_ppl' in val_metrics:
                for dname, dppl in val_metrics['domain_ppl'].items():
                    short = dname.replace('RedPajama', '')
                    wandb.summary[f'domain_ppl_{short}'] = dppl

        results = {
            'config': self.cfg,
            'best_val_ppl': round(self.best_val_ppl, 2),
            'final_val': val_metrics,
            'epoch_history': all_epoch_results,
            'total_time_h': round(total_time / 3600, 2),
        }

        results_path = os.path.join(self.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

        # Training completed successfully: drop latest.pt (it exists only for
        # mid-run resume; best.pt + results.json are the persistent artifacts).
        # ~120 MB per 30M-param NLP run × hundreds of runs adds up fast on a
        # single-volume training server.
        latest_path = os.path.join(self.output_dir, 'latest.pt')
        if os.path.exists(latest_path):
            try:
                os.remove(latest_path)
                print(f"Removed redundant latest.pt ({latest_path})")
            except OSError as e:
                print(f"[warn] could not remove {latest_path}: {e}")

        return results

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        last_lr = self.lr

        # SMoE-Dropout k-schedule: advance once per epoch.
        from models.smoe_dropout_transformer_lm import SMoEDropoutTransformerLM
        if isinstance(self.model, SMoEDropoutTransformerLM):
            self.model.update_k(epoch=epoch, total_epochs=self.epochs)

        faithful = _is_faithful_lm(self.model)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for input_ids, domain_ids in pbar:
            input_ids = input_ids.to(self.device)
            domain_ids = domain_ids.to(self.device)

            last_lr = self._update_lr()
            self.global_step += 1

            # Expose step counter to the algorithm before get_mask. Consumed by
            # SoftSpecDrop(warmup_unit='step'); ignored by 'epoch' (legacy) and
            # by algorithms without a current_step attribute (e.g. no_dropout).
            if self.algorithm is not None and hasattr(self.algorithm, 'current_step'):
                self.algorithm.current_step = self.global_step

            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()

            if faithful:
                with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    logits, model_aux = _faithful_lm_forward(
                        self.model, inputs, domain_ids)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-1,
                    ) + model_aux
                mask = None
            else:
                mask = None
                use_two_phase = (self.algorithm is not None
                                 and hasattr(self.model, 'num_branches')
                                 and getattr(self.algorithm, 'needs_features', False))

                if use_two_phase:
                    with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                        stem = self.model.get_stem_features(inputs)
                        pooled = stem.mean(dim=1)
                    mask = self.algorithm.get_mask(domain_ids, training=True,
                                                    features=pooled.float())
                elif self.algorithm is not None and hasattr(self.model, 'num_branches'):
                    route_ids = (domain_ids[torch.randperm(domain_ids.size(0), device=domain_ids.device)]
                                 if self.shuffle_category_labels else domain_ids)
                    mask = self.algorithm.get_mask(route_ids, training=True)

                with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    if use_two_phase:
                        logits = self.model.forward_from_stem(stem, branch_mask=mask)
                    elif mask is not None:
                        logits = self.model(inputs, branch_mask=mask)
                    else:
                        logits = self.model(inputs)

                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-1,
                    )
                    if self.algorithm is not None:
                        aux_loss = self.algorithm.get_auxiliary_loss()
                        if aux_loss != 0.0:
                            loss = loss + aux_loss

            # Backward — bf16 has fp32 range so GradScaler is disabled
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm > 0:
                all_params = [p for g in self.optimizer.param_groups for p in g['params']]
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    all_params, self.max_grad_norm).item()
            else:
                grad_norm = 0.0

            # NaN-skip with abort budget: occasional bf16 spikes recover, but
            # a permanently broken trajectory (compile + AMP edge case, etc.)
            # would otherwise burn the full epoch. After consecutive or total
            # budget is exceeded, raise so the dispatcher moves on.
            if not torch.isfinite(loss).item():
                self._nan_consecutive += 1
                self._nan_total += 1
                print(f"\n  [DEBUG] NaN/Inf loss at step {self.global_step}: "
                      f"loss={loss.item()} — skipping optimizer.step() "
                      f"(consec={self._nan_consecutive}, total={self._nan_total})")
                with torch.no_grad():
                    if mask is not None:
                        print(f"    mask stats: min={mask.min().item():.4f} max={mask.max().item():.4f} "
                              f"mean={mask.float().mean().item():.4f} has_nan={torch.isnan(mask).any().item()}")
                    print(f"    logits stats: min={logits.float().min().item():.2f} "
                          f"max={logits.float().max().item():.2f} has_nan={torch.isnan(logits).any().item()}")
                    print(f"    grad_norm before clip: {grad_norm:.2f}")
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

            self.optimizer.step()

            num_tokens = (targets != -1).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Per-step wandb logging (every 50 steps to avoid overhead)
            if self.global_step % 50 == 0:
                self._log_wandb({
                    'train/loss': loss.item(),
                    'train/ppl': math.exp(min(loss.item(), 20)),
                    'lr': last_lr,
                    'grad_norm': grad_norm,
                    'global_step': self.global_step,
                }, step=self.global_step)

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                ppl=f"{math.exp(min(loss.item(), 20)):.1f}",
                lr=f"{last_lr:.6f}")

        avg_loss = total_loss / max(total_tokens, 1)
        return {
            'loss': avg_loss,
            'ppl': math.exp(min(avg_loss, 20)),
            'lr': last_lr,
        }

    @torch.no_grad()
    def _eval(self, dataloader, per_domain=False):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        # Per-domain tracking
        domain_loss = {}
        domain_tokens = {}

        for input_ids, domain_ids in tqdm(dataloader, desc="Eval", leave=False):
            input_ids = input_ids.to(self.device)
            domain_ids = domain_ids.to(self.device)

            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()

            if _is_faithful_lm(self.model):
                from models.demix_transformer_lm import DemixTransformerLM
                # Faithful DEMix inference protocol (Gururangan 2022 Section 3.3):
                # mixture-of-experts with Bayes posterior P(d|x) ∝ P(x|d)·P(d).
                # 'oracle' (using domain label) is an upper-bound reference only.
                demix_eval = self.cfg.get('training', {}).get('demix_eval_mode',
                                                               'mixture')
                if isinstance(self.model, DemixTransformerLM) and demix_eval == 'mixture':
                    # Mixture-of-Experts inference (DEMix Section 3.3 Eq. 3):
                    # runs the LM N times and weighs by Bayesian domain posterior.
                    logits, _posts = self.model.forward_mixture(inputs)
                else:
                    logits, _ = _faithful_lm_forward(self.model, inputs, domain_ids)
            else:
                use_two_phase = (self.algorithm is not None
                                 and hasattr(self.model, 'num_branches')
                                 and getattr(self.algorithm, 'needs_features', False))

                if use_two_phase:
                    stem = self.model.get_stem_features(inputs)
                    pooled = stem.mean(dim=1)
                    mask = self.algorithm.get_mask(domain_ids, training=False,
                                                    features=pooled.float())
                    logits = self.model.forward_from_stem(stem, branch_mask=mask)
                elif self.algorithm is not None and hasattr(self.model, 'num_branches'):
                    route_ids = (domain_ids[torch.randperm(domain_ids.size(0), device=domain_ids.device)]
                                 if self.shuffle_category_labels else domain_ids)
                    mask = self.algorithm.get_mask(route_ids, training=False)
                    logits = self.model(inputs, branch_mask=mask)
                else:
                    logits = self.model(inputs)

            # Per-sample loss for domain tracking
            B, T, V = logits.shape
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, V), targets.reshape(-1),
                ignore_index=-1, reduction='none'
            ).reshape(B, T)

            total_loss += loss_per_token.sum().item()
            total_tokens += (targets != -1).sum().item()

            if per_domain:
                for b in range(B):
                    d = domain_ids[b].item()
                    valid = (targets[b] != -1)
                    if valid.any():
                        dl = loss_per_token[b][valid].sum().item()
                        dt = valid.sum().item()
                        domain_loss[d] = domain_loss.get(d, 0.0) + dl
                        domain_tokens[d] = domain_tokens.get(d, 0) + dt

        avg_loss = total_loss / max(total_tokens, 1)
        result = {
            'loss': avg_loss,
            'ppl': math.exp(min(avg_loss, 20)),
        }

        if per_domain:
            try:
                from data.slimpajama import DOMAIN_NAMES
                dnames = DOMAIN_NAMES
            except ImportError:
                dnames = [str(i) for i in range(max(domain_loss.keys()) + 1)]

            result['domain_ppl'] = {}
            for d in sorted(domain_loss.keys()):
                dl = domain_loss[d] / max(domain_tokens[d], 1)
                name = dnames[d] if d < len(dnames) else str(d)
                result['domain_ppl'][name] = round(math.exp(min(dl, 20)), 2)

        return result

    def _save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_ppl': self.best_val_ppl,
            'global_step': self.global_step,
            'scaler_state_dict': self.scaler.state_dict(),
            'rng_states': {
                'torch': torch.get_rng_state(),
                'numpy': np.random.get_state(),
                'python': random.getstate(),
            },
        }
        if self.algorithm is not None:
            ckpt['algorithm_state'] = self.algorithm.state_dict()
        if torch.cuda.is_available():
            ckpt['rng_states']['cuda'] = torch.cuda.get_rng_state_all()
        name = 'best.pt' if is_best else 'latest.pt'
        path = os.path.join(self.output_dir, name)
        torch.save(ckpt, path)

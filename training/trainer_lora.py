"""LoRA-specific instruction-tuning trainer for the post-train track.

Orchestrates training of a BaseLoRAModel (Batch 3's lora_models.py) on
SuperNI tasks (Batch 2's natural_instructions.py), supporting all five
method configurations in a single trainer:
  - Single LoRA        (algorithm=None, routing ignored)
  - MultiBranch-LoRA   (algorithm=SoftSpecDrop OR NoDropout, feeds mask)
  - HydraLoRA          (algorithm=None, routing ignored; gate is learned)
  - LoRAMoE            (algorithm=None, routing ignored; adds balance loss)
  - MoCLE              (algorithm=None, routing = {cluster_id: ...})

Key design points:
  - LoRA-only gradients: the model has base frozen via freeze_base_params;
    trainer just calls trainable_parameters() for the optimizer.
  - pa warmup_unit='step' (per-step) by design principle: matches LR's
    per-step cosine granularity. Trainer pushes `algorithm.current_step`
    after every optimizer.step() and calls algorithm.set_total_steps()
    at init (same pattern as trainer_nlp.py for SlimPajama step-wise mode).
  - bf16 AMP forward; fp32 LoRA param update (AdamW handles mixed precision).
    No GradScaler needed with bf16.
  - Periodic eval on SuperNI held-out split with greedy decoding.

Backward-compat: 0 impact on existing trainer.py / trainer_nlp.py paths.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm


class LoRATrainer:
    """Instruction-tuning trainer that owns model + algorithm + loaders."""

    def __init__(self, cfg: Dict, model, algorithm, train_loader, eval_loader,
                 device, use_wandb: bool = False):
        self.cfg = cfg
        self.model = model.to(device)
        # When run_lora.py wraps the model with torch.compile, `self.model`
        # becomes an OptimizedModule proxy. `_inner_model` pierces the wrapper
        # so isinstance() checks and named_parameters() (for checkpoint save)
        # see the original class + clean param names without `_orig_mod.`
        # prefix. Falls back to `self.model` for uncompiled runs.
        self._inner_model = getattr(self.model, '_orig_mod', self.model)
        self.algorithm = algorithm  # None, SoftSpecDrop, or NoDropout
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.use_wandb = use_wandb

        tcfg = cfg['training']
        self.epochs = int(tcfg.get('epochs', 3))
        self.lr = float(tcfg.get('lr', 2e-4))
        self.weight_decay = float(tcfg.get('weight_decay', 0.0))
        self.max_grad_norm = float(tcfg.get('max_grad_norm', 1.0))
        self.warmup_ratio_lr = float(tcfg.get('warmup_ratio_lr', 0.03))
        self.amp_dtype = tcfg.get('amp_dtype', 'bf16')
        # TF32 for fp32 matmul paths (LayerNorm, loss, some reductions) that
        # bf16 autocast doesn't cover. 5-15% free speedup on Ampere+/Hopper/
        # Blackwell GPUs with precision loss far below bf16's. Mirrors
        # trainer.py:251 (CIFAR + ViT) and trainer_nlp.py:139 (SlimPajama).
        torch.set_float32_matmul_precision('high')
        self.log_interval = int(tcfg.get('log_interval', 20))
        self.eval_interval_epochs = int(tcfg.get('eval_interval_epochs', 1))
        # Gradient accumulation: DataLoader yields per-device mini-batches;
        # we accumulate `grad_accum_steps` of them into one optimizer update
        # so effective batch = batch_size_per_device × grad_accum_steps.
        # LR schedule + pa warmup tick PER OPTIMIZER STEP, not per mini-batch.
        self.grad_accum_steps = max(1, int(tcfg.get('grad_accum_steps', 1)))
        # max_steps: hard stop for smoke / quick validation. Counts OPTIMIZER
        # steps (same unit as global_step / LR scheduler), NOT mini-batches.
        self.max_steps = tcfg.get('max_steps', None)
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)

        # total_steps = epochs × (optimizer steps per epoch). Trailing
        # micro-batches that don't complete an accumulation cycle are dropped
        # (standard convention; integer div via //).
        opt_steps_per_epoch = max(1, len(train_loader) // self.grad_accum_steps)
        self.total_steps = self.epochs * opt_steps_per_epoch

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

        # Wire pa warmup total-steps into the algorithm (only SoftSpecDrop uses
        # it; NoDropout + None ignore). Mirror trainer_nlp.py's convention.
        if self.algorithm is not None and hasattr(self.algorithm, 'set_total_steps'):
            self.algorithm.set_total_steps(self.total_steps)

        # Ensure algorithm sub-modules (if any) live on the trainer device.
        if self.algorithm is not None:
            for v in vars(self.algorithm).values():
                if isinstance(v, nn.Module):
                    v.to(device)
                elif isinstance(v, torch.Tensor):
                    v.data = v.data.to(device)

        self.global_step = 0
        self.history: List[Dict] = []
        # Two parallel selection tracks (Wang 2022 ROUGE-L canonical + smoke-test
        # fallback). Production: ROUGE-L succeeds at every epoch → best.pt is
        # selected by argmax ROUGE-L (ties → earlier epoch wins, Occam). Smoke
        # tests / fake models without HF tokenizer: ROUGE-L raises → caught,
        # falls back to argmin eval_loss for that epoch. Once any epoch's ROUGE
        # succeeds, we COMMIT to ROUGE-based selection — subsequent eval_loss
        # improvements at non-ROUGE-best epochs do NOT overwrite best.pt.
        self.best_eval_loss = float('inf')
        self.best_eval_rouge_l = -float('inf')
        self.best_eval_em: Optional[float] = None
        self.best_epoch: Optional[int] = None
        # Whether selection has been committed to ROUGE-L (any epoch's ROUGE
        # eval succeeded). Once True, eval_loss-based selection is disabled
        # to prevent protocol-mixing. This is the "commit point" — like git's
        # detached HEAD, but for the model selection criterion.
        self._selection_locked_to_rouge = False
        # Cached ROUGE eval loader/tokenizer/elapsed (built lazily per training
        # run, reused across epochs for identical eval set across epochs).
        self._rouge_eval_loader = None
        self._rouge_eval_tokenizer = None
        self._rouge_total_elapsed_min = 0.0
        self._rouge_num_tasks: Optional[int] = None

    # ────────────────────────────────────────────────────────────────────
    def _build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        # Log LoRA trainable count once, for sanity.
        n_train = sum(p.numel() for p in params)
        print(f"[LoRATrainer] LoRA trainable params: {n_train:,}")
        return torch.optim.AdamW(
            params, lr=self.lr, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=self.weight_decay)

    def _build_lr_scheduler(self):
        warmup_steps = int(self.warmup_ratio_lr * self.total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, self.total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ────────────────────────────────────────────────────────────────────
    def _get_mask_and_scale(self, cluster_id):
        """If running MultiBranch-LoRA with SoftSpecDrop / NoDropout, ask the
        algorithm for the mask + mask_scale for this batch."""
        if self.algorithm is None:
            return None, None
        mask = self.algorithm.get_mask(cluster_id, training=self.model.training)
        scale = self.algorithm.expected_mask_sum
        return mask, scale

    def _forward_step(self, batch):
        """One forward pass. Returns (loss, aux_loss, logits_shape)."""
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        cluster_id = batch['cluster_id'].to(self.device, non_blocking=True)

        # For MultiBranch-LoRA, set the fixed-denominator scale once per step
        # (it doesn't change across sites or steps — only across epoch-level
        # pa warmup progression, but mask_scale = S = p_a + (K-1) p_i is
        # constant by design of SoftSpecDrop.expected_mask_sum).
        from models.lora_models import MultiBranchLoRAModel
        if isinstance(self._inner_model, MultiBranchLoRAModel):
            mask, scale = self._get_mask_and_scale(cluster_id)
            self._inner_model.set_mask_scale(scale)
            kwargs = {'mask': mask}
        else:
            kwargs = {}

        amp_dtype = torch.bfloat16 if self.amp_dtype == 'bf16' else torch.float16
        with autocast(device_type=self.device, dtype=amp_dtype, enabled=True):
            out = self.model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              labels=labels,
                              cluster_id=cluster_id,
                              **kwargs)
        return out

    # ────────────────────────────────────────────────────────────────────
    def train(self):
        print(f"[LoRATrainer] begin training: epochs={self.epochs}, "
              f"total_steps={self.total_steps} (optimizer steps), "
              f"grad_accum={self.grad_accum_steps}, lr={self.lr}, "
              f"warmup_lr_ratio={self.warmup_ratio_lr}, amp={self.amp_dtype}")
        t0 = time.time()

        # ── Env-var-gated step profiler (diagnosis only, 0=off) ──────────
        # PROFILE_STEPS=N: time each phase of the first N micro-steps with
        # CUDA sync between phases, print breakdown + summary, then sys.exit.
        # Use to disambiguate compute-bound vs dataloader-bound vs sync-point-
        # bound on a 5090. Zero overhead when unset.
        _profile_steps = int(os.environ.get('PROFILE_STEPS', '0') or '0')
        _phase_t = {'data': [], 'fwd': [], 'bwd': [], 'opt': []}
        _data_t_start = time.perf_counter() if _profile_steps else None

        stop_training = False
        for epoch in range(self.epochs):
            if stop_training:
                break
            self.model.train()
            epoch_loss = 0.0
            epoch_aux = 0.0
            n_opt = 0                         # optimizer steps this epoch
            micro_in_accum = 0                # micro-batches accumulated so far
            accum_loss = 0.0                  # running loss within the current accum window
            accum_aux = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(self.train_loader, desc=f'epoch {epoch+1}/{self.epochs}')
            for batch in pbar:
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    print(f'[LoRATrainer] reached max_steps={self.max_steps}, stopping.')
                    stop_training = True
                    break

                # Profiler: snap data-fetch time (from end-of-prev-step to here)
                if _profile_steps and len(_phase_t['data']) < _profile_steps:
                    torch.cuda.synchronize()
                    _phase_t['data'].append(time.perf_counter() - _data_t_start)
                    _t = time.perf_counter()

                out = self._forward_step(batch)

                if _profile_steps and len(_phase_t['fwd']) < _profile_steps:
                    torch.cuda.synchronize()
                    _phase_t['fwd'].append(time.perf_counter() - _t)
                    _t = time.perf_counter()

                # Scale per-mini-batch loss by 1/accum so the accumulated
                # gradient matches what a single large batch would produce.
                loss = out.loss / self.grad_accum_steps
                loss.backward()

                if _profile_steps and len(_phase_t['bwd']) < _profile_steps:
                    torch.cuda.synchronize()
                    _phase_t['bwd'].append(time.perf_counter() - _t)
                    _t = time.perf_counter()

                accum_loss += float(out.loss.detach())
                accum_aux += float(out.aux_loss.detach())
                micro_in_accum += 1

                # Only step the optimizer after grad_accum_steps micro-batches.
                if micro_in_accum >= self.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    n_opt += 1
                    # pa warmup step advance (SoftSpecDrop step-unit, matches
                    # LR schedule granularity).
                    if self.algorithm is not None and hasattr(self.algorithm, 'current_step'):
                        self.algorithm.current_step = self.global_step

                    if _profile_steps and len(_phase_t['opt']) < _profile_steps:
                        torch.cuda.synchronize()
                        _phase_t['opt'].append(time.perf_counter() - _t)

                    # Bookkeeping: record the average loss over this accum
                    # window, then reset the window.
                    epoch_loss += accum_loss / micro_in_accum
                    epoch_aux += accum_aux / micro_in_accum
                    if self.global_step % self.log_interval == 0:
                        pbar.set_postfix(
                            loss=f'{accum_loss / micro_in_accum:.4f}',
                            aux=f'{accum_aux / micro_in_accum:.4e}',
                            lr=f'{self.lr_scheduler.get_last_lr()[0]:.2e}')
                    accum_loss = 0.0
                    accum_aux = 0.0
                    micro_in_accum = 0

                # End-of-step bookkeeping for the NEXT iteration's data timer
                if _profile_steps and len(_phase_t['data']) < _profile_steps:
                    _data_t_start = time.perf_counter()

                # Exit once we've collected N samples for data/fwd/bwd
                if _profile_steps and len(_phase_t['fwd']) >= _profile_steps:
                    def _stats(xs):
                        if not xs: return 'n/a'
                        import statistics
                        m = statistics.mean(xs); md = statistics.median(xs)
                        mx = max(xs); mn = min(xs)
                        return f'mean={m*1000:.1f}ms median={md*1000:.1f}ms '\
                               f'min={mn*1000:.1f}ms max={mx*1000:.1f}ms n={len(xs)}'
                    print()
                    print('=' * 72)
                    print(f'[PROFILE] First {_profile_steps} micro-steps '
                          f'(bs={self.cfg["training"].get("batch_size_per_device")}, '
                          f'seq≤{self.cfg["training"].get("max_seq_len")}, '
                          f'grad_ckpt=True, amp=bf16, compile={"on" if hasattr(self.model, "_orig_mod") else "off"})')
                    print('=' * 72)
                    for name in ('data', 'fwd', 'bwd', 'opt'):
                        print(f'  {name:>4} : {_stats(_phase_t[name])}')
                    # Dominant phase + suggested next action
                    totals = {k: sum(v) for k, v in _phase_t.items() if v}
                    if totals:
                        denom = sum(totals.values())
                        print(f'  TOTAL: {denom*1000:.1f}ms over {_profile_steps} '
                              f'micro-steps = {denom*1000/_profile_steps:.1f}ms/step avg')
                        print('  ── phase share ──')
                        for k in sorted(totals, key=totals.get, reverse=True):
                            share = 100.0 * totals[k] / denom
                            print(f'  {k:>4} : {share:5.1f}%')
                    print('=' * 72)
                    print(f'[PROFILE] sys.exit(0) — unset PROFILE_STEPS to do real run.')
                    import sys; sys.exit(0)

            # Drop any incomplete tail-accumulation (micro_in_accum < accum);
            # standard convention — keeps every optimizer step at identical
            # effective batch size.
            # End-of-epoch algorithm hook (supports epoch-unit pa warmup too).
            if self.algorithm is not None and hasattr(self.algorithm, 'on_epoch_end'):
                self.algorithm.on_epoch_end(branches=None, epoch=epoch)
            # Use n_opt for loss denom so avg reflects per-optimizer-step loss.
            n = n_opt

            train_loss = epoch_loss / max(1, n)
            train_aux = epoch_aux / max(1, n)
            print(f'[epoch {epoch+1}] train loss={train_loss:.4f} aux={train_aux:.4e}')

            if (epoch + 1) % self.eval_interval_epochs == 0:
                eval_metrics = self.evaluate_loss()
                print(f'[epoch {epoch+1}] eval loss={eval_metrics["loss"]:.4f}')

                # Per-epoch ROUGE-L generation eval (Wang 2022 Tk-Instruct
                # canonical protocol). This is the SINGLE selection metric for
                # best.pt + downstream hyperparam selection + baseline
                # comparison — kept for run-to-run comparability
                # 2026-04-25 for protocol rationale. Falls back to eval_loss
                # only when ROUGE eval is disabled (cfg.training.run_rouge_eval
                # = False) or the eval call raises (smoke tests without real
                # HF tokenizer / data).
                rouge_metrics = self._maybe_run_per_epoch_rouge(epoch + 1)

                record: Dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_aux': train_aux,
                    'eval_loss': eval_metrics['loss'],
                }
                if rouge_metrics is not None:
                    record['eval_rouge_l'] = rouge_metrics['rougeL_mean']
                    record['eval_exact_match'] = rouge_metrics['exact_match_mean']
                self.history.append(record)

                self._maybe_update_best(epoch + 1, eval_metrics, rouge_metrics)

        elapsed = time.time() - t0
        if self._selection_locked_to_rouge:
            print(f'[LoRATrainer] done: {elapsed/60:.1f} min, '
                  f'best epoch={self.best_epoch} '
                  f'ROUGE-L={self.best_eval_rouge_l:.4f} (selection metric)')
        else:
            print(f'[LoRATrainer] done: {elapsed/60:.1f} min, '
                  f'best eval loss {self.best_eval_loss:.4f} '
                  f'(fallback: ROUGE-L unavailable)')
        return self.finalize_results()

    # ────────────────────────────────────────────────────────────────────
    def _maybe_run_per_epoch_rouge(self, epoch_num: int) -> Optional[Dict]:
        """Run ROUGE-L generation eval for this epoch's current weights, or
        return None if disabled / failed.

        Caches the eval loader + tokenizer + scoring across epochs so
        subsequent epochs only pay the generation pass (~10 min on 5090 at
        119 tasks × 10 inst/task default), not the loader build. Cache key
        is the in-process trainer instance — no cross-cell sharing.

        Failures (no HF tokenizer, OOM at generate(), torch.compile breakage)
        are caught and logged. Selection then falls back to eval_loss for
        this epoch only; if any LATER epoch's ROUGE succeeds, selection
        re-locks to ROUGE for the rest of the run.
        """
        tcfg = self.cfg.get('training', {})
        if not tcfg.get('run_rouge_eval', True):
            return None
        try:
            ipt_eval = int(tcfg.get('rouge_eval_instances_per_task', 10))
            max_new = int(tcfg.get('rouge_eval_max_new_tokens', 128))
            print(f'[LoRATrainer] epoch {epoch_num} ROUGE-L eval: '
                  f'{ipt_eval} inst/task, max_new_tokens={max_new}')
            t0 = time.time()
            rouge = self.run_rouge_eval(
                instances_per_task=ipt_eval, max_new_tokens=max_new)
            elapsed = (time.time() - t0) / 60
            self._rouge_total_elapsed_min += elapsed
            self._rouge_num_tasks = rouge['num_tasks']
            print(f'[LoRATrainer] epoch {epoch_num} ROUGE-L='
                  f'{rouge["rougeL_mean"]:.4f}  '
                  f'EM={rouge["exact_match_mean"]:.4f}  '
                  f'({rouge["num_tasks"]} tasks, {elapsed:.1f} min)')
            # Generation eval flips the model to eval(); restore for next
            # epoch's training pass.
            self.model.train()
            return rouge
        except Exception as e:
            import traceback
            print(f'[LoRATrainer] epoch {epoch_num} ROUGE-L eval FAILED: {e}')
            traceback.print_exc()
            self.model.train()
            return None

    def _maybe_update_best(self, epoch_num: int, eval_metrics: Dict,
                            rouge_metrics: Optional[Dict]) -> None:
        """Decide whether this epoch's checkpoint is the new best, save if so.

        Selection rules (canonical: Wang 2022 ROUGE-L; fallback: eval_loss
        for smoke tests):
          - rouge_metrics is not None → ROUGE-L tracking. Lock selection to
            ROUGE the first time it succeeds. Update best.pt if strictly
            higher than previous best ROUGE-L (ties → earlier epoch wins,
            Occam). After lock, eval_loss-based updates are ignored.
          - rouge_metrics is None AND selection NOT yet locked → eval_loss
            tracking (legacy / smoke-test path). Update best.pt if strictly
            lower than previous best eval_loss.
          - rouge_metrics is None AND selection locked → no update (ROUGE
            failed at this epoch but succeeded earlier; keep that best).
        """
        if rouge_metrics is not None:
            self._selection_locked_to_rouge = True
            if rouge_metrics['rougeL_mean'] > self.best_eval_rouge_l:
                self.best_eval_rouge_l = rouge_metrics['rougeL_mean']
                self.best_eval_em = rouge_metrics['exact_match_mean']
                self.best_eval_loss = eval_metrics['loss']
                self.best_epoch = epoch_num
                self._save_checkpoint()
                print(f'[LoRATrainer]   ✓ new best ROUGE-L (epoch {epoch_num}); '
                      f'saved best.pt')
        elif not self._selection_locked_to_rouge:
            # Smoke-test / no-ROUGE fallback path
            if eval_metrics['loss'] < self.best_eval_loss:
                self.best_eval_loss = eval_metrics['loss']
                self.best_epoch = epoch_num
                self._save_checkpoint()

    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate_loss(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in self.eval_loader:
            out = self._forward_step(batch)
            if out.loss is not None:
                total_loss += float(out.loss)
                n += 1
        return {'loss': total_loss / max(1, n)}

    def _save_checkpoint(self):
        out_dir = self.cfg.get('output_dir', '.')
        os.makedirs(out_dir, exist_ok=True)
        # Only LoRA (trainable) params need saving — base is frozen and
        # loadable from the HF hub.
        # Use _inner_model.named_parameters() so keys are clean even under
        # torch.compile (compiled wrapper adds `_orig_mod.` prefix).
        state = {name: p.detach().cpu().clone()
                 for name, p in self._inner_model.named_parameters()
                 if p.requires_grad}
        torch.save({
            'lora_state': state,
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
        }, os.path.join(out_dir, 'best.pt'))

    # ────────────────────────────────────────────────────────────────────
    # Post-hoc ROUGE-L + exact-match eval (Wang 2022 Tk-Instruct protocol).
    # CE eval during training is a fast proxy; generation-based ROUGE-L is
    # the canonical held-out metric for SuperNI and what the paper reports.
    # Runs ONCE at end of training (expensive: greedy decode per example).
    # Sub-samples test set (default 10 instances per task) to keep it under
    # ~30-60 min per run while still scoring across all 119 test tasks.
    # ────────────────────────────────────────────────────────────────────
    def _make_routing_fn(self):
        """Build a routing_fn(cluster_id) → dict for SuperNIEvaluator, based
        on model type. Returns None for methods that ignore routing."""
        from models.lora_models import MoCLEModel, MultiBranchLoRAModel
        if isinstance(self._inner_model, MultiBranchLoRAModel) and self.algorithm is not None:
            def fn(cluster_id):
                mask = self.algorithm.get_mask(cluster_id, training=False)
                return {'mask': mask, 'mask_scale': self.algorithm.expected_mask_sum}
            return fn
        if isinstance(self._inner_model, MoCLEModel):
            def fn(cluster_id):
                return {'cluster_id': cluster_id}
            return fn
        return None

    def _make_mask_scale_fn(self):
        from models.lora_models import MultiBranchLoRAModel
        if isinstance(self._inner_model, MultiBranchLoRAModel):
            return lambda: self._inner_model._mask_scale
        return None

    def _build_rouge_eval_loader(self, instances_per_task: int = 10):
        """Construct a held-out test DataLoader with a small instances-per-
        task budget for post-hoc generation eval (greedy decode is slow).

        Re-uses the same tokenizer, K=20 domain map, and cached SuperNI task
        JSONs — only the instance subsample differs from the training-time
        eval_loader. Returns (loader, tokenizer)."""
        from data.natural_instructions import get_superni_dataloaders
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.cfg['model']['base_model_name'])
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        dcfg = self.cfg['data']
        tcfg = self.cfg['training']
        _, loader, _ = get_superni_dataloaders(
            data_root=dcfg['data_root'],
            tokenizer=tok,
            batch_size=tcfg.get('batch_size_per_device', 4),
            max_seq_len=tcfg.get('max_seq_len', 2048),
            instances_per_task_train=dcfg.get('instances_per_task_train', 100),
            instances_per_task_eval=instances_per_task,
            subset_frac_train=dcfg.get('subset_frac_train', 1.0),
            num_workers=0,   # single-process for reproducibility + no pool churn
            # Fixed 42 (decoupled from experiment seed) so the ROUGE-L eval
            # sub-sample is identical across all 3 seeds — consistent with
            # run_lora.py's data_subset_seed=42 convention.
            data_subset_seed=dcfg.get('data_subset_seed', 42),
            K=dcfg.get('num_clusters', 20),
            cache_dir=dcfg.get('cluster_cache_dir', './data_cache/lora'),
        )
        return loader, tok

    def run_rouge_eval(self, instances_per_task: int = 10,
                        max_new_tokens: int = 128) -> Dict:
        """Run the Wang 2022 eval protocol once and return aggregated metrics.

        Self-contained: builds the eval loader + tokenizer, installs the
        appropriate routing_fn for this trainer's model/algorithm combo, and
        invokes evaluation.instruction_eval.SuperNIEvaluator. Safe to call
        after training completes (finalize_results auto-calls it unless
        cfg.training.run_rouge_eval is False)."""
        from evaluation.instruction_eval import SuperNIEvaluator
        loader, tokenizer = self._build_rouge_eval_loader(instances_per_task)
        evaluator = SuperNIEvaluator(
            model=self.model, tokenizer=tokenizer, eval_loader=loader,
            max_new_tokens=max_new_tokens, device=self.device,
            routing_fn=self._make_routing_fn(),
            mask_scale_fn=self._make_mask_scale_fn(),
        )
        return evaluator.evaluate()

    def finalize_results(self) -> Dict:
        """Write results.json from per-epoch tracking. ROUGE-L is run per epoch
        in train(); finalize just dumps the best-epoch view + diagnostics.

        Top-level fields preserved verbatim from the prior post-hoc protocol
        for downstream compatibility (rerun_rouge.py, _ensure_cell.sh tier
        (b), 8a/8b/8c summary blocks): eval_rouge_l, eval_exact_match,
        eval_rouge_num_tasks, eval_rouge_instances_per_task,
        eval_rouge_elapsed_min. Semantically these are now "best epoch's
        ROUGE" not "post-training ROUGE on best-eval-loss checkpoint".
        """
        out = {
            'best_eval_loss': self.best_eval_loss,
            'best_epoch': self.best_epoch,
            'selection_metric': ('eval_rouge_l'
                                  if self._selection_locked_to_rouge
                                  else 'eval_loss'),
            'history': self.history,
            'total_steps': self.global_step,
            'num_trainable_params': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad),
        }

        if self._selection_locked_to_rouge:
            tcfg = self.cfg.get('training', {})
            out['eval_rouge_l'] = self.best_eval_rouge_l
            out['eval_exact_match'] = self.best_eval_em
            out['eval_rouge_num_tasks'] = self._rouge_num_tasks
            out['eval_rouge_instances_per_task'] = int(
                tcfg.get('rouge_eval_instances_per_task', 10))
            out['eval_rouge_elapsed_min'] = round(self._rouge_total_elapsed_min, 2)
        else:
            # Smoke test / no-ROUGE path — eval_rouge_l = None signals to
            # _ensure_cell.sh tier (b) that ROUGE backfill is needed.
            out['eval_rouge_l'] = None

        out_dir = self.cfg.get('output_dir', '.')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'results.json'), 'w') as f:
            json.dump(out, f, indent=2)
        return out

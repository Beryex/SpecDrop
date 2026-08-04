"""Smoke tests for training/trainer_lora.py.

We don't spin up a real 3B HF model here (that's a 5090 integration test);
instead we mock just enough to verify:
  - Trainer builds without HF network (algorithm wiring works)
  - Per-step LR scheduler does linear warmup then cosine decay
  - pa warmup step advance happens after optimizer.step
  - LoRA-only grad: optimizer param list matches requires_grad=True
"""
import math
import os
import tempfile

import pytest
import torch
import torch.nn as nn


def _fake_algorithm():
    """Minimal SoftSpecDrop stand-in with the trainer's required API."""
    class _A:
        num_modules = 4
        num_categories = 10
        current_step = 0
        current_epoch = 0
        total_warmup_steps = 0
        expected_mask_sum = 2.2

        def set_total_steps(self, total_steps):
            self.total_warmup_steps = total_steps

        def get_mask(self, category_ids, training=True, **kw):
            return torch.rand(category_ids.shape[0], self.num_modules)

        def on_epoch_end(self, branches, epoch):
            self.current_epoch = epoch + 1
            return {}
    return _A()


def _fake_lora_model():
    """A pretend BaseLoRAModel: a linear + a ParallelLoRA wrap + .base w/ forward."""
    from models.lora_adapters import MultiBranchSoftSpecDropAdapter
    from models.lora_base import LoRAInjectedLinear, freeze_base_params

    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 8)
            self.l1 = nn.Linear(8, 8, bias=False)  # will be wrapped
            self.head = nn.Linear(8, 100, bias=False)

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            h = self.emb(input_ids)
            h = self.l1(h)
            logits = self.head(h)
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, 100), labels.view(-1), ignore_index=-100)
            else:
                loss = torch.tensor(0.0)
            from types import SimpleNamespace
            return SimpleNamespace(loss=loss, logits=logits)

    class _Model(nn.Module):
        """Pretend MultiBranchLoRAModel — subclass so isinstance() works."""
        def __init__(self):
            super().__init__()
            self.base = _Base()
            # Wrap l1 with LoRAInjectedLinear
            adapter = MultiBranchSoftSpecDropAdapter(8, 8, num_experts=4, rank=2)
            self.base.l1 = LoRAInjectedLinear(
                self.base.l1, adapter, site_role='', target_name='l1')
            freeze_base_params(self.base)
            self._mask_scale = None

        def set_mask_scale(self, s): self._mask_scale = s

        def forward(self, input_ids, attention_mask=None, labels=None,
                    cluster_id=None, mask=None, **kwargs):
            if mask is not None:
                from models.lora_base import set_routing_all
                set_routing_all(
                    self.base,
                    {'mask': mask, 'mask_scale': self._mask_scale})
            out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                             labels=labels)
            from types import SimpleNamespace
            return SimpleNamespace(loss=out.loss, logits=out.logits,
                                   aux_loss=torch.tensor(0.0))

        def trainable_parameters(self):
            return [p for p in self.parameters() if p.requires_grad]

    return _Model()


def _fake_loader(n_batches=4):
    """Yield dicts matching the SuperNI collate format."""
    data = []
    for _ in range(n_batches):
        data.append({
            'input_ids': torch.randint(0, 100, (2, 5)),
            'attention_mask': torch.ones(2, 5, dtype=torch.long),
            'labels': torch.randint(0, 100, (2, 5)),
            'cluster_id': torch.randint(0, 4, (2,)),
        })
    return data


# ── Per-step LR scheduler sanity ────────────────────────────────────────────

def test_lr_scheduler_warmup_then_cosine():
    """Trainer's LR schedule: linear warmup for `warmup_ratio_lr × total_steps`,
    then cosine decay to 0."""
    from training.trainer_lora import LoRATrainer

    # Need a minimal trainer, but the expensive bit is the optimizer build
    # on trainable params. Build directly on a toy model + skip HF.
    cfg = {
        'training': {
            'epochs': 2, 'lr': 1e-3, 'warmup_ratio_lr': 0.2,
            'max_grad_norm': 1.0, 'weight_decay': 0.0,
            'amp_dtype': 'bf16', 'log_interval': 1000,
        },
        'output_dir': tempfile.mkdtemp(),
    }
    model = _fake_lora_model()
    algo = _fake_algorithm()
    loader = _fake_loader(n_batches=10)
    trainer = LoRATrainer(cfg, model=model, algorithm=algo,
                           train_loader=loader, eval_loader=loader,
                           device='cpu')

    # total_steps = 2 × 10 = 20, warmup = 0.2 × 20 = 4 steps.
    assert trainer.total_steps == 20
    # LR at step 0 should be ~0
    assert trainer.lr_scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-9)
    # Step through warmup
    for _ in range(4):
        trainer.lr_scheduler.step()
    # After warmup: LR should be ~ max LR
    assert trainer.lr_scheduler.get_last_lr()[0] == pytest.approx(1e-3, rel=1e-2)
    # Step through remainder: LR → 0 at end
    for _ in range(16):
        trainer.lr_scheduler.step()
    assert trainer.lr_scheduler.get_last_lr()[0] < 1e-5


def test_trainer_pa_warmup_step_advance():
    """One train epoch: algorithm.current_step should equal global_step."""
    from training.trainer_lora import LoRATrainer

    cfg = {
        'training': {
            'epochs': 1, 'lr': 1e-4, 'warmup_ratio_lr': 0.0,
            'max_grad_norm': 1.0, 'amp_dtype': 'bf16',
            'log_interval': 1000, 'eval_interval_epochs': 999,
        },
        'output_dir': tempfile.mkdtemp(),
    }
    model = _fake_lora_model()
    algo = _fake_algorithm()
    loader = _fake_loader(n_batches=3)

    trainer = LoRATrainer(cfg, model=model, algorithm=algo,
                           train_loader=loader, eval_loader=loader, device='cpu')
    # Initial state
    assert algo.current_step == 0
    assert algo.total_warmup_steps == 3  # set_total_steps(total_steps)

    # Run training
    trainer.train()

    # After 3 batches, global_step and current_step must match
    assert algo.current_step == trainer.global_step == 3


def test_trainer_grad_accum():
    """Gradient accumulation: 8 mini-batches × accum=4 → 2 optimizer steps.

    Regression test for a bug where `grad_accum_steps` was documented in
    configs (32 for SuperNI) but unimplemented in the train loop, silently
    running every mini-batch as its own optimizer step (→ 32× too many LR
    ticks, wrong warmup, wrong effective batch).
    """
    from training.trainer_lora import LoRATrainer

    cfg = {
        'training': {
            'epochs': 1, 'lr': 1e-4, 'warmup_ratio_lr': 0.0,
            'grad_accum_steps': 4,
            'max_grad_norm': 1.0, 'amp_dtype': 'bf16',
            'log_interval': 1000, 'eval_interval_epochs': 999,
        },
        'output_dir': tempfile.mkdtemp(),
    }
    model = _fake_lora_model()
    algo = _fake_algorithm()
    loader = _fake_loader(n_batches=8)

    trainer = LoRATrainer(cfg, model=model, algorithm=algo,
                           train_loader=loader, eval_loader=loader, device='cpu')
    # total_steps counts OPTIMIZER steps, not mini-batches: 1 epoch × (8//4) = 2.
    assert trainer.total_steps == 2
    # set_total_steps wired through to algorithm for pa warmup granularity.
    assert algo.total_warmup_steps == 2

    trainer.train()

    # After 8 mini-batches at accum=4, exactly 2 optimizer steps happened.
    assert trainer.global_step == 2
    # pa warmup step mirrors optimizer step, NOT mini-batch count.
    assert algo.current_step == 2


def test_trainer_grad_accum_drops_incomplete_tail():
    """Trailing mini-batches that don't complete an accum window are dropped.

    9 mini-batches × accum=4 → 2 complete windows (steps 1-4, 5-8), 1 tail
    batch (step 9) left uncommitted. Standard convention — keeps every
    optimizer step at identical effective batch size.
    """
    from training.trainer_lora import LoRATrainer

    cfg = {
        'training': {
            'epochs': 1, 'lr': 1e-4, 'warmup_ratio_lr': 0.0,
            'grad_accum_steps': 4,
            'max_grad_norm': 1.0, 'amp_dtype': 'bf16',
            'log_interval': 1000, 'eval_interval_epochs': 999,
        },
        'output_dir': tempfile.mkdtemp(),
    }
    model = _fake_lora_model()
    algo = _fake_algorithm()
    loader = _fake_loader(n_batches=9)

    trainer = LoRATrainer(cfg, model=model, algorithm=algo,
                           train_loader=loader, eval_loader=loader, device='cpu')
    assert trainer.total_steps == 2  # 9 // 4 = 2, tail dropped
    trainer.train()
    assert trainer.global_step == 2


def test_trainer_optimizer_only_has_lora_params():
    """Optimizer param list equals `filter(requires_grad, model.parameters())`."""
    from training.trainer_lora import LoRATrainer

    cfg = {
        'training': {
            'epochs': 1, 'lr': 1e-4, 'warmup_ratio_lr': 0.0,
            'max_grad_norm': 1.0, 'amp_dtype': 'bf16',
            'log_interval': 1000, 'eval_interval_epochs': 999,
        },
        'output_dir': tempfile.mkdtemp(),
    }
    model = _fake_lora_model()
    algo = _fake_algorithm()
    loader = _fake_loader(n_batches=1)
    trainer = LoRATrainer(cfg, model=model, algorithm=algo,
                           train_loader=loader, eval_loader=loader, device='cpu')
    opt_params = trainer.optimizer.param_groups[0]['params']
    req_grad = [p for p in model.parameters() if p.requires_grad]
    assert len(opt_params) == len(req_grad)
    assert all(p1 is p2 for p1, p2 in zip(opt_params, req_grad))


def test_trainer_saves_only_lora_weights():
    """Checkpoint should include only trainable (LoRA) params, not frozen base."""
    from training.trainer_lora import LoRATrainer

    tmpdir = tempfile.mkdtemp()
    cfg = {
        'training': {
            'epochs': 1, 'lr': 1e-4, 'warmup_ratio_lr': 0.0,
            'max_grad_norm': 1.0, 'amp_dtype': 'bf16',
            'log_interval': 1000, 'eval_interval_epochs': 1,
        },
        'output_dir': tmpdir,
    }
    model = _fake_lora_model()
    algo = _fake_algorithm()
    loader = _fake_loader(n_batches=2)
    trainer = LoRATrainer(cfg, model=model, algorithm=algo,
                           train_loader=loader, eval_loader=loader, device='cpu')
    trainer.train()
    ckpt_path = os.path.join(tmpdir, 'best.pt')
    assert os.path.exists(ckpt_path)
    ckpt = torch.load(ckpt_path, weights_only=False)
    # All saved param names must correspond to trainable (LoRA) params.
    saved_names = list(ckpt['lora_state'].keys())
    # Must have at least some LoRA params
    assert len(saved_names) > 0
    for name in saved_names:
        # Frozen params (emb, head, l1.base) should NOT be saved.
        assert not name.startswith('base.emb'), name
        assert not name.startswith('base.head'), name
        # The LoRAInjectedLinear wraps l1; its `.base` is frozen, and only
        # `.adapter.*` is trainable. So "base.l1.base" (frozen) not allowed,
        # but "base.l1.adapter.*" (trainable) IS expected.
        assert 'l1.base' not in name, name
    # At least one LoRA adapter param present
    assert any('adapter' in n for n in saved_names), saved_names
    # Results JSON exists
    assert os.path.exists(os.path.join(tmpdir, 'results.json'))


# ── Per-epoch ROUGE-L selection (Wang 2022 protocol) ──────────────────────
#
# These tests verify the new SINGLE-METRIC convention: ROUGE-L is the only
# selection criterion across (a) best.pt checkpoint selection, (b)
# hyperparameter sweep argmax, (c) baseline comparison. The eval_loss
# fallback only activates when ROUGE eval can't run (e.g. fake-model smoke
# tests without a real HF tokenizer).
#
# Strategy: monkey-patch trainer.run_rouge_eval to return a controlled
# sequence of per-epoch ROUGE values, verify selection picks the argmax.

def _make_rouge_test_trainer(tmpdir, epochs=3, eval_loss_seq=None):
    """Build a LoRATrainer ready to run on the fake LoRA model + fake loader.

    Returns (trainer, model, algo). Caller monkey-patches `run_rouge_eval`
    and `evaluate_loss` on the trainer to control per-epoch return values.
    """
    from training.trainer_lora import LoRATrainer
    cfg = {
        'training': {
            'epochs': epochs, 'lr': 1e-4, 'warmup_ratio_lr': 0.0,
            'max_grad_norm': 1.0, 'amp_dtype': 'bf16',
            'log_interval': 1000, 'eval_interval_epochs': 1,
            'run_rouge_eval': True,
            'rouge_eval_instances_per_task': 5,
        },
        'output_dir': tmpdir,
    }
    model = _fake_lora_model()
    algo = _fake_algorithm()
    loader = _fake_loader(n_batches=2)
    trainer = LoRATrainer(cfg, model=model, algorithm=algo,
                           train_loader=loader, eval_loader=loader, device='cpu')
    return trainer, model, algo


def test_per_epoch_rouge_picks_argmax():
    """best.pt should be saved at the epoch with the HIGHEST ROUGE-L, not the
    lowest eval_loss. Simulates 3 epochs where eval_loss is monotone (epoch 1
    best on loss) but ROUGE-L peaks at epoch 2 (the realistic instruction-
    tuning pattern: token CE diverges past peak generation quality)."""
    tmpdir = tempfile.mkdtemp()
    trainer, model, algo = _make_rouge_test_trainer(tmpdir, epochs=3)

    # Eval-loss path (decoy): epoch-1 wins.
    eval_loss_seq = iter([1.00, 1.10, 1.20])
    rouge_seq = iter([
        {'rougeL_mean': 0.40, 'exact_match_mean': 0.20, 'num_tasks': 5},
        {'rougeL_mean': 0.50, 'exact_match_mean': 0.30, 'num_tasks': 5},
        {'rougeL_mean': 0.45, 'exact_match_mean': 0.25, 'num_tasks': 5},
    ])
    trainer.evaluate_loss = lambda: {'loss': next(eval_loss_seq)}
    trainer.run_rouge_eval = lambda **kw: next(rouge_seq)

    trainer.train()

    assert trainer._selection_locked_to_rouge, "Selection should lock to ROUGE"
    assert trainer.best_epoch == 2, f"Expected epoch 2 (max ROUGE), got {trainer.best_epoch}"
    assert trainer.best_eval_rouge_l == pytest.approx(0.50)
    assert trainer.best_eval_em == pytest.approx(0.30)
    assert trainer.best_eval_loss == pytest.approx(1.10)  # eval_loss AT best epoch

    # results.json: top-level eval_rouge_l matches the best-epoch value.
    import json
    res = json.load(open(os.path.join(tmpdir, 'results.json')))
    assert res['eval_rouge_l'] == pytest.approx(0.50)
    assert res['eval_exact_match'] == pytest.approx(0.30)
    assert res['best_epoch'] == 2
    assert res['selection_metric'] == 'eval_rouge_l'
    assert res['eval_rouge_num_tasks'] == 5
    assert res['eval_rouge_instances_per_task'] == 5
    # History has eval_rouge_l + eval_exact_match per epoch.
    assert len(res['history']) == 3
    for i, ep in enumerate(res['history']):
        assert 'eval_rouge_l' in ep, f"epoch {i+1} missing eval_rouge_l"
        assert 'eval_exact_match' in ep
    assert res['history'][1]['eval_rouge_l'] == pytest.approx(0.50)


def test_per_epoch_rouge_tie_keeps_earlier_epoch():
    """Strict argmax with Occam's razor: identical ROUGE-L → earlier epoch wins.

    Mirrors scripts/ablation_chain.py's tie-break (smaller key wins). Important
    because LoRA's heavy overfitting may produce ROUGE plateaus across epochs
    — we don't want to silently flip best.pt at last-epoch noise."""
    tmpdir = tempfile.mkdtemp()
    trainer, model, algo = _make_rouge_test_trainer(tmpdir, epochs=3)

    eval_loss_seq = iter([1.00, 1.10, 1.20])
    rouge_seq = iter([
        {'rougeL_mean': 0.50, 'exact_match_mean': 0.30, 'num_tasks': 5},
        {'rougeL_mean': 0.50, 'exact_match_mean': 0.32, 'num_tasks': 5},
        {'rougeL_mean': 0.50, 'exact_match_mean': 0.31, 'num_tasks': 5},
    ])
    trainer.evaluate_loss = lambda: {'loss': next(eval_loss_seq)}
    trainer.run_rouge_eval = lambda **kw: next(rouge_seq)
    trainer.train()
    assert trainer.best_epoch == 1, "Tie should resolve to earliest epoch"


def test_rouge_failure_falls_back_to_eval_loss():
    """When ROUGE eval raises (no HF tokenizer / OOM), trainer should fall
    back to eval_loss-based selection silently. Required for smoke tests
    that use the fake LoRA model without a real `model.base_model_name`."""
    tmpdir = tempfile.mkdtemp()
    trainer, model, algo = _make_rouge_test_trainer(tmpdir, epochs=2)

    # ROUGE always fails. eval_loss decreases → epoch 2 wins on loss.
    eval_loss_seq = iter([1.50, 1.20])

    def _fail(**kw): raise RuntimeError("simulated ROUGE failure")
    trainer.evaluate_loss = lambda: {'loss': next(eval_loss_seq)}
    trainer.run_rouge_eval = _fail

    trainer.train()

    assert not trainer._selection_locked_to_rouge, \
        "Selection should NOT lock to ROUGE if eval never succeeded"
    assert trainer.best_epoch == 2
    assert trainer.best_eval_loss == pytest.approx(1.20)

    import json
    res = json.load(open(os.path.join(tmpdir, 'results.json')))
    assert res['selection_metric'] == 'eval_loss'
    # eval_rouge_l = None signals to _ensure_cell.sh tier (b) that ROUGE
    # backfill is needed (rerun_rouge.py path).
    assert res['eval_rouge_l'] is None


def test_rouge_partial_failure_after_lock_keeps_best():
    """If ROUGE succeeds at epoch 1 then fails at epoch 2, best.pt should
    stay at epoch 1's checkpoint — selection is LOCKED to ROUGE the moment
    any epoch's ROUGE succeeds, and a later eval_loss improvement does NOT
    overwrite it (prevents protocol mixing within a single run)."""
    tmpdir = tempfile.mkdtemp()
    trainer, model, algo = _make_rouge_test_trainer(tmpdir, epochs=3)

    eval_loss_seq = iter([1.50, 0.80, 0.50])  # monotone DOWN — would pick epoch 3 by loss

    rouge_calls = {'n': 0}
    def _rouge(**kw):
        rouge_calls['n'] += 1
        if rouge_calls['n'] == 1:
            return {'rougeL_mean': 0.40, 'exact_match_mean': 0.20, 'num_tasks': 5}
        raise RuntimeError("simulated late failure")
    trainer.evaluate_loss = lambda: {'loss': next(eval_loss_seq)}
    trainer.run_rouge_eval = _rouge

    trainer.train()

    assert trainer._selection_locked_to_rouge
    assert trainer.best_epoch == 1, "Should stay locked to epoch-1 ROUGE win"
    assert trainer.best_eval_rouge_l == pytest.approx(0.40)


def test_run_rouge_eval_disabled_skips_per_epoch():
    """cfg.training.run_rouge_eval=False → no ROUGE call at any epoch,
    eval_loss-based selection. Backward-compat path for any caller that
    explicitly disables ROUGE."""
    from training.trainer_lora import LoRATrainer
    tmpdir = tempfile.mkdtemp()
    cfg = {
        'training': {
            'epochs': 2, 'lr': 1e-4, 'warmup_ratio_lr': 0.0,
            'max_grad_norm': 1.0, 'amp_dtype': 'bf16',
            'log_interval': 1000, 'eval_interval_epochs': 1,
            'run_rouge_eval': False,
        },
        'output_dir': tmpdir,
    }
    model = _fake_lora_model(); algo = _fake_algorithm()
    loader = _fake_loader(n_batches=2)
    trainer = LoRATrainer(cfg, model=model, algorithm=algo,
                           train_loader=loader, eval_loader=loader, device='cpu')

    rouge_calls = {'n': 0}
    def _rouge(**kw):
        rouge_calls['n'] += 1
        return {'rougeL_mean': 0.5, 'exact_match_mean': 0.3, 'num_tasks': 5}
    trainer.run_rouge_eval = _rouge

    trainer.train()
    assert rouge_calls['n'] == 0, "ROUGE should not be called when disabled"
    assert not trainer._selection_locked_to_rouge


def test_history_includes_per_epoch_rouge():
    """Per-epoch ROUGE numbers should land in results.json's history list,
    enabling post-hoc analysis like 'did ROUGE peak at epoch 2 or 3 across pa
    values?' without needing the trainer back."""
    tmpdir = tempfile.mkdtemp()
    trainer, model, algo = _make_rouge_test_trainer(tmpdir, epochs=3)

    eval_loss_seq = iter([1.00, 1.10, 1.20])
    rouge_seq = iter([
        {'rougeL_mean': 0.40, 'exact_match_mean': 0.20, 'num_tasks': 5},
        {'rougeL_mean': 0.50, 'exact_match_mean': 0.30, 'num_tasks': 5},
        {'rougeL_mean': 0.45, 'exact_match_mean': 0.25, 'num_tasks': 5},
    ])
    trainer.evaluate_loss = lambda: {'loss': next(eval_loss_seq)}
    trainer.run_rouge_eval = lambda **kw: next(rouge_seq)
    trainer.train()

    import json
    res = json.load(open(os.path.join(tmpdir, 'results.json')))
    rouges = [ep['eval_rouge_l'] for ep in res['history']]
    assert rouges == [pytest.approx(0.40), pytest.approx(0.50), pytest.approx(0.45)]
    losses = [ep['eval_loss'] for ep in res['history']]
    assert losses == [pytest.approx(1.00), pytest.approx(1.10), pytest.approx(1.20)]

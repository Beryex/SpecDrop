"""Routing-state helpers used by the LoRA diagnostics (scripts/_diag_helpers.py)."""
import json
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from algorithms.soft_specdrop import SoftSpecDrop
from scripts._diag_helpers import (set_softspecdrop_to_checkpoint_state,
                                   superni_frac_per_category)


def _algo(fr=None):
    return SoftSpecDrop(num_modules=4, num_categories=4, p_active=0.8, p_inactive=0.2,
                        assignment='round_robin', warmup_ratio=1.0, total_epochs=3,
                        frac_per_category=fr, amplification_beta=1.0,
                        warmup_schedule='cosine', warmup_unit='step')


def test_restores_best_epoch_state_after_trainer_reset(tmp_path):
    """After LoRATrainer resets the step count, the helper restores the warmup
    progress at which the best epoch was scored during training."""
    (tmp_path / 'results.json').write_text(json.dumps({'total_steps': 30, 'best_epoch': 2}))
    a = _algo()
    a.set_total_steps(7)  # what LoRATrainer.__init__ does for the eval loader
    p = set_softspecdrop_to_checkpoint_state(a, str(tmp_path), 3)
    ref = _algo()
    ref.set_total_steps(30)
    ref.current_step = 20  # state in which epoch 2 was scored during training
    assert abs(p - 2 / 3) < 1e-12
    for c in range(4):
        assert torch.allclose(a.get_mask(torch.tensor([c]), training=False),
                              ref.get_mask(torch.tensor([c]), training=False))


def test_superni_frac_per_category():
    m = {'task_to_cluster': {'a': 0, 'b': 0, 'c': 2, 'z': 1}, 'train_tasks': ['a', 'b', 'c']}
    assert superni_frac_per_category(m, 3) == [2 / 3, 0.0, 1 / 3]

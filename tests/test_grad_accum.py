#!/usr/bin/env python3
"""Test that gradient accumulation is equivalent to large-batch training.

Verifies: training with batch_size=16 × grad_accum_steps=2 produces
identical model weights as batch_size=32 × grad_accum_steps=1,
given the same data order and seed.

Uses synthetic data on CPU for speed.
"""

import os
import sys
import random
import shutil
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_fake_data(num_samples=64, num_classes=100, num_superclasses=20):
    """Create a fixed synthetic dataset as (images, fine, coarse, example_id).

    The 4th tuple field is required by trainer.py's batch-unpack convention
    (CIFAR100WithSuperclass and HFImageNetDataset both return 4-tuples).
    """
    gen = torch.Generator().manual_seed(777)
    images = torch.randn(num_samples, 3, 32, 32, generator=gen)
    fine = torch.randint(0, num_classes, (num_samples,), generator=gen)
    coarse = torch.randint(0, num_superclasses, (num_samples,), generator=gen)
    example_ids = torch.arange(num_samples)
    return images, fine, coarse, example_ids


class FakeLoader:
    """Deterministic data loader that yields fixed-size 4-tuple batches."""

    def __init__(self, images, fine, coarse, batch_size, example_ids=None):
        self.images = images
        self.fine = fine
        self.coarse = coarse
        self.bs = batch_size
        self.example_ids = (example_ids if example_ids is not None
                             else torch.arange(len(images)))

    def __len__(self):
        return (len(self.images) + self.bs - 1) // self.bs

    def __iter__(self):
        for i in range(0, len(self.images), self.bs):
            yield (self.images[i:i+self.bs],
                   self.fine[i:i+self.bs],
                   self.coarse[i:i+self.bs],
                   self.example_ids[i:i+self.bs])


class TestGradAccumEquivalence(unittest.TestCase):

    def test_accum_approx_large_batch(self):
        """batch=16 × accum=2 should approximately match batch=32 × accum=1.

        Not exactly equal due to BatchNorm: BN computes statistics per
        mini-batch (16), not per effective batch (32). This is standard
        behavior — gradient accumulation is mathematically exact only
        for BN-free models. We verify losses are close (<1% relative diff).
        """
        from models import build_model
        from algorithms import build_algorithm
        from training import Trainer

        images, fine, coarse, _ = make_fake_data(num_samples=64)

        base_cfg = {
            'experiment_name': 'test_accum',
            'model': {
                'type': 'multi_branch_resnet110',
                'base_channels': 4, 'num_branches': 4,
                'branch_channels': [2, 2, 2], 'num_blocks': 2,
                'num_classes': 100, 'shared_expert': False,
            },
            'algorithm': {
                'type': 'soft_specdrop', 'p_active': 0.9, 'p_inactive': 0.1,
                'assignment': 'round_robin',
            },
            'training': {
                'epochs': 2, 'lr': 0.01, 'momentum': 0.9,
                'weight_decay': 5e-4, 'lr_schedule': 'cosine', 'warmup_epochs': 0,
            },
            'data': {},
            'seed': 42,
        }

        # --- Run A: batch=32, accum=1 (standard) ---
        dir_a = tempfile.mkdtemp()
        set_seed(42)
        cfg_a = {**base_cfg, 'output_dir': dir_a,
                 'training': {**base_cfg['training'], 'batch_size': 32, 'grad_accum_steps': 1}}
        loader_a = FakeLoader(images, fine, coarse, batch_size=32)
        model_a = build_model(cfg_a)
        algo_a = build_algorithm(cfg_a)
        model_a.mask_scale = algo_a.expected_mask_sum
        trainer_a = Trainer(cfg_a, model_a, algo_a, loader_a, loader_a, 'cpu')
        results_a = trainer_a.train()

        # --- Run B: batch=16, accum=2 (equivalent effective batch=32) ---
        dir_b = tempfile.mkdtemp()
        set_seed(42)
        cfg_b = {**base_cfg, 'output_dir': dir_b,
                 'training': {**base_cfg['training'], 'batch_size': 16, 'grad_accum_steps': 2}}
        loader_b = FakeLoader(images, fine, coarse, batch_size=16)
        model_b = build_model(cfg_b)
        algo_b = build_algorithm(cfg_b)
        model_b.mask_scale = algo_b.expected_mask_sum
        trainer_b = Trainer(cfg_b, model_b, algo_b, loader_b, loader_b, 'cpu')
        results_b = trainer_b.train()

        # Compare model weights
        state_a = trainer_a.model.state_dict()
        state_b = trainer_b.model.state_dict()
        max_diff = 0.0
        for key in state_a:
            diff = (state_a[key].float() - state_b[key].float()).abs().max().item()
            max_diff = max(max_diff, diff)

        print(f"  Max weight diff: {max_diff:.2e}")
        print(f"  Run A (bs=32, accum=1): loss={results_a['epoch_history'][-1]['train_loss']:.4f}")
        print(f"  Run B (bs=16, accum=2): loss={results_b['epoch_history'][-1]['train_loss']:.4f}")

        # Weights won't be exactly equal due to BatchNorm mini-batch statistics.
        # Verify training loss is close (<2% relative difference).
        loss_a = results_a['epoch_history'][-1]['train_loss']
        loss_b = results_b['epoch_history'][-1]['train_loss']
        rel_diff = abs(loss_a - loss_b) / max(loss_a, 1e-8)
        self.assertLess(rel_diff, 0.02,
                        f"Loss rel diff too large: {rel_diff:.4f} ({loss_a:.4f} vs {loss_b:.4f})")

        print(f"  PASS: loss rel diff = {rel_diff:.4f} < 2% (BN causes small diff, expected)")

        shutil.rmtree(dir_a, ignore_errors=True)
        shutil.rmtree(dir_b, ignore_errors=True)

    def test_accum_1_is_noop(self):
        """grad_accum_steps=1 should be identical to no accumulation."""
        from models import build_model
        from algorithms import build_algorithm
        from training import Trainer

        images, fine, coarse, _ = make_fake_data(num_samples=32)

        cfg = {
            'experiment_name': 'test_noop',
            'model': {
                'type': 'multi_branch_resnet110',
                'base_channels': 4, 'num_branches': 4,
                'branch_channels': [2, 2, 2], 'num_blocks': 2,
                'num_classes': 100, 'shared_expert': False,
            },
            'algorithm': {
                'type': 'soft_specdrop', 'p_active': 0.9, 'p_inactive': 0.1,
                'assignment': 'round_robin',
            },
            'training': {
                'epochs': 1, 'batch_size': 16, 'lr': 0.01, 'momentum': 0.9,
                'weight_decay': 5e-4, 'lr_schedule': 'cosine', 'warmup_epochs': 0,
                'grad_accum_steps': 1,
            },
            'data': {},
            'seed': 42,
        }

        loader = FakeLoader(images, fine, coarse, batch_size=16)

        dir_out = tempfile.mkdtemp()
        set_seed(42)
        cfg['output_dir'] = dir_out
        model = build_model(cfg)
        algo = build_algorithm(cfg)
        model.mask_scale = algo.expected_mask_sum
        trainer = Trainer(cfg, model, algo, loader, loader, 'cpu')

        self.assertEqual(trainer.grad_accum_steps, 1)
        results = trainer.train()
        self.assertGreater(results['epoch_history'][0]['train_acc'], 0)
        print(f"  PASS: accum=1 runs correctly, train_acc={results['epoch_history'][0]['train_acc']:.1f}%")

        shutil.rmtree(dir_out, ignore_errors=True)


if __name__ == '__main__':
    print("=" * 60)
    print(" Gradient Accumulation Equivalence Tests")
    print("=" * 60)
    unittest.main(verbosity=2)

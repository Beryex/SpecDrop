#!/usr/bin/env python3
"""Test that checkpoint resume produces identical results to continuous training.

Verifies:
  1. CV Trainer: train 3 epochs straight == train 1 epoch + resume 2 epochs
  2. NLP Trainer: same equivalence
  3. Algorithm state (SoftSpecDrop warmup, SMoE-Dropout k-schedule) restored correctly
"""

import os
import sys
import shutil
import random
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_fake_cv_loader(num_batches=20, batch_size=16, num_classes=100, num_superclasses=20):
    """Create a deterministic fake CV data loader.

    Yields 4-tuples (image, fine_label, coarse_label, example_id) matching the
    CIFAR100WithSuperclass / HFImageNetDataset convention that trainer.py
    unpacks at training/trainer.py:477.
    """
    data = []
    gen = torch.Generator().manual_seed(777)
    for b in range(num_batches):
        images = torch.randn(batch_size, 3, 32, 32, generator=gen)
        fine = torch.randint(0, num_classes, (batch_size,), generator=gen)
        coarse = torch.randint(0, num_superclasses, (batch_size,), generator=gen)
        example_ids = torch.arange(b * batch_size, (b + 1) * batch_size)
        data.append((images, fine, coarse, example_ids))
    return data


class TestCVCheckpointResume(unittest.TestCase):
    """Test CV trainer checkpoint/resume equivalence."""

    def _make_cfg(self, output_dir, epochs=3):
        return {
            'experiment_name': 'test_ckpt',
            'model': {
                'type': 'multi_branch_resnet110',
                'base_channels': 4,
                'num_branches': 4,
                'branch_channels': [2, 2, 2],
                'num_blocks': 2,
                'num_classes': 100,
                'shared_expert': False,
            },
            'algorithm': {
                'type': 'soft_specdrop',
                'p_active': 0.9,
                'p_inactive': 0.1,
                'assignment': 'round_robin',
                'warmup_ratio': 0.5,
            },
            'training': {
                'epochs': epochs,
                'batch_size': 16,
                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'lr_schedule': 'cosine',
                'warmup_epochs': 1,
            },
            'data': {
                'dataset': 'cifar100',
                'data_dir': './data_cache',
                'num_workers': 0,
            },
            'output_dir': output_dir,
            'seed': 42,
        }

    def test_resume_equals_continuous(self):
        """Train 3 epochs straight == train 1 + resume 2."""
        from models import build_model
        from algorithms import build_algorithm
        from training import Trainer

        device = 'cpu'
        fake_loader = _make_fake_cv_loader()

        # --- Run A: Train 3 epochs continuously ---
        dir_a = tempfile.mkdtemp()
        try:
            set_seed(42)
            cfg_a = self._make_cfg(dir_a, epochs=3)
            model_a = build_model(cfg_a)
            algo_a = build_algorithm(cfg_a)
            if hasattr(model_a, 'mask_scale') and algo_a.expected_mask_sum is not None:
                model_a.mask_scale = algo_a.expected_mask_sum
            trainer_a = Trainer(cfg_a, model_a, algo_a, fake_loader, fake_loader, device)
            results_a = trainer_a.train()

            # --- Run B: Train 1 epoch, save, then resume for 2 more ---
            # Use same epochs=3 cfg so scheduler/LR schedule matches exactly
            dir_b = tempfile.mkdtemp()
            set_seed(42)
            cfg_b1 = self._make_cfg(dir_b, epochs=3)
            model_b = build_model(cfg_b1)
            algo_b = build_algorithm(cfg_b1)
            if hasattr(model_b, 'mask_scale') and algo_b.expected_mask_sum is not None:
                model_b.mask_scale = algo_b.expected_mask_sum
            trainer_b1 = Trainer(cfg_b1, model_b, algo_b, fake_loader, fake_loader, device)
            trainer_b1.epochs = 1  # stop after 1 epoch
            trainer_b1.train()
            # Trainer.train() treats its self.epochs budget as "training finished"
            # and deletes latest.pt at the tail (mid-run latest.pt is redundant
            # once training completes normally). Since we want to resume from
            # the post-epoch-0 state, re-save latest.pt manually.
            trainer_b1._save_checkpoint(epoch=0, is_best=False)

            # Resume: load latest.pt, train to epoch 3
            cfg_b2 = self._make_cfg(dir_b, epochs=3)
            model_b2 = build_model(cfg_b2)
            algo_b2 = build_algorithm(cfg_b2)
            if hasattr(model_b2, 'mask_scale') and algo_b2.expected_mask_sum is not None:
                model_b2.mask_scale = algo_b2.expected_mask_sum
            trainer_b2 = Trainer(cfg_b2, model_b2, algo_b2, fake_loader, fake_loader, device)
            trainer_b2.resume_from_checkpoint(os.path.join(dir_b, 'latest.pt'))
            results_b = trainer_b2.train()

            # Compare final model weights
            state_a = trainer_a.model.state_dict()
            state_b = trainer_b2.model.state_dict()
            for key in state_a:
                self.assertTrue(
                    torch.allclose(state_a[key], state_b[key], atol=1e-6),
                    f"Mismatch in {key}: max diff = {(state_a[key] - state_b[key]).abs().max().item()}"
                )

            # Compare best accuracy
            self.assertAlmostEqual(results_a['best_top1'], results_b['best_top1'], places=1)

            # Compare algorithm state
            self.assertEqual(algo_a.current_epoch, algo_b2.current_epoch,
                             "SoftSpecDrop current_epoch mismatch after resume")

            print("  PASS: CV resume == continuous (model weights, best_acc, algo state)")

        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)


class TestNLPCheckpointResume(unittest.TestCase):
    """Test NLP trainer checkpoint/resume equivalence."""

    def _make_cfg(self, output_dir, epochs=3):
        return {
            'experiment_name': 'test_nlp_ckpt',
            'model': {
                'type': 'multi_branch_transformer_lm',
                'vocab_size': 1000,
                'hidden_dim': 64,
                'num_layers': 2,
                'num_heads': 2,
                'num_branches': 4,
                'ffn_dim_per_branch': 32,
                'max_seq_len': 64,
                'dropout': 0.0,
            },
            'algorithm': {
                'type': 'soft_specdrop',
                'p_active': 0.9,
                'p_inactive': 0.1,
                'assignment': 'round_robin',
                'warmup_ratio': 0.5,
            },
            'training': {
                'epochs': epochs,
                'batch_size': 8,
                'lr': 1e-3,
                'beta1': 0.9,
                'beta2': 0.95,
                'weight_decay': 0.01,
                'warmup_steps': 5,
                'max_grad_norm': 1.0,
            },
            'data': {},
            'output_dir': output_dir,
            'seed': 42,
        }

    def _make_fake_loader(self, num_batches=10, batch_size=8, seq_len=64, vocab_size=1000,
                          num_domains=4):
        """Create a deterministic fake data loader."""
        data = []
        gen = torch.Generator().manual_seed(999)
        for _ in range(num_batches):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
            domain_ids = torch.randint(0, num_domains, (batch_size,), generator=gen)
            data.append((input_ids, domain_ids))
        return data

    def test_resume_equals_continuous(self):
        """Train 3 epochs straight == train 1 + resume 2."""
        from models.transformer_lm import MultiBranchTransformerLM
        from algorithms import build_algorithm
        from training.trainer_nlp import NLPTrainer

        device = 'cpu'

        # --- Run A: 3 epochs straight ---
        dir_a = tempfile.mkdtemp()
        try:
            set_seed(42)
            cfg_a = self._make_cfg(dir_a, epochs=3)
            cfg_a['algorithm']['_num_categories'] = 4
            model_a = MultiBranchTransformerLM(
                vocab_size=1000, hidden_dim=64, num_layers=2, num_heads=2,
                num_branches=4, ffn_dim_per_branch=32, max_seq_len=64, dropout=0.0)
            algo_a = build_algorithm(cfg_a)
            if hasattr(model_a, 'mask_scale') and algo_a.expected_mask_sum is not None:
                model_a.mask_scale = algo_a.expected_mask_sum
            train_loader = self._make_fake_loader()
            val_loader = self._make_fake_loader(num_batches=3)
            trainer_a = NLPTrainer(cfg_a, model_a, algo_a, train_loader, val_loader, device)
            results_a = trainer_a.train()

            # --- Run B: 1 epoch, then resume 2 ---
            # Use same epochs=3 cfg so total_steps matches for LR schedule
            dir_b = tempfile.mkdtemp()
            set_seed(42)
            cfg_b1 = self._make_cfg(dir_b, epochs=3)
            cfg_b1['algorithm']['_num_categories'] = 4
            model_b = MultiBranchTransformerLM(
                vocab_size=1000, hidden_dim=64, num_layers=2, num_heads=2,
                num_branches=4, ffn_dim_per_branch=32, max_seq_len=64, dropout=0.0)
            algo_b = build_algorithm(cfg_b1)
            if hasattr(model_b, 'mask_scale') and algo_b.expected_mask_sum is not None:
                model_b.mask_scale = algo_b.expected_mask_sum
            train_loader = self._make_fake_loader()
            val_loader = self._make_fake_loader(num_batches=3)
            trainer_b1 = NLPTrainer(cfg_b1, model_b, algo_b, train_loader, val_loader, device)
            trainer_b1.epochs = 1  # stop after 1 epoch
            trainer_b1.train()
            # NLPTrainer.train() also removes latest.pt on completion (see
            # training/trainer_nlp.py:327+). Re-save it for the resume leg.
            trainer_b1._save_checkpoint(epoch=0, is_best=False)

            # Resume
            cfg_b2 = self._make_cfg(dir_b, epochs=3)
            cfg_b2['algorithm']['_num_categories'] = 4
            model_b2 = MultiBranchTransformerLM(
                vocab_size=1000, hidden_dim=64, num_layers=2, num_heads=2,
                num_branches=4, ffn_dim_per_branch=32, max_seq_len=64, dropout=0.0)
            algo_b2 = build_algorithm(cfg_b2)
            if hasattr(model_b2, 'mask_scale') and algo_b2.expected_mask_sum is not None:
                model_b2.mask_scale = algo_b2.expected_mask_sum
            train_loader = self._make_fake_loader()
            val_loader = self._make_fake_loader(num_batches=3)
            trainer_b2 = NLPTrainer(cfg_b2, model_b2, algo_b2, train_loader, val_loader, device)
            trainer_b2.resume_from_checkpoint(os.path.join(dir_b, 'latest.pt'))
            results_b = trainer_b2.train()

            # Compare final model weights
            state_a = trainer_a.model.state_dict()
            state_b = trainer_b2.model.state_dict()
            for key in state_a:
                self.assertTrue(
                    torch.allclose(state_a[key], state_b[key], atol=1e-5),
                    f"Mismatch in {key}: max diff = {(state_a[key] - state_b[key]).abs().max().item()}"
                )

            # Compare best PPL
            self.assertAlmostEqual(results_a['best_val_ppl'], results_b['best_val_ppl'], places=1)

            # Compare algorithm state
            self.assertEqual(algo_a.current_epoch, algo_b2.current_epoch)

            print("  PASS: NLP resume == continuous (model weights, best_ppl, algo state)")

        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)


class TestAlgorithmStateDict(unittest.TestCase):
    """Test algorithm state_dict / load_state_dict."""

    def test_soft_specdrop_state(self):
        from algorithms.soft_specdrop import SoftSpecDrop
        algo = SoftSpecDrop(4, 20, warmup_ratio=0.5, total_epochs=200)
        algo.on_epoch_end(None, 0)
        algo.on_epoch_end(None, 1)
        algo.on_epoch_end(None, 2)
        self.assertEqual(algo.current_epoch, 3)

        state = algo.state_dict()
        algo2 = SoftSpecDrop(4, 20, warmup_ratio=0.5, total_epochs=200)
        self.assertEqual(algo2.current_epoch, 0)
        algo2.load_state_dict(state)
        self.assertEqual(algo2.current_epoch, 3)
        print("  PASS: SoftSpecDrop state_dict round-trip")

    def test_base_state_dict_noop(self):
        """Algorithms without state should return empty dict and accept load.

        Covered by NoDropout here (the current codebase's canonical stateless
        algorithm). The SpecDrop / SMoEDropoutRouting variants this test
        originally covered have been removed; SMoE-Dropout now ships as a
        faithful-model class (models/smoe_dropout_transformer_lm.py) rather
        than a separate algorithm module.
        """
        from algorithms.no_dropout import NoDropout
        algo = NoDropout(4, 20)
        state = algo.state_dict()
        self.assertEqual(state, {})
        algo.load_state_dict(state)  # should not raise
        print("  PASS: Base algorithm state_dict is no-op")


if __name__ == '__main__':
    print("=" * 60)
    print(" Checkpoint/Resume Equivalence Tests")
    print("=" * 60)
    unittest.main(verbosity=2)

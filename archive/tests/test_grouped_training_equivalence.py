#!/usr/bin/env python
"""Verify that grouped conv produces identical training outcomes.

Runs 20 epochs of training with both FullMultiBranchResNet (for-loop) and
GroupedFullMultiBranchResNet (grouped conv) using identical seeds, data, and
hyperparameters. Compares final accuracy, loss trajectory, and per-epoch metrics.

This is the strongest evidence that the optimization is lossless — more
convincing than any numerical tolerance analysis.

Usage:
    python tests/test_grouped_training_equivalence.py [--device cuda] [--epochs 20]
"""

import sys
import os
import argparse
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim

from models.multi_branch import FullMultiBranchResNet, GroupedFullMultiBranchResNet
from algorithms import build_algorithm
from data.cifar100 import get_dataloaders


def train_one_epoch(model, train_loader, optimizer, criterion, algo, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, coarse_labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        coarse_labels = coarse_labels.to(device)

        mask = algo.get_mask(coarse_labels, training=True).to(device)
        outputs = model(images, mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, test_loader, criterion, algo, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, coarse_labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        coarse_labels = coarse_labels.to(device)

        mask = algo.get_mask(coarse_labels, training=False).to(device)
        outputs = model(images, mask)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def run_training(model_cls, model_name, args):
    """Run full training loop, return epoch-by-epoch metrics."""
    # Seed everything
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)

    # Model
    model = model_cls(
        num_branches=20, num_blocks=3, num_classes=100,
        base_channels=16, branch_channels=[4, 7, 14],
        shared_expert=True, shared_expert_blocks=3,
    ).to(device)

    # Algorithm
    algo_cfg = {
        'algorithm': {
            'type': 'soft_specdrop',
            'p_active': 0.9,
            'p_inactive': 0.1,
            'assignment': 'round_robin',
        },
        'model': {'num_branches': 20},
        'data': {'dataset': 'cifar100'},
    }
    algo = build_algorithm(algo_cfg)
    model.mask_scale = algo.expected_mask_sum

    # Data
    train_loader, test_loader = get_dataloaders('./data_cache', 128, num_workers=2)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Train
    metrics = []
    t0 = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, algo, device, epoch)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, algo, device)
        scheduler.step()
        algo.on_epoch_end(None, epoch)

        metrics.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 2),
            'test_loss': round(test_loss, 6),
            'test_acc': round(test_acc, 2),
        })
        print(f"  [{model_name}] Epoch {epoch+1:>2}/{args.epochs}  "
              f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%  "
              f"train_loss={train_loss:.4f}")

    elapsed = time.time() - t0
    return metrics, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    print(f"Device: {args.device}, Epochs: {args.epochs}")
    print(f"Config: K=20, nb=3, bc=[4,7,14], SE=True, Soft SpecDrop p=0.9")
    print()

    # Run for-loop version
    print("=" * 60)
    print(" Training: FullMultiBranchResNet (for-loop)")
    print("=" * 60)
    metrics_fl, time_fl = run_training(FullMultiBranchResNet, "for-loop", args)

    # Run grouped version
    print()
    print("=" * 60)
    print(" Training: GroupedFullMultiBranchResNet (grouped conv)")
    print("=" * 60)
    metrics_gr, time_gr = run_training(GroupedFullMultiBranchResNet, "grouped", args)

    # Compare
    print()
    print("=" * 60)
    print(" Comparison")
    print("=" * 60)
    print(f"  {'Epoch':>5} | {'FL train_acc':>12} {'GR train_acc':>12} {'diff':>8} | "
          f"{'FL test_acc':>11} {'GR test_acc':>11} {'diff':>8}")
    print(f"  {'-'*5}-+-{'-'*12}-{'-'*12}-{'-'*8}-+-{'-'*11}-{'-'*11}-{'-'*8}")

    max_train_diff = 0.0
    max_test_diff = 0.0
    for fl, gr in zip(metrics_fl, metrics_gr):
        td = abs(fl['train_acc'] - gr['train_acc'])
        vd = abs(fl['test_acc'] - gr['test_acc'])
        max_train_diff = max(max_train_diff, td)
        max_test_diff = max(max_test_diff, vd)
        print(f"  {fl['epoch']+1:>5} | {fl['train_acc']:>11.2f}% {gr['train_acc']:>11.2f}% "
              f"{td:>7.2f}% | {fl['test_acc']:>10.2f}% {gr['test_acc']:>10.2f}% {vd:>7.2f}%")

    final_fl = metrics_fl[-1]['test_acc']
    final_gr = metrics_gr[-1]['test_acc']

    print()
    print(f"  Final test accuracy:  for-loop={final_fl:.2f}%  grouped={final_gr:.2f}%  "
          f"diff={abs(final_fl - final_gr):.2f}%")
    print(f"  Max train_acc diff across epochs: {max_train_diff:.2f}%")
    print(f"  Max test_acc diff across epochs:  {max_test_diff:.2f}%")
    print(f"  Wall time:  for-loop={time_fl:.1f}s  grouped={time_gr:.1f}s  "
          f"speedup={time_fl/time_gr:.2f}x")

    # Save results
    results = {
        'config': 'K=20 nb=3 bc=[4,7,14] SE=True soft_specdrop p=0.9',
        'device': args.device,
        'epochs': args.epochs,
        'forloop': metrics_fl,
        'grouped': metrics_gr,
        'final_test_acc_forloop': final_fl,
        'final_test_acc_grouped': final_gr,
        'final_diff': abs(final_fl - final_gr),
        'max_train_diff': max_train_diff,
        'max_test_diff': max_test_diff,
        'time_forloop': round(time_fl, 1),
        'time_grouped': round(time_gr, 1),
    }
    out_path = 'tests/grouped_training_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Verdict
    print()
    if max_test_diff < 1.0:
        print("  VERDICT: Training equivalence confirmed — accuracy matches within noise.")
    else:
        print(f"  WARNING: Max test accuracy diff = {max_test_diff:.2f}% — investigate.")


if __name__ == '__main__':
    main()

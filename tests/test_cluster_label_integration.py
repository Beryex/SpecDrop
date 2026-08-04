"""Unit tests for cluster-label integration with data/slimpajama.py.

Validates:
  (1) load_cluster_labels schema — required keys present, correct types.
  (2) compute_category_fractions(..., cluster_label_path=...) returns fractions
      computed from CLUSTER IDs, not SlimPajama domain_ids.
  (3) Domain-path vs cluster-path produce DIFFERENT fractions (the override
      actually takes effect).
  (4) Chunk count mismatch between cluster file and token cache raises.
  (5) get_slimpajama_dataloaders' _apply_cluster_labels swap path works when
      both paths given (simulated via direct call on the private helper).

Run:  python tests/test_cluster_label_integration.py
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.slimpajama import (
    compute_category_fractions,
    load_cluster_labels,
    _count_chunks_per_domain,
)


def _make_token_cache(path, domain_ids, seq_len=8):
    """Minimal token cache with given per-chunk domain_ids."""
    N = len(domain_ids)
    input_ids = torch.randint(0, 100, (N, seq_len), dtype=torch.long)
    torch.save({
        'input_ids': input_ids,
        'domain_ids': torch.tensor(domain_ids, dtype=torch.long),
    }, path)


def _make_cluster_cache(path, cluster_ids, n_clusters=7, embedder='test',
                        seed=42, source_cache='train.pt'):
    torch.save({
        'cluster_ids': torch.tensor(cluster_ids, dtype=torch.long),
        'n_clusters': n_clusters,
        'embedder': embedder,
        'seed': seed,
        'source_cache': source_cache,
        'num_chunks': len(cluster_ids),
    }, path)


class TestLoadClusterLabels(unittest.TestCase):

    def test_schema_required_keys(self):
        with tempfile.TemporaryDirectory() as td:
            cpath = os.path.join(td, 'cluster.pt')
            _make_cluster_cache(cpath, [0, 1, 2, 0], n_clusters=3)
            blob = load_cluster_labels(cpath)
            for key in ('cluster_ids', 'n_clusters', 'embedder', 'seed',
                        'source_cache', 'num_chunks'):
                self.assertIn(key, blob)
            self.assertIsInstance(blob['cluster_ids'], torch.Tensor)
            self.assertEqual(blob['cluster_ids'].dtype, torch.long)
            self.assertEqual(blob['n_clusters'], 3)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_cluster_labels('/nonexistent/path.pt')

    def test_missing_required_key_raises(self):
        with tempfile.TemporaryDirectory() as td:
            cpath = os.path.join(td, 'bad.pt')
            torch.save({'cluster_ids': torch.tensor([0, 1])}, cpath)  # no n_clusters
            with self.assertRaises(ValueError):
                load_cluster_labels(cpath)


class TestComputeCategoryFractionsWithClusters(unittest.TestCase):

    def test_cluster_labels_override_domain_ids(self):
        """When cluster_label_path is given, fractions reflect CLUSTER IDs
        even if the token cache has different domain_ids.
        """
        with tempfile.TemporaryDirectory() as td:
            # Token cache: 6 chunks, domain_ids = [0, 0, 0, 1, 1, 1] (uniform 2-way)
            tc = os.path.join(td, 'train.pt')
            _make_token_cache(tc, domain_ids=[0, 0, 0, 1, 1, 1])

            # Cluster cache on SAME 6 chunks, cluster_ids = [0, 0, 1, 1, 2, 2]
            # (uniform 3-way) — DIFFERENT distribution from domain_ids.
            cc = os.path.join(td, 'cluster.pt')
            _make_cluster_cache(cc, [0, 0, 1, 1, 2, 2], n_clusters=3)

            # Without cluster_label_path: reflects domain_ids (2-way split)
            fracs_d = compute_category_fractions(tc, num_categories=3)
            self.assertAlmostEqual(fracs_d[0], 3/6)  # domain 0: 3 chunks
            self.assertAlmostEqual(fracs_d[1], 3/6)  # domain 1: 3 chunks
            self.assertAlmostEqual(fracs_d[2], 0/6)  # no domain 2 in token cache

            # With cluster_label_path: reflects cluster_ids (3-way split)
            fracs_c = compute_category_fractions(
                tc, num_categories=3, cluster_label_path=cc)
            self.assertAlmostEqual(fracs_c[0], 2/6)
            self.assertAlmostEqual(fracs_c[1], 2/6)
            self.assertAlmostEqual(fracs_c[2], 2/6)

            # Different → override is in effect
            self.assertNotEqual(fracs_d, fracs_c)

    def test_chunk_count_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            tc = os.path.join(td, 'train.pt')
            _make_token_cache(tc, domain_ids=[0]*6)
            cc = os.path.join(td, 'cluster.pt')
            _make_cluster_cache(cc, [0, 1, 2], n_clusters=3)  # only 3 labels vs 6 chunks

            with self.assertRaises(ValueError):
                compute_category_fractions(
                    tc, num_categories=3, cluster_label_path=cc)

    def test_fractions_sum_to_one_either_way(self):
        with tempfile.TemporaryDirectory() as td:
            tc = os.path.join(td, 'train.pt')
            _make_token_cache(tc, domain_ids=[0]*100 + [1]*100)
            cc = os.path.join(td, 'cluster.pt')
            _make_cluster_cache(cc, [0]*50 + [1]*50 + [2]*100, n_clusters=3)

            fracs_d = compute_category_fractions(tc, num_categories=7)
            fracs_c = compute_category_fractions(tc, num_categories=7,
                                                   cluster_label_path=cc)
            self.assertAlmostEqual(sum(fracs_d), 1.0, places=6)
            self.assertAlmostEqual(sum(fracs_c), 1.0, places=6)

    def test_cluster_ids_beyond_num_categories_ignored(self):
        """If cluster file has IDs >= num_categories, they are silently ignored
        (counts stay 0), matching existing _count_chunks_per_domain semantics.
        This is a defensive property; real usage should set M = num_categories."""
        with tempfile.TemporaryDirectory() as td:
            tc = os.path.join(td, 'train.pt')
            _make_token_cache(tc, domain_ids=[0]*10)
            cc = os.path.join(td, 'cluster.pt')
            _make_cluster_cache(cc, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                n_clusters=10)

            # num_categories=5 → IDs 5..9 ignored; fracs = [1/10]*5
            fracs = compute_category_fractions(
                tc, num_categories=5, cluster_label_path=cc)
            for f in fracs:
                self.assertAlmostEqual(f, 0.1)


class TestGetSlimpajamaDataloaderSwap(unittest.TestCase):
    """Smoke-test the cluster-label swap path in get_slimpajama_dataloaders.

    We don't invoke the full dataloader builder (which needs the HF dataset
    cache available), but we can exercise the critical _apply_cluster_labels
    logic by shimming it via _count_chunks_per_domain with the cluster path.
    """

    def test_integration_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            tc = os.path.join(td, 'train.pt')
            # Imbalanced domain_ids: domain 0 has 80%, domain 1 has 20%
            _make_token_cache(tc, domain_ids=[0]*80 + [1]*20)

            cc = os.path.join(td, 'cluster.pt')
            # Cluster rebalances: 40 / 40 / 20 across 3 clusters
            _make_cluster_cache(cc, [0]*40 + [1]*40 + [2]*20, n_clusters=3)

            counts_d, total_d = _count_chunks_per_domain(tc, num_categories=3)
            self.assertEqual(counts_d, [80, 20, 0])
            self.assertEqual(total_d, 100)

            counts_c, total_c = _count_chunks_per_domain(
                tc, num_categories=3, cluster_label_path=cc)
            self.assertEqual(counts_c, [40, 40, 20])
            self.assertEqual(total_c, 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)

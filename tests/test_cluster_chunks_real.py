"""Real end-to-end smoke test for data/cluster_chunks.py.

Unlike tests/test_cluster_chunks.py (which uses a callable mock embedder to
stay dependency-free and fast), this test exercises the actual
SentenceTransformer code path with a small real model
(sentence-transformers/all-MiniLM-L6-v2, 22 MB). Its purpose is to guard
against regressions that mocked tests would miss — e.g., changes in the
SentenceTransformer API, missing requirements.txt entries, wrong import
paths, wrong argument names.

Skipped automatically if sentence-transformers isn't installed, so the
default test suite stays runnable without the extra dependency.

Run:  python tests/test_cluster_chunks_real.py
"""
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import sentence_transformers  # noqa: F401
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False


from data.cluster_chunks import (
    build_cluster_cache, _derive_text_cache_path,
    _derive_embedding_cache_path, _embedder_short_name,
)


TINY_EMBEDDER = 'sentence-transformers/all-MiniLM-L6-v2'  # 22 MB, fast


@unittest.skipUnless(_HAS_SBERT,
                      'sentence-transformers not installed; '
                      'run `pip install -r requirements.txt`')
class TestRealEndToEnd(unittest.TestCase):

    def test_full_pipeline_with_real_sbert_model(self):
        """Full train+val build_cluster_cache with a real SentenceTransformer.
        Verifies all 6 cache artifacts exist with correct shapes and dtypes.
        """
        with tempfile.TemporaryDirectory() as td:
            # Tiny realistic-shape caches (24 + 8 chunks × 32 tokens each)
            train_cache = os.path.join(
                td, 'tokenized_train_seq32_tok768_vocabdead.pt')
            val_cache = os.path.join(
                td, 'tokenized_val_seq32_tok256_vocabdead.pt')
            torch.save({
                'input_ids': torch.randint(100, 5000, (24, 32), dtype=torch.long),
                'domain_ids': torch.zeros(24, dtype=torch.long),
            }, train_cache)
            torch.save({
                'input_ids': torch.randint(100, 5000, (8, 32), dtype=torch.long),
                'domain_ids': torch.zeros(8, dtype=torch.long),
            }, val_cache)

            tr_out = os.path.join(td, 'clusters_train.pt')
            va_out = os.path.join(td, 'clusters_val.pt')

            result = build_cluster_cache(
                train_cache_path=train_cache, train_output_path=tr_out,
                val_cache_path=val_cache, val_output_path=va_out,
                n_clusters=3, embedder=TINY_EMBEDDER, seed=42,
                batch_size=8, device='cpu', detok_batch_size=16,
            )
            self.assertEqual(set(result.keys()), {'train', 'val'})

            # All 6 cache artifacts must exist
            short = _embedder_short_name(TINY_EMBEDDER)
            expected = {
                'train cluster':  (tr_out, 24, 'cluster_ids'),
                'val cluster':    (va_out, 8, 'cluster_ids'),
                'train text':     (_derive_text_cache_path(train_cache), 24, 'texts'),
                'val text':       (_derive_text_cache_path(val_cache), 8, 'texts'),
                'train emb':      (_derive_embedding_cache_path(train_cache, short), 24, 'embeddings'),
                'val emb':        (_derive_embedding_cache_path(val_cache, short), 8, 'embeddings'),
            }
            for name, (path, expected_n, key) in expected.items():
                self.assertTrue(os.path.exists(path), f'{name} missing: {path}')
                blob = torch.load(path, weights_only=False)
                self.assertIn(key, blob, f'{name}: missing key {key}')
                if key == 'cluster_ids':
                    self.assertEqual(len(blob[key]), expected_n)
                    self.assertEqual(blob['n_clusters'], 3)
                elif key == 'texts':
                    self.assertEqual(len(blob[key]), expected_n)
                elif key == 'embeddings':
                    emb = blob[key]
                    self.assertEqual(emb.shape, (expected_n, 384),
                                     msg=f'{name}: MiniLM-L6 should produce 384-dim embeddings')

    def test_second_call_is_noop(self):
        """Skip behavior when all outputs already exist."""
        with tempfile.TemporaryDirectory() as td:
            train_cache = os.path.join(
                td, 'tokenized_train_seq32_tok256_vocabdead.pt')
            torch.save({
                'input_ids': torch.randint(100, 5000, (8, 32), dtype=torch.long),
                'domain_ids': torch.zeros(8, dtype=torch.long),
            }, train_cache)
            out = os.path.join(td, 'clusters_train.pt')

            build_cluster_cache(
                train_cache_path=train_cache, train_output_path=out,
                n_clusters=2, embedder=TINY_EMBEDDER, seed=42,
                batch_size=4, device='cpu', detok_batch_size=8,
            )
            mtime1 = os.path.getmtime(out)
            import time; time.sleep(0.05)

            build_cluster_cache(
                train_cache_path=train_cache, train_output_path=out,
                n_clusters=2, embedder=TINY_EMBEDDER, seed=42,
                batch_size=4, device='cpu', detok_batch_size=8,
            )
            self.assertEqual(mtime1, os.path.getmtime(out),
                              msg='idempotent skip should not rewrite the file')


if __name__ == '__main__':
    unittest.main(verbosity=2)

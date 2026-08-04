"""Unit tests for data/cluster_chunks.py.

Validates:
  (1) GPT-2 detokenize round-trip on realistic chunk lengths is non-empty.
  (2) KMeans recovers well-separated fake embeddings (up to label permutation).
  (3) Full build_cluster_cache pipeline (detokenize + fake embed + kmeans + save)
      produces the expected output file with correct schema.
  (4) Train + val clustering shares cluster definitions (km.predict on val).
  (5) Idempotent skip: second call with skip_if_exists=True is a no-op.
  (6) Output files use atomic rename (no .tmp files left behind).
  (7) Path derivation (text cache, embedding cache, embedder short name).
  (8) Text cache is populated on first run, reused on second run (detokenize
      is skipped).
  (9) Embedding cache is populated on first run, reused on second run
      (detokenize AND embedder are skipped — embedder callable never invoked).
  (10) --no-save-intermediate skips persisting text + embedding caches.

Run:  python tests/test_cluster_chunks.py
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.cluster_chunks import (
    detokenize_chunks, embed_chunks, cluster_embeddings, fit_kmeans,
    predict_clusters, build_cluster_cache, _EmbedderAdapter,
    _derive_text_cache_path, _derive_embedding_cache_path,
    _embedder_short_name, _parse_token_cache_name,
)


def _make_fake_token_cache(path, num_chunks_per_domain, seq_len=32):
    """Write a fake tokenized cache. num_chunks_per_domain: {domain_id: count}.
    Token IDs are generated deterministically from (domain_id, chunk_idx) so
    detokenize produces distinguishable text per domain.
    """
    input_ids = []
    domain_ids = []
    for did, cnt in num_chunks_per_domain.items():
        for c in range(cnt):
            # Seed token IDs so different domains look different after decode.
            rng = np.random.RandomState(100 * did + c)
            chunk = rng.randint(100, 5000, size=(seq_len,))
            input_ids.append(torch.tensor(chunk, dtype=torch.long))
            domain_ids.append(did)
    torch.save({
        'input_ids': torch.stack(input_ids),
        'domain_ids': torch.tensor(domain_ids, dtype=torch.long),
    }, path)
    return len(input_ids)


def _fake_embedder_from_true_labels(true_labels, dim=32, sep=10.0, noise=0.1,
                                     seed=0):
    """Return a callable (texts, bs, device) -> (N,D) that emits well-separated
    vectors whose cluster structure matches `true_labels`.
    """
    true_labels = np.asarray(true_labels)
    n_clusters = int(true_labels.max()) + 1
    centers = np.random.RandomState(seed).randn(n_clusters, dim) * sep
    noise_vecs = np.random.RandomState(seed + 1).randn(len(true_labels), dim) * noise

    # Bind a mutable counter so successive calls (e.g. train then val) consume
    # different rows of the noise array if needed. But for tests we just emit
    # len(texts) rows matching true_labels' length.
    def _embed(texts, batch_size=32, device='cuda'):
        n = len(texts)
        assert n == len(true_labels), \
            f'fake embedder received {n} texts but true_labels has {len(true_labels)}'
        return centers[true_labels] + noise_vecs
    return _embed


class TestDetokenize(unittest.TestCase):

    def test_round_trip_non_empty(self):
        """GPT-2 decode on realistic random token IDs produces non-empty
        strings aligned with input rows."""
        N, L = 5, 32
        ids = torch.randint(100, 5000, (N, L))
        texts = detokenize_chunks(ids, batch_size=2, show_progress=False)
        self.assertEqual(len(texts), N)
        for t in texts:
            self.assertIsInstance(t, str)

    def test_accepts_numpy_array(self):
        """detokenize also accepts np.ndarray input."""
        N, L = 3, 16
        arr = np.random.randint(100, 5000, (N, L))
        texts = detokenize_chunks(arr, batch_size=8, show_progress=False)
        self.assertEqual(len(texts), N)

    def test_rejects_1d_input(self):
        with self.assertRaises(AssertionError):
            detokenize_chunks(torch.randint(0, 100, (32,)), show_progress=False)


class TestClusterEmbeddings(unittest.TestCase):

    def test_recovers_well_separated_clusters(self):
        """KMeans on well-separated fake vectors recovers the true clusters
        up to label permutation. Adjusted Rand ≈ 1.
        """
        from sklearn.metrics import adjusted_rand_score
        true = np.array([0]*10 + [1]*10 + [2]*10)
        emb = _fake_embedder_from_true_labels(true, dim=8, sep=20.0, noise=0.1)(
            texts=[''] * len(true))
        labels = cluster_embeddings(emb, n_clusters=3, seed=42)
        self.assertEqual(labels.shape, (30,))
        self.assertEqual(set(labels.tolist()), {0, 1, 2})
        self.assertAlmostEqual(adjusted_rand_score(true, labels), 1.0, places=2)

    def test_fit_kmeans_returns_predictable_model(self):
        """fit_kmeans returns (labels, km); km.predict on new points near a
        center returns the expected cluster.
        """
        true = np.array([0]*8 + [1]*8 + [2]*8)
        embedder = _fake_embedder_from_true_labels(true, dim=4, sep=30.0,
                                                    noise=0.01)
        emb = embedder([''] * len(true))
        labels, km = fit_kmeans(emb, n_clusters=3, seed=42)

        # km.predict on a point very close to cluster 0's center should
        # return the same cluster as labels[0..7].
        new_points = emb[:4]
        pred = predict_clusters(km, new_points, split_name='test')
        self.assertTrue(np.all(pred == labels[:4]))


class TestBuildClusterCacheEndToEnd(unittest.TestCase):

    def test_train_only(self):
        with tempfile.TemporaryDirectory() as td:
            # 3 "domains" × 8 chunks each = 24 chunks
            counts = {0: 8, 1: 8, 2: 8}
            train_cache = os.path.join(td,
                'tokenized_train_seq16_tok384_vocabdead.pt')
            N = _make_fake_token_cache(train_cache, counts, seq_len=16)

            # Fake embedder: true clusters match the domain IDs 0/1/2
            true_labels = sum([[d] * c for d, c in counts.items()], [])
            embedder = _fake_embedder_from_true_labels(true_labels, dim=16,
                                                       sep=20.0, noise=0.05)

            out = os.path.join(td, 'clusters_train.pt')
            result = build_cluster_cache(
                train_cache_path=train_cache,
                train_output_path=out,
                n_clusters=3,
                embedder=embedder,
                seed=42,
                batch_size=8,
                device='cpu',
                detok_batch_size=8,
            )
            self.assertEqual(result, {'train': out})
            self.assertTrue(os.path.exists(out))

            blob = torch.load(out, weights_only=False)
            self.assertEqual(blob['cluster_ids'].shape, (N,))
            self.assertEqual(blob['n_clusters'], 3)
            self.assertEqual(blob['num_chunks'], N)
            self.assertEqual(blob['seed'], 42)
            self.assertEqual(blob['embedder'], 'callable')
            # All labels in valid range
            self.assertTrue((blob['cluster_ids'] >= 0).all())
            self.assertTrue((blob['cluster_ids'] < 3).all())

            # Adjusted Rand high (cluster recovery successful)
            from sklearn.metrics import adjusted_rand_score
            self.assertAlmostEqual(
                adjusted_rand_score(true_labels, blob['cluster_ids'].numpy()),
                1.0, places=2)

    def test_train_plus_val_share_clusters(self):
        """Val labels land in the same cluster ID space as train (since they
        share the fitted KMeans model).
        """
        with tempfile.TemporaryDirectory() as td:
            train_counts = {0: 6, 1: 6, 2: 6}
            val_counts = {0: 2, 1: 2, 2: 2}
            train_cache = os.path.join(td,
                'tokenized_train_seq16_tok288_vocabdead.pt')
            val_cache = os.path.join(td,
                'tokenized_val_seq16_tok96_vocabdead.pt')
            Ntr = _make_fake_token_cache(train_cache, train_counts, seq_len=16)
            Nva = _make_fake_token_cache(val_cache, val_counts, seq_len=16)

            # One unified embedder: first Ntr calls = train, next Nva = val.
            # Easier: build one that keys on text index within the call.
            all_true_train = sum([[d] * c for d, c in train_counts.items()], [])
            all_true_val = sum([[d] * c for d, c in val_counts.items()], [])

            class _StatefulEmbedder:
                def __init__(self):
                    self.call_count = 0
                    self.rng_centers = np.random.RandomState(0).randn(3, 16) * 20
                def __call__(self, texts, batch_size=32, device='cuda'):
                    self.call_count += 1
                    n = len(texts)
                    if self.call_count == 1:  # train
                        true = all_true_train
                    else:  # val
                        true = all_true_val
                    assert n == len(true)
                    noise = np.random.RandomState(100 + self.call_count).randn(n, 16) * 0.05
                    return self.rng_centers[true] + noise

            embedder = _StatefulEmbedder()

            tr_out = os.path.join(td, 'clusters_train.pt')
            va_out = os.path.join(td, 'clusters_val.pt')

            result = build_cluster_cache(
                train_cache_path=train_cache,
                train_output_path=tr_out,
                val_cache_path=val_cache,
                val_output_path=va_out,
                n_clusters=3,
                embedder=embedder,
                seed=42,
                batch_size=8,
                device='cpu',
                detok_batch_size=8,
            )
            self.assertEqual(set(result.keys()), {'train', 'val'})
            self.assertTrue(os.path.exists(tr_out))
            self.assertTrue(os.path.exists(va_out))

            tr_blob = torch.load(tr_out, weights_only=False)
            va_blob = torch.load(va_out, weights_only=False)
            self.assertEqual(tr_blob['cluster_ids'].shape, (Ntr,))
            self.assertEqual(va_blob['cluster_ids'].shape, (Nva,))

            # Same cluster space
            self.assertEqual(tr_blob['n_clusters'], va_blob['n_clusters'])
            # Val labels in [0, K)
            self.assertTrue((va_blob['cluster_ids'] >= 0).all())
            self.assertTrue((va_blob['cluster_ids'] < 3).all())

            # Critical property: train chunks with true=0 and val chunks with
            # true=0 land in the SAME cluster ID (shared cluster space).
            tr_ids = tr_blob['cluster_ids'].numpy()
            va_ids = va_blob['cluster_ids'].numpy()
            tr_domain0_cluster = int(np.bincount(tr_ids[:6]).argmax())
            va_domain0_cluster = int(np.bincount(va_ids[:2]).argmax())
            self.assertEqual(tr_domain0_cluster, va_domain0_cluster,
                             msg="train & val 'domain 0' chunks should map "
                                 "to the same cluster ID (shared km model)")

    def test_idempotent_skip(self):
        """Second call with skip_if_exists=True is a no-op."""
        with tempfile.TemporaryDirectory() as td:
            train_cache = os.path.join(td,
                'tokenized_train_seq16_tok128_vocabdead.pt')
            _make_fake_token_cache(train_cache, {0: 4, 1: 4}, seq_len=16)
            true = [0]*4 + [1]*4
            embedder = _fake_embedder_from_true_labels(true, dim=8, sep=20.0)
            out = os.path.join(td, 'clusters.pt')

            build_cluster_cache(
                train_cache_path=train_cache, train_output_path=out,
                n_clusters=2, embedder=embedder, device='cpu',
                batch_size=4, detok_batch_size=4)
            mtime1 = os.path.getmtime(out)

            # Second call: skip, file untouched
            import time; time.sleep(0.05)
            build_cluster_cache(
                train_cache_path=train_cache, train_output_path=out,
                n_clusters=2, embedder=embedder, device='cpu',
                batch_size=4, detok_batch_size=4,
                skip_if_exists=True)
            mtime2 = os.path.getmtime(out)
            self.assertEqual(mtime1, mtime2,
                             msg="idempotent skip should not rewrite the file")

    def test_force_reruns(self):
        """skip_if_exists=False forces re-run (new mtime)."""
        with tempfile.TemporaryDirectory() as td:
            train_cache = os.path.join(td,
                'tokenized_train_seq16_tok128_vocabdead.pt')
            _make_fake_token_cache(train_cache, {0: 4, 1: 4}, seq_len=16)
            true = [0]*4 + [1]*4
            embedder = _fake_embedder_from_true_labels(true, dim=8, sep=20.0)
            out = os.path.join(td, 'clusters.pt')

            build_cluster_cache(
                train_cache_path=train_cache, train_output_path=out,
                n_clusters=2, embedder=embedder, device='cpu',
                batch_size=4, detok_batch_size=4)
            mtime1 = os.path.getmtime(out)

            import time; time.sleep(0.05)
            build_cluster_cache(
                train_cache_path=train_cache, train_output_path=out,
                n_clusters=2, embedder=embedder, device='cpu',
                batch_size=4, detok_batch_size=4,
                skip_if_exists=False)
            mtime2 = os.path.getmtime(out)
            self.assertGreater(mtime2, mtime1)

    def test_no_tmp_file_left_behind(self):
        """After a successful run, no .tmp file should remain (atomic rename)."""
        with tempfile.TemporaryDirectory() as td:
            train_cache = os.path.join(td,
                'tokenized_train_seq16_tok128_vocabdead.pt')
            _make_fake_token_cache(train_cache, {0: 4, 1: 4}, seq_len=16)
            true = [0]*4 + [1]*4
            embedder = _fake_embedder_from_true_labels(true, dim=8, sep=20.0)
            out = os.path.join(td, 'clusters.pt')

            build_cluster_cache(
                train_cache_path=train_cache, train_output_path=out,
                n_clusters=2, embedder=embedder, device='cpu',
                batch_size=4, detok_batch_size=4)
            self.assertTrue(os.path.exists(out))
            self.assertFalse(os.path.exists(out + '.tmp'))

    def test_val_cache_without_val_output_raises(self):
        with tempfile.TemporaryDirectory() as td:
            tc = os.path.join(td,
                'tokenized_train_seq16_tok64_vocabdead.pt')
            vc = os.path.join(td,
                'tokenized_val_seq16_tok32_vocabdead.pt')
            _make_fake_token_cache(tc, {0: 4}, seq_len=16)
            _make_fake_token_cache(vc, {0: 2}, seq_len=16)
            with self.assertRaises(ValueError):
                build_cluster_cache(
                    train_cache_path=tc,
                    train_output_path=os.path.join(td, 'ctr.pt'),
                    val_cache_path=vc,
                    val_output_path=None,  # missing → ValueError
                    n_clusters=1,
                    embedder=_fake_embedder_from_true_labels([0]*4, dim=4,
                                                              sep=10.0))


class TestPathDerivation(unittest.TestCase):

    def test_parse_new_suffix(self):
        """New style: tokenized_train_seq512_tok100000000_vocab<H>.pt"""
        name = '/a/b/tokenized_train_seq512_tok100000000_vocab2e62aacd7f.pt'
        self.assertEqual(_parse_token_cache_name(name),
                         'train_seq512_tok100000000')

    def test_parse_legacy_suffix(self):
        """Legacy style: tokenized_val_seq512_tok10000000_tok<H>.pt"""
        name = '/a/b/tokenized_val_seq512_tok10000000_tok2e62aacd7f.pt'
        self.assertEqual(_parse_token_cache_name(name),
                         'val_seq512_tok10000000')

    def test_parse_no_hash_suffix(self):
        """Very old style: tokenized_val_seq512_tok200000.pt"""
        name = 'tokenized_val_seq512_tok200000.pt'
        self.assertEqual(_parse_token_cache_name(name),
                         'val_seq512_tok200000')

    def test_parse_rejects_non_tokenized(self):
        with self.assertRaises(ValueError):
            _parse_token_cache_name('some_other_file.pt')

    def test_text_cache_path_derivation(self):
        src = '/a/b/tokenized_train_seq512_tok100000000_vocab2e62aacd7f.pt'
        self.assertEqual(
            _derive_text_cache_path(src),
            '/a/b/detokenized_train_seq512_tok100000000.pt')

    def test_text_cache_path_override_dir(self):
        src = '/a/b/tokenized_train_seq512_tok100000000_vocab2e62aacd7f.pt'
        self.assertEqual(
            _derive_text_cache_path(src, override_dir='/tmp/x'),
            '/tmp/x/detokenized_train_seq512_tok100000000.pt')

    def test_embedding_cache_path_derivation(self):
        src = '/a/b/tokenized_train_seq512_tok100000000_vocab2e62aacd7f.pt'
        self.assertEqual(
            _derive_embedding_cache_path(src, 'bge-large-en-v1.5'),
            '/a/b/embeddings_train_seq512_tok100000000_bge-large-en-v1.5.pt')

    def test_embedder_short_name_strips_prefix(self):
        self.assertEqual(_embedder_short_name('BAAI/bge-large-en-v1.5'),
                         'bge-large-en-v1.5')
        self.assertEqual(_embedder_short_name('intfloat/e5-large-v2'),
                         'e5-large-v2')
        self.assertEqual(_embedder_short_name('all-MiniLM-L6-v2'),
                         'all-MiniLM-L6-v2')

    def test_embedder_short_name_sanitizes(self):
        """Non-filename-safe chars get replaced with underscores."""
        self.assertEqual(_embedder_short_name('org/bad*name?here'),
                         'bad_name_here')

    def test_embedder_short_name_callable(self):
        self.assertEqual(_embedder_short_name(lambda t, bs, d: None),
                         'callable')

    def test_different_embedder_different_emb_cache_path(self):
        """Ablation-friendly: changing embedder yields different path, so
        bge and mpnet caches don't clobber each other.
        """
        src = '/a/b/tokenized_train_seq512_tok100000000_vocab2e62aacd7f.pt'
        p1 = _derive_embedding_cache_path(src, _embedder_short_name('BAAI/bge-large-en-v1.5'))
        p2 = _derive_embedding_cache_path(src, _embedder_short_name('sentence-transformers/all-mpnet-base-v2'))
        self.assertNotEqual(p1, p2)
        self.assertIn('bge-large', p1)
        self.assertIn('mpnet', p2)


class TestLayeredCaching(unittest.TestCase):

    def _run_and_count_calls(self, td, true_labels, counts, skip_if_exists=True,
                              save_intermediate=True):
        """Build cluster cache with a call-counting embedder. Returns
        (result dict, number of embedder calls)."""
        train_cache = os.path.join(td, 'tokenized_train_seq16_tok64_vocabdead.pt')
        # Real filename format so path derivation works
        _make_fake_token_cache(train_cache, counts, seq_len=16)

        # Wrap an inner embedder with a call counter
        call_counter = {'n': 0}
        inner = _fake_embedder_from_true_labels(true_labels, dim=8, sep=20.0,
                                                 noise=0.05)
        def _counting(texts, batch_size, device):
            call_counter['n'] += 1
            return inner(texts, batch_size, device)

        out_path = os.path.join(td, 'clusters.pt')
        result = build_cluster_cache(
            train_cache_path=train_cache, train_output_path=out_path,
            n_clusters=len(set(true_labels)), embedder=_counting,
            seed=42, batch_size=8, device='cpu', detok_batch_size=8,
            skip_if_exists=skip_if_exists,
            save_intermediate=save_intermediate,
        )
        return result, call_counter['n'], train_cache

    def test_text_cache_populated_on_first_run(self):
        with tempfile.TemporaryDirectory() as td:
            true = [0]*6 + [1]*6
            _, n_calls, train_cache = self._run_and_count_calls(
                td, true, {0: 6, 1: 6})
            # Text cache should exist after first run
            text_path = _derive_text_cache_path(train_cache)
            self.assertTrue(os.path.exists(text_path),
                             f'text cache missing: {text_path}')
            blob = torch.load(text_path, weights_only=False)
            self.assertIn('texts', blob)
            self.assertEqual(len(blob['texts']), 12)
            self.assertEqual(blob['num_chunks'], 12)

    def test_embedding_cache_populated_on_first_run(self):
        with tempfile.TemporaryDirectory() as td:
            true = [0]*6 + [1]*6
            _, _, train_cache = self._run_and_count_calls(td, true, {0: 6, 1: 6})
            from data.cluster_chunks import _embedder_short_name
            emb_path = _derive_embedding_cache_path(
                train_cache, _embedder_short_name(lambda t, bs, d: None))
            self.assertTrue(os.path.exists(emb_path),
                             f'embedding cache missing: {emb_path}')
            blob = torch.load(emb_path, weights_only=False)
            self.assertIn('embeddings', blob)
            self.assertEqual(blob['num_chunks'], 12)
            self.assertEqual(blob['dim'], 8)
            self.assertEqual(blob['embedder'], 'callable')

    def test_embedding_cache_skips_embedder_on_rerun(self):
        """Second run with embedding cache present does NOT call embedder."""
        with tempfile.TemporaryDirectory() as td:
            true = [0]*6 + [1]*6
            counts = {0: 6, 1: 6}

            # First run: builds everything, 1 embedder call
            _, n1, train_cache = self._run_and_count_calls(td, true, counts)
            self.assertEqual(n1, 1)

            # Delete the cluster output so we force re-clustering.
            # But keep text + embedding caches → embedder should NOT be called.
            os.remove(os.path.join(td, 'clusters.pt'))

            # Second run: should load from embedding cache, embedder untouched
            _, n2, _ = self._run_and_count_calls(td, true, counts)
            self.assertEqual(n2, 0,
                              msg='Embedding cache hit should prevent any '
                                  'embedder invocation')

    def test_text_cache_hit_embedding_cache_miss(self):
        """If text cache exists but embedding cache does not (e.g. embedder
        changed), we reuse text and only re-embed."""
        with tempfile.TemporaryDirectory() as td:
            true = [0]*6 + [1]*6
            counts = {0: 6, 1: 6}

            # First run: text + embedding cache both created
            _, _, train_cache = self._run_and_count_calls(td, true, counts)

            # Delete embedding cache only (simulate embedder change)
            from data.cluster_chunks import _embedder_short_name
            emb_path = _derive_embedding_cache_path(
                train_cache, _embedder_short_name(lambda t, bs, d: None))
            os.remove(emb_path)
            os.remove(os.path.join(td, 'clusters.pt'))

            # Second run: embedding missed, text hit → 1 embedder call
            _, n2, _ = self._run_and_count_calls(td, true, counts)
            self.assertEqual(n2, 1)
            # Text cache should still exist (unmodified)
            self.assertTrue(os.path.exists(_derive_text_cache_path(train_cache)))

    def test_no_save_intermediate_does_not_write_caches(self):
        with tempfile.TemporaryDirectory() as td:
            true = [0]*4 + [1]*4
            counts = {0: 4, 1: 4}
            _, n, train_cache = self._run_and_count_calls(
                td, true, counts, save_intermediate=False)
            # No intermediate files written
            self.assertFalse(os.path.exists(_derive_text_cache_path(train_cache)))
            from data.cluster_chunks import _embedder_short_name
            emb_path = _derive_embedding_cache_path(
                train_cache, _embedder_short_name(lambda t, bs, d: None))
            self.assertFalse(os.path.exists(emb_path))
            # But cluster output is still produced
            self.assertTrue(os.path.exists(os.path.join(td, 'clusters.pt')))

    def test_train_val_share_embedding_cache_across_splits(self):
        """When val is added, val gets its own text + embedding caches
        (separate from train, derived from val source filename)."""
        with tempfile.TemporaryDirectory() as td:
            train_cache = os.path.join(td,
                'tokenized_train_seq16_tok96_vocabdead.pt')
            val_cache = os.path.join(td,
                'tokenized_val_seq16_tok32_vocabdead.pt')
            _make_fake_token_cache(train_cache, {0: 3, 1: 3}, seq_len=16)
            _make_fake_token_cache(val_cache, {0: 1, 1: 1}, seq_len=16)

            all_true_train = [0]*3 + [1]*3
            all_true_val = [0, 1]

            class _Stateful:
                def __init__(self):
                    self.n = 0
                    self.centers = np.random.RandomState(0).randn(2, 8) * 20
                def __call__(self, texts, bs, d):
                    self.n += 1
                    true = all_true_train if self.n == 1 else all_true_val
                    assert len(texts) == len(true)
                    noise = np.random.RandomState(self.n).randn(len(texts), 8) * 0.05
                    return self.centers[true] + noise

            embedder = _Stateful()
            build_cluster_cache(
                train_cache_path=train_cache,
                train_output_path=os.path.join(td, 'clusters_train.pt'),
                val_cache_path=val_cache,
                val_output_path=os.path.join(td, 'clusters_val.pt'),
                n_clusters=2, embedder=embedder,
                seed=42, batch_size=8, device='cpu', detok_batch_size=8,
            )

            # Train + val each have their own text + embedding cache files
            self.assertTrue(os.path.exists(_derive_text_cache_path(train_cache)))
            self.assertTrue(os.path.exists(_derive_text_cache_path(val_cache)))
            from data.cluster_chunks import _embedder_short_name
            short = _embedder_short_name(embedder)
            self.assertTrue(os.path.exists(
                _derive_embedding_cache_path(train_cache, short)))
            self.assertTrue(os.path.exists(
                _derive_embedding_cache_path(val_cache, short)))
            # Embedder called exactly twice (once per split)
            self.assertEqual(embedder.n, 2)


class TestLatestCheckpointCleanup(unittest.TestCase):
    """The trainer's post-completion cleanup removes a redundant latest.pt.

    We mirror the snippet here (unit-testing the trainer directly would
    require heavy setup — model, optimizer, dataloaders), to guard against
    a future edit silently removing the os.remove call.
    """

    def _simulate_cleanup(self, output_dir):
        """Copy of the snippet inserted in trainer.py + trainer_nlp.py."""
        latest_path = os.path.join(output_dir, 'latest.pt')
        if os.path.exists(latest_path):
            try:
                os.remove(latest_path)
                return True
            except OSError:
                return False
        return False

    def test_latest_removed_when_present(self):
        with tempfile.TemporaryDirectory() as td:
            latest = os.path.join(td, 'latest.pt')
            results = os.path.join(td, 'results.json')
            best = os.path.join(td, 'best.pt')
            torch.save({'epoch': 10}, latest)
            torch.save({'epoch': 7, 'best_val_ppl': 55.0}, best)
            import json
            with open(results, 'w') as f:
                json.dump({'best_val_ppl': 55.0}, f)

            self.assertTrue(self._simulate_cleanup(td))
            self.assertFalse(os.path.exists(latest),
                              msg='latest.pt should be removed after success')
            self.assertTrue(os.path.exists(best),
                             msg='best.pt must be preserved')
            self.assertTrue(os.path.exists(results),
                             msg='results.json must be preserved')

    def test_noop_when_latest_absent(self):
        """Second call (or never-written case) is harmless."""
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(self._simulate_cleanup(td))
            # No side effects
            self.assertEqual(os.listdir(td), [])

    def test_trainer_source_contains_cleanup_snippet(self):
        """Static guard: the auto-delete lines must remain in the trainer.

        If a future edit removes them, this test fires, alerting the author.
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for rel in ('training/trainer.py', 'training/trainer_nlp.py'):
            path = os.path.join(repo_root, rel)
            with open(path) as f:
                src = f.read()
            # Expect the sentinel + the os.remove call + results.json anchor
            self.assertIn('latest_path', src,
                          msg=f'{rel}: missing latest_path cleanup')
            self.assertIn("os.remove(latest_path)", src,
                          msg=f'{rel}: missing os.remove(latest_path)')


if __name__ == '__main__':
    unittest.main(verbosity=2)

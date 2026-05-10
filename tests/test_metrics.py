"""Tests for evaluation metrics (synthetic data with known ground truth)."""

import numpy as np
import pytest


# -------------------------------------------------------------------------
# Fixtures — synthetic embeddings with known structure
# -------------------------------------------------------------------------


@pytest.fixture
def perfect_embeddings():
    """Create embeddings that perfectly separate 5 identities.

    Each identity gets a unit vector along a different axis in 5D space,
    with 4 samples per identity (slight noise added).
    """
    rng = np.random.RandomState(42)
    n_ids = 5
    samples_per_id = 4
    dim = 32

    embeddings = []
    labels = []

    for i in range(n_ids):
        center = np.zeros(dim)
        center[i] = 1.0  # distinct axis
        for _ in range(samples_per_id):
            noise = rng.randn(dim) * 0.01
            emb = center + noise
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
            labels.append(i)

    return np.array(embeddings), np.array(labels)


@pytest.fixture
def random_embeddings():
    """Random normalized embeddings (should give ~chance accuracy)."""
    rng = np.random.RandomState(123)
    n = 100
    dim = 32
    embeddings = rng.randn(n, dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.repeat(np.arange(10), 10)
    return embeddings, labels


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


class TestRank1Accuracy:
    """Test rank-based identification."""

    def test_perfect_separation(self, perfect_embeddings):
        """With well-separated embeddings, Rank-1 should be 1.0."""
        from src.evaluation.metrics import rank1_accuracy

        embeddings, labels = perfect_embeddings

        # Use first sample per identity as gallery, rest as probe
        unique = np.unique(labels)
        gallery_idx = np.array([np.where(labels == l)[0][0] for l in unique])
        probe_mask = np.ones(len(labels), dtype=bool)
        probe_mask[gallery_idx] = False
        probe_idx = np.where(probe_mask)[0]

        acc = rank1_accuracy(
            embeddings[gallery_idx], labels[gallery_idx],
            embeddings[probe_idx], labels[probe_idx],
        )
        assert acc == pytest.approx(1.0), f"Expected 1.0, got {acc}"

    def test_rank5_gte_rank1(self, random_embeddings):
        """Rank-5 accuracy should always be >= Rank-1."""
        from src.evaluation.metrics import rank1_accuracy, rank_k_accuracy

        embeddings, labels = random_embeddings

        unique = np.unique(labels)
        gallery_idx = np.array([np.where(labels == l)[0][0] for l in unique])
        probe_mask = np.ones(len(labels), dtype=bool)
        probe_mask[gallery_idx] = False
        probe_idx = np.where(probe_mask)[0]

        r1 = rank1_accuracy(
            embeddings[gallery_idx], labels[gallery_idx],
            embeddings[probe_idx], labels[probe_idx],
        )
        r5 = rank_k_accuracy(
            embeddings[gallery_idx], labels[gallery_idx],
            embeddings[probe_idx], labels[probe_idx], k=5,
        )
        assert r5 >= r1 - 1e-7


class TestFARFRR:
    """Test FAR/FRR computation."""

    def test_shapes(self):
        """FAR and FRR arrays should match threshold count."""
        from src.evaluation.metrics import compute_far_frr

        genuine = np.array([0.8, 0.9, 0.85, 0.95])
        impostor = np.array([0.1, 0.2, 0.15, 0.05])

        thresholds, far, frr = compute_far_frr(genuine, impostor, n_thresholds=500)

        assert len(thresholds) == 500
        assert len(far) == 500
        assert len(frr) == 500

    def test_perfect_separation(self):
        """With perfectly separated scores, we should achieve TAR=1.0 at some FAR=0."""
        from src.evaluation.metrics import compute_far_frr

        genuine = np.array([0.9, 0.95, 0.92, 0.91])
        impostor = np.array([0.1, 0.05, 0.08, 0.12])

        thresholds, far, frr = compute_far_frr(genuine, impostor)

        # At some threshold, FAR should be 0 and FRR should be 0
        # (i.e., perfect separation is achievable)
        # Find where FAR ≈ 0
        min_far_idx = np.argmin(far)
        # At that threshold, FRR should also be near 0 (for perfectly separated data)
        assert far[min_far_idx] < 0.01

    def test_far_frr_crossover(self):
        """FAR should decrease and FRR increase as threshold increases."""
        from src.evaluation.metrics import compute_far_frr

        rng = np.random.RandomState(42)
        genuine = rng.normal(0.7, 0.1, 1000)
        impostor = rng.normal(0.3, 0.1, 1000)

        thresholds, far, frr = compute_far_frr(genuine, impostor)

        # FAR at lowest threshold should be high
        assert far[0] > 0.5
        # FAR at highest threshold should be low
        assert far[-1] < 0.01
        # FRR at lowest threshold should be low
        assert frr[0] < 0.01


class TestTARatFAR:
    """Test TAR@FAR=1e-4."""

    def test_returns_three_values(self):
        """tar_at_far should return (tar, actual_far, threshold)."""
        from src.evaluation.metrics import tar_at_far

        genuine = np.random.RandomState(42).normal(0.7, 0.1, 1000)
        impostor = np.random.RandomState(42).normal(0.3, 0.1, 1000)

        result = tar_at_far(genuine, impostor, target_far=1e-3)
        assert len(result) == 3
        tar_val, actual_far, threshold = result
        assert 0.0 <= tar_val <= 1.0
        assert 0.0 <= actual_far <= 1.0


class TestKFoldVerification:
    """Test 10-fold cross-validation verification accuracy."""

    def test_perfect_pairs(self):
        """With perfectly separable scores, kfold accuracy should be ~1.0."""
        from src.evaluation.metrics import kfold_verification_accuracy

        rng = np.random.RandomState(42)
        n_pairs = 1000
        labels = np.array([1] * (n_pairs // 2) + [0] * (n_pairs // 2))
        # Genuine pairs: high score, impostor pairs: low score
        scores = np.where(labels == 1,
                          rng.normal(0.9, 0.02, n_pairs),
                          rng.normal(0.1, 0.02, n_pairs))

        acc = kfold_verification_accuracy(labels, scores, n_folds=5)
        assert acc > 0.95

    def test_random_scores(self):
        """With random scores, accuracy should be near 0.5."""
        from src.evaluation.metrics import kfold_verification_accuracy

        rng = np.random.RandomState(42)
        n_pairs = 1000
        labels = np.array([1] * (n_pairs // 2) + [0] * (n_pairs // 2))
        scores = rng.uniform(0, 1, n_pairs)  # Random scores

        acc = kfold_verification_accuracy(labels, scores, n_folds=5)
        assert 0.3 < acc < 0.7  # Should be near chance


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_self_similarity_is_one(self):
        """Cosine similarity of a vector with itself should be 1."""
        from src.evaluation.metrics import cosine_similarity_matrix

        embeddings = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])

        sim = cosine_similarity_matrix(embeddings, embeddings)

        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)

    def test_orthogonal_similarity_is_zero(self):
        """Orthogonal vectors should have cosine similarity ≈ 0."""
        from src.evaluation.metrics import cosine_similarity_matrix

        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])

        sim = cosine_similarity_matrix(a, b)

        assert abs(sim[0, 0]) < 1e-6

    def test_matrix_shape(self):
        """Output shape should be (N, M)."""
        from src.evaluation.metrics import cosine_similarity_matrix

        a = np.random.randn(10, 64)
        b = np.random.randn(5, 64)

        sim = cosine_similarity_matrix(a, b)

        assert sim.shape == (10, 5)


class TestRunFullEvaluation:
    """Test the full evaluation pipeline."""

    def test_returns_required_keys(self, perfect_embeddings):
        """run_full_evaluation should return all expected metric keys."""
        from src.evaluation.metrics import run_full_evaluation

        embeddings, labels = perfect_embeddings
        results = run_full_evaluation(embeddings, labels, "test-experiment")

        assert "experiment" in results
        assert results["experiment"] == "test-experiment"
        # With ≥2 images per identity, rank metrics should be present
        assert "rank1_accuracy" in results

    def test_perfect_data_high_accuracy(self, perfect_embeddings):
        """Well-separated embeddings should yield high accuracy."""
        from src.evaluation.metrics import run_full_evaluation

        embeddings, labels = perfect_embeddings
        results = run_full_evaluation(embeddings, labels, "perfect")

        assert results["rank1_accuracy"] > 0.9


class TestPairGeneration:
    """Test pair generation from embeddings."""

    def test_pair_balance(self, perfect_embeddings):
        """Generated pairs should be roughly balanced genuine/impostor."""
        from src.evaluation.metrics import generate_pairs_from_embeddings

        embeddings, labels = perfect_embeddings

        pair_labels, pair_scores = generate_pairs_from_embeddings(
            embeddings, labels, n_pairs=100,
        )

        n_genuine = np.sum(pair_labels == 1)
        n_impostor = np.sum(pair_labels == 0)

        # Should each be roughly n_pairs / 2
        assert n_genuine > 0
        assert n_impostor > 0

    def test_genuine_scores_higher(self, perfect_embeddings):
        """Genuine pair scores should be higher than impostor scores on average."""
        from src.evaluation.metrics import generate_pairs_from_embeddings

        embeddings, labels = perfect_embeddings

        pair_labels, pair_scores = generate_pairs_from_embeddings(
            embeddings, labels, n_pairs=100,
        )

        genuine_mean = pair_scores[pair_labels == 1].mean()
        impostor_mean = pair_scores[pair_labels == 0].mean()

        assert genuine_mean > impostor_mean

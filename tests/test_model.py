"""Tests for the FaceRecModel (forward pass, embedding extraction)."""

import numpy as np
import pytest
import torch


# We guard the test with an import check because timm may not be
# available in all test environments.
timm = pytest.importorskip("timm")


@pytest.fixture
def small_model():
    """Create a small FaceRecModel for testing."""
    from src.downstream.train_downstream import FaceRecModel

    model = FaceRecModel(
        backbone_name="convnext_atto",
        n_classes=10,
        loss_type="cosface",
        embedding_dim=128,
    )
    model.eval()
    return model


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


class TestFaceRecModel:
    """Test the FaceRecModel class."""

    def test_model_creation(self, small_model):
        """Model should be creatable with convnext_atto backbone."""
        assert small_model.embedding_dim == 128

    def test_forward_returns_logits_and_embeddings(self, small_model):
        """Forward pass should return (logits, embeddings)."""
        batch = torch.randn(4, 3, 112, 112)
        labels = torch.tensor([0, 1, 2, 3])

        logits, emb = small_model(batch, labels)

        assert logits.shape == (4, 10)  # (batch, n_classes)
        assert emb.shape == (4, 128)  # (batch, embedding_dim)

    def test_forward_without_labels(self, small_model):
        """Forward without labels should return unmodified cosine logits."""
        batch = torch.randn(4, 3, 112, 112)

        logits, emb = small_model(batch, labels=None)

        assert logits.shape == (4, 10)
        assert emb.shape == (4, 128)

    def test_get_embedding_normalized(self, small_model):
        """get_embedding() should return L2-normalized vectors."""
        batch = torch.randn(4, 3, 112, 112)

        with torch.no_grad():
            emb = small_model.get_embedding(batch)

        norms = torch.norm(emb, dim=1)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_embedding_dim_matches(self, small_model):
        """Embedding dimension should match the configured value."""
        batch = torch.randn(2, 3, 112, 112)

        with torch.no_grad():
            emb = small_model.get_embedding(batch)

        assert emb.shape[1] == 128

    def test_different_loss_types(self):
        """Model should work with cosface, arcface, and adaface."""
        from src.downstream.train_downstream import FaceRecModel

        for loss_type in ["cosface", "arcface", "adaface"]:
            model = FaceRecModel(
                backbone_name="convnext_atto",
                n_classes=5,
                loss_type=loss_type,
                embedding_dim=64,
            )
            model.eval()

            batch = torch.randn(2, 3, 112, 112)
            labels = torch.tensor([0, 1])
            logits, emb = model(batch, labels)

            assert logits.shape == (2, 5)
            assert emb.shape == (2, 64)


class TestArcFaceHead:
    """Test the ArcFace/CosFace classification head."""

    def test_cosface_margin_applied(self):
        """CosFace should subtract margin from the correct class logit."""
        from src.downstream.train_downstream import ArcFaceHead

        head = ArcFaceHead(embedding_dim=8, n_classes=3, loss_type="cosface",
                           s=1.0, m=0.4)

        emb = torch.randn(2, 8)
        emb = torch.nn.functional.normalize(emb, dim=1)
        labels = torch.tensor([0, 2])

        logits = head(emb, labels)

        # Without labels
        logits_no_label = head(emb, None)

        # The logit for the correct class should be lower by margin
        for i in range(2):
            target_class = labels[i].item()
            diff = logits_no_label[i, target_class] - logits[i, target_class]
            # diff should be approximately m (0.4) since s=1.0
            assert abs(diff.item() - 0.4) < 1e-5

    def test_arcface_margin_applied(self):
        """ArcFace should apply angular margin."""
        from src.downstream.train_downstream import ArcFaceHead

        head = ArcFaceHead(embedding_dim=8, n_classes=3, loss_type="arcface",
                           s=1.0, m=0.5)

        emb = torch.randn(2, 8)
        emb = torch.nn.functional.normalize(emb, dim=1)
        labels = torch.tensor([0, 1])

        logits_with_labels = head(emb, labels)
        logits_no_labels = head(emb, None)

        # ArcFace modifies the target class logit
        for i in range(2):
            target = labels[i].item()
            # The modified logit should be different from the unmodified
            assert logits_with_labels[i, target] != logits_no_labels[i, target]


class TestCosineSchedule:
    """Test the learning rate scheduler."""

    def test_warmup_starts_at_zero(self):
        """LR should start at 0 during warmup."""
        from src.downstream.train_downstream import CosineSchedule

        sched = CosineSchedule(base_lr=1e-3, total_steps=1000, warmup_frac=0.1)

        assert sched.get_lr(0) == 0.0

    def test_warmup_reaches_base_lr(self):
        """LR should reach base_lr at end of warmup."""
        from src.downstream.train_downstream import CosineSchedule

        sched = CosineSchedule(base_lr=1e-3, total_steps=1000, warmup_frac=0.1)

        lr_at_warmup_end = sched.get_lr(100)  # warmup_steps = 100
        assert abs(lr_at_warmup_end - 1e-3) < 1e-7

    def test_cosine_decay_to_zero(self):
        """LR should decay to ~0 at the end of training."""
        from src.downstream.train_downstream import CosineSchedule

        sched = CosineSchedule(base_lr=1e-3, total_steps=1000, warmup_frac=0.1)

        lr_at_end = sched.get_lr(999)
        assert lr_at_end < 1e-5

    def test_lr_monotonically_decreasing_after_warmup(self):
        """After warmup, LR should monotonically decrease."""
        from src.downstream.train_downstream import CosineSchedule

        sched = CosineSchedule(base_lr=1e-3, total_steps=1000, warmup_frac=0.1)

        prev_lr = sched.get_lr(100)
        for step in range(101, 1000):
            lr = sched.get_lr(step)
            assert lr <= prev_lr + 1e-10
            prev_lr = lr

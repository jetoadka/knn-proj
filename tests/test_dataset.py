"""Tests for FolderFaceDataset (flat and nested layouts)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _create_dummy_image(path: Path, size=(112, 112)):
    """Create a random solid-color JPEG image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(path), "JPEG")


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def nested_dataset_dir(tmp_path: Path) -> Path:
    """Create a folder-per-identity dataset.

    Structure:
        tmp_path/
        ├── alice/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── bob/
        │   └── img1.jpg
        └── carol/
            ├── img1.jpg
            ├── img2.jpg
            └── img3.jpg
    """
    for name, count in [("alice", 2), ("bob", 1), ("carol", 3)]:
        for i in range(count):
            _create_dummy_image(tmp_path / name / f"img{i}.jpg")
    return tmp_path


@pytest.fixture
def flat_dataset_dir(tmp_path: Path) -> Path:
    """Create a flat (WebFace4M-style) dataset.

    Structure:
        tmp_path/
        ├── 008633_018.jpg
        ├── 008633_019.jpg
        ├── 013254_017.jpg
        └── 013254_018.jpg
    """
    for identity in ["008633", "013254"]:
        for seq in range(2):
            _create_dummy_image(tmp_path / f"{identity}_{seq:03d}.jpg")
    return tmp_path


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


def test_nested_dataset_loads(nested_dataset_dir: Path):
    """FolderFaceDataset should detect nested layout and load all images."""
    from src.downstream.train_downstream import FolderFaceDataset

    ds = FolderFaceDataset(nested_dataset_dir)

    assert len(ds) == 6  # alice(2) + bob(1) + carol(3)
    assert ds.n_classes == 3


def test_nested_dataset_labels(nested_dataset_dir: Path):
    """Each image should have the correct label for its identity folder."""
    from src.downstream.train_downstream import FolderFaceDataset

    ds = FolderFaceDataset(nested_dataset_dir)

    labels_by_identity: dict[str, set] = {}
    for path, label in ds.samples:
        identity = path.parent.name
        labels_by_identity.setdefault(identity, set()).add(label)

    # Each identity should map to exactly one label
    for identity, label_set in labels_by_identity.items():
        assert len(label_set) == 1, f"{identity} has multiple labels: {label_set}"

    # All labels should be distinct
    all_labels = [next(iter(s)) for s in labels_by_identity.values()]
    assert len(set(all_labels)) == 3


def test_flat_dataset_loads(flat_dataset_dir: Path):
    """FolderFaceDataset should detect flat layout and parse identity from filename."""
    from src.downstream.train_downstream import FolderFaceDataset

    ds = FolderFaceDataset(flat_dataset_dir)

    assert len(ds) == 4
    assert ds.n_classes == 2


def test_flat_dataset_labels(flat_dataset_dir: Path):
    """Flat layout: identity is everything before the last underscore."""
    from src.downstream.train_downstream import FolderFaceDataset

    ds = FolderFaceDataset(flat_dataset_dir)

    labels = [label for _, label in ds.samples]
    # Two images per identity → [a, a, b, b] or similar
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_dataset_getitem_returns_tensor(nested_dataset_dir: Path):
    """__getitem__ should return (image_tensor, label) when transform is provided."""
    from torchvision import transforms

    from src.downstream.train_downstream import FolderFaceDataset

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    ds = FolderFaceDataset(nested_dataset_dir, transform=transform)

    img, label = ds[0]
    assert img.shape == (3, 112, 112)
    assert isinstance(label, int)


def test_empty_directory(tmp_path: Path):
    """Dataset should handle empty directories gracefully."""
    from src.downstream.train_downstream import FolderFaceDataset

    ds = FolderFaceDataset(tmp_path)
    assert len(ds) == 0
    assert ds.n_classes == 0

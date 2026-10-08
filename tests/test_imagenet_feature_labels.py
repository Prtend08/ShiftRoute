from pathlib import Path

import pytest
from torchvision import datasets

from scripts.extract_pretrained_imagenet_features import restore_numeric_folder_labels


def make_numeric_image_folder(root: Path, classes: int) -> None:
    for index in range(classes):
        directory = root / str(index)
        directory.mkdir(parents=True)
        (directory / "sample.JPEG").write_bytes(b"not-decoded-during-indexing")


def test_numeric_folders_map_to_classifier_indices(tmp_path):
    make_numeric_image_folder(tmp_path, 11)
    dataset = datasets.ImageFolder(tmp_path)
    assert dataset.class_to_idx["10"] != 10  # demonstrates lexicographic default
    restore_numeric_folder_labels(dataset)
    assert dataset.class_to_idx == {str(index): index for index in range(11)}
    assert all(target == int(Path(path).parent.name) for path, target in dataset.samples)


def test_numeric_folders_must_be_contiguous(tmp_path):
    for name in ("0", "2"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "sample.JPEG").write_bytes(b"placeholder")
    dataset = datasets.ImageFolder(tmp_path)
    with pytest.raises(ValueError, match="contiguous"):
        restore_numeric_folder_labels(dataset)

from pathlib import Path

import pytest
import pandas as pd

from src.data.AmazonReviewTokenized import AmazonReviewTokenized, ToTensor


def test_constructor():
    dataset = AmazonReviewTokenized(data_path='tests/test_files/', transform=ToTensor())
    assert type(dataset) == AmazonReviewTokenized
    assert type(dataset.transform) == ToTensor


def test_len():
    dataset = AmazonReviewTokenized(data_path='tests/test_files/')
    assert len(dataset) == 100


def test_getitem():
    dataset = AmazonReviewTokenized(data_path='tests/test_files/', transform=ToTensor())

    item = next(iter(dataset))
    assert item is not None
    assert item['labels'].item() == 0

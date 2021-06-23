from pathlib import Path

import pytest
import pandas as pd

from src.data.AmazonReviewTokenized import AmazonReviewTokenized, ToTensor


@pytest.fixture(autouse=True)
def mock_read_json(monkeypatch):
    project_dir = Path(__file__).resolve().parents[2]

    def load_data():
        return pd.read_json(project_dir.joinpath('tests/test_files/test.json'))

    data = load_data()
    monkeypatch.setattr(pd, 'read_json', lambda x: data)


def test_constructor():
    dataset = AmazonReviewTokenized(data_path='tests/test_files/', transform=ToTensor())
    assert type(dataset) == AmazonReviewTokenized
    assert type(dataset.transform) == ToTensor


def test_len():
    dataset = AmazonReviewTokenized(data_path='tests/test_files/')
    assert len(dataset) == 50000


def test_getitem():
    dataset = AmazonReviewTokenized(data_path='tests/test_files/', transform=ToTensor())

    item = next(iter(dataset))
    assert item is not None
    assert item['labels'].item() == 0

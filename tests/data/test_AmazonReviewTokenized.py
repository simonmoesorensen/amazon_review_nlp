import pytest
import pandas as pd

from src.data.AmazonReviewTokenized import AmazonReviewTokenized, ToTensor


@pytest.fixture(autouse=True)
def mock_read_json(monkeypatch):
    def load_data():
        return pd.read_json('../test_files/test.json')

    data = load_data()
    monkeypatch.setattr(pd, 'read_json', lambda x: data)


def test_constructor():
    dataset = AmazonReviewTokenized(transform=ToTensor())
    assert type(dataset) == AmazonReviewTokenized
    assert type(dataset.transform) == ToTensor


def test_len():
    dataset = AmazonReviewTokenized()
    assert len(dataset) == 50000


def test_getitem():
    dataset = AmazonReviewTokenized(transform=ToTensor())

    item = next(iter(dataset))
    assert item is not None
    assert item['labels'].item() == 1

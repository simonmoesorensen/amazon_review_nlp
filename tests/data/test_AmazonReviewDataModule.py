from pathlib import Path

import pandas as pd
import pytest

from src.data.AmazonReviewDataModule import AmazonReviewDataModule


def test_constructor():
    data = AmazonReviewDataModule(data_path='tests/test_files/')
    assert type(data) == AmazonReviewDataModule


@pytest.mark.parametrize("val_size", [0.05, 0.1, 0.4])
def test_setup(val_size):
    data = AmazonReviewDataModule(data_path='tests/test_files/', val_size=val_size)
    assert data.test_set.data.columns.to_list() == ['input_ids', 'attention_mask', 'labels']
    assert data.train_set.data.columns.to_list() == ['input_ids', 'attention_mask', 'labels']
    assert len(data.test_set) == 100
    assert len(data.train_sampler) == int(len(data.train_set) * (1 - data.val_size))


def test_train_dataloader():
    data = AmazonReviewDataModule(data_path='tests/test_files/')
    data.setup()
    data_loader = data.train_dataloader()
    assert list(next(iter(data_loader)).keys()) == ['input_ids', 'attention_mask', 'labels']


def test_val_dataloader():
    data = AmazonReviewDataModule(data_path='tests/test_files/')
    data.setup()
    data_loader = data.val_dataloader()
    assert list(next(iter(data_loader)).keys()) == ['input_ids', 'attention_mask', 'labels']


def test_test_dataloader():
    data = AmazonReviewDataModule(data_path='tests/test_files/', val_size=0.2)
    data.setup()
    data_loader = data.test_dataloader()
    assert list(next(iter(data_loader)).keys()) == ['input_ids', 'attention_mask', 'labels']


@pytest.mark.parametrize("batch_size", [4, 16, 64])
def test_test_dataloader_shape(batch_size):
    data_loader = AmazonReviewDataModule(
        val_size=0.2,
        batch_size=batch_size,
        data_path='tests/test_files/'
    ).test_dataloader()

    batch = next(iter(data_loader))

    assert batch["labels"].shape[0] == batch_size

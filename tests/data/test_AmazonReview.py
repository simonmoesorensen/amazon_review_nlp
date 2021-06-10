import pytest

from src.data.AmazonReview import AmazonReviewFullDataModule


def test_constructor():
    data = AmazonReviewFullDataModule()
    assert type(data) == AmazonReviewFullDataModule


@pytest.mark.parametrize('val_size', [0.05, 0.1, 0.4])
def test_setup(val_size):
    data = AmazonReviewFullDataModule(val_size=val_size)
    data.setup()
    assert len(data.amazon_test) == 650000
    assert len(data.amazon_train) == int(3000000 * (1 - data.val_size))
    assert len(data.amazon_val) == int(3000000 * data.val_size)


def test_train_dataloader():
    data = AmazonReviewFullDataModule(val_size=0.2)
    data.setup()
    data_loader = data.train_dataloader()
    assert len(data_loader.dataset) == 2400000


def test_val_dataloader():
    data = AmazonReviewFullDataModule(val_size=0.2)
    data.setup()
    data_loader = data.val_dataloader()
    assert len(data_loader.dataset) == 600000


def test_test_dataloader():
    data = AmazonReviewFullDataModule(val_size=0.2)
    data.setup()
    data_loader = data.test_dataloader()
    assert len(data_loader.dataset) == 650000

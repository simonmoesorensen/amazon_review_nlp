import pytest
import torch

from src.data.AmazonReviewData import AmazonReviewFullDataModule


def test_constructor():
    data = AmazonReviewFullDataModule()
    assert type(data) == AmazonReviewFullDataModule


@pytest.mark.parametrize("val_size", [0.05, 0.1, 0.4])
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


# @pytest.mark.parametrize("batch_size", [16, 17, 32, 29, 64, 128])
def test_test_dataloader_shape():
    batch_size = 15

    data_loader = AmazonReviewFullDataModule(
        val_size=0.2,
        batch_size=batch_size
    ).test_dataloader()

    batch = next(iter(data_loader))

    max_sequence_length = 128

    labels_shape = torch.Size([batch_size])
    assert batch["labels"].shape == labels_shape

    input_ids_shape = torch.Size([batch_size, max_sequence_length])
    assert batch["input_ids"].shape == input_ids_shape

    token_type_ids_shape = torch.Size([batch_size, max_sequence_length])
    assert batch["token_type_ids"].shape == token_type_ids_shape

    attention_mask_shape = torch.Size([batch_size, max_sequence_length])
    assert batch["attention_mask"].shape == attention_mask_shape

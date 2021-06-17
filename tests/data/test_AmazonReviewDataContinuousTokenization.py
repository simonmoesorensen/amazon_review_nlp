import pytest
import torch

from src.data.AmazonReviewDataContinuousTokenization import AmazonReviewFullDataModule
from transformers import DistilBertTokenizer


@pytest.fixture
def model_name():
    return "distilbert-base-cased"


@pytest.fixture
def data(model_name):
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    data = AmazonReviewFullDataModule(tokenizer)
    return data


def test_constructor(data):
    assert type(data) == AmazonReviewFullDataModule


def test_setup(data):
    assert len(data.amazon_train) == 3600000
    assert len(data.amazon_test) == 400000


def test_train_dataloader(data):
    data_loader = data.train_dataloader()
    assert len(data_loader.dataset) == 3600000


def test_test_dataloader(data):
    data_loader = data.test_dataloader()
    assert len(data_loader.dataset) == 400000


def test_test_dataloader_shape(data):
    data_loader = data.test_dataloader()

    labels, X = next(iter(data_loader))

    max_sequence_length = 128
    batch_size = 32

    labels_shape = torch.Size([batch_size])
    assert labels.shape == labels_shape

    input_ids_shape = torch.Size([batch_size, max_sequence_length])
    assert X["input_ids"].shape == input_ids_shape

    attention_mask_shape = torch.Size([batch_size, max_sequence_length])
    assert X["attention_mask"].shape == attention_mask_shape

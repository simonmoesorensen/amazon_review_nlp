import pytest
import torch

from src.models.distilbertsentimentclassifier import DistilBertSentimentClassifier
from src.data.AmazonReviewDataContinuousTokenization import AmazonReviewFullDataModule
from transformers import DistilBertTokenizer


@pytest.fixture
def model_name():
    return "distilbert-base-cased"


@pytest.fixture
def model(model_name):
    return DistilBertSentimentClassifier(model_name)


@pytest.fixture
def tokenizer(model_name):
    return DistilBertTokenizer.from_pretrained(model_name)


@pytest.fixture
def data(tokenizer):
    return AmazonReviewFullDataModule(tokenizer)


def test_constructor(model):
    assert type(model) == DistilBertSentimentClassifier


def test_model_output_dim(model, data):
    labels, batch = next(iter(data.test_dataloader()))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    batch_size = 32
    n_classes = 2

    assert logits.shape == torch.Size([batch_size, n_classes])


def test_model_predictions(model, data):
    labels, batch = next(iter(data.test_dataloader()))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    preds = model.get_prediction(logits)

    batch_size = 32

    assert preds.shape == torch.Size([batch_size])

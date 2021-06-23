from pathlib import Path

import pytest
import torch
import pandas as pd

from src.models.distilbertsentimentclassifier import DistilBertSentimentClassifier
from src.data.AmazonReviewDataModule import AmazonReviewDataModule
from transformers import DistilBertTokenizer


@pytest.fixture
def model_name():
    return "distilbert-base-uncased"


@pytest.fixture
def model(model_name):
    return DistilBertSentimentClassifier(model_name)


@pytest.fixture
def tokenizer(model_name):
    return DistilBertTokenizer.from_pretrained(model_name)


@pytest.fixture
def data(tokenizer):
    return AmazonReviewDataModule(data_path='tests/test_files/')


def test_constructor(model):
    assert type(model) == DistilBertSentimentClassifier


def test_model_output_dim(model, data):
    batch = next(iter(data.test_dataloader()))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch['labels']

    loss, logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    batch_size = 32
    n_classes = 2

    assert logits.shape == torch.Size([batch_size, n_classes])
    assert loss is not None


def test_model_predictions(model, data):
    batch = next(iter(data.test_dataloader()))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch['labels']

    loss, logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    preds = model.get_prediction(logits)

    batch_size = 32

    assert preds.shape == torch.Size([batch_size])

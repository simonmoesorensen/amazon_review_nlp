from pathlib import Path

import pytest
import torch
import pandas as pd

from src.models.distilbertsentimentclassifier import DistilBertSentimentClassifier
from src.data.AmazonReviewDataModule import AmazonReviewDataModule
from transformers import DistilBertTokenizer


@pytest.fixture(autouse=True)
def mock_read_json(monkeypatch):
    project_dir = Path(__file__).resolve().parents[2]

    def load_data():
        return pd.read_json(project_dir.joinpath('tests/test_files/test.json'))

    data = load_data()
    monkeypatch.setattr(pd, 'read_json', lambda x: data)

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
    return AmazonReviewDataModule()


def test_constructor(model):
    assert type(model) == DistilBertSentimentClassifier


def test_model_output_dim(model, data):
    batch = next(iter(data.test_dataloader()))

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
    batch = next(iter(data.test_dataloader()))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    preds = model.get_prediction(logits)

    batch_size = 32

    assert preds.shape == torch.Size([batch_size])

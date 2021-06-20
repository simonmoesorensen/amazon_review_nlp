from pathlib import Path

import pytest
import torch
import pandas as pd

from src.models.bertsentimentclassifier import BertSentimentClassifier
from src.data.AmazonReviewDataModule import AmazonReviewDataModule


model = BertSentimentClassifier("bert-base-cased")

data = AmazonReviewDataModule().test_dataloader()


@pytest.fixture(autouse=True)
def mock_read_json(monkeypatch):
    project_dir = Path(__file__).resolve().parents[2]

    def load_data():
        return pd.read_json(project_dir.joinpath('tests/test_files/test.json'))

    data = load_data()
    monkeypatch.setattr(pd, 'read_json', lambda x: data)


def test_constructor():
    assert type(model) == BertSentimentClassifier


def test_bert_output_dim():
    batch = next(iter(data))

    input_ids = batch["input_ids"]
    # token_type_ids = batch["token_type_ids"]
    attention_mask = batch["attention_mask"]

    out = model.bert(
        input_ids=input_ids,
        # token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )

    pooler_output = out.pooler_output

    batch_size = 32
    bert_output_size = 768

    assert pooler_output.shape == torch.Size([batch_size, bert_output_size])


def test_model_output_dim():
    batch = next(iter(data))

    input_ids = batch["input_ids"]
    token_type_ids = batch["token_type_ids"]
    attention_mask = batch["attention_mask"]

    out = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )

    batch_size = 32
    n_classes = 5

    assert out.shape == torch.Size([batch_size, n_classes])


def test_model_predictions():
    batch = next(iter(data))

    input_ids = batch["input_ids"]
    token_type_ids = batch["token_type_ids"]
    attention_mask = batch["attention_mask"]

    out = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )
    preds = model.get_prediction(out)

    batch_size = 32

    assert preds.shape == torch.Size([batch_size])

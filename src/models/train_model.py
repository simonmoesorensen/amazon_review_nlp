from src.data.AmazonReviewDataContinuousTokenization import AmazonReviewFullDataModule
from src.models.distilbertsentimentclassifier import DistilBertSentimentClassifier
from transformers import DistilBertTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch


def collate_fn(batch):
    labels = torch.stack([x["labels"] for x in batch], axis=0)
    input_ids = torch.stack([x["input_ids"] for x in batch], axis=0)
    token_type_ids = torch.stack([x["token_type_ids"] for x in batch], axis=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], axis=0)
    return {
        "labels": labels,
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }


model_name = "distilbert-base-cased"
model = DistilBertSentimentClassifier(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

data = AmazonReviewFullDataModule(tokenizer)

labels, x = next(iter(data.train_dataloader()))
out = model(**x)

trainer_params = {
    "gpus": 0,
    "max_epochs": 3,
    # "precision": 16,
    "progress_bar_refresh_rate": 20,
    "log_every_n_steps": 10,
    "callbacks": [
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=5,
            verbose=False,
            mode="min"
        ),
        ModelCheckpoint(
            dirpath="models/BertSentimentClassifier/checkpoints/weights",
            verbose=True,
            monitor="val_loss",
            mode="min",
        ),
    ],
}

trainer = pl.Trainer(**trainer_params)
trainer.fit(
    model,
    train_dataloader=data.train_dataloader(),
    val_dataloaders=data.test_dataloader(),
)

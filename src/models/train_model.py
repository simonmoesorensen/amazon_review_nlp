from src.data.AmazonReviewNew import AmazonReviewFullDataModule
from src.models.bertsentimentclassifier import BertSentimentClassifier
from transformers import BertTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch


def collate_fn(batch):
    labels = torch.stack([x["labels"] for x in batch],
                         axis=0)
    input_ids = torch.stack([x["input_ids"] for x in batch],
                            axis=0)
    token_type_ids = torch.stack([x["token_type_ids"] for x in batch],
                                 axis=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch],
                                 axis=0)
    return {
        "labels": labels,
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }


model_name = "bert-base-cased"
model = BertSentimentClassifier(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

data = AmazonReviewFullDataModule(tokenizer)

trainer_params = {
        'gpus': 0,
        'max_epochs': 3,
        'progress_bar_refresh_rate': 20,
        'log_every_n_steps': 10,
        'logger': WandbLogger(save_dir="lightning_logs/"),
        'callbacks': [
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.00,
                patience=5,
                verbose=False,
                mode='min'
                ),
            ModelCheckpoint(
                dirpath='models/BertSentimentClassifier/checkpoints/weights',
                verbose=True,
                monitor='val_loss',
                mode='min'
                ),
        ]
    }

trainer = pl.Trainer(**trainer_params)
trainer.fit(
    model,
    train_dataloader=data.train_dataloader(),
    val_dataloaders=data.val_dataloader()
    )

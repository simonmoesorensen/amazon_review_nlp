from src.data.AmazonReviewDataContinuousTokenization import AmazonReviewFullDataModule
from src.models.distilbertsentimentclassifier import DistilBertSentimentClassifier
from transformers import DistilBertTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


model_name = "distilbert-base-cased"
model = DistilBertSentimentClassifier(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

data = AmazonReviewFullDataModule(tokenizer)

trainer_params = {
    "gpus": 0,
    "max_epochs": 3,
    # "precision": 16,  # only on GPU
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

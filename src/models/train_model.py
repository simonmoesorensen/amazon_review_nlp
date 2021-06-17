from src.data.AmazonReviewDataContinuousTokenization import AmazonReviewFullDataModule
from src.models.distilbertsentimentclassifier import DistilBertSentimentClassifier
from src.models.AzureMLLogger import AzureMLLogger
from transformers import DistilBertTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import argparse


def main():
    args = parse_args()
    train_model(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Amazon review sentiment classification task"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        metavar="N",
        help="number of GPUs (default: 0)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to train (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="batch size (default: 128)",
    )
    parser.add_argument(
        "--azure",
        action="store_true",
        default=False,
        help="Run from Azure.",
    )
    args = parser.parse_args()
    return args


def train_model(args):
    model_name = "distilbert-base-cased"
    model = DistilBertSentimentClassifier(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    data = AmazonReviewFullDataModule(tokenizer, batch_size=args.batch_size)

    trainer_params = {
        "gpus": args.gpus,
        "max_epochs": args.epochs,
        "precision": 16 if args.gpus > 0 else 32,
        "progress_bar_refresh_rate": 20,
        "log_every_n_steps": 10,
        "logger": AzureMLLogger() if args.azure else None,
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
        train_dataloader=data.test_dataloader(),
        val_dataloaders=data.test_dataloader(),
    )

if __name__ == "__main__":
    main()

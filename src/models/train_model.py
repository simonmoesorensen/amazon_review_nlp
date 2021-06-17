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
    # parser.add_argument(
    #     "--batch-size",
    #     type=int,
    #     default=64,
    #     metavar="N",
    #     help="input batch size for training (default: 64)",
    # )
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
    # parser.add_argument(
    #     "--lr",
    #     type=float,
    #     default=1.0,
    #     metavar="LR",
    #     help="learning rate (default: 1.0)",
    # )
    # parser.add_argument(
    #     "--dry-run",
    #     action="store_true",
    #     default=False,
    #     help="quickly check a single pass",
    # )
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=1,
    #     metavar="S",
    #     help="random seed (default: 1)"
    # )
    # parser.add_argument(
    #     "--log-interval",
    #     type=int,
    #     default=10,
    #     metavar="N",
    #     help="batches to wait before logging training",
    # )
    # parser.add_argument(
    #     "--no-test",
    #     action="store_false",
    #     default=True,
    #     help="run a test pass after training",
    # )
    # parser.add_argument(
    #     "--logger",
    #     type=str,
    #     default="tensorboard",
    #     metavar="N",
    #     help="logger to use (wandb or tensorboard)",
    # )
    args = parser.parse_args()
    return args


def train_model(args):
    model_name = "distilbert-base-cased"
    model = DistilBertSentimentClassifier(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    data = AmazonReviewFullDataModule(tokenizer, batch_size=128)

    logger = AzureMLLogger()

    trainer_params = {
        "gpus": args.gpus,
        "max_epochs": args.epochs,
        "precision": 16 if args.gpus > 0 else 32,
        "progress_bar_refresh_rate": 20,
        "log_every_n_steps": 10,
        "logger": logger,
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

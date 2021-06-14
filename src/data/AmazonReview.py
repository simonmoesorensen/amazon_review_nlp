# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

from torch.utils.data import random_split, DataLoader
from torchtext.datasets import AmazonReviewFull
import pytorch_lightning as pl

project_dir = Path(__file__).resolve().parents[2]


class AmazonReviewFullDataModule(pl.LightningDataModule):
    def __init__(
        self,
        val_size: float = 0.2,
        data_dir: str = project_dir.joinpath("data"),
        batch_size: int = 32,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size

    def setup(self, stage: Optional[str] = None):
        self.amazon_test = AmazonReviewFull(self.data_dir, split="test")
        amazon_full = AmazonReviewFull(self.data_dir, split="train")

        val_size = int(len(amazon_full) * self.val_size)
        train_size = int(len(amazon_full) - val_size)
        self.amazon_train, self.amazon_val = random_split(
            amazon_full, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(self.amazon_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.amazon_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.amazon_test, batch_size=self.batch_size)


if __name__ == "__main__":
    amazon = AmazonReviewFull(project_dir.joinpath("data"), split="train")
    print(amazon)

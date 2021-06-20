# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
import numpy as np

from src.data.AmazonReviewTokenized import AmazonReviewTokenized, ToTensor

project_dir = Path(__file__).resolve().parents[2]


class AmazonReviewDataModule(pl.LightningDataModule):
    def __init__(
            self,
            val_size: float = 0.2,
            batch_size: int = 32,
            num_workers: int = 1
    ):
        super().__init__()
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers

        train_path = "data/processed/train.json"
        test_path = "data/processed/test.json"
        if (not project_dir.joinpath(train_path).exists() or
                not project_dir.joinpath(test_path).exists()):
            raise FileNotFoundError('Could not find processed data. '
                                    'Did you run make_dataset.py and '
                                    'build_features.py?')

        self.setup()
        self.train_sampler, self.valid_sampler = self._get_samplers()

    def _get_samplers(self):
        n = len(self.train_set)
        indices = list(range(n))
        val_size = self.val_size
        split = int(np.floor(val_size * n))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        return train_sampler, valid_sampler

    def setup(self, stage: Optional[str] = None):
        print('Setting up train and test set')
        transform = ToTensor()

        self.train_set = AmazonReviewTokenized(train=True,
                                               transform=transform)
        self.test_set = AmazonReviewTokenized(train=False,
                                              transform=transform)
        print('Finished setup')

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=self.valid_sampler,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


if __name__ == '__main__':
    datamodule = AmazonReviewDataModule(num_workers=1)

    i = 0
    train_dataloader = datamodule.train_dataloader()
    valid_dataloader = datamodule.val_dataloader()
    for item in train_dataloader:
        # if i > 10:
        #     break

        # print(item)

        i += 1

    print(i)

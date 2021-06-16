# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchtext.datasets import AmazonReviewPolarity
import pytorch_lightning as pl
import torch
import numpy as np


project_dir = Path(__file__).resolve().parents[2]


class AmazonReviewFullDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        n_train: float = 20000,
        n_val: float = 1000,
        n_test: float = 1000,
        # val_size: float = 0.2,
        data_dir: str = project_dir.joinpath("data"),
        batch_size: int = 32,
        max_seq_length: int = 128,
        load_train: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test

        self.tokenizer = tokenizer

        self.setup()

        # self.train_sampler, self.valid_sampler = self.get_samplers()

    def get_samplers(self):

        n = len(self.amazon_train)
        indices = list(range(n))
        split = int(np.floor(self.n_train + self.n_val))
        np.random.shuffle(indices)

        train_and_val = indices[split:]

        n = len(train_and_val)
        indices = list(range(n))
        split = int(np.floor(self.n_val))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        return train_sampler, valid_sampler

    def setup(self, stage: Optional[str] = None):
        print("Downloading data...")
        self.amazon_train = AmazonReviewPolarity(self.data_dir, split="train")
        self.amazon_test = AmazonReviewPolarity(self.data_dir, split="test")

    def collate_fn(self, batch):

        labels = torch.tensor([x[0] for x in batch])
        sentences = [x[1] for x in batch]

        x = self.tokenize(sentences)

        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        # labels = torch.stack([x["labels"] for x in batch], axis=0)
        # input_ids = torch.stack([x["input_ids"] for x in batch], axis=0)
        # token_type_ids = torch.stack([x["token_type_ids"] for x in batch],
        #                              axis=0)
        # attention_mask = torch.stack([x["attention_mask"] for x in batch],
        #                              axis=0)
        return labels, {
            # "labels": labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def train_dataloader(self):
        return DataLoader(
            self.amazon_train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            # sampler=self.train_sampler,
            # num_workers=16,
        )

    # def valid_dataloader(self):
    #     return DataLoader(
    #         self.amazon_train,
    #         batch_size=self.batch_size,
    #         collate_fn=self.collate_fn,
    #         sampler=self.valid_sampler,
    #         num_workers=16,
    #     )

    def test_dataloader(self):
        return DataLoader(
            self.amazon_test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=16,
        )

    def tokenize(self, example):
        # label, sentence = example

        encoded_sentence = self.tokenizer(
            example,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
        )

        return {
            "input_ids": encoded_sentence["input_ids"],
            "attention_mask": encoded_sentence["attention_mask"],
        }

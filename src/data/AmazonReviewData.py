# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional
from os.path import exists

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchtext.datasets import AmazonReviewFull
import pytorch_lightning as pl
import torch
import numpy as np


project_dir = Path(__file__).resolve().parents[2]


class AmazonReviewFullDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer=None,
        val_size: float = 0.2,
        
        self.val_size = val_size
        data_dir: str = project_dir.joinpath("data"),
        batch_size: int = 32,
        max_seq_length: int = 128,
        load_train: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.max_seq_length = max_seq_length

        self.tokenizer = tokenizer

        self.processed_data_path = "data/processed/"

        train_path = "data/processed/train.pt"
        test_path = "data/processed/test.pt"
        if not exists(train_path) or not exists(test_path):
            self.setup()

        self.tokenized_test = torch.load(test_path)

        if load_train:
            self.tokenized_train = torch.load(train_path)
            self.train_sampler, self.valid_sampler = self.get_samplers()

    def get_samplers(self):
        n = len(self.tokenized_train)
        indices = list(range(n))
        val_size = self.val_size
        split = int(np.floor(val_size * n))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        return train_sampler, valid_sampler

    def setup(self, stage: Optional[str] = None):
        print("Downloading data...")
        self.amazon_train = AmazonReviewFull(self.data_dir, split="train")
        self.amazon_test = AmazonReviewFull(self.data_dir, split="test")

        print("Tokenizing training set...")
        self.tokenize_all_and_save(self.amazon_train, "train.pt")

        # TODO: host from cloud and download from there first
        # TODO: store dataset parameters batch_size and max_seq_length in
        # dictionary in saved tensor, compare when downloading
        # TODO: Compare parameters for data loading with respect to performance

        print("Tokenizing test set...")
        self.tokenize_all_and_save(self.amazon_test, "test.pt")

    def collate_fn(self, batch):
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

    def tokenize_all_and_save(self, data, filename):
        dataloader = DataLoader(
            data,
            batch_size=512,
        )

        data = [self.tokenize(x) for x in dataloader]

        torch.save(data, self.processed_data_path + filename)

    def train_dataloader(self):
        return DataLoader(
            self.tokenized_train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            sampler=self.train_sampler,
            num_workers=16,
        )

    def valid_dataloader(self):
        return DataLoader(
            self.tokenized_train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            sampler=self.valid_sampler,
            num_workers=16,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tokenized_test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=16,
        )

    def tokenize(self, example):
        label, sentence = example

        encoded_sentence = self.tokenizer(
            sentence,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        return {
            "label": label,
            "input_ids": encoded_sentence["input_ids"],
            "token_type_ids": encoded_sentence["token_type_ids"],
            "attention_mask": encoded_sentence["attention_mask"],
        }

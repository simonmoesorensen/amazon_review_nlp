# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
from torchtext.datasets import AmazonReviewPolarity
import pytorch_lightning as pl
import torch


project_dir = Path(__file__).resolve().parents[2]


class AmazonReviewFullDataModule(pl.LightningDataModule):
    def __init__(self,
                 tokenizer,
                 val_size: float = 0.2,
                 data_dir: str = project_dir.joinpath('data'),
                 batch_size: int = 32,):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size

        self.tokenizer = tokenizer

        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.amazon_train = AmazonReviewPolarity(self.data_dir, split='train')
        self.amazon_test = AmazonReviewPolarity(self.data_dir, split='test')

        processed_data_path = "data/processed/"

        tokenized = self.tokenize_all(self.amazon_train)
        torch.save(tokenized, processed_data_path + 'train.pt')

    def collate_fn(self, batch):
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

    def tokenize_all(self, data):
        dl1 = DataLoader(
            data,
            batch_size=512,
        )

        data = [self.tokenize(x) for x in dl1]
        labels = [x[0] for x in data]
        input_ids = [x[1]["input_ids"] for x in data]
        token_type_ids = [x[1]["token_type_ids"] for x in data]
        attention_mask = [x[1]["attention_mask"] for x in data]

        labels = torch.cat(labels, axis=0)
        input_ids = torch.cat(input_ids, axis=0)
        token_type_ids = torch.cat(token_type_ids, axis=0)
        attention_mask = torch.cat(attention_mask, axis=0)

        data = {
            "labels": labels,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }
        return data

    def train_dataloader(self):
        return DataLoader(
            self.amazon_train,
            batch_size=self.batch_size,
            # collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.amazon_test,
            batch_size=self.batch_size,
            # collate_fn=self.collate_fn
        )

    def tokenize(self, example):
        label, sentence = example

        encoded_sentence = self.tokenizer(
            sentence,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        return (label, encoded_sentence)


if __name__ == '__main__':
    amazon = AmazonReviewFullDataModule()
    labels, sentences = next(iter(amazon.train_dataloader()))
    print()

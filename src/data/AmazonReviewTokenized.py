from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

project_dir = Path(__file__).resolve().parents[2]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_ids, attention_mask, label = sample['input_ids'], sample['attention_mask'], sample['labels']

        return {'input_ids': torch.from_numpy(np.array(input_ids)),
                'attention_mask': torch.from_numpy(np.array(attention_mask)),
                'label': torch.from_numpy(np.array(label))}


class AmazonReviewTokenized(Dataset):
    """Amazon Review dataset with a tokenizer"""

    def __init__(self, train=True, transform=None):
        name = 'train' if train else 'test'
        self.data = pd.read_json(project_dir.joinpath(
            f'data/processed/{name}.json'))

        self.transfrom = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx, :]

        if self.transfrom:
            sample = self.transfrom(sample)

        return sample
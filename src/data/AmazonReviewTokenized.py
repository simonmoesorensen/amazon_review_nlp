from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

project_dir = Path(__file__).resolve().parents[2]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_ids, attention_mask, label, token_type_ids = sample['input_ids'], sample['attention_mask'], sample['labels'], sample['token_type_ids']

        return {'input_ids': torch.from_numpy(np.array(input_ids)),
                'attention_mask': torch.from_numpy(np.array(attention_mask)),
                'labels': torch.from_numpy(np.array(label)),
                'token_type_ids': torch.from_numpy(np.array(token_type_ids))}


class AmazonReviewTokenized(Dataset):
    """Amazon Review dataset with a tokenizer"""

    def __init__(self,
                 train: bool = True,
                 transform=None,
                 data_path=None):
        name = 'train' if train else 'test'

        if not data_path:
            file_path = project_dir.joinpath(f'{data_path + name}.json')
        else:
            file_path = Path(data_path).joinpath(f'{name}.json')

        if not file_path.exists():
            raise FileNotFoundError(f'Could not find the file: {file_path}')

        self.data = pd.read_json(file_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx, :]

        if self.transform:
            sample = self.transform(sample)

        return sample

"""
Tokenize the raw dataset
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import transformers
from transformers import DistilBertTokenizer

from src.data.make_dataset import data_exists

# Define paths
project_dir = Path(__file__).resolve().parents[2]
raw_dir = project_dir.joinpath(
    'data/raw/AmazonReviewFull/amazon_review_full_csv/')
processed_dir = project_dir.joinpath(
    'data/processed/'
)


def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize the AmazonReview dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='distilbert-base-uncased',
        help="Model name from: https://huggingface.co/models"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Maximum length of a sequence"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Size of batch to tokenize pr iteration"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=.025e6,  # 3.5e6,
        help="Ability to tokenize a subset of the total data"
    )
    args = parser.parse_args(args)
    return args


def tokenize(tokenizer: transformers.PreTrainedTokenizer,
             sentence: List[str],
             max_seq_length: int) -> Dict:
    encoded_sentence = tokenizer(
        sentence,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
    )

    return {
        "input_ids": encoded_sentence["input_ids"],
        "attention_mask": encoded_sentence["attention_mask"],
    }


def tokenize_data(args: argparse.Namespace,
                  data: pd.io.parsers.TextFileReader) -> Dict:
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)

    data_dict = {'input_ids': [],
                 'attention_mask': [],
                 'labels': []}

    parsed_rows = 0
    for batch in data:
        print(f'...Tokenizing batch (parsed_rows={parsed_rows})')
        labels = batch.rating.to_list()

        tokenized_values = tokenize(tokenizer,
                                    batch.sentence.to_list(),
                                    args.max_seq_length)

        data_dict['labels'] += labels
        data_dict['input_ids'] += tokenized_values['input_ids']
        data_dict['attention_mask'] += tokenized_values['attention_mask']
        parsed_rows += len(batch)

        if parsed_rows >= args.max_rows:
            print(f'Reached maximum size of {args.max_rows}. '
                  f'Stopping tokenization')
            break

    return data_dict


def save_dict(data: Dict, file_name: str) -> None:
    file_path = processed_dir.joinpath(file_name)
    print(f'Saving data to {file_path}')
    with open(file_path, 'x') as file:
        json.dump(data, file)
    print('File saved')


def main():
    args = parse_args(sys.argv[1:])

    print('Tokenizing dataset...')
    print(f'Data exists: {data_exists()}')
    if not data_exists():
        print('Dataset is not downloaded. \n'
              'Please run "src/data/make_dataset.py"')

    print('Tokenizing train data...')

    train_data = pd.read_csv(raw_dir.joinpath('train.csv'),
                             chunksize=args.chunk_size,
                             names=['rating', 'title', 'sentence'])

    tokenized_data = tokenize_data(args, train_data)
    save_dict(tokenized_data, 'train.json')

    print('Tokenizing test data...')
    train_data = pd.read_csv(raw_dir.joinpath('test.csv'),
                             chunksize=args.chunk_size,
                             names=['rating', 'title', 'sentence'])

    tokenized_data = tokenize_data(args, train_data)
    save_dict(tokenized_data, 'test.json')

    print('Dataset tokenized')


if __name__ == '__main__':
    main()

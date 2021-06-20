"""
Download the AmazonReviewFull dataset from the torchtext.dataset library
and insert the untokenized data into {root}/data/raw
"""

import logging
from pathlib import Path

from torchtext.datasets import AmazonReviewFull

project_dir = Path(__file__).resolve().parents[2]


def data_exists() -> bool:
    """Check if the data is downloaded"""
    return project_dir.joinpath(
        'data/raw/AmazonReviewFull/amazon_review_full_csv/test.csv'
    ).is_file() and project_dir.joinpath(
        'data/raw/AmazonReviewFull/amazon_review_full_csv/train.csv'
    ).is_file()


def download_data() -> None:
    AmazonReviewFull(project_dir.joinpath('data/raw'))


def main():
    logging.info('Making raw dataset...')
    logging.info(f'Data exists: {data_exists()}')
    if not data_exists():
        logging.info('Downloading data')
        download_data()
    logging.info('Raw dataset created')


if __name__ == '__main__':
    main()

"""
Download the AmazonReviewFull dataset from the torchtext.dataset library
and insert the untokenized data into {root}/data/raw
"""

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
    print('Making raw dataset...')

    print('Creating data folder with raw and processed if they dont exist')
    project_dir.joinpath('data').mkdir(parents=True, exist_ok=True)
    project_dir.joinpath('data/raw/').mkdir(parents=True, exist_ok=True)
    project_dir.joinpath('data/processed/').mkdir(parents=True,
                                                  exist_ok=True)

    print(f'Amazon data exists: {data_exists()}')
    if not data_exists():
        print('Downloading data')
        download_data()

    print('Raw dataset created')


if __name__ == '__main__':
    main()

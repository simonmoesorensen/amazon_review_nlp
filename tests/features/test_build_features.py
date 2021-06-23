import pandas as pd
from transformers import DistilBertTokenizer

from src.features.build_features import parse_args, tokenize, tokenize_data


def test_parse_args():
    args = parse_args(['--chunk-size', '10'])
    assert args.chunk_size == 10
    assert args.max_seq_length == 256


def test_tokenize():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    tokenized = tokenize(tokenizer, ['hey how are you', 'I am good'], 10)
    print(tokenized)
    assert tokenized['input_ids'][0] == [101, 4931, 2129, 2024, 2017, 102, 0, 0, 0, 0]
    assert tokenized['input_ids'][1] == [101, 1045, 2572, 2204, 102, 0, 0, 0, 0, 0]
    assert list(tokenized.keys()) == ['input_ids', 'attention_mask']


def test_tokenize_data():
    sub_batch = [pd.DataFrame({'rating': [0, 1], 'sentence': ['hey how are you', 'I am good']})]
    args = parse_args(['--chunk-size', '1',
                       '--max-rows', '2'])

    data_dict = tokenize_data(args, sub_batch)
    assert data_dict['input_ids'][0][0:6] == [101, 4931, 2129, 2024, 2017, 102]
    assert data_dict['input_ids'][1][0:6] == [101, 1045, 2572, 2204, 102, 0]
from src.data.AmazonReviewNew import AmazonReviewFullDataModule
from transformers import BertTokenizer

model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
data = AmazonReviewFullDataModule(tokenizer)

# from pathlib import Path
# from torch.utils.data import DataLoader
# from torchtext.datasets import AmazonReviewPolarity
# import torch


# def tokenize_all(self, data):
#     dl1 = DataLoader(
#         data,
#         batch_size=512,
#     )

#     data = [self.tokenize(x) for x in dl1]
#     labels = [x[0] for x in data]
#     input_ids = [x[1]["input_ids"] for x in data]
#     token_type_ids = [x[1]["token_type_ids"] for x in data]
#     attention_mask = [x[1]["attention_mask"] for x in data]

#     labels = torch.cat(labels, axis=0)
#     input_ids = torch.cat(input_ids, axis=0)
#     token_type_ids = torch.cat(token_type_ids, axis=0)
#     attention_mask = torch.cat(attention_mask, axis=0)

#     data = {
#         "labels": labels,
#         "input_ids": input_ids,
#         "token_type_ids": token_type_ids,
#         "attention_mask": attention_mask
#     }
#     return data


# project_dir = Path(__file__).resolve().parents[2]
# data_dir = project_dir.joinpath('data')

# amazon_train = AmazonReviewPolarity(data_dir, split='train')
# amazon_test = AmazonReviewPolarity(data_dir, split='test')

# processed_data_path = "data/processed/"

# tokenized = tokenize_all(amazon_train)
# torch.save(tokenized, processed_data_path + 'train.pt')

# data = tokenized
# data_transposed = [{
#             "labels": data["labels"][i],
#             "input_ids": data["input_ids"][i],
#             "token_type_ids": data["token_type_ids"][i],
#             "attention_mask": data["attention_mask"][i],
#             } for i in range(len(data["labels"]))]
# torch.save(data_transposed, processed_data_path + "train2.pt")

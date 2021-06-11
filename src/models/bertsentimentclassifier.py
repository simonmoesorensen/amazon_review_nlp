from torch import nn, optim
from transformers import BertModel
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class BertSentimentClassifier(pl.LightningModule):
    def __init__(self, model_name):
        super(BertSentimentClassifier, self).__init__()
        n_classes = 2

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
            )

        output = self.dropout(out.pooler_output)
        return self.out(output)

    def step(self, batch):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]-1

        logits = self(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        probs = F.softmax(logits, dim=1)
        loss = self.criterion(probs, labels)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_prediction(logits: torch.Tensor):
        _, preds = torch.max(logits, dim=1)
        return preds

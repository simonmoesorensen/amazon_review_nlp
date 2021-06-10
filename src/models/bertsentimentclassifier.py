from torch import nn, optim
from transformers import BertModel
import pytorch_lightning as pl
import torch


class BertSentimentClassifier(pl.LightningModule):
    def __init__(self, n_classes):
        super(BertSentimentClassifier, self).__init__()

        PRE_TRAINED_MODEL_NAME = "bert-base-cased"
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
        output = self.dropout(pooled_output)
        return self.out(output)

    def step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(logits, dim=1)
        loss = self.criterion(logits, targets)

        return loss

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.step(batch)
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.step(batch)
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.step(batch)
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

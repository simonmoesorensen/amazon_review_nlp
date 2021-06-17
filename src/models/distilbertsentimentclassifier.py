from torch import nn, optim
from transformers import DistilBertForSequenceClassification  # Only 2 classes
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class DistilBertSentimentClassifier(pl.LightningModule):
    def __init__(self, model_name):
        super(DistilBertSentimentClassifier, self).__init__()

        self.bert = DistilBertForSequenceClassification\
            .from_pretrained(model_name)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return out.logits

    def step(self, batch):
        labels, tokens = batch
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        labels = labels - 1

        logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probs = F.softmax(logits, dim=1)
        loss = self.criterion(probs, labels)

        preds = self.get_prediction(logits)
        accuracy = self.flat_accuracy(preds, labels)

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.step(batch)
        self.log("train_loss", loss)
        self.log("train_acc", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.step(batch)
        self.log("val_loss", loss)
        self.log("val_acc", accuracy)

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", accuracy)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_prediction(self, logits: torch.Tensor):
        _, preds = torch.max(logits, dim=1)
        return preds

    def flat_accuracy(self, preds, labels):
        pred_flat = preds.flatten()
        labels_flat = labels.flatten()
        return torch.sum(pred_flat == labels_flat) / len(labels_flat)

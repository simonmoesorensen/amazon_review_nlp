import json
import pickle
import joblib
import numpy as np
from azureml.core.model import Model
from torch.nn.functional import softmax
import torch
import transformers
from azureml.core import Workspace


def init():
    global model
    global tokenizer

    authparams = {
        "subscription_id": "99b1be03-6ed4-4d71-90d6-7b927354d06c",
        "resource_group": "mlops",
        "workspace_name": "mlops"
    }
    ws = Workspace(**authparams)

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer_name = model_name + "-tokenizer"

    model_path = Model.get_model_path(model_name, _workspace=ws)
    tokenizer_path = Model.get_model_path(tokenizer_name, _workspace=ws)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)


def run(raw_data):
    data = json.loads(raw_data)['data']
    x = tokenizer(data, return_tensors="pt")
    out = model(**x)
    probs = softmax(out.logits, dim=1)
    pred = torch.argmax(probs, dim=1)
    classnames = ['Negative', 'Positive']
    predicted_classes = classnames[pred]
    return json.dumps(predicted_classes)

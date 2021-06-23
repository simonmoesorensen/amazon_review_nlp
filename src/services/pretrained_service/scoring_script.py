import json
import pickle
import joblib
import numpy as np
from azureml.core.model import Model
from torch.nn.functional import softmax
import torch
import transformers
from azureml.core import Workspace

# Called when the service is loaded
def init():
    # global model
    # global tokenizer
    # ws = Workspace.from_config()
    # model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    # tokenizer_name = model_name + "-tokenizer"
    # # Get the path to the deployed model file and load it
    # model_path = Model.get_model_path(model_name, _workspace=ws)
    # tokenizer_path = Model.get_model_path(tokenizer_name, _workspace=ws)
    # model = joblib.load(model_path)
    # tokenizer = joblib.load(tokenizer_path)

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

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    # data = np.array(json.loads(raw_data)['data'])
    data = json.loads(raw_data)['data']
    # print(data)
    # print(tokenizer)
    # print(torch.__version__)
    # print(transformers.__version__)

    x = tokenizer(data, return_tensors="pt")
    # print(x)
    out = model(**x)
    # print(out)
    probs = softmax(out.logits, dim=1)
    # print(probs)
    pred = torch.argmax(probs, dim=1)
    # print(pred)

    classnames = ['Negative', 'Positive']
    predicted_classes = classnames[pred]
    # print(predicted_classes)
    # predicted_classes = ["NegativeTest"]
    # predicted_classes = []
    # for prediction in predictions:
    #     predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)
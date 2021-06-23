# from azureml.core import Workspace, Webservice
import requests
import json
import argparse


def main():
    args = parse_args()
    text = args.text
    out = call_service(text)

    print(out)


def call_service(text):
    input_json = json.dumps({"data": text})

    # service_output = call_service_directly(input_json, "pretrained-service")
    service_output = call_service_request(input_json, "http://f08440fb-67f2-4b23-bec4-8fda7060fd2d.northeurope.azurecontainer.io/score")

    predicted_classes = json.loads(service_output)
    return predicted_classes


def call_service_directly(input_json, service_name):
    ws = Workspace.from_config()
    service = Webservice(ws, name=service_name)
    prediction = service.run(input_data=input_json)
    return prediction


def call_service_request(input_json, endpoint):
    headers = {'Content-Type': 'application/json'}
    prediction = requests.post(endpoint, input_json, headers=headers)
    return prediction.json()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Amazon review sentiment classification task"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        metavar="N",
        help="input text to classify"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

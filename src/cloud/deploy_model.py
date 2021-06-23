from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.model import Model
from azureml.core import Workspace
import os

ws = Workspace.from_config()

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = ws.models[model_name]

service_path = "src/services/pretrained_service/"
env_file = os.path.join(service_path, "service_env.yml")
script_file = os.path.join(service_path, "scoring_script.py")

# Configure the scoring environment
inference_config = InferenceConfig(runtime= "python",
                                   entry_script=script_file,
                                   conda_file=env_file)

deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service_name = "pretrained-service"

service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

service.wait_for_deployment(True)
print(service.state)

# from azureml.core import Workspace
# ws = Workspace.from_config()
# from azureml.core import Webservice
# Webservice(ws, name="pretrained-service").delete()
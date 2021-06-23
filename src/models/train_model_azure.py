from pathlib import Path

from azureml.core import Workspace, ScriptRunConfig, Experiment, Environment
from azureml.core.authentication import InteractiveLoginAuthentication

project_dir = Path(__file__).resolve().parents[2]

interactive_auth = InteractiveLoginAuthentication(
    tenant_id="f251f123-c9ce-448e-9277-34bb285911d9")

ws = Workspace.from_config()
requirements = project_dir.joinpath("conda_dependencies.yml")

env = Environment.from_conda_specification(name='experiment_env',
                                           file_path=str(requirements))

env.docker.enabled = True
env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04'

compute_target = ws.compute_targets['mlops-sbsosfftt']

datastore = ws.get_default_datastore()
data_ref = datastore.path('amazon-review/data').as_mount()

script_config = ScriptRunConfig(
    source_directory=str(project_dir),
    compute_target=compute_target,
    script='src/models/train_model.py',
    arguments=["--epochs", 2,
               "--gpus", 1,
               "--azure",
               "--batch-size", 128,
               "--data-path", str(data_ref)],
    environment=env
)

script_config.run_config.data_references[data_ref.data_reference_name] = data_ref.to_config()

experiment = Experiment(workspace=ws,
                        name="distilbert-amazon-review-classification")

print('Submitting experiment')
run = experiment.submit(config=script_config)
print('Submitted')

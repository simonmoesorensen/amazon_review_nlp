from pathlib import Path

from azureml.core import Workspace, ScriptRunConfig, Experiment, Environment, Dataset
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
dataset = Dataset.File.from_files(path=(datastore, 'amazon-review/data/'))

command = 'pip install -e . && ' \
          f'python src/models/train_model.py ' \
          f'--epochs 2 --gpus 1 --azure --batch-size 32' \
          f'--data-path {dataset.as_named_input("input").as_mount()}'

script_config = ScriptRunConfig(
    command=command.split(),
    source_directory=str(project_dir),
    compute_target=compute_target,
    # script='src/models/train_model.py',
    # arguments=["--epochs", 2, "--gpus", 1, "--azure"],
    environment=env
)

experiment = Experiment(workspace=ws,
                        name="bert-amazon-review-classification")

print('Submitting experiment')
run = experiment.submit(config=script_config)
print('Submitted')

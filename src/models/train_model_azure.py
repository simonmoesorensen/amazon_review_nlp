from pathlib import Path

from azureml.core import Workspace, ScriptRunConfig, Experiment, Environment
from azureml.core.authentication import InteractiveLoginAuthentication

project_dir = Path(__file__).resolve().parents[2]

interactive_auth = InteractiveLoginAuthentication(
    tenant_id="f251f123-c9ce-448e-9277-34bb285911d9")

ws = Workspace.from_config()
requirements = project_dir.joinpath("requirements.txt")
env = Environment.from_pip_requirements("experiment_env",
                                        file_path=str(requirements))

env.docker.enabled = True
env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04'

compute_target = ws.compute_targets['mlops-sbsosfftt']

command = 'conda install cudetoolkit=11.0 && ' \
          'python -m src/data/make_dataset.py &&' \
          'python -m src/features/build_features.py --max_rows 3500000 &&' \
          'python -m src/models/train_model.py --epochs 2 --gpus 1 --azure'

script_config = ScriptRunConfig(
    command=command,
    source_directory=str(project_dir),
    compute_target=compute_target,
    # arguments=["--epochs", 2, "--gpus", 1, "--azure"],
    environment=env
)

experiment = Experiment(workspace=ws,
                        name="bert-amazon-review-classification")

print('Submitting experiment')
run = experiment.submit(config=script_config)
print('Submitted')

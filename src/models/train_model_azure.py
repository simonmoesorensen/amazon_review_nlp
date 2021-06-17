from azureml.core import Workspace, ScriptRunConfig, Experiment, Environment
from azureml.core.conda_dependencies import CondaDependencies
# from azureml.widgets import RunDetails


ws = Workspace.from_config()
env = Environment.from_pip_requirements("experiment_env", file_path="requirements.txt")

script_config = ScriptRunConfig(
    source_directory=".",
    script="src/models/train_model.py",
    arguments=["--epochs", 2, "--gpus", 1, "--azure"],
    environment=env,
    
)

experiment = Experiment(workspace=ws,
                        name="distilbert-amazon-review-classification")
run = experiment.submit(config=script_config)

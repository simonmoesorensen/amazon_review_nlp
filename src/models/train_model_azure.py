from azureml.core import Workspace, ScriptRunConfig, Experiment, Environment
from azureml.core.conda_dependencies import CondaDependencies
# from azureml.widgets import RunDetails


ws = Workspace.from_config()

env = Environment("experiment_env")

packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults'])
env.python.conda_dependencies = packages


script_config = ScriptRunConfig(
    source_directory="src/models/",
    script="train_model.py",
    arguments=["--epochs", 2, "--gpus", 1],
    environment=env,
)

experiment = Experiment(workspace=ws,
                        name="distilbert-amazon-review-classification")
run = experiment.submit(config=script_config)

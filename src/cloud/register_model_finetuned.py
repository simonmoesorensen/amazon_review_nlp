
from azureml.core import Experiment
from azureml.core import Workspace
import joblib
# from src.models.


# Load the workspace from the saved config file
ws = Workspace.from_config()

# Create an Azure ML experiment in your workspace
experiment = Experiment(workspace=ws, name="register-pretrained-model")
run = experiment.start_logging()
print("Starting experiment:", experiment.name)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name,
                                                            num_labels=2)
# Save the trained model
model_file = model_name + ".pkl"
model_folder = "outputs/"
model_path = model_folder + model_file
joblib.dump(value=model, filename=model_file)
run.upload_file(name=model_path, path_or_stream='./' + model_file)

# Save the tokenizer
tokenizer_name = model_name + "-tokenizer"
tokenizer_file = tokenizer_name + ".pkl"
tokenizer_folder = "outputs/"
tokenizer_path = tokenizer_folder + tokenizer_file
joblib.dump(value=tokenizer, filename=tokenizer_file)
run.upload_file(name=tokenizer_path, path_or_stream='./' + tokenizer_file)

# Complete the run
run.complete()

# Register the model
run.register_model(model_path=model_path,
                   model_name=model_name)

# Register the tokenizer
run.register_model(model_path=tokenizer_path,
                   model_name=tokenizer_name)

print('Pretrained model registered.')

from pathlib import Path

from azureml.core import Workspace

project_dir = Path(__file__).resolve().parents[2]

ws = Workspace.from_config()

datastore = ws.get_default_datastore()
print('Uploading data/processed to blobstorage @ amazon-review/data/')
datastore.upload(src_dir=str(project_dir.joinpath('data/processed')),
                 target_path='amazon-review/data/',
                 overwrite=True)

print('Upload complete')

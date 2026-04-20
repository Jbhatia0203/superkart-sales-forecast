# to upload any folder/files on hf portal
from huggingface_hub import HfApi
# work with local files related to any system like hf
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# upload files in deployment folder - dockerfile,app.py, dependencies to hf space
api.upload_folder(
    folder_path="/content/mlops/deployment",
    repo_id="JaiBhatia020373/mlops",
    repo_type="space",
    path_in_repo=""
)

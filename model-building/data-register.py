from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# target repository: <login_id> / <repo master folder>
repo_id = "JaiBhatia020373" + "/" + "mlops"

# to manage/store text, table or image data
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
  api.repo_info(repo_id=repo_id, repo_type=repo_type)
  print(f"Data repository {repo_id} already exists")
except RepositoryNotFoundError:
    api.create_repo(repo_id=repo_id, repo_type=repo_type)
    print(f"Data repository {repo_id} created successfully")

# upload colab folder contents to hf space
api.upload_folder(
    folder_path="mlops/data",
    repo_id=repo_id,
    repo_type=repo_type)

print("Data uploaded successfully")

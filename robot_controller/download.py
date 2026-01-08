# download from huggingface: nvidia/GR00T-N1.5-3B

from huggingface_hub import snapshot_download
import os

repo_id = "nvidia/GR00T-N1.5-3B"
local_dir = "./models/GR00T-N1.5-3B"

# Create models directory if it doesn't exist
os.makedirs(os.path.dirname(local_dir), exist_ok=True)

print(f"Downloading model {repo_id} to {local_dir}...")
snapshot_download(repo_id, local_dir=local_dir, repo_type="model")

print(f"Model downloaded to {os.path.abspath(local_dir)}")
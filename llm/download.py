from huggingface_hub import hf_hub_download, HfApi
import os, shutil

REPO_ID = "lmsys/vicuna-7b-v1.5"
output_dir = "vicuna-7b-v1.5"
os.makedirs(output_dir, exist_ok=True)

api = HfApi()
files = api.list_repo_files(repo_id=REPO_ID)

for filename in files:
    print(f"Downloading {filename}...")
    file_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    # Copy to flat output_dir
    shutil.copy(file_path, os.path.join(output_dir, os.path.basename(filename)))
    print(f"Saved to {output_dir}/{os.path.basename(filename)}")

print("All files downloaded successfully.")
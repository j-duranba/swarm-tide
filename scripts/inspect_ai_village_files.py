
from huggingface_hub import HfApi

REPO_ID = "aidigestorg/ai-village"

api = HfApi()

print("Inspecting repository (metadata only)...")

files = api.list_repo_tree(
    repo_id=REPO_ID,
    repo_type="dataset",
    recursive=True,
)

for item in files:
    if hasattr(item, "size"):
        size_mb = item.size / (1024 * 1024)
        print(f"{size_mb:10.2f} MB  {item.path}")

print("\nInspection complete. No dataset files downloaded.")

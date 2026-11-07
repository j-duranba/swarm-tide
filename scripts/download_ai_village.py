
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "aidigestorg/ai-village"
OUTPUT_DIR = Path("data/ai-village")

files = [
    "README.md",
    "SCHEMA.md",
    "CHANGELOG.md",
    "events.jsonl.gz",
    "chat_messages.jsonl.gz",
    "computer_use_sessions.jsonl.gz",
    "agents.jsonl.gz",
    "agent_goals.jsonl.gz",
    "village_goals.jsonl.gz",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for filename in files:
    print(f"\nDownloading {filename}...")

    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(OUTPUT_DIR),
        )
        print(f"Saved: {path}")
    except Exception as error:
        print(f"Could not download {filename}: {error}")

print("\nDownload attempt complete.")

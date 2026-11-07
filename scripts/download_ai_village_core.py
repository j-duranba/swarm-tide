
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "aidigestorg/ai-village"
OUTPUT_DIR = Path("data/ai-village")


def should_download(filename: str) -> bool:
    path = PurePosixPath(filename)

    # Exclude images directories in every village:
    # main-village/images/
    # game-jam-village/images/
    # swarm-village/images/
    return "images" not in path.parts


def main():
    api = HfApi()

    print(f"Listing files in {REPO_ID}...")
    all_files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    files_to_download = [
        filename for filename in all_files
        if should_download(filename)
    ]

    excluded_files = [
        filename for filename in all_files
        if not should_download(filename)
    ]

    print(f"\nFound {len(all_files)} files in the repository.")
    print(f"Will download: {len(files_to_download)}")
    print(f"Skipping images: {len(excluded_files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    failed = []

    for i, filename in enumerate(files_to_download, start=1):
        print(f"\n[{i}/{len(files_to_download)}] {filename}")

        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(OUTPUT_DIR),
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed.append((filename, str(exc)))

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Successfully downloaded: {len(files_to_download) - len(failed)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped image files: {len(excluded_files)}")
    print(f"Output directory: {OUTPUT_DIR}")

    if failed:
        print("\nFailed files:")
        for filename, error in failed:
            print(f"  {filename}: {error}")


if __name__ == "__main__":
    main()


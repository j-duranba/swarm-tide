
import argparse
import gzip
import shutil
from pathlib import Path

DATA_DIR = Path("data/ai-village")
EXTRACTED_DIR = Path("data/ai-village-extracted")
SAMPLES_FILE = Path("reports/ai_village_gz_samples.txt")


def process_gz_file(source: Path, overwrite: bool = False):
    relative_path = source.relative_to(DATA_DIR)

    # Preserve the directory structure and remove only the .gz suffix.
    output_path = EXTRACTED_DIR / relative_path.with_suffix("")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []

    print(f"Processing: {relative_path}")

    with gzip.open(source, "rt", encoding="utf-8", errors="replace") as infile:
        # Read and retain the first three lines.
        for _ in range(3):
            line = infile.readline()
            if not line:
                break
            samples.append(line)

        # Extract the complete file.
        if overwrite or not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as outfile:
                for line in samples:
                    outfile.write(line)
                shutil.copyfileobj(infile, outfile)

            print(f"  Extracted to: {output_path}")
        else:
            print(f"  Already extracted, skipping: {output_path}")

    return relative_path, samples


def main():
    parser = argparse.ArgumentParser(
        description="Extract AI Village .gz files and collect sample records."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that have already been extracted.",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    gz_files = sorted(
        path for path in DATA_DIR.rglob("*.gz")
        if ".cache" not in path.parts
    )

    print(f"Found {len(gz_files)} compressed files.")

    SAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)

    total = len(gz_files)

    with SAMPLES_FILE.open("w", encoding="utf-8") as sample_out:
        sample_out.write("AI VILLAGE — FIRST THREE LINES OF EACH GZ FILE\n")
        sample_out.write("=" * 80 + "\n\n")

        for index, source in enumerate(gz_files, start=1):
            print(f"\n[{index}/{total}]")
            relative_path, samples = process_gz_file(
                source, overwrite=args.overwrite
            )

            sample_out.write(f"\n{'=' * 80}\n")
            sample_out.write(f"FILE: {relative_path}\n")
            sample_out.write(f"{'=' * 80}\n")

            for line in samples:
                sample_out.write(line)
                if not line.endswith("\n"):
                    sample_out.write("\n")

    print("\n" + "=" * 60)
    print("FINISHED")
    print("=" * 60)
    print(f"Compressed files processed: {total}")
    print(f"Extracted files directory: {EXTRACTED_DIR}")
    print(f"Samples file: {SAMPLES_FILE}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditReport:
    dataset_dir: str
    file_count: int
    tables: dict[str, Any]
    id_hints: dict[str, list[str]]
    timestamp_hints: dict[str, list[str]]
    missingness: dict[str, dict[str, float]]
    hf_access_state: dict[str, Any]
    legal_notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "file_count": self.file_count,
            "tables": self.tables,
            "id_hints": self.id_hints,
            "timestamp_hints": self.timestamp_hints,
            "missingness": self.missingness,
            "hf_access_state": self.hf_access_state,
            "legal_notes": self.legal_notes,
        }


def _scan_jsonl(path: Path, max_rows: int = 2000) -> tuple[list[str], dict[str, float], int]:
    keys: Counter[str] = Counter()
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                total += 1
                for k in obj.keys():
                    keys[k] += 1
            if total >= max_rows:
                break
    cols = sorted(keys)
    missing = {k: 1.0 - keys[k] / max(1, total) for k in cols}
    return cols, missing, total


def _scan_csv(path: Path, max_rows: int = 2000) -> tuple[list[str], dict[str, float], int]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], {}, 0
        counts = Counter({name: 0 for name in reader.fieldnames})
        total = 0
        for row in reader:
            total += 1
            for k, v in row.items():
                if v not in (None, ""):
                    counts[k] += 1
            if total >= max_rows:
                break
    cols = list(reader.fieldnames)
    missing = {k: 1.0 - counts[k] / max(1, total) for k in cols}
    return cols, missing, total


def _hf_access_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "dataset": "aidigestorg/ai-village",
        "token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
        "datasets_library_installed": False,
        "status": "unknown",
        "error": None,
    }
    try:
        from datasets import load_dataset_builder  # type: ignore

        state["datasets_library_installed"] = True
        try:
            load_dataset_builder("aidigestorg/ai-village")
            state["status"] = "reachable_or_cached"
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            state["status"] = "gated_or_unavailable"
            state["error"] = msg[:500]
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"datasets library unavailable: {exc}"
    return state


def audit_dataset(dataset_dir: str) -> AuditReport:
    root = Path(dataset_dir)
    files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
    candidate = [p for p in files if p.suffix.lower() in {".jsonl", ".csv"}]

    tables: dict[str, Any] = {}
    missingness: dict[str, dict[str, float]] = {}
    id_hints: dict[str, list[str]] = {}
    ts_hints: dict[str, list[str]] = {}

    for p in sorted(candidate):
        name = p.stem
        if p.suffix.lower() == ".jsonl":
            cols, missing, sampled = _scan_jsonl(p)
        else:
            cols, missing, sampled = _scan_csv(p)
        tables[name] = {
            "path": str(p),
            "sampled_rows": sampled,
            "column_count": len(cols),
            "columns": cols,
        }
        missingness[name] = missing
        id_hints[name] = [c for c in cols if "id" in c.lower()]
        ts_hints[name] = [c for c in cols if any(t in c.lower() for t in ("time", "timestamp", "created", "date"))]

    legal_notes = [
        "Do not bypass gated access requirements.",
        "Do not train or fine-tune on AI Village data without written permission.",
        "Preserve privacy obligations and avoid re-identification attempts.",
        "Treat convenience summaries as secondary evidence when computer-use traces exist.",
    ]
    return AuditReport(
        dataset_dir=str(root),
        file_count=len(files),
        tables=tables,
        id_hints=id_hints,
        timestamp_hints=ts_hints,
        missingness=missingness,
        hf_access_state=_hf_access_state(),
        legal_notes=legal_notes,
    )


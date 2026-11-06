from __future__ import annotations

import csv
import json
from pathlib import Path

from tide.normalization import normalize_record
from tide.schema import EventRecord


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield str(i), obj


def _iter_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            yield str(i), dict(row)


def load_table_rows(path: str, limit: int | None = None) -> list[tuple[str, dict]]:
    p = Path(path)
    iterator = _iter_jsonl(p) if p.suffix.lower() == ".jsonl" else _iter_csv(p)
    out: list[tuple[str, dict]] = []
    for record_id, raw in iterator:
        out.append((record_id, raw))
        if limit is not None and len(out) >= limit:
            break
    return out


def normalize_table(path: str, source_table: str | None = None, limit: int | None = None) -> list[EventRecord]:
    p = Path(path)
    source_table = source_table or p.stem
    rows = load_table_rows(path, limit=limit)
    out: list[EventRecord] = []
    for record_id, raw in rows:
        event = normalize_record(raw, source_table=source_table, source_record_id=record_id)
        out.append(event)
    return out

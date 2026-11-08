from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator
from urllib.parse import urlsplit

import adaption
from adaption import Adaption
from tqdm import tqdm

from tide.pass2 import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    PUBLIC_RERANK_SYSTEM_PROMPT,
    RERANK_SYSTEM_PROMPT,
    _installed_version,
    _load_embedding_model,
    _parse_timestamp,
    _read_checkpoint,
    _write_errors,
    _write_jsonl_atomic,
    file_sha256,
    generate_candidates,
    load_agent_registry,
    load_agent_identities,
    load_events,
    load_pass1_annotations,
    request_rerank,
)


@contextmanager
def _stage_context(
    output_dir: Path,
    stage: str,
    config: dict[str, Any],
    output_names: tuple[str, ...],
    overwrite: bool,
) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / f"pass2_{stage}.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(f"Pass 2 {stage} stage is locked; check for a running process before removing {lock_path.name}") from None
    try:
        with lock:
            lock.write(str(os.getpid()))
        config_path = output_dir / f"pass2_{stage}_config.json"
        if overwrite:
            if config_path.exists():
                previous_config = json.loads(config_path.read_text(encoding="utf-8"))
                previous_mode = previous_config.get("context_mode", "private")
                if previous_mode != config.get("context_mode", "private"):
                    raise ValueError("Public and private Pass 2 stages require separate output directories")
            for name in output_names:
                (output_dir / name).unlink(missing_ok=True)
        if config_path.exists():
            previous_config = json.loads(config_path.read_text(encoding="utf-8"))
            if previous_config.get("context_mode", "private") != config.get("context_mode", "private"):
                raise ValueError("Public and private Pass 2 stages require separate output directories")
            if previous_config != config:
                raise ValueError(f"Pass 2 {stage} configuration or inputs changed; use a new output directory or --overwrite")
        else:
            if any((output_dir / name).exists() for name in output_names if name != config_path.name):
                raise ValueError(f"Pass 2 {stage} cache exists without a configuration; choose a new output directory")
            temporary = config_path.with_name(config_path.name + ".tmp")
            temporary.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            temporary.replace(config_path)
        yield
    finally:
        lock.close()
        lock_path.unlink(missing_ok=True)


def _event_inputs(dataset_dir: str | Path) -> tuple[Path, Path, dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    dataset = Path(dataset_dir).resolve()
    events_path = dataset / "events.jsonl"
    agents_path = dataset / "agents.jsonl"
    if context_mode == "public":
        agents, _ = load_agent_identities(agents_path)
    else:
        agents, _ = load_agent_registry(agents_path)
    events, skipped_consolidations = load_events(events_path, agents, {})
    return events_path, agents_path, agents, events, skipped_consolidations


def _context_event_inputs(
    dataset_dir: str | Path,
    context_mode: str,
    pass1_dir: str | Path | None,
    *,
    include_relations: bool,
) -> tuple[Path, Path, dict[str, dict[str, Any]], list[dict[str, Any]], int, Path | None]:
    if context_mode not in {"private", "public"}:
        raise ValueError("context_mode must be private or public")
    if context_mode == "public" and pass1_dir is None:
        raise ValueError("Public Pass 2 requires Pass 1 summaries")
    dataset = Path(dataset_dir).resolve()
    events_path = dataset / "events.jsonl"
    agents_path = dataset / "agents.jsonl"
    agents, _ = load_agent_registry(agents_path)
    annotations: dict[str, dict[str, Any]] = {}
    annotations_path: Path | None = None
    if pass1_dir is not None:
        annotations, annotations_path = load_pass1_annotations(
            Path(pass1_dir).resolve(), include_relations=include_relations
        )
    events, skipped_consolidations = load_events(
        events_path, agents, annotations, representation_mode=context_mode
    )
    return events_path, agents_path, agents, events, skipped_consolidations, annotations_path


def _rerank_target(
    client: Adaption,
    model: str,
    retrieval: dict[str, Any],
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    candidate_group_size: int,
) -> dict[str, Any]:
    if candidate_group_size < 1:
        raise ValueError("candidate_group_size must be positive")
    public_only = "public_summary" in target
    dependency_map: dict[str, dict[str, Any]] = {}
    for start in range(0, len(candidates), candidate_group_size):
        group = candidates[start:start + candidate_group_size]
        dependencies = request_rerank(client, model, target, group, public_only=public_only)["dependencies"]
        overlap = dependency_map.keys() & dependencies.keys()
        if overlap:
            raise ValueError("Candidate groups returned duplicate dependency IDs")
        dependency_map.update(dependencies)
    if set(dependency_map) != {candidate["source_event_id"] for candidate in candidates}:
        raise ValueError("Candidate groups did not return every dependency exactly once")
    scored = []
    for candidate in candidates:
        score = dependency_map[candidate["source_event_id"]]
        result_candidate = {
            "source_event_id": candidate["source_event_id"],
            "author_id": candidate["author_id"],
            "author_name": candidate["author_name"],
            "timestamp": candidate["timestamp"],
            "retrieval_sources": candidate["retrieval_sources"],
            "similarity_score": candidate["similarity_score"],
            "dependency_score": score["dependency_score"],
            "reason": score["reason"],
        }
        if public_only:
            result_candidate["public_summary"] = candidate["public_summary"]
        else:
            result_candidate["agent_goal"] = candidate["agent_goal"]
        scored.append(result_candidate)
    result = {
        "event_id": retrieval["event_id"],
        "source_index": retrieval["source_index"],
        "timestamp": retrieval["timestamp"],
        "author_id": retrieval["author_id"],
        "candidate_event_ids": retrieval["candidate_event_ids"],
        "candidates": scored,
    }
    if public_only:
        result["public_summary"] = target["public_summary"]
    else:
        result["agent_goal"] = retrieval["agent_goal"]
    return result


def filter_candidates_for_rerank(
    retrieval: dict[str, Any],
    candidate_filter: str,
    *,
    context_mode: str = "private",
) -> list[dict[str, Any]]:
    candidates = retrieval["candidates"]
    if candidate_filter == "none":
        return list(candidates)
    if candidate_filter == "top10":
        if context_mode == "public":
            same_agent = [
                candidate for candidate in candidates
                if "same_agent_history" in candidate["retrieval_sources"]
            ]
            selected_ids = {candidate["source_event_id"] for candidate in same_agent}
            semantic = sorted(
                (
                    candidate for candidate in candidates
                    if candidate["source_event_id"] not in selected_ids
                ),
                key=lambda candidate: candidate["similarity_score"],
                reverse=True,
            )
            selected_ids.update(candidate["source_event_id"] for candidate in semantic[:max(0, 10 - len(same_agent))])
            return [candidate for candidate in candidates if candidate["source_event_id"] in selected_ids]
        return sorted(
            candidates,
            key=lambda candidate: candidate["similarity_score"],
            reverse=True,
        )[:10]
    if candidate_filter != "proposed":
        raise ValueError("candidate_filter must be none, top10, or proposed")

    selected_ids: set[str] = set()

    def newest_first(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: _parse_timestamp(row.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

    same_agent = newest_first([
        candidate for candidate in candidates
        if "same_agent_history" in candidate["retrieval_sources"]
    ])[:3]
    selected_ids.update(candidate["source_event_id"] for candidate in same_agent)

    source_candidates = [
        candidate for candidate in candidates
        if "identified_information_source" in candidate["retrieval_sources"]
    ]
    latest_by_author: dict[str, dict[str, Any]] = {}
    for candidate in newest_first(source_candidates):
        author_id = candidate.get("author_id")
        if author_id and author_id not in latest_by_author:
            latest_by_author[author_id] = candidate
    selected_ids.update(candidate["source_event_id"] for candidate in latest_by_author.values())

    target_time = _parse_timestamp(retrieval.get("timestamp"))
    last_same_agent_time = max(
        (_parse_timestamp(candidate.get("timestamp")) for candidate in same_agent),
        default=None,
    )
    incoming_lower_bound = last_same_agent_time
    if incoming_lower_bound is None and target_time is not None:
        incoming_lower_bound = target_time - timedelta(hours=12)
    for candidate in candidates:
        candidate_time = _parse_timestamp(candidate.get("timestamp"))
        if (
            "incoming_communication" in candidate["retrieval_sources"]
            and candidate["source_event_id"] not in selected_ids
            and candidate_time is not None
            and target_time is not None
            and incoming_lower_bound is not None
            and incoming_lower_bound < candidate_time < target_time
        ):
            selected_ids.add(candidate["source_event_id"])

    semantic = sorted(
        (
            candidate for candidate in candidates
            if "semantic" in candidate["retrieval_sources"]
            and candidate["source_event_id"] not in selected_ids
        ),
        key=lambda candidate: candidate["similarity_score"],
        reverse=True,
    )
    selected_ids.update(candidate["source_event_id"] for candidate in semantic[:5])
    return [candidate for candidate in candidates if candidate["source_event_id"] in selected_ids]


def run_pass2_embed(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    embedding_model: str = EMBEDDING_MODEL,
    embedding_revision: str = EMBEDDING_REVISION,
    precision: str = "auto",
    batch_size: int = 32,
    context_mode: str = "private",
    pass1_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    events_path, agents_path, _, events, skipped_consolidations, annotations_path = _context_event_inputs(
        dataset_dir, context_mode, pass1_dir, include_relations=False
    )
    destination = Path(output_dir).resolve()
    output_names = (
        "pass2_embed_config.json", "pass2_embed_config.json.tmp",
        "event_embeddings.jsonl", "event_embeddings.jsonl.tmp",
        "event_embeddings_checkpoint.jsonl", "event_embeddings_checkpoint.jsonl.tmp",
        "pass2_embed_errors.jsonl", "pass2_embed_errors.jsonl.tmp",
    )
    config = {
        "version": 1,
        "context_mode": context_mode,
        "events_sha256": file_sha256(events_path),
        "agents_sha256": file_sha256(agents_path),
        "embedding_model": embedding_model,
        "embedding_revision": embedding_revision,
        "precision": precision,
        "batch_size": batch_size,
        "representation_version": 2,
        "excluded_action_types": ["CONSOLIDATE"],
        "runtime_versions": {
            "numpy": _installed_version("numpy"),
            "torch": _installed_version("torch"),
            "transformers": _installed_version("transformers"),
        },
    }
    if annotations_path is not None:
        config["pass1_sha256"] = file_sha256(annotations_path)
    with _stage_context(destination, "embed", config, output_names, overwrite):
        checkpoint_path = destination / "event_embeddings_checkpoint.jsonl"
        embeddings = _read_checkpoint(checkpoint_path, "event_id")
        events_by_id = {event["event_id"]: event for event in events}
        for event_id, row in embeddings.items():
            event = events_by_id.get(event_id)
            if event is None or row.get("source_index") != event["source_index"]:
                raise ValueError("Embedding checkpoint does not match the source events")
            for key in ("query_embedding", "document_embedding"):
                vector = row.get(key)
                if not isinstance(vector, list) or not vector or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in vector
                ):
                    raise ValueError(f"Invalid {key} for {event_id}")
        errors_path = destination / "pass2_embed_errors.jsonl"
        errors = _read_checkpoint(errors_path, "event_id")
        for event_id in embeddings:
            errors.pop(event_id, None)
        missing = [event for event in events if event["event_id"] not in embeddings]
        _write_errors(errors_path, errors)
        print("Pass 2 / embed: creating event vectors...", file=sys.stderr, flush=True)
        if missing:
            model_object = _load_embedding_model(embedding_model, embedding_revision, precision)
            with checkpoint_path.open("a", encoding="utf-8", newline="\n") as journal, tqdm(
                total=len(events), initial=len(embeddings), desc="Pass 2 embeddings", unit="event",
                ascii=True, dynamic_ncols=True, file=sys.stderr,
            ) as progress:
                for start in range(0, len(missing), batch_size):
                    batch = missing[start:start + batch_size]
                    try:
                        texts = [event["representation"] for event in batch]
                        query_rows = model_object.encode(
                            texts=texts, task="retrieval", prompt_name="query"
                        ).tolist()
                        document_rows = model_object.encode(
                            texts=texts, task="retrieval", prompt_name="document"
                        ).tolist()
                        if len(query_rows) != len(batch) or len(document_rows) != len(batch):
                            raise ValueError("Embedding model returned an unexpected batch size")
                        for event, query_vector, document_vector in zip(batch, query_rows, document_rows):
                            row = {
                                "event_id": event["event_id"],
                                "source_index": event["source_index"],
                                "author_id": event["author_id"],
                                "author_name": event["author_name"],
                                "timestamp": event["timestamp"],
                                "query_embedding": query_vector,
                                "document_embedding": document_vector,
                            }
                            if context_mode == "public":
                                row["public_summary"] = event["public_summary"]
                            else:
                                row["agent_goal"] = event["agent_goal"]
                            journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                            embeddings[row["event_id"]] = row
                            errors.pop(row["event_id"], None)
                        journal.flush()
                        os.fsync(journal.fileno())
                    except Exception as error:
                        for event in batch:
                            errors[event["event_id"]] = {
                                "event_id": event["event_id"],
                                "source_index": event["source_index"],
                                "stage": "embedding",
                                "error": str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__,
                            }
                    progress.update(len(batch))
                    progress.set_postfix(failed=len(errors))
                    _write_errors(errors_path, errors)
        ordered = sorted(embeddings.values(), key=lambda row: row["source_index"])
        _write_jsonl_atomic(destination / "event_embeddings.jsonl", ordered)
        return {
            "stage": "embed", "events": len(events), "skipped_consolidations": skipped_consolidations,
            "embedded": len(embeddings), "failures": sorted(errors.values(), key=lambda row: row["source_index"]),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "output": str(destination / "event_embeddings.jsonl"),
        }


def run_pass2_retrieve(
    dataset_dir: str | Path,
    pass1_dir: str | Path,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    context_mode: str = "private",
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    events_path, agents_path, agents, events, skipped_consolidations, annotations_path = _context_event_inputs(
        dataset_dir, context_mode, pass1_dir, include_relations=context_mode != "public"
    )
    if annotations_path is None:
        raise ValueError("Pass 1 annotations are required for candidate retrieval")
    _, names_to_ids = load_agent_registry(agents_path)
    destination = Path(output_dir).resolve()
    embeddings_path = destination / "event_embeddings.jsonl"
    if not embeddings_path.exists():
        raise FileNotFoundError("Embedding cache is missing; run pass2-embed first")
    embedding_config = json.loads((destination / "pass2_embed_config.json").read_text(encoding="utf-8"))
    if embedding_config.get("context_mode", "private") != context_mode:
        raise ValueError("Embedding cache context does not match retrieval context")
    embeddings = _read_checkpoint(embeddings_path, "event_id")
    expected_ids = {event["event_id"] for event in events}
    missing_embeddings = expected_ids - set(embeddings)
    if missing_embeddings:
        raise ValueError(f"Embedding cache is incomplete ({len(missing_embeddings)} events); finish pass2-embed first")
    embedding_vectors = {
        event_id: {"query_embedding": row["query_embedding"], "document_embedding": row["document_embedding"]}
        for event_id, row in embeddings.items()
    }
    target_events = [event for event in events if event["has_pass1_annotation"]]
    if limit is not None:
        target_events = target_events[:limit]
    candidate_limits = {
        "same_agent_history": 3,
        "identified_information_source": 5,
        "incoming_communication": 5,
        "semantic": 10,
    }
    config = {
        "version": 1,
        "context_mode": context_mode,
        "events_sha256": file_sha256(events_path),
        "agents_sha256": file_sha256(agents_path),
        "pass1_sha256": file_sha256(annotations_path),
        "embeddings_sha256": file_sha256(embeddings_path),
        "time_window_hours": 12,
        "candidate_limits": candidate_limits,
        "same_agent_history_exception": True,
        "excluded_action_types": ["CONSOLIDATE"],
        "limit": limit,
    }
    output_names = (
        "pass2_retrieve_config.json", "pass2_retrieve_config.json.tmp",
        "retrieval_candidates.jsonl", "retrieval_candidates.jsonl.tmp",
        "retrieval_checkpoint.jsonl", "retrieval_checkpoint.jsonl.tmp",
    )
    with _stage_context(destination, "retrieve", config, output_names, overwrite):
        checkpoint_path = destination / "retrieval_checkpoint.jsonl"
        completed = _read_checkpoint(checkpoint_path, "event_id")
        for event_id, row in completed.items():
            if event_id not in {event["event_id"] for event in target_events}:
                raise ValueError("Retrieval checkpoint contains a target outside this run")
            if not isinstance(row.get("candidates"), list):
                raise ValueError(f"Invalid retrieval checkpoint for {event_id}")
        by_id = {event["event_id"]: event for event in events}
        ordered_events = sorted(events, key=lambda event: (
            event["timestamp_dt"] is None,
            event["timestamp_dt"] or datetime.max.replace(tzinfo=timezone.utc),
            event["source_index"],
        ))
        event_cursor = 0
        recent_events: list[dict[str, Any]] = []
        with checkpoint_path.open("a", encoding="utf-8", newline="\n") as journal, tqdm(
            total=len(target_events), initial=len(completed), desc="Pass 2 retrieval", unit="event",
            ascii=True, dynamic_ncols=True, file=sys.stderr,
        ) as progress:
            for target in target_events:
                if target["event_id"] in completed:
                    continue
                timestamp = target["timestamp_dt"]
                if timestamp is None:
                    earlier = []
                else:
                    cutoff = timestamp - timedelta(hours=12)
                    while event_cursor < len(ordered_events):
                        previous = ordered_events[event_cursor]
                        previous_time = previous["timestamp_dt"]
                        if previous_time is None or previous_time >= timestamp:
                            break
                        if previous_time >= cutoff:
                            recent_events.append(previous)
                        event_cursor += 1
                    recent_events = [event for event in recent_events if event["timestamp_dt"] >= cutoff]
                    earlier = recent_events
                candidates = generate_candidates(target, earlier, embedding_vectors, names_to_ids)
                record = {
                    "event_id": target["event_id"],
                    "source_index": target["source_index"],
                    "timestamp": target["timestamp"],
                    "author_id": target["author_id"],
                    "candidate_event_ids": [candidate["source_event_id"] for candidate in candidates],
                    "candidates": [
                        {
                            "source_event_id": candidate["source_event_id"],
                            "author_id": candidate["author_id"],
                            "author_name": candidate["author_name"],
                            "timestamp": candidate["timestamp"],
                            "retrieval_sources": candidate["retrieval_sources"],
                            "similarity_score": candidate["similarity_score"],
                        }
                        for candidate in candidates
                    ],
                }
                if context_mode == "public":
                    record["public_summary"] = target["public_summary"]
                else:
                    record["agent_goal"] = target["agent_goal"]
                journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                completed[record["event_id"]] = record
                progress.update(1)
        ordered = sorted(completed.values(), key=lambda row: row["source_index"])
        _write_jsonl_atomic(destination / "retrieval_candidates.jsonl", ordered)
        return {
            "stage": "retrieve", "events": len(events), "skipped_consolidations": skipped_consolidations,
            "targets": len(target_events), "retrieved": len(completed),
            "candidate_count": sum(len(row["candidates"]) for row in ordered),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "output": str(destination / "retrieval_candidates.jsonl"),
        }


def run_pass2_rerank(
    dataset_dir: str | Path,
    output_dir: str | Path,
    api_key: str,
    *,
    model: str = "deepseek-v4-flash",
    timeout: float = 120,
    retries: int = 3,
    concurrency: int = 1,
    candidate_group_size: int = 4,
    context_mode: str = "private",
    pass1_dir: str | Path | None = None,
    candidate_filter: str = "none",
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    if not api_key or not model:
        raise ValueError("DeepSeek API key and model are required")
    if timeout <= 0 or retries < 0 or concurrency < 1 or candidate_group_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Invalid timeout, retries, concurrency, candidate group size, or limit")
    if candidate_filter not in {"none", "top10", "proposed"}:
        raise ValueError("candidate_filter must be none, top10, or proposed")
    events_path, agents_path, agents, events, skipped_consolidations, annotations_path = _context_event_inputs(
        dataset_dir, context_mode, pass1_dir, include_relations=False
    )
    destination = Path(output_dir).resolve()
    retrieval_path = destination / "retrieval_candidates.jsonl"
    if not retrieval_path.exists():
        raise FileNotFoundError("Retrieval cache is missing; run pass2-retrieve first")
    retrieval_rows = _read_checkpoint(retrieval_path, "event_id")
    retrieval_config = json.loads((destination / "pass2_retrieve_config.json").read_text(encoding="utf-8"))
    if retrieval_config.get("context_mode", "private") != context_mode:
        raise ValueError("Retrieval cache context does not match reranking context")
    event_by_id = {event["event_id"]: event for event in events}
    for event_id, row in retrieval_rows.items():
        if event_id not in event_by_id or row.get("source_index") != event_by_id[event_id]["source_index"]:
            raise ValueError("Retrieval cache does not match the event data")
        for candidate in row.get("candidates", []):
            if candidate.get("source_event_id") not in event_by_id:
                raise ValueError("Retrieval cache contains an unknown candidate event")
    selected_rows = list(retrieval_rows.values())
    if limit is not None:
        selected_rows.sort(key=lambda row: (
            event_by_id[row["event_id"]]["timestamp_dt"] is None,
            event_by_id[row["event_id"]]["timestamp_dt"] or datetime.max.replace(tzinfo=timezone.utc),
            row["source_index"],
        ))
        selected_rows = selected_rows[:limit]
    filtered_candidates = {
        row["event_id"]: filter_candidates_for_rerank(row, candidate_filter, context_mode=context_mode)
        for row in selected_rows
    }
    candidate_count_before = sum(len(row["candidates"]) for row in selected_rows)
    candidate_count_after = sum(len(rows) for rows in filtered_candidates.values())
    client = Adaption(api_key=api_key, timeout=timeout, max_retries=retries)
    endpoint = str(client.base_url)
    url = urlsplit(endpoint)
    if (url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1"))) or not url.hostname or url.username or url.password or url.query or url.fragment:
        client.close()
        raise ValueError("SDK base URL must be HTTPS without credentials or query parameters")
    config = {
        "version": 1,
        "context_mode": context_mode,
        "events_sha256": file_sha256(events_path),
        "agents_sha256": file_sha256(agents_path),
        "retrieval_sha256": file_sha256(retrieval_path),
        "rerank_model": model,
        "endpoint": endpoint,
        "timeout_seconds": timeout,
        "retries": retries,
        "candidate_group_size": candidate_group_size,
        "candidate_filter": candidate_filter,
        "temperature": 0,
        "prompt_sha256": hashlib.sha256(RERANK_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }
    if annotations_path is not None:
        config["pass1_sha256"] = file_sha256(annotations_path)
    if context_mode == "public":
        config["prompt_sha256"] = hashlib.sha256(PUBLIC_RERANK_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    output_names = (
        "pass2_rerank_config.json", "pass2_rerank_config.json.tmp",
        "dependencies.jsonl", "dependencies.jsonl.tmp",
        "rerank_checkpoint.jsonl", "rerank_checkpoint.jsonl.tmp",
        "pass2_rerank_errors.jsonl", "pass2_rerank_errors.jsonl.tmp",
    )
    try:
        with _stage_context(destination, "rerank", config, output_names, overwrite):
            checkpoint_path = destination / "rerank_checkpoint.jsonl"
            completed = _read_checkpoint(checkpoint_path, "event_id")
            errors_path = destination / "pass2_rerank_errors.jsonl"
            errors = _read_checkpoint(errors_path, "event_id")
            for event_id in completed:
                errors.pop(event_id, None)
            _write_errors(errors_path, errors)
            ordered_retrieval = sorted(selected_rows, key=lambda row: row["source_index"])
            target_ids = {row["event_id"] for row in ordered_retrieval}
            if not set(completed).issubset(retrieval_rows):
                raise ValueError("Reranking checkpoint contains a target outside this retrieval cache")
            pending: dict[Any, dict[str, Any]] = {}
            retrieval_iter = iter(ordered_retrieval)
            source_exhausted = False
            with checkpoint_path.open("a", encoding="utf-8", newline="\n") as journal, ThreadPoolExecutor(max_workers=concurrency) as executor, tqdm(
                total=len(ordered_retrieval), initial=len(target_ids.intersection(completed)), desc="Pass 2 reranking", unit="event",
                ascii=True, dynamic_ncols=True, file=sys.stderr,
            ) as progress:
                while pending or not source_exhausted:
                    while not source_exhausted and len(pending) < concurrency:
                        try:
                            retrieval = next(retrieval_iter)
                        except StopIteration:
                            source_exhausted = True
                            break
                        event_id = retrieval["event_id"]
                        if event_id in completed:
                            continue
                        target = event_by_id[event_id]
                        candidates = []
                        cached_candidates = filtered_candidates[event_id]
                        for cached in cached_candidates:
                            candidate = dict(event_by_id[cached["source_event_id"]])
                            candidate.update({
                                "source_event_id": cached["source_event_id"],
                                "retrieval_sources": cached["retrieval_sources"],
                                "similarity_score": cached["similarity_score"],
                            })
                            candidates.append(candidate)
                        target_retrieval = dict(retrieval)
                        target_retrieval["candidate_event_ids"] = [
                            candidate["source_event_id"] for candidate in cached_candidates
                        ]
                        if not candidates:
                            result = _rerank_target(client, model, target_retrieval, target, candidates, candidate_group_size)
                            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                            journal.flush()
                            os.fsync(journal.fileno())
                            completed[event_id] = result
                            errors.pop(event_id, None)
                            progress.update(1)
                            continue
                        future = executor.submit(
                            _rerank_target, client, model, target_retrieval, target, candidates, candidate_group_size
                        )
                        pending[future] = retrieval
                    if not pending:
                        continue
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        retrieval = pending.pop(future)
                        event_id = retrieval["event_id"]
                        had_error = event_id in errors
                        try:
                            result = future.result()
                            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                            journal.flush()
                            os.fsync(journal.fileno())
                            completed[event_id] = result
                            errors.pop(event_id, None)
                        except Exception as error:
                            errors[event_id] = {
                                "event_id": event_id,
                                "source_index": retrieval["source_index"],
                                "stage": "reranking",
                                "error": str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__,
                            }
                        progress.set_postfix(failed=sum(event_id in target_ids for event_id in errors))
                        if had_error or event_id in errors:
                            _write_errors(errors_path, errors)
                        progress.update(1)
            ordered = sorted(completed.values(), key=lambda row: row["source_index"])
            _write_jsonl_atomic(destination / "dependencies.jsonl", ordered)
            _write_errors(errors_path, errors)
            return {
                "stage": "rerank", "targets": len(ordered_retrieval), "completed_targets": len(target_ids.intersection(completed)),
                "candidate_filter": candidate_filter,
                "candidate_count_before": candidate_count_before,
                "candidate_count_after": candidate_count_after,
                "candidate_reduction": candidate_count_before - candidate_count_after,
                "candidate_reduction_percent": round(
                    100 * (candidate_count_before - candidate_count_after) / candidate_count_before, 2
                ) if candidate_count_before else 0.0,
                "unresolved_targets": len(target_ids - set(completed)),
                "failures": sorted((row for event_id, row in errors.items() if event_id in target_ids), key=lambda row: row["source_index"]),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "output": str(destination / "dependencies.jsonl"),
            }
    finally:
        client.close()


def run_pass2_all(
    dataset_dir: str | Path,
    pass1_dir: str | Path,
    output_dir: str | Path,
    api_key: str,
    *,
    model: str = "deepseek-v4-flash",
    embedding_model: str = EMBEDDING_MODEL,
    embedding_revision: str = EMBEDDING_REVISION,
    precision: str = "auto",
    batch_size: int = 32,
    limit: int | None = None,
    timeout: float = 120,
    retries: int = 3,
    concurrency: int = 1,
    candidate_group_size: int = 4,
    candidate_filter: str = "none",
    overwrite: bool = False,
) -> dict[str, Any]:
    embedded = run_pass2_embed(
        dataset_dir, output_dir, embedding_model=embedding_model,
        embedding_revision=embedding_revision, precision=precision,
        batch_size=batch_size, overwrite=overwrite,
    )
    if embedded["failures"]:
        return {
            "events": embedded["events"], "skipped_consolidations": embedded["skipped_consolidations"],
            "targets": 0, "embedded": embedded["embedded"], "completed_targets": 0,
            "unresolved_targets": 0, "failures": embedded["failures"],
            "elapsed_seconds": embedded["elapsed_seconds"],
            "embeddings_output": embedded["output"], "output": str(Path(output_dir) / "dependencies.jsonl"),
        }
    retrieved = run_pass2_retrieve(
        dataset_dir, pass1_dir, output_dir, limit=limit, overwrite=overwrite,
    )
    reranked = run_pass2_rerank(
        dataset_dir, output_dir, api_key, model=model, timeout=timeout, retries=retries,
        concurrency=concurrency, candidate_group_size=candidate_group_size,
        candidate_filter=candidate_filter, overwrite=overwrite,
    )
    return {
        "events": retrieved["events"],
        "skipped_consolidations": retrieved["skipped_consolidations"],
        "targets": reranked["targets"],
        "embedded": embedded["embedded"],
        "retrieved": retrieved["retrieved"],
        "candidate_count": retrieved["candidate_count"],
        "completed_targets": reranked["completed_targets"],
        "unresolved_targets": reranked["unresolved_targets"],
        "failures": reranked["failures"],
        "elapsed_seconds": round(embedded["elapsed_seconds"] + retrieved["elapsed_seconds"] + reranked["elapsed_seconds"], 2),
        "embeddings_output": embedded["output"],
        "retrieval_output": retrieved["output"],
        "output": reranked["output"],
    }
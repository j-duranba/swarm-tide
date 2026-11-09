from __future__ import annotations

import math
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.special import expit

from tide.graph_analysis import read_snapshot
from tide.io_utils import write_json, write_jsonl
from tide.pass2 import file_sha256, load_agent_registry, load_events


def mean_squared_error(pairs: list[dict[str, Any]], decay_lambda: float) -> float | None:
    if not pairs:
        return None
    return sum(
        (pair["public_score"] * math.exp(-decay_lambda * pair["elapsed_hours"]) - pair["private_score"]) ** 2
        for pair in pairs
    ) / len(pairs)


def fit_temporal_decay(pairs: list[dict[str, Any]], *, lambda_max: float = 5.0, steps: int = 1000) -> dict[str, Any]:
    if not pairs:
        raise ValueError("No matched calibration pairs")
    if not math.isfinite(lambda_max) or lambda_max <= 0 or steps < 1:
        raise ValueError("lambda_max and steps must be positive")
    candidates = [lambda_max * index / steps for index in range(steps + 1)]
    decay_lambda = min(candidates, key=lambda value: mean_squared_error(pairs, value))
    return {
        "lambda_per_hour": decay_lambda,
        "half_life_hours": math.log(2) / decay_lambda if decay_lambda else None,
        "lambda_max": lambda_max,
        "grid_steps": steps,
        "at_upper_bound": decay_lambda == lambda_max,
        "baseline_mse": mean_squared_error(pairs, 0),
        "decayed_mse": mean_squared_error(pairs, decay_lambda),
    }


def sigmoid_decay_score(public_score: float, elapsed_hours: float, alpha: float, intercept: float, decay_lambda: float) -> float:
    return float(expit(alpha * public_score + intercept)) * math.exp(-decay_lambda * elapsed_hours)


def fit_sigmoid_decay(pairs: list[dict[str, Any]], *, lambda_max: float = 5.0) -> dict[str, Any]:
    if not pairs:
        raise ValueError("No matched calibration pairs")
    if not math.isfinite(lambda_max) or lambda_max <= 0:
        raise ValueError("lambda_max must be positive and finite")
    public = np.array([pair["public_score"] for pair in pairs], dtype=float)
    private = np.array([pair["private_score"] for pair in pairs], dtype=float)
    elapsed = np.array([pair["elapsed_hours"] for pair in pairs], dtype=float)
    starts = [(0.0, 0.0, 0.0), (2.0, -1.0, 0.0), (8.0, -4.0, min(1.0, lambda_max)),
              (-2.0, 1.0, min(1.0, lambda_max))]
    runs = [least_squares(
        lambda params: expit(params[0] * public + params[1]) * np.exp(-params[2] * elapsed) - private,
        start, bounds=([-30.0, -30.0, 0.0], [30.0, 30.0, lambda_max]), max_nfev=2000,
    ) for start in starts]
    converged = [run for run in runs if run.success and np.isfinite(run.cost)]
    if not converged:
        raise RuntimeError("Sigmoid-decay optimization failed to converge")
    best = min(converged, key=lambda run: run.cost)
    alpha, intercept, decay_lambda = (float(value) for value in best.x)
    return {
        "alpha": alpha, "intercept": intercept, "lambda_per_hour": decay_lambda,
        "half_life_hours": math.log(2) / decay_lambda if decay_lambda else None,
        "training_mse": float(2 * best.cost / len(pairs)),
        "optimizer": "scipy.optimize.least_squares", "scipy_version": scipy.__version__,
        "starts": len(starts), "converged_starts": len(converged), "best_evaluations": best.nfev,
        "parameter_bounds": {"alpha": [-30.0, 30.0], "intercept": [-30.0, 30.0], "lambda_per_hour": [0.0, lambda_max]},
        "active_bounds": [name for name, active in zip(("alpha", "intercept", "lambda_per_hour"), best.active_mask) if active],
    }


def fit_sigmoid_ranking(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise ValueError("No matched calibration pairs")
    public = np.array([pair["public_score"] for pair in pairs], dtype=float)
    private = np.array([pair["private_score"] for pair in pairs], dtype=float)
    starts = [(0.0, 0.0), (2.0, -1.0), (8.0, -4.0), (-2.0, 1.0)]
    runs = [least_squares(
        lambda params: expit(params[0] * public + params[1]) - private,
        start, bounds=([-30.0, -30.0], [30.0, 30.0]), max_nfev=2000,
    ) for start in starts]
    converged = [run for run in runs if run.success and np.isfinite(run.cost)]
    if not converged:
        raise RuntimeError("Sigmoid ranking optimization failed to converge")
    best = min(converged, key=lambda run: run.cost)
    alpha, intercept = (float(value) for value in best.x)
    return {
        "alpha": alpha, "intercept": intercept,
        "training_mse": float(2 * best.cost / len(pairs)),
        "optimizer": "scipy.optimize.least_squares", "scipy_version": scipy.__version__,
        "starts": len(starts), "converged_starts": len(converged), "best_evaluations": best.nfev,
        "parameter_bounds": {"alpha": [-30.0, 30.0], "intercept": [-30.0, 30.0]},
        "active_bounds": [name for name, active in zip(("alpha", "intercept"), best.active_mask) if active],
    }


def sigmoid_ranking_score(public_score: float, alpha: float, intercept: float) -> float:
    return float(expit(alpha * public_score + intercept))


def score_metrics(pairs: list[dict[str, Any]], decay_lambda: float, threshold: float, *,
                  alpha: float | None = None, intercept: float = 0.0) -> dict[str, Any]:
    if not pairs:
        return {"pairs": 0, "mse": None, "mae": None, "true_positive": 0, "false_positive": 0,
                "false_negative": 0, "true_negative": 0, "precision": None, "recall": None}
    errors = []
    true_positive = false_positive = false_negative = true_negative = 0
    for pair in pairs:
        score = (pair["public_score"] * math.exp(-decay_lambda * pair["elapsed_hours"]) if alpha is None else
             sigmoid_decay_score(pair["public_score"], pair["elapsed_hours"], alpha, intercept, decay_lambda))
        errors.append(abs(score - pair["private_score"]))
        predicted, reference = score >= threshold, pair["private_score"] >= threshold
        if predicted and reference:
            true_positive += 1
        elif predicted:
            false_positive += 1
        elif reference:
            false_negative += 1
        else:
            true_negative += 1
    return {
        "pairs": len(pairs), "mse": sum(error * error for error in errors) / len(errors), "mae": sum(errors) / len(errors),
        "true_positive": true_positive, "false_positive": false_positive,
        "false_negative": false_negative, "true_negative": true_negative,
        "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else None,
        "recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else None,
    }


def _checked_rerank_inputs(directory: Path, context: str, dataset: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    retrieval_path = directory / "retrieval_candidates.jsonl"
    retrieval_config = json.loads((directory / "pass2_retrieve_config.json").read_text(encoding="utf-8"))
    rerank_config = json.loads((directory / "pass2_rerank_config.json").read_text(encoding="utf-8"))
    if retrieval_config.get("context_mode", "private") != context or rerank_config.get("context_mode", "private") != context:
        raise ValueError(f"Expected {context} retrieval and rerank inputs in {directory}")
    for key, path in (("events_sha256", dataset / "events.jsonl"), ("agents_sha256", dataset / "agents.jsonl")):
        expected = file_sha256(path)
        if retrieval_config.get(key) != expected or rerank_config.get(key) != expected:
            raise ValueError(f"{context} inputs do not match {key}")
    if rerank_config.get("retrieval_sha256") != file_sha256(retrieval_path):
        raise ValueError(f"{context} rerank configuration does not match retrieval")
    return retrieval_config, rerank_config, retrieval_path


def _candidate_scores(row: dict[str, Any], event_by_id: dict[str, dict[str, Any]], cohort_ids: set[str]) -> dict[str, dict[str, Any]]:
    target_id = row["event_id"]
    target = event_by_id.get(target_id)
    if target is None or row.get("source_index") != target["source_index"]:
        raise ValueError(f"Rerank target does not match dataset: {target_id}")
    if not isinstance(row.get("candidates"), list):
        raise ValueError(f"Invalid candidates for {target_id}")
    scores = {}
    seen = set()
    for candidate in row["candidates"]:
        source_id = candidate["source_event_id"]
        source = event_by_id.get(source_id)
        score = candidate["dependency_score"]
        if source is None or source_id in seen:
            raise ValueError(f"Unknown or duplicate candidate for {target_id}: {source_id}")
        seen.add(source_id)
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid dependency score for {source_id} -> {target_id}")
        if source["timestamp_dt"] is None or target["timestamp_dt"] is None or source["timestamp_dt"] >= target["timestamp_dt"]:
            raise ValueError(f"Candidate is not strictly earlier: {source_id} -> {target_id}")
        if source_id in cohort_ids:
            scores[source_id] = {"score": float(score),
                                 "elapsed_hours": (target["timestamp_dt"] - source["timestamp_dt"]).total_seconds() / 3600}
    return scores


def run_calibration(dataset_dir: str | Path, public_dir: str | Path, private_dir: str | Path,
                    output_dir: str | Path, *, train_targets: int = 200, cohort_limit: int = 1200,
                    lambda_max: float = 5.0, grid_steps: int = 1000, threshold: float = 0.5,
                    model: str = "sigmoid-decay") -> dict[str, Any]:
    if train_targets < 1 or cohort_limit <= train_targets:
        raise ValueError("cohort_limit must exceed the positive train_targets count")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if model not in {"sigmoid-decay", "decay-only"}:
        raise ValueError("model must be sigmoid-decay or decay-only")
    dataset, public, private, output = (Path(path).resolve() for path in (dataset_dir, public_dir, private_dir, output_dir))
    if any(output == source or source in output.parents for source in (dataset, public, private)):
        raise ValueError("Calibration output must be outside input directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Calibration output directory must be empty")
    public_retrieval_config, public_config, public_retrieval_path = _checked_rerank_inputs(public, "public", dataset)
    private_retrieval_config, private_config, _ = _checked_rerank_inputs(private, "private", dataset)
    if public_retrieval_config.get("pass1_sha256") != private_retrieval_config.get("pass1_sha256"):
        raise ValueError("Public and private retrieval use different Pass 1 annotations")
    agents, _ = load_agent_registry(dataset / "agents.jsonl")
    events, _ = load_events(dataset / "events.jsonl", agents, {})
    event_by_id = {event["event_id"]: event for event in events}
    retrieval, retrieval_hash = read_snapshot(public_retrieval_path)
    for row in retrieval:
        event = event_by_id.get(row["event_id"])
        if event is None or row.get("source_index") != event["source_index"]:
            raise ValueError("Public retrieval cache does not match the dataset")
    target_ids = {row["event_id"] for row in retrieval}
    cohort = [event for event in events if event["event_id"] in target_ids][:cohort_limit]
    if len(cohort) != cohort_limit:
        raise ValueError("Public retrieval has fewer targets than cohort_limit")
    cohort_ids = {event["event_id"] for event in cohort}
    public_rows, public_hash = read_snapshot(public / "rerank_checkpoint.jsonl")
    private_rows, private_hash = read_snapshot(private / "rerank_checkpoint.jsonl")
    public_by_id = {row["event_id"]: row for row in public_rows if row["event_id"] in cohort_ids}
    private_by_id = {row["event_id"]: row for row in private_rows if row["event_id"] in cohort_ids}
    pairs: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": []}
    coverage: dict[str, dict[str, int]] = {}
    public_edges = []
    for split, selected in (("train", cohort[:train_targets]), ("holdout", cohort[train_targets:])):
        stats = {"targets": len(selected), "public_completed_targets": 0, "private_completed_targets": 0,
                 "both_completed_targets": 0, "matched_pairs": 0, "public_only_pairs": 0, "private_only_pairs": 0}
        for event in selected:
            target_id = event["event_id"]
            public_row, private_row = public_by_id.get(target_id), private_by_id.get(target_id)
            stats["public_completed_targets"] += public_row is not None
            stats["private_completed_targets"] += private_row is not None
            public_scores = _candidate_scores(public_row, event_by_id, cohort_ids) if public_row is not None else {}
            private_scores = _candidate_scores(private_row, event_by_id, cohort_ids) if private_row is not None else {}
            for source_id, value in public_scores.items():
                public_edges.append({"source_event_id": source_id, "target_event_id": target_id, "split": split,
                                     "elapsed_hours": value["elapsed_hours"], "public_score": value["score"]})
            if public_row is None or private_row is None:
                continue
            stats["both_completed_targets"] += 1
            common = public_scores.keys() & private_scores.keys()
            stats["public_only_pairs"] += len(public_scores.keys() - private_scores.keys())
            stats["private_only_pairs"] += len(private_scores.keys() - public_scores.keys())
            for source_id in sorted(common):
                public_score, private_score = public_scores[source_id], private_scores[source_id]
                pairs[split].append({"source_event_id": source_id, "target_event_id": target_id, "split": split,
                                     "elapsed_hours": public_score["elapsed_hours"], "public_score": public_score["score"],
                                     "private_score": private_score["score"]})
        stats["matched_pairs"] = len(pairs[split])
        coverage[split] = stats
    decay_fit = fit_temporal_decay(pairs["train"], lambda_max=lambda_max, steps=grid_steps)
    sigmoid_decay_fit = fit_sigmoid_decay(pairs["train"], lambda_max=lambda_max)
    ranking_fit = fit_sigmoid_ranking(pairs["train"])
    fit = sigmoid_decay_fit if model == "sigmoid-decay" else decay_fit
    decay_lambda = decay_fit["lambda_per_hour"]
    alpha = fit["alpha"] if model == "sigmoid-decay" else None
    intercept = fit["intercept"] if model == "sigmoid-decay" else 0.0
    for row in public_edges:
        row["decayed_public_score"] = row["public_score"] * math.exp(-decay_lambda * row["elapsed_hours"])
        row["sigmoid_decay_public_score"] = sigmoid_decay_score(
            row["public_score"], row["elapsed_hours"], sigmoid_decay_fit["alpha"],
            sigmoid_decay_fit["intercept"], sigmoid_decay_fit["lambda_per_hour"]
        )
        row["ranking_calibrated_public_score"] = sigmoid_ranking_score(
            row["public_score"], ranking_fit["alpha"], ranking_fit["intercept"]
        )
        row["calibrated_public_score"] = row["sigmoid_decay_public_score"] if model == "sigmoid-decay" else row["decayed_public_score"]
    for row in pairs["train"] + pairs["holdout"]:
        row["decayed_public_score"] = row["public_score"] * math.exp(-decay_lambda * row["elapsed_hours"])
        row["sigmoid_decay_public_score"] = sigmoid_decay_score(
            row["public_score"], row["elapsed_hours"], sigmoid_decay_fit["alpha"],
            sigmoid_decay_fit["intercept"], sigmoid_decay_fit["lambda_per_hour"]
        )
        row["ranking_calibrated_public_score"] = sigmoid_ranking_score(
            row["public_score"], ranking_fit["alpha"], ranking_fit["intercept"]
        )
        row["calibrated_public_score"] = row["sigmoid_decay_public_score"] if model == "sigmoid-decay" else row["decayed_public_score"]
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "method": "pair-weighted training MSE; elapsed time in hours",
        "cohort": {"targets": len(cohort), "training_targets": train_targets,
                   "first_timestamp": cohort[0]["timestamp"], "last_timestamp": cohort[-1]["timestamp"],
                   "event_ids_sha256": hashlib.sha256("\n".join(sorted(cohort_ids)).encode("utf-8")).hexdigest()},
        "input_hashes": {"events_sha256": public_retrieval_config["events_sha256"],
                         "agents_sha256": public_retrieval_config["agents_sha256"],
                         "pass1_sha256": public_retrieval_config["pass1_sha256"],
                         "public_retrieval_sha256": retrieval_hash, "public_rerank_snapshot_sha256": public_hash,
                         "private_retrieval_sha256": private_config["retrieval_sha256"],
                         "private_rerank_snapshot_sha256": private_hash},
        "candidate_filters": {"public": public_config.get("candidate_filter", "none"),
                              "private": private_config.get("candidate_filter", "none")},
        "rerank_models": {"public": public_config.get("rerank_model"), "private": private_config.get("rerank_model")},
        "threshold": threshold, "coverage": coverage, "fit": fit,
        "sigmoid_decay_fit": sigmoid_decay_fit, "decay_only_fit": decay_fit, "ranking_only_fit": ranking_fit,
        "metrics": {split: {
            "baseline": score_metrics(rows, 0, threshold),
            "decayed": score_metrics(rows, decay_lambda, threshold),
            "sigmoid_decay": score_metrics(rows, sigmoid_decay_fit["lambda_per_hour"], threshold,
                                            alpha=sigmoid_decay_fit["alpha"], intercept=sigmoid_decay_fit["intercept"]),
            "ranking_only": score_metrics(rows, 0, threshold, alpha=ranking_fit["alpha"],
                                           intercept=ranking_fit["intercept"]),
        } for split, rows in pairs.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "calibration.json", result)
    write_jsonl(output / "matched_pairs.jsonl", pairs["train"] + pairs["holdout"])
    write_jsonl(output / "public_decayed_edges.jsonl", public_edges)
    return result
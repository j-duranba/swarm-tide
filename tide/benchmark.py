from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
import time
from typing import Any

from tide.graphing import aggregate_agent_matrix, build_event_graph
from tide.io_utils import ensure_dir, write_json, write_jsonl
from tide.retrieval import RetrievalConfig, retrieve_candidates, strictly_precedes
from tide.schema import EventRecord, build_event
from tide.support import SupportConfig, score_candidates
from tide.synthetic import GroundTruthRelation, SyntheticBundle, generate_synthetic_trace


def _truth_direct(relations: list[GroundTruthRelation]) -> set[tuple[str, str]]:
    return {
        (r.source_event_id, r.target_event_id)
        for r in relations
        if r.direct and r.support_type != "indirect_path_only"
    }


def _truth_by_target(relations: list[GroundTruthRelation]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in relations:
        if r.direct and r.support_type != "indirect_path_only":
            out[r.target_event_id].add(r.source_event_id)
    return out


def _event_index(events: list[EventRecord]) -> dict[str, EventRecord]:
    return {e.event_id: e for e in events}


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _average_precision(y_true: list[int], y_score: list[float]) -> float:
    paired = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    total_pos = sum(y_true)
    if total_pos == 0:
        return 0.0
    ap = 0.0
    for i, (_, label) in enumerate(paired, start=1):
        if label == 1:
            tp += 1
            ap += tp / i
        else:
            fp += 1
    return ap / total_pos


def _to_event_dict(event: EventRecord) -> dict[str, Any]:
    d = event.as_dict()
    d["timestamp"] = d.pop("timestamp")
    return d


def _apply_ablation(events: list[EventRecord], mode: str, seed: int) -> list[EventRecord]:
    rng = random.Random(seed)
    out: list[EventRecord] = []
    for event in events:
        row = _to_event_dict(event)
        if mode == "no_nl_outputs":
            row["action_summary"] = None
            row["agent_output"] = None
        elif mode == "no_action_arguments":
            row["input"] = None
        elif mode == "no_action_results":
            row["result"] = None
        elif mode == "no_artifact_ids":
            row["artifact_refs"] = []
        elif mode == "only_action_and_time":
            row["action_summary"] = None
            row["tool"] = None
            row["input"] = None
            row["result"] = None
            row["agent_output"] = None
            row["artifact_refs"] = []
            row["agent_goal"] = None
            row["shared_goal"] = None
        elif mode == "missing_selected_fields_30pct":
            for field in ("input", "result", "agent_output", "action_summary"):
                if rng.random() < 0.3:
                    row[field] = None
        elif mode == "timestamp_noise":
            if row["timestamp"]:
                dt = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
                dt += timedelta(seconds=rng.randint(-8, 8))
                row["timestamp"] = dt.isoformat()
        elif mode == "increased_delay":
            if row["event_id"] == "e23" and row["timestamp"]:
                dt = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
                row["timestamp"] = (dt + timedelta(seconds=180)).isoformat()
        out.append(build_event(row, source_table=event.source_table, source_record_id=event.source_record_id))
    if mode == "missing_events_20pct":
        keep = []
        for e in out:
            if rng.random() >= 0.2 or e.agent_id == "ENV":
                keep.append(e)
        out = keep
    return out


def _retrieval_configs() -> dict[str, RetrievalConfig]:
    base = RetrievalConfig()
    return {
        "semantic_only": RetrievalConfig(top_k_cross_agent=base.top_k_cross_agent, same_agent_history_k=base.same_agent_history_k, max_time_lag_seconds=base.max_time_lag_seconds, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0),
        "semantic_plus_temporal": RetrievalConfig(top_k_cross_agent=base.top_k_cross_agent, same_agent_history_k=base.same_agent_history_k, max_time_lag_seconds=base.max_time_lag_seconds, alpha=0.8, beta=0.2, gamma=0.0, delta=0.0),
        "semantic_temporal_artifact": base,
        "recent_history_baseline": RetrievalConfig(top_k_cross_agent=0, same_agent_history_k=6, max_time_lag_seconds=base.max_time_lag_seconds, alpha=0.0, beta=0.0, gamma=0.0, delta=1.0),
    }


def default_retrieval_config() -> RetrievalConfig:
    return _retrieval_configs()["semantic_temporal_artifact"]


def _oracle_candidates(events: list[EventRecord], truth: list[GroundTruthRelation]) -> dict[str, list[str]]:
    by_target = _truth_by_target(truth)
    ev = _event_index(events)
    out: dict[str, list[str]] = {}
    for target_id, sources in by_target.items():
        target = ev[target_id]
        valid = [s for s in sources if s in ev and strictly_precedes(ev[s], target)]
        out[target_id] = sorted(valid)
    return out


def _recall_at_k(candidate_map: dict[str, list[str]], by_target_truth: dict[str, set[str]], k: int) -> float:
    vals = []
    for target, true_sources in by_target_truth.items():
        if not true_sources:
            continue
        preds = candidate_map.get(target, [])[:k]
        vals.append(len(set(preds) & true_sources) / len(true_sources))
    return _avg(vals)


def _retrieval_eval(events: list[EventRecord], truth: list[GroundTruthRelation], strategy_name: str, config: RetrievalConfig | None) -> dict[str, Any]:
    by_target_truth = _truth_by_target(truth)
    ev = _event_index(events)
    t0 = time.perf_counter()
    if strategy_name == "oracle":
        candidate_ids = _oracle_candidates(events, truth)
        avg_size = _avg([float(len(v)) for v in candidate_ids.values()]) if candidate_ids else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000
    else:
        raw = retrieve_candidates(events, config=config)
        candidate_ids = {k: [c.source_event_id for c in v] for k, v in raw.items()}
        avg_size = _avg([float(len(v)) for v in raw.values()]) if raw else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000

    delayed_targets = {r.target_event_id for r in truth if r.scenario == "delayed_use" and r.direct}
    artifact_targets = {r.target_event_id for r in truth if r.support_type == "artifact-mediated"}

    same_truth_targets: dict[str, set[str]] = defaultdict(set)
    cross_truth_targets: dict[str, set[str]] = defaultdict(set)
    for target, srcs in by_target_truth.items():
        for s in srcs:
            if ev[s].agent_id == ev[target].agent_id:
                same_truth_targets[target].add(s)
            else:
                cross_truth_targets[target].add(s)

    out = {
        "strategy": strategy_name,
        "candidate_avg_size": avg_size,
        "runtime_ms": elapsed_ms,
        "recall_at": {str(k): _recall_at_k(candidate_ids, by_target_truth, k) for k in (3, 5, 10, 20)},
        "same_agent_recall_at_10": _recall_at_k(candidate_ids, same_truth_targets, 10),
        "cross_agent_recall_at_10": _recall_at_k(candidate_ids, cross_truth_targets, 10),
        "delayed_recall_at_10": _recall_at_k(candidate_ids, {t: by_target_truth[t] for t in delayed_targets if t in by_target_truth}, 10),
        "artifact_recall_at_10": _recall_at_k(candidate_ids, {t: by_target_truth[t] for t in artifact_targets if t in by_target_truth}, 10),
    }
    return out


def _pipeline_eval(
    events: list[EventRecord],
    truth: list[GroundTruthRelation],
    retrieval_cfg: RetrievalConfig,
    threshold: float = 0.5,
    support_cfg: SupportConfig | None = None,
) -> dict[str, Any]:
    by_id = _event_index(events)
    candidates = retrieve_candidates(events, retrieval_cfg)
    all_candidates = [c for x in candidates.values() for c in x]
    scored = score_candidates(by_id, all_candidates, config=support_cfg)

    gt = {
        (s, t)
        for (s, t) in _truth_direct(truth)
        if s in by_id and t in by_id and strictly_precedes(by_id[s], by_id[t])
    }
    y_true = [1 if (e.source_event_id, e.target_event_id) in gt else 0 for e in scored]
    y_score = [float(e.support_score) for e in scored]
    y_pred = [1 if s >= threshold and not e.abstain else 0 for s, e in zip(y_score, scored)]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    ev = by_id
    edge_lookup = {(e.source_event_id, e.target_event_id): e for e in scored}
    graph = build_event_graph(events, scored, min_score=threshold)
    agent_matrix = aggregate_agent_matrix(events, graph)

    # Scenario-focused metrics.
    def scenario_pairs(name: str) -> list[tuple[str, str]]:
        ids = {e.event_id for e in events if e.episode_id == name}
        return [pair for pair in edge_lookup if pair[0] in ids and pair[1] in ids]

    independent_scenarios = {e.episode_id for e in events if e.episode_id == "independent_identical_task" or e.episode_id.startswith("hardneg_")}
    independent_pairs = [pair for name in independent_scenarios for pair in scenario_pairs(name)]
    shared_pairs = scenario_pairs("shared_external_event")
    explicit_pairs = scenario_pairs("explicit_communication")
    resource_pairs = scenario_pairs("shared_resource")
    broker_pairs = scenario_pairs("broker_aggregator")
    misleading_pairs = scenario_pairs("misleading_information")
    delayed_pairs = scenario_pairs("delayed_use")

    def pair_precision(pairs: list[tuple[str, str]]) -> float:
        selected = [p for p in pairs if edge_lookup[p].support_score >= threshold and not edge_lookup[p].abstain]
        if not selected:
            return 0.0
        good = sum(1 for p in selected if p in gt)
        return good / len(selected)

    def fp_rate_negative(pairs: list[tuple[str, str]], forbid_env: bool = False) -> float:
        negatives = []
        false_positives = 0
        for s, t in pairs:
            if forbid_env and ("ENV" in {ev[s].agent_id, ev[t].agent_id}):
                continue
            if (s, t) in gt:
                continue
            negatives.append((s, t))
            edge = edge_lookup[(s, t)]
            if edge.support_score >= threshold and not edge.abstain:
                false_positives += 1
        if not negatives:
            return 0.0
        return false_positives / len(negatives)

    broker_detected = all(p in {(e["source_event_id"], e["target_event_id"]) for e in graph.edges} for p in [("e14", "e16"), ("e15", "e16"), ("e16", "e17")])
    misleading_detected = ("e18", "e19") in {(e["source_event_id"], e["target_event_id"]) for e in graph.edges}
    delayed_detected = ("e20", "e23") in {(e["source_event_id"], e["target_event_id"]) for e in graph.edges}
    incorrect_direct_ac = ("e24", "e27") in {(e["source_event_id"], e["target_event_id"]) for e in graph.edges}

    # Error examples.
    false_pos = [
        {
            "source_event_id": e.source_event_id,
            "target_event_id": e.target_event_id,
            "score": e.support_score,
            "support_type": e.support_type,
            "scenario": by_id[e.target_event_id].episode_id,
            "source_agent": by_id[e.source_event_id].agent_id,
            "target_agent": by_id[e.target_event_id].agent_id,
            "evidence": list(e.evidence),
        }
        for e in sorted(scored, key=lambda x: x.support_score, reverse=True)
        if e.support_score >= threshold and not e.abstain and (e.source_event_id, e.target_event_id) not in gt
    ][:8]
    false_neg = []
    scored_lookup = {(e.source_event_id, e.target_event_id): e for e in scored}
    for pair in sorted(gt):
        edge = scored_lookup.get(pair)
        if edge is None or edge.abstain or edge.support_score < threshold:
            false_neg.append(
                {
                    "source_event_id": pair[0],
                    "target_event_id": pair[1],
                    "scenario": by_id[pair[1]].episode_id if pair[1] in by_id else "missing_target",
                    "reason": "not retrieved or below threshold",
                }
            )
    false_neg = false_neg[:8]

    return {
        "threshold": threshold,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": _f1(precision, recall),
        "event_auprc": _average_precision(y_true, y_score),
        "false_positive_rate_independent_identical_task": fp_rate_negative(independent_pairs),
        "false_positive_rate_shared_external_non_env": fp_rate_negative(shared_pairs, forbid_env=True),
        "precision_explicit_communication": pair_precision(explicit_pairs),
        "precision_shared_resource": pair_precision(resource_pairs),
        "precision_broker_aggregator": pair_precision(broker_pairs),
        "precision_misleading_transfer": pair_precision(misleading_pairs),
        "precision_delayed": pair_precision(delayed_pairs),
        "broker_pattern_detected": broker_detected,
        "misleading_dependency_detected": misleading_detected,
        "delayed_dependency_detected": delayed_detected,
        "incorrect_direct_A_to_C_in_indirect_chain": incorrect_direct_ac,
        "candidate_count": len(all_candidates),
        "scored_pair_count": len(scored),
        "graph_edge_count": len(graph.edges),
        "false_positive_examples": false_pos,
        "false_negative_examples": false_neg,
        "agent_matrix": agent_matrix,
    }


def _aggregate_metric(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    return _avg(vals)


def run_extended_validation(output_dir: str, seeds: list[int], runs_per_seed: int = 3) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    bundles: list[SyntheticBundle] = []
    for seed in seeds:
        for run_id in range(runs_per_seed):
            bundles.append(generate_synthetic_trace(seed=seed, run_id=run_id))

    retrieval_rows: list[dict[str, Any]] = []
    pipeline_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    detailed_errors: list[dict[str, Any]] = []

    retrieval_cfgs = _retrieval_configs()
    ablations = [
        "full",
        "no_nl_outputs",
        "no_action_arguments",
        "no_action_results",
        "no_artifact_ids",
        "only_action_and_time",
        "missing_events_20pct",
        "missing_selected_fields_30pct",
        "timestamp_noise",
        "increased_delay",
    ]

    for bundle in bundles:
        events = list(bundle.events)
        truth = list(bundle.ground_truth_relations)

        # Retrieval experiments.
        for name, cfg in retrieval_cfgs.items():
            row = _retrieval_eval(events, truth, name, cfg)
            row.update({"seed": bundle.metadata["seed"], "run_id": bundle.metadata["run_id"]})
            retrieval_rows.append(row)
        oracle = _retrieval_eval(events, truth, "oracle", None)
        oracle.update({"seed": bundle.metadata["seed"], "run_id": bundle.metadata["run_id"]})
        retrieval_rows.append(oracle)

        # Full pipeline.
        p = _pipeline_eval(events, truth, retrieval_cfgs["semantic_temporal_artifact"], threshold=0.5)
        p.update({"seed": bundle.metadata["seed"], "run_id": bundle.metadata["run_id"], "ablation": "full"})
        pipeline_rows.append(p)
        detailed_errors.append(
            {
                "seed": bundle.metadata["seed"],
                "run_id": bundle.metadata["run_id"],
                "false_positive_examples": p["false_positive_examples"],
                "false_negative_examples": p["false_negative_examples"],
            }
        )

        for ab in ablations[1:]:
            ab_events = _apply_ablation(events, ab, seed=bundle.metadata["seed"] + bundle.metadata["run_id"])
            ap = _pipeline_eval(ab_events, truth, retrieval_cfgs["semantic_temporal_artifact"], threshold=0.5)
            ap.update({"seed": bundle.metadata["seed"], "run_id": bundle.metadata["run_id"], "ablation": ab})
            ablation_rows.append(ap)

    # Aggregates.
    retrieval_summary: dict[str, Any] = {}
    for strategy in sorted({r["strategy"] for r in retrieval_rows}):
        rows = [r for r in retrieval_rows if r["strategy"] == strategy]
        retrieval_summary[strategy] = {
            "recall@3": _avg([r["recall_at"]["3"] for r in rows]),
            "recall@5": _avg([r["recall_at"]["5"] for r in rows]),
            "recall@10": _avg([r["recall_at"]["10"] for r in rows]),
            "recall@20": _avg([r["recall_at"]["20"] for r in rows]),
            "same_agent_recall@10": _avg([r["same_agent_recall_at_10"] for r in rows]),
            "cross_agent_recall@10": _avg([r["cross_agent_recall_at_10"] for r in rows]),
            "delayed_recall@10": _avg([r["delayed_recall_at_10"] for r in rows]),
            "artifact_recall@10": _avg([r["artifact_recall_at_10"] for r in rows]),
            "avg_candidate_size": _avg([r["candidate_avg_size"] for r in rows]),
            "avg_runtime_ms": _avg([r["runtime_ms"] for r in rows]),
        }

    full_rows = [r for r in pipeline_rows if r["ablation"] == "full"]
    full_summary = {
        "event_precision": _aggregate_metric(full_rows, "event_precision"),
        "event_recall": _aggregate_metric(full_rows, "event_recall"),
        "event_f1": _aggregate_metric(full_rows, "event_f1"),
        "event_auprc": _aggregate_metric(full_rows, "event_auprc"),
        "fpr_independent_identical_task": _aggregate_metric(full_rows, "false_positive_rate_independent_identical_task"),
        "fpr_shared_external_non_env": _aggregate_metric(full_rows, "false_positive_rate_shared_external_non_env"),
        "precision_explicit_communication": _aggregate_metric(full_rows, "precision_explicit_communication"),
        "precision_shared_resource": _aggregate_metric(full_rows, "precision_shared_resource"),
        "precision_broker_aggregator": _aggregate_metric(full_rows, "precision_broker_aggregator"),
        "precision_misleading_transfer": _aggregate_metric(full_rows, "precision_misleading_transfer"),
        "precision_delayed": _aggregate_metric(full_rows, "precision_delayed"),
        "broker_pattern_detected_rate": _avg([1.0 if r["broker_pattern_detected"] else 0.0 for r in full_rows]),
        "misleading_dependency_detected_rate": _avg([1.0 if r["misleading_dependency_detected"] else 0.0 for r in full_rows]),
        "delayed_dependency_detected_rate": _avg([1.0 if r["delayed_dependency_detected"] else 0.0 for r in full_rows]),
        "incorrect_direct_A_to_C_rate": _avg([1.0 if r["incorrect_direct_A_to_C_in_indirect_chain"] else 0.0 for r in full_rows]),
    }

    ablation_summary: dict[str, Any] = {}
    for ab in sorted({r["ablation"] for r in ablation_rows}):
        rows = [r for r in ablation_rows if r["ablation"] == ab]
        ablation_summary[ab] = {
            "event_f1": _aggregate_metric(rows, "event_f1"),
            "event_recall": _aggregate_metric(rows, "event_recall"),
            "event_precision": _aggregate_metric(rows, "event_precision"),
            "event_auprc": _aggregate_metric(rows, "event_auprc"),
            "delayed_detection_rate": _avg([1.0 if r["delayed_dependency_detected"] else 0.0 for r in rows]),
        }

    summary = {
        "config": {"seeds": seeds, "runs_per_seed": runs_per_seed, "total_runs": len(bundles), "threshold": 0.5},
        "retrieval_summary": retrieval_summary,
        "pipeline_full_summary": full_summary,
        "ablation_summary": ablation_summary,
    }

    write_jsonl(out / "retrieval_results.jsonl", retrieval_rows)
    write_jsonl(out / "pipeline_results.jsonl", pipeline_rows)
    write_jsonl(out / "ablation_results.jsonl", ablation_rows)
    write_jsonl(out / "error_examples.jsonl", detailed_errors)
    write_json(out / "extended_summary.json", summary)

    return {
        "summary": summary,
        "retrieval_rows": retrieval_rows,
        "pipeline_rows": pipeline_rows,
        "ablation_rows": ablation_rows,
        "errors": detailed_errors,
    }


def analyze_independent_false_positives(
    seeds: list[int], runs_per_seed: int, threshold: float, retrieval_cfg: RetrievalConfig, support_cfg: SupportConfig | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    support_cfg = support_cfg or SupportConfig()
    for seed in seeds:
        for run_id in range(runs_per_seed):
            bundle = generate_synthetic_trace(seed=seed, run_id=run_id)
            events = list(bundle.events)
            by_id = _event_index(events)
            truth = _truth_direct(list(bundle.ground_truth_relations))
            candidates = retrieve_candidates(events, retrieval_cfg)
            all_candidates = [c for cs in candidates.values() for c in cs]
            scored = score_candidates(by_id, all_candidates, config=support_cfg)

            rank_lookup: dict[tuple[str, str], tuple[int, float, float, float, float]] = {}
            for target_id, cs in candidates.items():
                for rank, c in enumerate(cs, start=1):
                    rank_lookup[(c.source_event_id, target_id)] = (
                        rank,
                        c.retrieval_score,
                        c.semantic_similarity,
                        c.temporal_score,
                        c.artifact_overlap,
                    )

            for edge in scored:
                pair = (edge.source_event_id, edge.target_event_id)
                if pair in truth or edge.abstain or edge.support_score < threshold:
                    continue
                src = by_id[edge.source_event_id]
                tgt = by_id[edge.target_event_id]
                if not (tgt.episode_id == "independent_identical_task" or tgt.episode_id.startswith("hardneg_")):
                    continue
                if src.agent_id == tgt.agent_id:
                    continue
                same_agent_alt = [
                    e
                    for e in scored
                    if e.target_event_id == tgt.event_id
                    and by_id[e.source_event_id].agent_id == tgt.agent_id
                ]
                rank, retrieval_score, sem, temp, art = rank_lookup.get(pair, (None, None, None, None, None))
                lag = None
                if src.timestamp_utc and tgt.timestamp_utc:
                    lag = (
                        datetime.fromisoformat(tgt.timestamp_utc.replace("Z", "+00:00"))
                        - datetime.fromisoformat(src.timestamp_utc.replace("Z", "+00:00"))
                    ).total_seconds()
                category = "other_or_unclear"
                evidence_text = " ".join(edge.evidence).lower()
                if "same public tool/input" in evidence_text or "same-agent explanation stronger" in " ".join(edge.alternative_explanations).lower():
                    category = "parallel_independent_execution"
                elif src.shared_goal and src.shared_goal == tgt.shared_goal:
                    category = "shared_task_or_instruction_effect"
                elif "contextual" in edge.support_type:
                    category = "temporal_proximity_without_transfer"
                elif sem is not None and sem > 0.8:
                    category = "coincidental_semantic_similarity"

                rows.append(
                    {
                        "seed": seed,
                        "run_id": run_id,
                        "scenario": tgt.episode_id,
                        "source_event_id": src.event_id,
                        "target_event_id": tgt.event_id,
                        "source_agent": src.agent_id,
                        "target_agent": tgt.agent_id,
                        "different_agents": src.agent_id != tgt.agent_id,
                        "time_lag_seconds": lag,
                        "semantic_similarity": sem,
                        "retrieval_score": retrieval_score,
                        "retrieval_rank": rank,
                        "temporal_score": temp,
                        "artifact_overlap": art,
                        "support_score": edge.support_score,
                        "support_type": edge.support_type,
                        "evidence": list(edge.evidence),
                        "alternative_explanations": list(edge.alternative_explanations),
                        "same_agent_explanations": [
                            {
                                "source_event_id": s.source_event_id,
                                "support_score": s.support_score,
                                "support_type": s.support_type,
                            }
                            for s in sorted(same_agent_alt, key=lambda x: x.support_score, reverse=True)[:3]
                        ],
                        "shared_goal_match": bool(src.shared_goal and src.shared_goal == tgt.shared_goal),
                        "classification": category,
                    }
                )
    return rows


def run_heldout_method_comparison(
    train_seeds: list[int], eval_seeds: list[int], runs_per_seed: int = 3
) -> dict[str, Any]:
    retrieval_cfg = _retrieval_configs()["semantic_temporal_artifact"]
    profiles = [
        {"name": "baseline_t05", "threshold": 0.5, "cfg": SupportConfig()},
        {"name": "baseline_t07", "threshold": 0.7, "cfg": SupportConfig()},
        {
            "name": "strict_cross_agent",
            "threshold": 0.6,
            "cfg": SupportConfig(require_strong_channel_for_cross_agent=True),
        },
        {
            "name": "context_aware_abstain",
            "threshold": 0.6,
            "cfg": SupportConfig(require_strong_channel_for_cross_agent=True, abstain_on_ambiguous=True),
        },
    ]

    def eval_on(seeds: list[int]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for profile in profiles:
            rows = []
            for seed in seeds:
                for run_id in range(runs_per_seed):
                    bundle = generate_synthetic_trace(seed=seed, run_id=run_id)
                    row = _pipeline_eval(
                        list(bundle.events),
                        list(bundle.ground_truth_relations),
                        retrieval_cfg,
                        threshold=profile["threshold"],
                        support_cfg=profile["cfg"],
                    )
                    rows.append(row)
            out[profile["name"]] = {
                "event_precision": _aggregate_metric(rows, "event_precision"),
                "event_recall": _aggregate_metric(rows, "event_recall"),
                "event_f1": _aggregate_metric(rows, "event_f1"),
                "event_auprc": _aggregate_metric(rows, "event_auprc"),
                "fpr_independent": _aggregate_metric(rows, "false_positive_rate_independent_identical_task"),
                "fpr_shared_external_non_env": _aggregate_metric(rows, "false_positive_rate_shared_external_non_env"),
                "broker_detect_rate": _avg([1.0 if r["broker_pattern_detected"] else 0.0 for r in rows]),
                "delayed_detect_rate": _avg([1.0 if r["delayed_dependency_detected"] else 0.0 for r in rows]),
                "indirect_direct_error_rate": _avg([1.0 if r["incorrect_direct_A_to_C_in_indirect_chain"] else 0.0 for r in rows]),
            }
        return out

    return {
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "runs_per_seed": runs_per_seed,
        "train_results": eval_on(train_seeds),
        "heldout_results": eval_on(eval_seeds),
    }


def write_false_positive_report(path: str, rows: list[dict[str, Any]]) -> None:
    unique = []
    seen = set()
    for row in rows:
        key = (row["scenario"], row["source_event_id"], row["target_event_id"], row["classification"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    by_class: dict[str, int] = defaultdict(int)
    by_scenario: dict[str, int] = defaultdict(int)
    for row in rows:
        by_class[row["classification"]] += 1
        by_scenario[row["scenario"]] += 1
    lines = [
        "# False Positive Forensic Analysis (Independent-Agent Controls)\n\n",
        f"- Total independent-scenario false positives inspected: **{len(rows)}**\n",
        "- Classification categories:\n",
    ]
    for k, v in sorted(by_class.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  - {k}: {v}\n")
    lines.append("\n- Scenario distribution:\n")
    for k, v in sorted(by_scenario.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  - {k}: {v}\n")
    lines.append(
        "\n## Representative cases\n\n"
        "| Scenario | Source -> Target | Lag(s) | SemSim | Retrieval rank | Retrieval score | Support score/type | Classification |\n"
        "|---|---|---:|---:|---:|---:|---|---|\n"
    )
    for row in unique[:30]:
        lines.append(
            f"| {row['scenario']} | {row['source_event_id']}({row['source_agent']}) -> {row['target_event_id']}({row['target_agent']}) | "
            f"{(row['time_lag_seconds'] if row['time_lag_seconds'] is not None else 'NA')} | "
            f"{(row['semantic_similarity'] if row['semantic_similarity'] is not None else 0):.3f} | "
            f"{row['retrieval_rank']} | {(row['retrieval_score'] if row['retrieval_score'] is not None else 0):.3f} | "
            f"{row['support_score']:.3f} ({row['support_type']}) | {row['classification']} |\n"
        )
    lines.append(
        "\n## Failure mode conclusion\n\n"
        "The current independent-agent FPR is driven primarily by systematic shared-task/parallel-execution patterns, "
        "not isolated random outliers. Cross-agent edges are still inferred when both agents independently surface the same fact.\n"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(lines), encoding="utf-8")


def write_hard_negative_report(path: str, method_results: dict[str, Any], summary_after: dict[str, Any]) -> None:
    heldout = method_results["heldout_results"]
    lines = [
        "# Hard-Negative Benchmark and Held-out Method Comparison\n\n",
        "- Baseline pre-hard-control results preserved at `outputs/extended_validation_baseline_v1/`.\n",
        f"- Train seeds: `{method_results['train_seeds']}`\n",
        f"- Held-out seeds: `{method_results['eval_seeds']}`\n",
        f"- Runs per seed: `{method_results['runs_per_seed']}`\n\n",
        "## Held-out profile comparison\n\n",
        "| Profile | Precision | Recall | F1 | AUPRC | Independent FPR | Shared-external FPR | Delayed detect | Broker detect | Indirect A->C error |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for name, row in heldout.items():
        lines.append(
            f"| {name} | {row['event_precision']:.3f} | {row['event_recall']:.3f} | {row['event_f1']:.3f} | {row['event_auprc']:.3f} | "
            f"{row['fpr_independent']:.3f} | {row['fpr_shared_external_non_env']:.3f} | {row['delayed_detect_rate']:.3f} | "
            f"{row['broker_detect_rate']:.3f} | {row['indirect_direct_error_rate']:.3f} |\n"
        )
    lines.append(
        "\n## Updated synthetic benchmark summary\n\n"
        f"- Current full benchmark precision/recall/F1: "
        f"{summary_after['pipeline_full_summary']['event_precision']:.3f} / "
        f"{summary_after['pipeline_full_summary']['event_recall']:.3f} / "
        f"{summary_after['pipeline_full_summary']['event_f1']:.3f}\n"
        f"- Independent-scenario FPR: {summary_after['pipeline_full_summary']['fpr_independent_identical_task']:.3f}\n"
        f"- Indirect direct-edge error rate: {summary_after['pipeline_full_summary']['incorrect_direct_A_to_C_rate']:.3f}\n"
    )
    lines.append(
        "\n## Generalization note\n\n"
        "Train and held-out scores are currently very close because the synthetic generator randomizes values more than structure. "
        "So this held-out split is still useful for leakage checks, but not a strong stress-test of structural generalization.\n"
    )
    lines.append(
        "\n## First real-data protocol (no download in this step)\n\n"
        "1. Run local schema audit first: `python -m tide.cli audit-dataset --dataset-dir <LOCAL_DATA_DIR> --output reports/real_data_audit.json`.\n"
        "2. Normalize one small table slice (100-500 rows) while preserving raw records: "
        "`python -m tide.cli normalize-events --input <TABLE.jsonl|csv> --limit 300 --output outputs/normalized_slice.jsonl --raw-output outputs/normalized_slice_raw.jsonl`.\n"
        "3. Manually inspect normalized events for timestamp/agent/provenance/missingness fields.\n"
        "4. Run dependency inference only on that slice and inspect top predicted cross-agent edges manually before scaling.\n"
        "5. Track observed missingness and unresolved ordering explicitly in the audit report.\n"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(lines), encoding="utf-8")


def write_extended_report(report_path: str, results: dict[str, Any]) -> None:
    summary = results["summary"]
    retrieval = summary["retrieval_summary"]
    full = summary["pipeline_full_summary"]
    ablations = summary["ablation_summary"]
    errors = results["errors"]

    md = []
    md.append("# TIDE Extended Synthetic Validation\n")
    md.append("## 1) Experimental setup\n")
    md.append(
        f"- Synthetic-only benchmark with **{summary['config']['total_runs']} runs** "
        f"({len(summary['config']['seeds'])} seeds × {summary['config']['runs_per_seed']} run IDs).\n"
    )
    md.append("- Ground truth includes direct edges, mediated-only paths, misleading transfers, and shared external causes.\n")
    md.append("- Retrieval and support scores were evaluated separately; support scores are **heuristic scores, not calibrated probabilities**.\n")
    seeds_str = ",".join(str(s) for s in summary["config"]["seeds"])
    md.append(
        f"- Repro command: `python -m tide.cli run-extended-validation --output-dir outputs/extended_validation "
        f"--report-path reports/extended_synthetic_validation.md --seeds {seeds_str} --runs-per-seed {summary['config']['runs_per_seed']}`\n"
    )

    md.append("\n## 2) Retrieval comparison\n")
    md.append("| Strategy | R@3 | R@5 | R@10 | R@20 | Same-agent R@10 | Cross-agent R@10 | Delayed R@10 | Artifact R@10 | Avg candidate size |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, row in retrieval.items():
        md.append(
            f"| {name} | {row['recall@3']:.3f} | {row['recall@5']:.3f} | {row['recall@10']:.3f} | {row['recall@20']:.3f} | "
            f"{row['same_agent_recall@10']:.3f} | {row['cross_agent_recall@10']:.3f} | {row['delayed_recall@10']:.3f} | "
            f"{row['artifact_recall@10']:.3f} | {row['avg_candidate_size']:.2f} |\n"
        )

    md.append("\n## 3) Full-pipeline event dependency results\n")
    md.append("| Metric | Value |\n|---|---:|\n")
    for key in (
        "event_precision",
        "event_recall",
        "event_f1",
        "event_auprc",
        "fpr_independent_identical_task",
        "fpr_shared_external_non_env",
        "precision_explicit_communication",
        "precision_shared_resource",
        "precision_broker_aggregator",
        "precision_misleading_transfer",
        "precision_delayed",
        "broker_pattern_detected_rate",
        "misleading_dependency_detected_rate",
        "delayed_dependency_detected_rate",
        "incorrect_direct_A_to_C_rate",
    ):
        md.append(f"| {key} | {full[key]:.3f} |\n")

    md.append("\n## 4) Observability ablations\n")
    md.append("| Ablation | F1 | Precision | Recall | AUPRC | Delayed detection rate |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for name, row in ablations.items():
        md.append(
            f"| {name} | {row['event_f1']:.3f} | {row['event_precision']:.3f} | {row['event_recall']:.3f} | "
            f"{row['event_auprc']:.3f} | {row['delayed_detection_rate']:.3f} |\n"
        )

    md.append("\n## 5) Agent-level directionality and brokerage checks\n")
    md.append("- Matrix convention used in all code/results: **W[target][source]**.\n")
    md.append("- Broker detection requires receiving from multiple upstream sources and relaying downstream, not centrality alone.\n")
    md.append("- Indirect chain check explicitly tracks A→B→C and records whether incorrect direct A→C edges are introduced.\n")

    sample_errors = errors[0] if errors else {"false_positive_examples": [], "false_negative_examples": []}
    md.append("\n## 6) Representative errors\n")
    md.append("### False positives (sample)\n")
    md.append("```json\n")
    md.append(json.dumps(sample_errors["false_positive_examples"][:3], indent=2))
    md.append("\n```\n")
    md.append("### False negatives (sample)\n")
    md.append("```json\n")
    md.append(json.dumps(sample_errors["false_negative_examples"][:3], indent=2))
    md.append("\n```\n")

    md.append("\n## 7) Bugs/risk fixes implemented\n")
    md.append("- Added strict temporal precedence checks to prevent retrieval/graph edges when ordering is ambiguous.\n")
    md.append("- Added explicit missing-field/provenance metadata on normalized events.\n")
    md.append("- Added tests that fail on direction reversal and temporal ambiguity.\n")

    md.append("\n## 8) Limitations and next steps\n")
    md.append("- Support estimator remains heuristic; scores are not calibrated probabilities.\n")
    md.append("- Synthetic benchmark cannot prove external validity for real-world traces.\n")
    md.append("- Next: run same protocol on real traces with access-compliant ingestion, then compare failure modes.\n")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("".join(md), encoding="utf-8")

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import igraph as ig

from tide.io_utils import write_json, write_jsonl
from tide.pass2 import file_sha256, load_agent_registry, load_events, load_pass1_annotations
from tide.pass2_stages import filter_candidates_for_rerank


def pass1_edges(events: list[dict[str, Any]], names_to_ids: dict[str, str], *, reference_event_ids: set[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["author_id"] is not None and event["timestamp_dt"] is not None:
            by_agent[event["author_id"]].append(event)
    times = {}
    for agent_id, rows in by_agent.items():
        rows.sort(key=lambda row: (row["timestamp_dt"], row["source_index"]))
        times[agent_id] = [row["timestamp_dt"] for row in rows]

    edges: dict[tuple[str, str], set[str]] = defaultdict(set)
    counts = {"references": 0, "unknown_agent": 0, "missing_timestamp": 0, "no_matching_event": 0}
    for event in events:
        if reference_event_ids is not None and event["event_id"] not in reference_event_ids:
            continue
        relations = event.get("pass1_relations") or {}
        for field in ("from_agent", "to_agent"):
            for name in relations.get(field, []):
                counts["references"] += 1
                agent_id = names_to_ids.get(name)
                if agent_id is None:
                    counts["unknown_agent"] += 1
                    continue
                timestamp = event["timestamp_dt"]
                if timestamp is None:
                    counts["missing_timestamp"] += 1
                    continue
                agent_times = times.get(agent_id, [])
                index = bisect_left(agent_times, timestamp) - 1 if field == "from_agent" else bisect_right(agent_times, timestamp)
                if index < 0 or index >= len(agent_times):
                    counts["no_matching_event"] += 1
                    continue
                other = by_agent[agent_id][index]["event_id"]
                pair = (other, event["event_id"]) if field == "from_agent" else (event["event_id"], other)
                edges[pair].add(field)
    return [
        {"source_event_id": source, "target_event_id": target, "reference_types": sorted(types)}
        for (source, target), types in sorted(edges.items())
    ], counts


def pass2_edges(rows: list[dict[str, Any]], score_field: str, threshold: float, event_ids: set[str]) -> list[dict[str, Any]]:
    edges = []
    seen = set()
    for row in rows:
        target = row["event_id"]
        if target not in event_ids:
            raise ValueError(f"Unknown target event: {target}")
        for candidate in row["candidates"]:
            source = candidate["source_event_id"]
            if source not in event_ids:
                raise ValueError(f"Unknown source event: {source}")
            pair = (source, target)
            if pair in seen:
                raise ValueError(f"Duplicate candidate edge: {source} -> {target}")
            seen.add(pair)
            score = candidate[score_field]
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f"Invalid {score_field} for {source} -> {target}")
            if score_field == "dependency_score" and not 0 <= score <= 1:
                raise ValueError(f"Invalid dependency_score for {source} -> {target}")
            if score >= threshold:
                edge = {"source_event_id": source, "target_event_id": target}
                if score_field == "dependency_score":
                    edge["weight"] = score
                else:
                    edge["similarity_score"] = score
                edges.append(edge)
    return edges


def induced_event_edges(edges: list[dict[str, Any]], event_ids: set[str]) -> list[dict[str, Any]]:
    return [edge for edge in edges if edge["source_event_id"] in event_ids and edge["target_event_id"] in event_ids]


def calibrated_event_edges(rows: list[dict[str, Any]], expected_pairs: dict[tuple[str, str], float],
                           threshold: float) -> list[dict[str, Any]]:
    edges = []
    seen = set()
    for row in rows:
        pair = (row["source_event_id"], row["target_event_id"])
        if pair not in expected_pairs or pair in seen or row.get("public_score") != expected_pairs[pair]:
            raise ValueError(f"Calibrated pair does not match public reranking: {pair}")
        seen.add(pair)
        score = row.get("calibrated_public_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid calibrated score for {pair}")
        if score >= threshold:
            edges.append({"source_event_id": pair[0], "target_event_id": pair[1], "weight": float(score)})
    if seen != set(expected_pairs):
        raise ValueError("Calibrated scores do not cover every public reranked pair in the cohort")
    return edges


def direct_reference_graph(events: list[dict[str, Any]], agents: dict[str, dict[str, Any]],
                           names_to_ids: dict[str, str], min_references: int = 1) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if min_references < 1:
        raise ValueError("minimum references must be positive")
    author_ids = {event["author_id"] for event in events if event["author_id"] is not None}
    node_ids = set(author_ids)
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    counts = {"references": 0, "unknown_agent": 0, "missing_author": 0}
    for event in events:
        author = event["author_id"]
        for field in ("from_agent", "to_agent"):
            for name in set((event.get("pass1_relations") or {}).get(field, [])):
                counts["references"] += 1
                referenced = names_to_ids.get(name)
                if referenced is None:
                    counts["unknown_agent"] += 1
                    continue
                node_ids.add(referenced)
                if author is None:
                    counts["missing_author"] += 1
                    continue
                pair = (referenced, author) if field == "from_agent" else (author, referenced)
                if pair not in pairs:
                    pairs[pair] = {"source_agent_id": pair[0], "target_agent_id": pair[1],
                                   "from_agent_count": 0, "to_agent_count": 0, "weight": 0, "references": []}
                pairs[pair][field + "_count"] += 1
                pairs[pair]["weight"] += 1
                pairs[pair]["references"].append({"event_id": event["event_id"], "type": field})
    nodes = [{"id": agent_id, "name": (agents.get(agent_id) or {}).get("name"),
              "has_cohort_event": agent_id in author_ids} for agent_id in sorted(node_ids)]
    edges = [pairs[pair] for pair in sorted(pairs) if pairs[pair]["weight"] >= min_references]
    counts["pairs_before_threshold"] = len(pairs)
    return nodes, edges, counts


def read_snapshot(path: Path) -> tuple[list[dict[str, Any]], str]:
    with path.open("rb") as source:
        data = source.read(os.fstat(source.fileno()).st_size)
    lines = data.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
    digest = hashlib.sha256(b"".join(lines)).hexdigest()
    rows = []
    seen = set()
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in seen:
            raise ValueError(f"Invalid or duplicate event ID in {path}")
        seen.add(event_id)
        rows.append(row)
    return rows, digest


def aggregate_agents(events: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    authors = {event["event_id"]: event["author_id"] for event in events}
    names = {event["author_id"]: event.get("author_name") for event in events if event["author_id"] is not None}
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    omitted = 0
    for edge in edges:
        source = authors[edge["source_event_id"]]
        target = authors[edge["target_event_id"]]
        if source is None or target is None:
            omitted += 1
            continue
        pair = (source, target)
        if pair not in aggregated:
            aggregated[pair] = {"source_agent_id": source, "target_agent_id": target, "event_edge_count": 0, "weight": 0.0}
        aggregated[pair]["event_edge_count"] += 1
        aggregated[pair]["weight"] += edge.get("weight", 1.0)
    nodes = [{"id": agent_id, "name": names[agent_id]} for agent_id in sorted(names)]
    return nodes, [aggregated[pair] for pair in sorted(aggregated)], omitted


def graph_statistics(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], *, level: str, weighted: bool, seed: int) -> dict[str, Any]:
    source_key, target_key = ("source_event_id", "target_event_id") if level == "event" else ("source_agent_id", "target_agent_id")
    indices = {node["id"]: index for index, node in enumerate(nodes)}
    graph = ig.Graph(n=len(nodes), edges=[(indices[edge[source_key]], indices[edge[target_key]]) for edge in edges], directed=True)
    weights = [float(edge["weight"]) for edge in edges] if weighted else None
    if weights is not None:
        graph.es["weight"] = weights
    weak = graph.connected_components(mode="weak")
    strong = graph.connected_components(mode="strong")
    undirected = graph.as_undirected(mode="collapse", combine_edges={"weight": "sum"} if weighted else None)
    if weighted:
        undirected.delete_edges([edge.index for edge in undirected.es if edge["weight"] <= 0])
    ig.set_random_number_generator(random.Random(seed))
    if undirected.ecount():
        communities = undirected.community_leiden(objective_function="modularity", weights="weight" if weighted else None, n_iterations=-1)
        membership = communities.membership
        modularity = undirected.modularity(membership, weights="weight" if weighted else None)
    else:
        membership = list(range(len(nodes)))
        modularity = None
    indegree = graph.degree(mode="in")
    outdegree = graph.degree(mode="out")
    in_strength = graph.strength(mode="in", weights=weights) if weighted else None
    out_strength = graph.strength(mode="out", weights=weights) if weighted else None
    node_stats = []
    for index, node in enumerate(nodes):
        row = {**node, "in_degree": indegree[index], "out_degree": outdegree[index],
               "weak_component": weak.membership[index], "strong_component": strong.membership[index],
               "community": membership[index]}
        if weighted:
            row.update({"in_strength": in_strength[index], "out_strength": out_strength[index]})
        node_stats.append(row)
    return {
        "node_count": graph.vcount(), "edge_count": graph.ecount(),
        "self_loops": sum(edge.source == edge.target for edge in graph.es),
        "density_including_loops": graph.density(loops=True) if nodes else 0.0,
        "isolates": sum(incoming == 0 and outgoing == 0 for incoming, outgoing in zip(indegree, outdegree)),
        "mean_in_degree": sum(indegree) / len(nodes) if nodes else 0.0,
        "in_degree_distribution": dict(sorted(Counter(indegree).items())),
        "out_degree_distribution": dict(sorted(Counter(outdegree).items())),
        "weak_component_sizes": sorted(weak.sizes(), reverse=True),
        "strong_component_sizes": sorted(strong.sizes(), reverse=True),
        "community_count": len(set(membership)),
        "community_sizes": sorted(Counter(membership).values(), reverse=True), "community_modularity": modularity,
        "nodes": node_stats,
    }


def run_graph_analysis(dataset_dir: str | Path, pass1_dir: str | Path, pass2_dir: str | Path,
                       output_dir: str | Path, *, similarity_threshold: float, dependency_threshold: float,
                       seed: int = 0, cohort_retrieval: str | Path | None = None,
                       cohort_limit: int | None = None, min_direct_references: int = 1,
                       calibration_dir: str | Path | None = None) -> dict[str, Any]:
    if not math.isfinite(similarity_threshold) or not -1 <= similarity_threshold <= 1:
        raise ValueError("similarity threshold must be between -1 and 1")
    if not math.isfinite(dependency_threshold) or not 0 <= dependency_threshold <= 1:
        raise ValueError("dependency threshold must be between 0 and 1")
    if (cohort_retrieval is None) != (cohort_limit is None) or (cohort_limit is not None and cohort_limit < 1):
        raise ValueError("cohort retrieval and a positive cohort limit must be provided together")
    if min_direct_references < 1:
        raise ValueError("minimum references must be positive")
    dataset, pass1, pass2, output = (Path(path).resolve() for path in (dataset_dir, pass1_dir, pass2_dir, output_dir))
    if any(output == source or source in output.parents for source in (dataset, pass1, pass2)):
        raise ValueError("Analysis output must be outside the input directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Analysis output directory must be empty")
    agents, names_to_ids = load_agent_registry(dataset / "agents.jsonl")
    annotations, annotations_path = load_pass1_annotations(pass1)
    events, skipped = load_events(dataset / "events.jsonl", agents, annotations)
    retrieval_path = pass2 / "retrieval_candidates.jsonl"
    retrieval_config = json.loads((pass2 / "pass2_retrieve_config.json").read_text(encoding="utf-8"))
    for field, path in (("events_sha256", dataset / "events.jsonl"), ("agents_sha256", dataset / "agents.jsonl"), ("pass1_sha256", annotations_path)):
        if retrieval_config.get(field) != file_sha256(path):
            raise ValueError(f"Retrieval configuration does not match {field}")
    rerank_config_path = pass2 / "pass2_rerank_config.json"
    rerank_path = pass2 / "rerank_checkpoint.jsonl"
    rerank_config = json.loads(rerank_config_path.read_text(encoding="utf-8")) if rerank_config_path.exists() else None
    if rerank_config is not None:
        if rerank_config.get("retrieval_sha256") != file_sha256(retrieval_path) or rerank_config.get("context_mode", "private") != retrieval_config.get("context_mode", "private"):
            raise ValueError("Rerank configuration does not match retrieval cache")
    retrieval, retrieval_snapshot_sha256 = read_snapshot(retrieval_path)
    reranked, rerank_snapshot_sha256 = read_snapshot(rerank_path) if rerank_path.exists() else ([], None)
    if reranked and rerank_config is None:
        raise ValueError("Rerank checkpoint exists without configuration")
    retrieval_ids = {row["event_id"] for row in retrieval}
    reranked_ids = {row["event_id"] for row in reranked}
    if not reranked_ids <= retrieval_ids:
        raise ValueError("Rerank checkpoint contains targets absent from retrieval")
    all_events = events
    all_event_ids = {event["event_id"] for event in all_events}
    cohort_snapshot_sha256 = None
    if cohort_retrieval is not None:
        cohort_path = Path(cohort_retrieval).resolve()
        if output == cohort_path.parent or cohort_path.parent in output.parents:
            raise ValueError("Analysis output must be outside the cohort input directory")
        cohort_config = json.loads((cohort_path.parent / "pass2_retrieve_config.json").read_text(encoding="utf-8"))
        for field in ("events_sha256", "agents_sha256", "pass1_sha256"):
            if cohort_config.get(field) != retrieval_config.get(field):
                raise ValueError(f"Cohort retrieval configuration does not match {field}")
        cohort_rows, cohort_snapshot_sha256 = read_snapshot(cohort_path)
        event_by_id = {event["event_id"]: event for event in all_events}
        for row in cohort_rows:
            event = event_by_id.get(row["event_id"])
            if event is None or row.get("source_index") != event["source_index"]:
                raise ValueError("Cohort retrieval does not match the dataset")
        cohort_targets = {row["event_id"] for row in cohort_rows}
        events = [event for event in all_events if event["event_id"] in cohort_targets][:cohort_limit]
        if len(events) != cohort_limit:
            raise ValueError("Cohort retrieval has fewer targets than requested")
    event_ids = {event["event_id"] for event in events}
    retrieval = [row for row in retrieval if row["event_id"] in event_ids]
    reranked = [row for row in reranked if row["event_id"] in event_ids]
    retrieval_ids = {row["event_id"] for row in retrieval}
    reranked_ids = {row["event_id"] for row in reranked}
    retrieval_by_id = {row["event_id"]: row for row in retrieval}
    matched_retrieval = []
    for row in reranked:
        retrieval_row = retrieval_by_id[row["event_id"]]
        selected = filter_candidates_for_rerank(
            retrieval_row, rerank_config.get("candidate_filter", "none"),
            context_mode=retrieval_config.get("context_mode", "private"),
        )
        expected = {candidate["source_event_id"]: candidate["similarity_score"] for candidate in selected}
        actual = {candidate["source_event_id"]: candidate["similarity_score"] for candidate in row["candidates"]}
        if expected != actual or len(actual) != len(row["candidates"]):
            raise ValueError(f"Rerank candidates do not match retrieval for {row['event_id']}")
        matched_retrieval.append({**retrieval_row, "candidates": selected})
    pass1_graph, reference_counts = pass1_edges(all_events, names_to_ids, reference_event_ids=event_ids)
    full_graphs = {
        "pass1": (pass1_graph, False),
        "pass2_retrieval": (pass2_edges(retrieval, "similarity_score", similarity_threshold, all_event_ids), False),
        "pass2_retrieval_matched": (pass2_edges(matched_retrieval, "similarity_score", similarity_threshold, all_event_ids), False),
        "pass2_reranked": (pass2_edges(reranked, "dependency_score", dependency_threshold, all_event_ids), True),
    }
    calibration_info = None
    if calibration_dir is not None:
        calibration = Path(calibration_dir).resolve()
        if output == calibration or calibration in output.parents:
            raise ValueError("Analysis output must be outside the calibration directory")
        if retrieval_config.get("context_mode", "private") != "public":
            raise ValueError("Calibrated public scores require public Pass 2 inputs")
        calibration_path = calibration / "calibration.json"
        calibration_edges_path = calibration / "public_decayed_edges.jsonl"
        calibration_info = json.loads(calibration_path.read_text(encoding="utf-8"))
        hashes = calibration_info["input_hashes"]
        for field in ("events_sha256", "agents_sha256", "pass1_sha256"):
            if hashes.get(field) != retrieval_config.get(field):
                raise ValueError(f"Calibration does not match {field}")
        if hashes.get("public_retrieval_sha256") != retrieval_snapshot_sha256 or hashes.get("public_rerank_snapshot_sha256") != rerank_snapshot_sha256:
            raise ValueError("Calibration does not match the public retrieval or rerank snapshot")
        cohort_hash = hashlib.sha256("\n".join(sorted(event_ids)).encode("utf-8")).hexdigest()
        if calibration_info.get("cohort", {}).get("event_ids_sha256") != cohort_hash:
            raise ValueError("Calibration cohort does not match analysis cohort")
        all_scored = induced_event_edges(pass2_edges(reranked, "dependency_score", 0, all_event_ids), event_ids)
        expected_pairs = {(edge["source_event_id"], edge["target_event_id"]): edge["weight"] for edge in all_scored}
        with calibration_edges_path.open(encoding="utf-8") as source:
            calibrated_rows = [json.loads(line) for line in source if line.strip()]
        full_graphs["pass2_calibrated"] = (calibrated_event_edges(calibrated_rows, expected_pairs, dependency_threshold), True)
    graphs = {name: (induced_event_edges(edges, event_ids), weighted) for name, (edges, weighted) in full_graphs.items()}
    direct_nodes, direct_edges, direct_counts = direct_reference_graph(events, agents, names_to_ids, min_direct_references)
    output.mkdir(parents=True, exist_ok=True)
    event_nodes = [{"id": event["event_id"], "agent_id": event["author_id"], "timestamp": event["timestamp"]} for event in events]
    summary = {
        "context_mode": retrieval_config.get("context_mode", "private"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "igraph_version": ig.__version__,
        "input_hashes": {"events_sha256": retrieval_config["events_sha256"], "agents_sha256": retrieval_config["agents_sha256"],
                 "pass1_sha256": retrieval_config["pass1_sha256"], "retrieval_snapshot_sha256": retrieval_snapshot_sha256,
                 "rerank_snapshot_sha256": rerank_snapshot_sha256},
        "similarity_threshold": similarity_threshold, "dependency_threshold": dependency_threshold,
        "cohort": {"selection": "first chronological retrieval targets" if cohort_retrieval else "all events",
               "limit": cohort_limit, "retrieval_snapshot_sha256": cohort_snapshot_sha256,
               "event_ids_sha256": hashlib.sha256("\n".join(sorted(event_ids)).encode("utf-8")).hexdigest(),
               "first_timestamp": events[0]["timestamp"] if events else None,
               "last_timestamp": events[-1]["timestamp"] if events else None},
        "leiden_seed": seed, "candidate_filter": rerank_config.get("candidate_filter") if rerank_config else None,
        "min_direct_references": min_direct_references, "direct_reference_counts": direct_counts,
        "events": len(events), "skipped_consolidations": skipped, "annotated_events": sum(event["has_pass1_annotation"] for event in events),
        "retrieval_targets": len(retrieval), "reranked_targets": len(reranked),
        "missing_retrieval_targets": len(event_ids - retrieval_ids),
        "unreranked_targets": len(event_ids - reranked_ids), "pass1_references": reference_counts,
        "graphs": {},
    }
    if calibration_info is not None:
        summary["calibration"] = {"model": calibration_info["model"], "fit": calibration_info["fit"],
                                  "training_targets": calibration_info["cohort"]["training_targets"],
                                  "metadata_sha256": file_sha256(calibration_path),
                                  "scores_sha256": file_sha256(calibration_edges_path)}
    for name, (edges, weighted) in graphs.items():
        agent_nodes, agent_edges, omitted = aggregate_agents(events, edges)
        event_stats = graph_statistics(event_nodes, edges, level="event", weighted=weighted, seed=seed)
        agent_stats = graph_statistics(agent_nodes, agent_edges, level="agent", weighted=True, seed=seed)
        for level, graph_edges, stats in (("event", edges, event_stats), ("agent", agent_edges, agent_stats)):
            write_jsonl(output / f"{name}_{level}_edges.jsonl", graph_edges)
            write_json(output / f"{name}_{level}_statistics.json", stats)
        summary["graphs"][name] = {"event_edges": len(edges), "agent_edges": len(agent_edges), "edges_with_missing_agent": omitted,
                        "edges_outside_cohort": len(full_graphs[name][0]) - len(edges)}
    direct_stats = graph_statistics(direct_nodes, direct_edges, level="agent", weighted=True, seed=seed)
    write_jsonl(output / "pass1_direct_agent_edges.jsonl", direct_edges)
    write_json(output / "pass1_direct_agent_statistics.json", direct_stats)
    summary["graphs"]["pass1_direct"] = {"agent_nodes": len(direct_nodes), "agent_edges": len(direct_edges)}
    write_json(output / "summary.json", summary)
    return summary
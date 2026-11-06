from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tide.graphing import aggregate_agent_matrix, build_event_graph, summarize_support
from tide.io_utils import ensure_dir, write_json, write_jsonl
from tide.retrieval import RetrievalConfig, retrieve_candidates
from tide.support import score_candidates
from tide.synthetic import SyntheticBundle, generate_synthetic_trace


@dataclass(frozen=True)
class PipelineOutputs:
    output_dir: str
    summary: dict[str, Any]


def run_synthetic_pipeline(output_dir: str, seed: int = 7, min_edge_score: float = 0.4) -> PipelineOutputs:
    out = ensure_dir(output_dir)
    bundle: SyntheticBundle = generate_synthetic_trace(seed=seed)
    events = list(bundle.events)
    event_by_id = {e.event_id: e for e in events}

    candidate_map = retrieve_candidates(events, config=RetrievalConfig())
    candidate_rows = [c.as_dict() for cands in candidate_map.values() for c in cands]

    support_edges = []
    for cands in candidate_map.values():
        support_edges.extend(score_candidates(event_by_id, cands))

    graph = build_event_graph(events, support_edges, min_score=min_edge_score)
    agent_matrix = aggregate_agent_matrix(events, graph)
    support_summary = summarize_support(support_edges)

    write_jsonl(out / "events.jsonl", (e.as_dict() for e in events))
    write_jsonl(out / "ground_truth_edges.jsonl", (e.as_dict() for e in bundle.ground_truth_edges))
    write_jsonl(out / "ground_truth_relations.jsonl", (r.as_dict() for r in bundle.ground_truth_relations))
    write_jsonl(out / "retrieval_candidates.jsonl", candidate_rows)
    write_jsonl(out / "support_edges.jsonl", (e.as_dict() for e in support_edges))
    write_json(out / "event_graph.json", graph.as_dict())
    write_json(out / "agent_matrix.json", agent_matrix)

    summary = {
        "mode": "synthetic",
        "output_dir": str(Path(output_dir)),
        "event_count": len(events),
        "retrieval_pair_count": len(candidate_rows),
        "support_pair_count": len(support_edges),
        "inferred_graph_edges": len(graph.edges),
        "synthetic_metadata": bundle.metadata,
        "support_summary": support_summary,
        "notes": [
            "Synthetic results are for controlled validation only.",
            "Support scores are not calibrated probabilities.",
            "Graph edges are inferred support/dependency, not proven causation.",
        ],
    }
    write_json(out / "summary.json", summary)
    return PipelineOutputs(output_dir=str(out), summary=summary)

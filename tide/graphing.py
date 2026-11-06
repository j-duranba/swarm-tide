from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from tide.schema import EventRecord, SupportEdge
from tide.retrieval import strictly_precedes


@dataclass(frozen=True)
class EventGraph:
    nodes: tuple[str, ...]
    edges: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {"nodes": list(self.nodes), "edges": list(self.edges)}


def build_event_graph(events: list[EventRecord], edges: list[SupportEdge], min_score: float = 0.4) -> EventGraph:
    event_index = {e.event_id: e for e in events}
    graph_edges: list[dict[str, Any]] = []

    for edge in edges:
        if edge.support_score < min_score or edge.abstain:
            continue
        source = event_index[edge.source_event_id]
        target = event_index[edge.target_event_id]
        if not strictly_precedes(source, target):
            continue
        if source.timestamp_utc and target.timestamp_utc:
            s = datetime.fromisoformat(source.timestamp_utc.replace("Z", "+00:00"))
            t = datetime.fromisoformat(target.timestamp_utc.replace("Z", "+00:00"))
            time_lag_seconds = (t - s).total_seconds()
        else:
            time_lag_seconds = None

        graph_edges.append(
            {
                "source_event_id": edge.source_event_id,
                "target_event_id": edge.target_event_id,
                "source_agent_id": source.agent_id,
                "target_agent_id": target.agent_id,
                "weight": edge.support_score,
                "support_probability": edge.support_probability,
                "support_type": edge.support_type,
                "evidence": list(edge.evidence),
                "alternative_explanations": list(edge.alternative_explanations),
                "time_lag_seconds": time_lag_seconds,
                "edge_origin": "inferred_support",
                "source_ref": {"table": source.source_table, "record_id": source.source_record_id},
                "target_ref": {"table": target.source_table, "record_id": target.source_record_id},
            }
        )

    return EventGraph(nodes=tuple(e.event_id for e in events), edges=tuple(graph_edges))


def aggregate_agent_matrix(events: list[EventRecord], graph: EventGraph) -> dict[str, Any]:
    agent_set = sorted({e.agent_id for e in events})
    matrix = {target: {source: 0.0 for source in agent_set} for target in agent_set}
    counts = {target: 0 for target in agent_set}

    for edge in graph.edges:
        t = edge["target_agent_id"]
        s = edge["source_agent_id"]
        w = float(edge["weight"])
        matrix[t][s] += w
        counts[t] += 1

    mean_per_target = {}
    for target in agent_set:
        denom = max(1, counts[target])
        mean_per_target[target] = {source: matrix[target][source] / denom for source in agent_set}

    return {
        "convention": "W[target_agent][source_agent] = evidence that source supports target",
        "agents": agent_set,
        "raw_sum": matrix,
        "mean_per_target_event": mean_per_target,
    }


def summarize_support(edges: list[SupportEdge]) -> dict[str, Any]:
    by_type: dict[str, int] = defaultdict(int)
    abstentions = 0
    for edge in edges:
        by_type[edge.support_type] += 1
        if edge.abstain:
            abstentions += 1
    return {"total_scored_pairs": len(edges), "abstentions": abstentions, "by_support_type": dict(sorted(by_type.items()))}

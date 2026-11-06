from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


def _to_utc_iso8601(value: Any) -> tuple[Optional[str], Optional[str]]:
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        return dt.isoformat(), str(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None, value
        candidate = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            return None, value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(), value
    return None, str(value)


@dataclass(frozen=True)
class ObservabilityFlags:
    has_goal: bool
    has_input: bool
    has_result: bool
    has_public_output: bool
    has_message_content: bool


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    episode_id: str
    agent_id: str
    timestamp_utc: Optional[str]
    timestamp_raw: Optional[str]
    event_index: Optional[int]
    goal_id: Optional[str]
    agent_goal: Optional[str]
    shared_goal: Optional[str]
    subgoal: Optional[str]
    subgoal_observed: bool
    state_before: Optional[str]
    action_type: Optional[str]
    action_summary: Optional[str]
    tool: Optional[str]
    input: Optional[str]
    result: Optional[str]
    agent_output: Optional[str]
    artifact_refs: tuple[str, ...]
    communication_metadata: Optional[dict[str, Any]]
    source_table: str
    source_record_id: str
    provenance_known: bool
    missing_fields: tuple[str, ...]
    observability_flags: ObservabilityFlags

    def sortable_time_key(self) -> tuple[int, int]:
        if self.timestamp_utc is None:
            return (1, self.event_index if self.event_index is not None else 10**12)
        return (0, self.event_index if self.event_index is not None else 10**12)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "episode_id": self.episode_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp_utc,
            "timestamp_raw": self.timestamp_raw,
            "event_index": self.event_index,
            "goal_id": self.goal_id,
            "agent_goal": self.agent_goal,
            "shared_goal": self.shared_goal,
            "subgoal": self.subgoal,
            "subgoal_observed": self.subgoal_observed,
            "state_before": self.state_before,
            "action_type": self.action_type,
            "action_summary": self.action_summary,
            "tool": self.tool,
            "input": self.input,
            "result": self.result,
            "agent_output": self.agent_output,
            "artifact_refs": list(self.artifact_refs),
            "communication_metadata": self.communication_metadata,
            "source_table": self.source_table,
            "source_record_id": self.source_record_id,
            "provenance_known": self.provenance_known,
            "missing_fields": list(self.missing_fields),
            "observability_flags": {
                "has_goal": self.observability_flags.has_goal,
                "has_input": self.observability_flags.has_input,
                "has_result": self.observability_flags.has_result,
                "has_public_output": self.observability_flags.has_public_output,
                "has_message_content": self.observability_flags.has_message_content,
            },
        }


@dataclass(frozen=True)
class CandidateRelation:
    source_event_id: str
    target_event_id: str
    retrieval_score: float
    semantic_similarity: float
    temporal_score: float
    artifact_overlap: float
    compatibility_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "retrieval_score": self.retrieval_score,
            "semantic_similarity": self.semantic_similarity,
            "temporal_score": self.temporal_score,
            "artifact_overlap": self.artifact_overlap,
            "compatibility_score": self.compatibility_score,
        }


@dataclass(frozen=True)
class SupportEdge:
    source_event_id: str
    target_event_id: str
    support_score: float
    support_probability: Optional[float]
    support_type: str
    evidence: tuple[str, ...]
    alternative_explanations: tuple[str, ...]
    abstain: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "support_score": self.support_score,
            "support_probability": self.support_probability,
            "support_type": self.support_type,
            "evidence": list(self.evidence),
            "alternative_explanations": list(self.alternative_explanations),
            "abstain": self.abstain,
        }


def build_event(record: dict[str, Any], source_table: str, source_record_id: str) -> EventRecord:
    ts_utc, ts_raw = _to_utc_iso8601(record.get("timestamp"))
    subgoal = record.get("subgoal")
    subgoal_observed = bool(record.get("subgoal_observed", subgoal is not None))

    artifact_refs = tuple(str(x) for x in (record.get("artifact_refs") or []))
    has_message_content = bool(
        (record.get("action_type") or "").lower() in {"message_send", "message"}
        and record.get("result")
    )

    missing_fields = []
    for field in ("timestamp", "event_index", "action_type", "action_summary", "input", "result", "agent_output"):
        if record.get(field) in (None, ""):
            missing_fields.append(field)

    return EventRecord(
        event_id=str(record["event_id"]),
        episode_id=str(record["episode_id"]),
        agent_id=str(record["agent_id"]),
        timestamp_utc=ts_utc,
        timestamp_raw=ts_raw,
        event_index=record.get("event_index"),
        goal_id=record.get("goal_id"),
        agent_goal=record.get("agent_goal"),
        shared_goal=record.get("shared_goal"),
        subgoal=subgoal,
        subgoal_observed=subgoal_observed,
        state_before=record.get("state_before"),
        action_type=record.get("action_type"),
        action_summary=record.get("action_summary"),
        tool=record.get("tool"),
        input=record.get("input"),
        result=record.get("result"),
        agent_output=record.get("agent_output"),
        artifact_refs=artifact_refs,
        communication_metadata=record.get("communication_metadata"),
        source_table=source_table,
        source_record_id=source_record_id,
        provenance_known=bool(source_table and source_record_id),
        missing_fields=tuple(missing_fields),
        observability_flags=ObservabilityFlags(
            has_goal=bool(record.get("goal_id") or record.get("agent_goal")),
            has_input=record.get("input") is not None,
            has_result=record.get("result") is not None,
            has_public_output=record.get("agent_output") is not None,
            has_message_content=has_message_content,
        ),
    )

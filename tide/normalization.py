from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from tide.schema import EventRecord, build_event


@dataclass(frozen=True)
class MappingHints:
    event_id: tuple[str, ...] = ("event_id", "id", "uuid")
    episode_id: tuple[str, ...] = ("episode_id", "village_id", "session_id", "task_id")
    agent_id: tuple[str, ...] = ("agent_id", "actor_id", "user_id")
    timestamp: tuple[str, ...] = ("timestamp", "created_at", "time")
    event_index: tuple[str, ...] = ("event_index", "index", "seq", "order")
    goal_id: tuple[str, ...] = ("goal_id",)
    agent_goal: tuple[str, ...] = ("agent_goal", "goal", "agent_goal_text")
    shared_goal: tuple[str, ...] = ("shared_goal", "village_goal", "system_goal")
    subgoal: tuple[str, ...] = ("subgoal",)
    state_before: tuple[str, ...] = ("state_before", "previous_state")
    action_type: tuple[str, ...] = ("action_type", "event_type", "type", "action")
    action_summary: tuple[str, ...] = ("action_summary", "summary", "description")
    tool: tuple[str, ...] = ("tool", "tool_name")
    input: tuple[str, ...] = ("input", "arguments", "args", "prompt")
    result: tuple[str, ...] = ("result", "output", "tool_result")
    agent_output: tuple[str, ...] = ("agent_output", "message", "public_output")
    artifact_refs: tuple[str, ...] = ("artifact_refs", "artifacts", "files")


def _first(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for n in names:
        if n in record and record[n] is not None:
            return record[n]
    return None


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _artifact_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def normalize_record(
    record: dict[str, Any], source_table: str, source_record_id: str, hints: MappingHints | None = None
) -> EventRecord:
    hints = hints or MappingHints()

    normalized = {
        "event_id": _as_text(_first(record, hints.event_id)),
        "episode_id": _as_text(_first(record, hints.episode_id)) or "unknown_episode",
        "agent_id": _as_text(_first(record, hints.agent_id)) or "unknown_agent",
        "timestamp": _first(record, hints.timestamp),
        "event_index": _first(record, hints.event_index),
        "goal_id": _as_text(_first(record, hints.goal_id)),
        "agent_goal": _as_text(_first(record, hints.agent_goal)),
        "shared_goal": _as_text(_first(record, hints.shared_goal)),
        "subgoal": _as_text(_first(record, hints.subgoal)),
        "state_before": _as_text(_first(record, hints.state_before)),
        "action_type": _as_text(_first(record, hints.action_type)),
        "action_summary": _as_text(_first(record, hints.action_summary)),
        "tool": _as_text(_first(record, hints.tool)),
        "input": _as_text(_first(record, hints.input)),
        "result": _as_text(_first(record, hints.result)),
        "agent_output": _as_text(_first(record, hints.agent_output)),
        "artifact_refs": _artifact_list(_first(record, hints.artifact_refs)),
        "communication_metadata": {
            k: record.get(k)
            for k in ("sender", "recipient", "room_id", "channel", "thread_id")
            if record.get(k) is not None
        }
        or None,
    }

    if normalized["event_id"] is None:
        normalized["event_id"] = f"{source_table}:{source_record_id}"

    return build_event(normalized, source_table=source_table, source_record_id=source_record_id)


from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


INPUT_FILES = (
    "events.jsonl",
    "chat_messages.jsonl",
    "agents.jsonl",
    "agent_goals.jsonl",
    "village_goals.jsonl",
    "chat_rooms.jsonl",
)


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON in {path} line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} line {line_number}")
            yield line_number, row


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    original = value.strip()
    try:
        result = datetime.fromisoformat(original.replace("Z", "+00:00"))
    except ValueError:
        for pattern in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(original, pattern).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None
    if result.tzinfo is None:
        return result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def utc_text(value: datetime | None) -> str | None:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z") if value else None


def collect_components(value: Any, component_type: str) -> list[dict[str, Any]]:
    found = []
    if isinstance(value, dict):
        if value.get("type") == component_type:
            found.append(value)
        for child in value.values():
            found.extend(collect_components(child, component_type))
    elif isinstance(value, list):
        for child in value:
            found.extend(collect_components(child, component_type))
    return found


def iter_typed_components(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("type"), str):
            yield value
        for child in value.values():
            yield from iter_typed_components(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_typed_components(child)


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def active_at(start: Any, end: Any, point: datetime) -> bool:
    starts = parse_timestamp(start)
    ends = parse_timestamp(end)
    return starts is not None and starts <= point and (ends is None or point < ends)


def create_version_directory(requested: Path, overwrite: bool = False) -> Path:
    requested.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if requested.exists() and not requested.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {requested}")
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    name = requested.name
    if not requested.exists():
        requested.mkdir(exist_ok=False)
        return requested
    if name.startswith("v") and name[1:].isdigit():
        base = requested.parent
        prefix = "v"
        version = int(name[1:]) + 1
    elif "_v" in name and name.rsplit("_v", 1)[1].isdigit():
        base = requested.with_name(name.rsplit("_v", 1)[0]).parent
        prefix = name.rsplit("_v", 1)[0] + "_v"
        version = int(name.rsplit("_v", 1)[1]) + 1
    else:
        base = requested.parent
        prefix = name + "_v"
        version = 1
    while True:
        candidate = base / f"{prefix}{version}"
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            version += 1


class SliceBuilder:
    def __init__(self, dataset_dir: Path, requested_output: Path, start: datetime, end: datetime, overwrite: bool = False) -> None:
        self.dataset_dir = dataset_dir
        self.requested_output = requested_output
        self.start = start
        self.end = end
        self.overwrite = overwrite
        self.output_dir: Path | None = None
        self.agents: dict[str, dict[str, Any]] = {}
        self.agent_goals: list[dict[str, Any]] = []
        self.village_goals: list[dict[str, Any]] = []
        self.rooms: dict[str, dict[str, Any]] = {}
        self.chat_by_message_id: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        self.input_counts: Counter[str] = Counter()
        self.output_counts: Counter[str] = Counter()
        self.metrics: Counter[str] = Counter()
        self.output_component_types_input: Counter[str] = Counter()
        self.output_component_types_output: Counter[str] = Counter()
        self.function_calls_input: Counter[str] = Counter()
        self.function_calls_output: Counter[str] = Counter()
        self.used_agents: set[str] = set()
        self.used_rooms: set[str] = set()
        self.used_goal_ids: set[str] = set()
        self.used_village_goal_ids: set[str] = set()
        self.matched_chat_sequences: set[int] = set()
        self.seen_chat_ids: Counter[str] = Counter()
        self.sender_checks: Counter[str] = Counter()
        self.room_checks: Counter[str] = Counter()
        self.content_checks: Counter[str] = Counter()
        self.event_ids: Counter[str] = Counter()

    def validate_inputs(self) -> None:
        if self.dataset_dir.name != "main-village" or not self.dataset_dir.is_dir():
            raise ValueError(f"Dataset directory must point to extracted main-village: {self.dataset_dir}")
        missing = [name for name in INPUT_FILES if not (self.dataset_dir / name).is_file()]
        if missing:
            raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))
        if self.end <= self.start:
            raise ValueError("End timestamp must be later than start timestamp")

    def load_dimensions(self) -> None:
        for name in ("agents", "agent_goals", "village_goals", "chat_rooms"):
            path = self.dataset_dir / f"{name}.jsonl"
            print(f"[slice] loading {path.name}", flush=True)
            for _, row in read_jsonl(path):
                self.input_counts[name] += 1
                if name == "agents" and row.get("id"):
                    self.agents[str(row["id"])] = row
                elif name == "agent_goals":
                    self.agent_goals.append(row)
                elif name == "village_goals":
                    self.village_goals.append(row)
                elif name == "chat_rooms" and row.get("id"):
                    self.rooms[str(row["id"])] = row
            print(f"[slice] {name}: {self.input_counts[name]:,} records", flush=True)

    def index_chat_messages(self) -> None:
        path = self.dataset_dir / "chat_messages.jsonl"
        print(f"[slice] indexing {path.name}", flush=True)
        for sequence, (_, row) in enumerate(read_jsonl(path), 1):
            self.input_counts["chat_messages"] += 1
            message_id = str(row["id"]) if row.get("id") is not None else None
            if message_id is not None:
                self.chat_by_message_id.setdefault(message_id, []).append((sequence, row))
            if sequence % 50000 == 0:
                print(f"[slice] chat messages indexed: {sequence:,}", flush=True)
        print(f"[slice] chat messages indexed: {self.input_counts['chat_messages']:,}", flush=True)

    def goal_context(self, agent_id: str | None, village_id: str | None, point: datetime) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        agent_goals = []
        if agent_id:
            for goal in self.agent_goals:
                if goal.get("agent_id") == agent_id and active_at(goal.get("start_time"), goal.get("end_time"), point):
                    agent_goals.append(goal)
                    if goal.get("id"):
                        self.used_goal_ids.add(str(goal["id"]))
        village_goals = []
        for goal in self.village_goals:
            if village_id and goal.get("village_id") != village_id:
                continue
            if active_at(goal.get("start_time"), goal.get("end_time"), point):
                village_goals.append(goal)
                if goal.get("id"):
                    self.used_village_goal_ids.add(str(goal["id"]))
        return agent_goals, village_goals

    def identity_context(self, event_data: dict[str, Any], chat_rows: list[dict[str, Any]]) -> dict[str, Any]:
        event_agent_id = event_data.get("agentId")
        event_speaker_id = event_data.get("speakerId")
        candidates = [event_agent_id, event_speaker_id]
        for chat in chat_rows:
            candidates.extend((chat.get("agent_speaker_id"), chat.get("user_speaker_id")))
        registered_id = next((str(candidate) for candidate in candidates if candidate and str(candidate) in self.agents), None)
        if registered_id:
            identity_class = "registered_agent"
            self.used_agents.add(registered_id)
        elif event_data.get("speakerType") == "HUMAN" or any(chat.get("speaker_type") == "user" for chat in chat_rows):
            identity_class = "human_or_user_speaker"
        else:
            identity_class = "unknown_or_system_actor"
        return {
            "class": identity_class,
            "registered_agent_id": registered_id,
            "event_agent_id": event_agent_id,
            "event_speaker_id": event_speaker_id,
            "event_speaker_type": event_data.get("speakerType"),
        }

    def build_event(self, event_row: dict[str, Any], event_line: int, timestamp: datetime, linked_rows: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
        event_data = event_row.get("data") if isinstance(event_row.get("data"), dict) else {}
        output = event_data.get("output")
        input_calls = collect_components(output, "function_call")
        self.function_calls_input.update(stable_digest(item) for item in input_calls)
        self.function_calls_output.update(stable_digest(item) for item in collect_components(output, "function_call"))
        for component in iter_typed_components(output):
            self.output_component_types_input[str(component["type"])] += 1
            self.output_component_types_output[str(component["type"])] += 1

        chat_rows = [row for _, row in linked_rows]
        event_message_id = event_data.get("messageId")
        for chat in chat_rows:
            self.matched_chat_sequences.add(next(sequence for sequence, candidate in linked_rows if candidate is chat))
            if chat.get("id") is not None:
                self.seen_chat_ids[str(chat["id"])] += 1
            self.metrics["event_message_pairs"] += 1
            if event_data.get("speakerType") == "agent":
                self.sender_checks["applicable"] += 1
                self.sender_checks["match"] += event_data.get("speakerId") == chat.get("agent_speaker_id")
            elif event_data.get("speakerType") == "HUMAN":
                self.sender_checks["applicable"] += 1
                self.sender_checks["match"] += event_data.get("speakerId") == chat.get("user_speaker_id")
            event_room = event_data.get("roomId")
            chat_room = chat.get("room_id")
            if event_room is not None and chat_room is not None:
                self.room_checks["applicable"] += 1
                self.room_checks["match"] += event_room == chat_room
            event_content = event_data.get("content")
            chat_content = chat.get("content")
            if isinstance(event_content, str) and isinstance(chat_content, str):
                self.content_checks["applicable"] += 1
                self.content_checks["match"] += event_content == chat_content

        primary_room_id = event_data.get("roomId") or next((chat.get("room_id") for chat in chat_rows if chat.get("room_id")), None)
        room = self.rooms.get(str(primary_room_id)) if primary_room_id else None
        if primary_room_id and room:
            self.used_rooms.add(str(primary_room_id))
        actor_id = self.identity_context(event_data, chat_rows)
        registered_id = actor_id["registered_agent_id"]
        village_id = event_row.get("village_id")
        agent_goals, village_goals = self.goal_context(registered_id, str(village_id) if village_id else None, timestamp)

        linked_messages = []
        for sequence, chat in linked_rows:
            linked_messages.append({
                "source_file": "chat_messages.jsonl",
                "source_record_sequence": sequence,
                "source_record_id": chat.get("id"),
                "timestamp_original": chat.get("created_at"),
                "timestamp_utc": utc_text(parse_timestamp(chat.get("created_at"))),
                "record": chat,
            })

        return {
            "record_type": "ai_village_event",
            "timestamp_original": event_row.get("created_at"),
            "timestamp_utc": utc_text(timestamp),
            "source_ids": {
                "event_id": event_row.get("id"),
                "event_line": event_line,
                "event_message_id": event_message_id,
                "chat_message_ids": [chat.get("id") for chat in chat_rows],
            },
            "identity": actor_id,
            "metadata": {
                "agent": self.agents.get(registered_id) if registered_id else None,
                "agent_goals_active_at_event": agent_goals,
                "village_goals_active_at_event": village_goals,
                "room": room,
                "room_id_from_event": event_data.get("roomId"),
                "room_ids_from_linked_messages": [chat.get("room_id") for chat in chat_rows],
            },
            "source_event": {
                "source_file": "events.jsonl",
                "source_record": event_row,
            },
            "linked_chat_messages": linked_messages,
        }

    def run(self) -> Path:
        self.validate_inputs()
        self.load_dimensions()
        self.index_chat_messages()

        output_directory = create_version_directory(self.requested_output, overwrite=self.overwrite)
        self.output_dir = output_directory
        print(f"[slice] writing to {output_directory}", flush=True)
        output_events: list[tuple[datetime, int, str]] = []
        event_path = self.dataset_dir / "events.jsonl"
        for event_line, event_row in read_jsonl(event_path):
            self.input_counts["events"] += 1
            timestamp = parse_timestamp(event_row.get("created_at"))
            if timestamp is None:
                self.metrics["event_timestamps_invalid"] += 1
                continue
            if not (self.start <= timestamp < self.end):
                continue
            self.metrics["events_in_window"] += 1
            if event_row.get("id") is not None:
                self.event_ids[str(event_row["id"])] += 1
            event_data = event_row.get("data") if isinstance(event_row.get("data"), dict) else {}
            message_id = event_data.get("messageId")
            linked_rows = []
            if message_id is not None:
                linked_rows = self.chat_by_message_id.get(str(message_id), [])
                if linked_rows:
                    self.metrics["events_with_chat_match"] += 1
                else:
                    self.metrics["events_with_unmatched_message_id"] += 1
            else:
                self.metrics["events_without_message_id"] += 1
            joined = self.build_event(event_row, event_line, timestamp, linked_rows)
            output_events.append((timestamp, event_line, json.dumps(joined, ensure_ascii=False, separators=(",", ":"))))
            if self.metrics["events_in_window"] % 5000 == 0:
                print(f"[slice] in-window events processed: {self.metrics['events_in_window']:,}", flush=True)

        events_output = output_directory / "events.jsonl"
        previous_timestamp = None
        output_count = 0
        equal_timestamp_pairs = 0
        with events_output.open("w", encoding="utf-8", newline="\n") as destination:
            output_events.sort(key=lambda item: (item[0], item[1]))
            for timestamp, _, row_json in output_events:
                timestamp_text = utc_text(timestamp)
                if previous_timestamp is not None:
                    if timestamp_text < previous_timestamp:
                        self.metrics["chronology_errors"] += 1
                    if timestamp_text == previous_timestamp:
                        equal_timestamp_pairs += 1
                previous_timestamp = timestamp_text
                emitted_timestamp = parse_timestamp(timestamp_text)
                if emitted_timestamp is None or not (self.start <= emitted_timestamp < self.end):
                    self.metrics["output_out_of_bounds"] += 1
                destination.write(row_json + "\n")
                output_count += 1
        self.output_counts["events"] = output_count

        for name in ("agents", "agent_goals", "village_goals", "chat_rooms"):
            output_path = output_directory / f"{name}.jsonl"
            with output_path.open("w", encoding="utf-8", newline="\n") as destination:
                if name == "agents":
                    rows = ((agent_id, self.agents[agent_id]) for agent_id in sorted(self.used_agents))
                elif name == "agent_goals":
                    rows = ((str(goal.get("id") or ""), goal) for goal in self.agent_goals if str(goal.get("id") or "") in self.used_goal_ids)
                elif name == "village_goals":
                    rows = ((str(goal.get("id") or ""), goal) for goal in self.village_goals if str(goal.get("id") or "") in self.used_village_goal_ids)
                else:
                    rows = ((room_id, self.rooms[room_id]) for room_id in sorted(self.used_rooms))
                count = 0
                for _, record in rows:
                    destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
                self.output_counts[name] = count

        self.metrics["chat_messages_matched_once_or_more"] = len(self.matched_chat_sequences)
        self.metrics["chat_messages_unmatched_in_input"] = max(0, self.input_counts["chat_messages"] - len(self.matched_chat_sequences))
        duplicate_event_ids = sum(count > 1 for count in self.event_ids.values())
        self.metrics["duplicate_event_ids_in_output"] = duplicate_event_ids
        self.metrics["equal_timestamp_adjacent_pairs"] = equal_timestamp_pairs
        self.metrics["function_call_components_input"] = sum(self.function_calls_input.values())
        self.metrics["function_call_components_output"] = sum(self.function_calls_output.values())
        self.metrics["function_call_arguments_preserved"] = self.function_calls_input == self.function_calls_output
        self.metrics["all_linked_messages_have_matching_content"] = self.content_checks["applicable"] == self.content_checks["match"]
        self.metrics["all_linked_messages_have_matching_rooms"] = self.room_checks["applicable"] == self.room_checks["match"]
        self.metrics["all_linked_messages_have_matching_senders"] = self.sender_checks["applicable"] == self.sender_checks["match"]
        self.metrics["all_output_components_preserved"] = self.output_component_types_input == self.output_component_types_output
        self.metrics["all_output_timestamps_within_window"] = self.metrics["output_out_of_bounds"] == 0
        self.metrics["all_output_rows_have_timestamp"] = output_count == self.metrics["events_in_window"]
        self.metrics["all_event_rows_written_once"] = output_count == self.metrics["events_in_window"] and duplicate_event_ids == 0
        self.metrics["chronologically_ordered"] = self.metrics["chronology_errors"] == 0

        manifest = {
            "dataset": "AI Village main-village",
            "generated_at_utc": utc_text(datetime.now(timezone.utc)),
            "dataset_dir": self.dataset_dir.as_posix(),
            "output_dir": output_directory.as_posix(),
            "input_files": {name: (self.dataset_dir / name).as_posix() for name in INPUT_FILES},
            "processing": {
                "window_start_inclusive_utc": utc_text(self.start),
                "window_end_exclusive_utc": utc_text(self.end),
                "timestamp_policy": "Timezone-aware timestamps normalized to UTC; timezone-naive timestamps are interpreted as UTC. Original timestamp fields are retained.",
                "join": "events.data.messageId -> chat_messages.id (exact string equality); multiple matching messages are retained together on the event record.",
                "output_unit": "One output JSONL record per in-window source event; linked chat messages are embedded, not emitted as separate events.",
                "identity_rule": "Agent identity is validated by exact match to agents.id. User/HUMAN speakers remain distinguishable in source fields and identity.class.",
                "event_output_policy": "All original event output components are preserved, including reasoning, thinking, redacted_reasoning, function calls, and arguments.",
                "metadata_join_policy": "Include the matching registered-agent record, agent goals active at event time, village goals active at event time and matching village_id, and referenced room metadata.",
                "ordering": "UTC timestamp ascending, then original events.jsonl line number ascending; equal-timestamp records are retained.",
                "version_selection": "Requested output directory is never overwritten; if it exists, the next numbered vN directory is created.",
            },
            "input_record_counts": dict(self.input_counts),
            "output_record_counts": dict(self.output_counts),
            "event_processing_counts": {
                "events_in_window": self.metrics["events_in_window"],
                "events_with_chat_match": self.metrics["events_with_chat_match"],
                "events_with_unmatched_message_id": self.metrics["events_with_unmatched_message_id"],
                "events_without_message_id": self.metrics["events_without_message_id"],
                "event_message_pairs": self.metrics["event_message_pairs"],
                "unique_matched_chat_records": self.metrics["chat_messages_matched_once_or_more"],
                "input_chat_records_not_linked_to_slice_events": self.metrics["chat_messages_unmatched_in_input"],
                "event_output_component_types_preserved": dict(self.output_component_types_output),
                "function_call_components_preserved": self.metrics["function_call_components_output"],
                "equal_timestamp_adjacent_pairs": equal_timestamp_pairs,
            },
            "validation": {
                "sender_match_count": self.sender_checks["match"],
                "sender_match_applicable_count": self.sender_checks["applicable"],
                "room_match_count": self.room_checks["match"],
                "room_match_applicable_count": self.room_checks["applicable"],
                "content_match_count": self.content_checks["match"],
                "content_match_applicable_count": self.content_checks["applicable"],
                "function_call_arguments_preserved": self.metrics["function_call_arguments_preserved"],
                "event_output_components_preserved": self.metrics["all_output_components_preserved"],
                "event_output_component_types_input": dict(self.output_component_types_input),
                "event_output_component_types_output": dict(self.output_component_types_output),
                "chronologically_ordered": self.metrics["chronologically_ordered"],
                "equal_timestamp_records_retained": True,
                "all_output_timestamps_within_requested_window": self.metrics["all_output_timestamps_within_window"],
                "output_records_outside_requested_window": self.metrics["output_out_of_bounds"],
                "output_event_rows_equal_in_window_event_rows": output_count == self.metrics["events_in_window"],
                "duplicate_event_ids_in_output": duplicate_event_ids,
                "all_linked_messages_have_matching_content": self.metrics["all_linked_messages_have_matching_content"],
                "all_linked_messages_have_matching_rooms": self.metrics["all_linked_messages_have_matching_rooms"],
                "all_linked_messages_have_matching_senders": self.metrics["all_linked_messages_have_matching_senders"],
            },
        }
        (output_directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.write_validation_report(output_directory, manifest, previous_timestamp)
        print(f"[slice] completed: {output_count:,} event rows", flush=True)
        return output_directory

    def write_validation_report(self, output_directory: Path, manifest: dict[str, Any], last_timestamp: str | None) -> None:
        counts = manifest["event_processing_counts"]
        checks = manifest["validation"]
        links = counts["event_message_pairs"]
        report = f"""# Main-Village Slice Validation

## Selection

- Window: `{utc_text(self.start)}` inclusive to `{utc_text(self.end)}` exclusive.
- Timezone policy: aware source timestamps are normalized to UTC; naive source timestamps are interpreted as UTC.
- Output version: `{output_directory.as_posix()}`.
- Event unit: one in-window source event row. Exact-linked chat records are embedded on that row and are not emitted as separate events.

## Record counts

| Source | Input records | Output records |
|---|---:|---:|
| events | {self.input_counts['events']:,} | {self.output_counts['events']:,} |
| chat_messages | {self.input_counts['chat_messages']:,} | {counts['unique_matched_chat_records']:,} linked at least once |
| agents | {self.input_counts['agents']:,} | {self.output_counts['agents']:,} referenced registered IDs |
| agent_goals | {self.input_counts['agent_goals']:,} | {self.output_counts['agent_goals']:,} active goals for included registered IDs |
| village_goals | {self.input_counts['village_goals']:,} | {self.output_counts['village_goals']:,} active matching-village goals |
| chat_rooms | {self.input_counts['chat_rooms']:,} | {self.output_counts['chat_rooms']:,} referenced rooms |

- In-window events with an exact chat match: **{counts['events_with_chat_match']:,}**.
- Event rows with a message ID but no chat match: **{counts['events_with_unmatched_message_id']:,}**.
- Event rows without a message ID: **{counts['events_without_message_id']:,}**.
- Exact event/message pairs embedded: **{links:,}**.
- Input chat rows not linked to any selected event: **{counts['input_chat_records_not_linked_to_slice_events']:,}**. These are not emitted as standalone events.
- Event output components, including reasoning/thinking/redacted-reasoning: **all preserved** (`{json.dumps(checks['event_output_component_types_output'], ensure_ascii=False)}`).
- Function-call output components preserved: **{counts['function_call_components_preserved']:,}**; full component digests match before/after filtering: **{checks['function_call_arguments_preserved']}**.

## Checks

| Check | Result |
|---|---|
| Exact `events.data.messageId -> chat_messages.id` matching | {links:,} pairs embedded |
| Linked sender equality where applicable | {checks['sender_match_count']:,}/{checks['sender_match_applicable_count']:,} |
| Linked room equality where applicable | {checks['room_match_count']:,}/{checks['room_match_applicable_count']:,} |
| Linked content equality where applicable | {checks['content_match_count']:,}/{checks['content_match_applicable_count']:,} |
| Linked source message content retained verbatim | Yes; embedded source records retain original fields |
| Function-call arguments preserved | {checks['function_call_arguments_preserved']} |
| All event output components preserved without filtering | {checks['event_output_components_preserved']} |
| Function-call component digests and arguments preserved | {checks['function_call_arguments_preserved']} |
| Output event rows equal in-window source events | {checks['output_event_rows_equal_in_window_event_rows']} |
| Duplicate event IDs in output | {checks['duplicate_event_ids_in_output']} |
| Chronological order by UTC timestamp then input line | {checks['chronologically_ordered']} |
| Equal-timestamp rows retained | {checks['equal_timestamp_records_retained']} ({counts['equal_timestamp_adjacent_pairs']:,} adjacent equal-time pairs) |
| Latest emitted timestamp (UTC) | {last_timestamp or 'none'} |

Sender, room, and content comparisons are reported only where both corresponding fields are present and applicable. Source event rows without a chat match remain in the output. Human/user classification is retained from the original speaker fields; registered identities are joined only by exact `agents.id` matches.
"""
        (output_directory / "validation_report.md").write_text(report, encoding="utf-8")

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an event-centered time slice from extracted AI Village main-village JSONL.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/ai-village-extracted/main-village"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v1"))
    parser.add_argument("--start", default="2026-10-24T00:00:00Z")
    parser.add_argument("--end", default="2026-11-01T00:00:00Z")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files in the requested output directory instead of selecting another version")
    args = parser.parse_args(argv)
    start, end = parse_timestamp(args.start), parse_timestamp(args.end)
    if start is None or end is None:
        parser.error("--start and --end must be valid ISO-8601 timestamps")
    builder = SliceBuilder(args.dataset_dir, args.output_dir, start, end, overwrite=args.overwrite)
    try:
        path = builder.run()
        print(f"[slice] output: {path}", flush=True)
    except (OSError, ValueError, MemoryError) as error:
        print(f"slice build failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
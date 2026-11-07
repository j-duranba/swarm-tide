from __future__ import annotations

import argparse
import calendar
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


SOURCE_FILES = {
    "events": "events.jsonl",
    "computer_use_turns": "computer_use_turns.jsonl",
    "computer_use_sessions": "computer_use_sessions.jsonl",
    "chat_messages": "chat_messages.jsonl",
    "chat_rooms": "chat_rooms.jsonl",
    "agents": "agents.jsonl",
    "agent_goals": "agent_goals.jsonl",
    "village_goals": "village_goals.jsonl",
}
ACTIVITY_SOURCES = ("events", "computer_use_turns", "chat_messages")
TIME_FIELDS = {
    "events": "created_at",
    "computer_use_turns": "created_at",
    "computer_use_sessions": "created_at",
    "chat_messages": "created_at",
    "chat_rooms": "created_at",
    "agents": "created_at",
    "agent_goals": "start_time",
    "village_goals": "start_time",
}
DURATIONS_HOURS = {"6 hours": 6, "1 day": 24, "3 days": 72, "1 week": 168, "1 month (30 days)": 720}


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def hour_number(value: datetime) -> int:
    return calendar.timegm(value.replace(minute=0, second=0, microsecond=0).timetuple()) // 3600


def hour_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value * 3600, tz=timezone.utc).replace(tzinfo=None)


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def walk_text(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(walk_text(item) for item in value.values())
    if isinstance(value, list):
        return sum(walk_text(item) for item in value)
    return 0


def format_timestamp(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") + "Z" if value else None


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


class Audit:
    def __init__(self, dataset_dir: Path, output_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.session_agents: dict[str, str] = {}
        self.agent_names: dict[str, str] = {}
        self.chat_ids: set[str] = set()
        self.message_lengths: list[int] = []
        self.inventory: dict[str, dict[str, Any]] = {}
        self.source_day: dict[str, Counter[str]] = defaultdict(Counter)
        self.source_hour: dict[str, Counter[int]] = defaultdict(Counter)
        self.hour_agents: dict[int, set[str]] = defaultdict(set)
        self.hour_sessions: dict[int, set[str]] = defaultdict(set)
        self.hour_rooms: dict[int, set[str]] = defaultdict(set)
        self.hour_text: Counter[int] = Counter()
        self.hour_event_content: Counter[int] = Counter()
        self.hour_event_output: Counter[int] = Counter()
        self.hour_agent_sessions: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.hour_agent_rooms: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.hour_agent_text: dict[str, dict[int, Counter[str]]] = {name: defaultdict(Counter) for name in ACTIVITY_SOURCES}
        self.hour_agent_counts: dict[str, dict[int, Counter[str]]] = {
            name: defaultdict(Counter) for name in ACTIVITY_SOURCES
        }
        self.agent_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.activity_row_counts: Counter[str] = Counter()
        self.activity_agent_attributed: Counter[str] = Counter()
        self.agent_days: dict[str, set[str]] = defaultdict(set)
        self.agent_first: dict[str, datetime] = {}
        self.agent_last: dict[str, datetime] = {}
        self.daily_agents: dict[str, Counter[str]] = defaultdict(Counter)
        self.records_per_session: Counter[str] = Counter()
        self.turns_with_session_id = 0
        self.turns_attributed = 0
        self.records_per_room: dict[str, Counter[str]] = defaultdict(Counter)
        self.action_types: Counter[str] = Counter()
        self.event_component_types: Counter[str] = Counter()
        self.event_message_ids: Counter[str] = Counter()
        self.linked_event_messages = 0
        self.event_message_references = 0
        self.recipient_field_rows = 0
        self.chat_room_counts: Counter[str] = Counter()
        self.chat_agent_counts: Counter[str] = Counter()
        self.event_action_types: Counter[str] = Counter()
        self.goal_intervals: dict[str, list[tuple[datetime | None, datetime | None]]] = defaultdict(list)
        self.village_goal_intervals: list[tuple[datetime | None, datetime | None]] = []
        self.malformed_examples: dict[str, str] = {}
        self.connection = sqlite3.connect("")
        self.connection.execute("PRAGMA journal_mode = OFF")
        self.connection.execute("PRAGMA synchronous = OFF")
        self.connection.execute("PRAGMA temp_store = FILE")
        self.connection.execute("CREATE TABLE ids (source TEXT NOT NULL, record_id TEXT NOT NULL, PRIMARY KEY (source, record_id))")

    def validate(self) -> None:
        if self.dataset_dir.name != "main-village":
            raise ValueError(f"Dataset directory must be named main-village, got: {self.dataset_dir}")
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")
        missing = [filename for filename in SOURCE_FILES.values() if not (self.dataset_dir / filename).is_file()]
        if missing:
            raise FileNotFoundError("Missing extracted JSONL source(s): " + ", ".join(missing))
        if self.output_dir.exists():
            raise FileExistsError(f"Output path already exists; refusing to overwrite: {self.output_dir}")

    def _inventory_start(self, source: str) -> dict[str, Any]:
        return {
            "file": SOURCE_FILES[source],
            "records": 0,
            "malformed_records": 0,
            "blank_lines": 0,
            "fields": {},
            "earliest_timestamp": None,
            "latest_timestamp": None,
            "chronologically_ordered": True,
            "timestamp_field": TIME_FIELDS[source],
            "timestamp_missing_or_invalid": 0,
        }

    def _record_inventory(self, source: str, row: dict[str, Any], stats: dict[str, Any]) -> None:
        stats["records"] += 1
        fields = stats["fields"]
        for key, value in row.items():
            entry = fields.setdefault(key, {"present": 0, "null": 0, "missing": 0, "types": Counter()})
            entry["present"] += 1
            entry["null"] += value is None
            entry["types"][json_type(value)] += 1
        nested_values: dict[str, Any] = {}
        if source == "events" and isinstance(row.get("data"), dict):
            data = row["data"]
            nested_values.update({f"data.{key}": value for key, value in data.items()})
            output = data.get("output")
            if isinstance(output, list):
                for component in output:
                    if isinstance(component, dict):
                        nested_values.update({f"data.output[].{key}": value for key, value in component.items() if key != "encrypted_content"})
        elif source == "computer_use_turns":
            for container in ("agent_action", "agent_messages"):
                value = row.get(container)
                if isinstance(value, dict):
                    nested_values.update({f"{container}.{key}": item for key, item in value.items() if key != "content"})
                    if isinstance(value.get("content"), list):
                        for item in value["content"]:
                            if isinstance(item, dict):
                                nested_values.update({f"{container}.content[].{key}": child for key, child in item.items() if key not in {"text", "input"}})
        for path, value in nested_values.items():
            entry = stats.setdefault("nested_fields", {}).setdefault(path, {"present": 0, "null": 0, "missing": 0, "types": Counter()})
            entry["present"] += 1
            entry["null"] += value is None
            entry["types"][json_type(value)] += 1
        timestamp = parse_timestamp(row.get(TIME_FIELDS[source]))
        if timestamp is None:
            stats["timestamp_missing_or_invalid"] += 1
        else:
            earliest = parse_timestamp(stats["earliest_timestamp"])
            latest = parse_timestamp(stats["latest_timestamp"])
            if earliest is None or timestamp < earliest:
                stats["earliest_timestamp"] = format_timestamp(timestamp)
            if latest is None or timestamp > latest:
                stats["latest_timestamp"] = format_timestamp(timestamp)
            previous = stats.get("_previous_timestamp")
            if previous is not None and timestamp < previous:
                stats["chronologically_ordered"] = False
            stats["_previous_timestamp"] = timestamp

    def _field_coverage(self) -> None:
        for stats in self.inventory.values():
            for entry in stats["fields"].values():
                entry["missing"] = stats["records"] - entry["present"]
                entry["null_coverage"] = entry["null"] / stats["records"] if stats["records"] else 0
                entry["missing_coverage"] = entry["missing"] / stats["records"] if stats["records"] else 0
                entry["types"] = dict(sorted(entry["types"].items()))
            for entry in stats.get("nested_fields", {}).values():
                entry["missing"] = stats["records"] - entry["present"]
                entry["null_coverage"] = entry["null"] / stats["records"] if stats["records"] else 0
                entry["missing_coverage"] = entry["missing"] / stats["records"] if stats["records"] else 0
                entry["types"] = dict(sorted(entry["types"].items()))
            stats.pop("_previous_timestamp", None)

    def _activity_identity(self, source: str, row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
        if source == "events":
            data = row.get("data") if isinstance(row.get("data"), dict) else {}
            speaker_id = data.get("speakerId")
            return (speaker_id if speaker_id in self.agent_names else None), None, data.get("roomId")
        if source == "computer_use_turns":
            session_id = row.get("session_id")
            agent_id = self.session_agents.get(session_id)
            return (agent_id if agent_id in self.agent_names else None), session_id, None
        if source == "chat_messages":
            speaker_id = row.get("agent_speaker_id") if row.get("speaker_type") == "agent" else None
            return (speaker_id if speaker_id in self.agent_names else None), None, row.get("room_id")
        return None, None, None

    def _record_activity(self, source: str, row: dict[str, Any]) -> None:
        agent_id, session_id, room_id = self._activity_identity(source, row)
        self.activity_row_counts[source] += 1
        self.activity_agent_attributed[source] += agent_id is not None
        text_size = 0
        if agent_id:
            self.agent_counts[agent_id][source] += 1
        if session_id:
            self.records_per_session[session_id] += 1
            self.turns_with_session_id += 1
            self.turns_attributed += agent_id is not None
        if room_id:
            self.records_per_room[source][room_id] += 1
        if source == "events":
            data = row.get("data") if isinstance(row.get("data"), dict) else {}
            action = data.get("actionType")
            if action:
                self.event_action_types[str(action)] += 1
            for component in data.get("output", []) if isinstance(data.get("output"), list) else []:
                if isinstance(component, dict):
                    self.event_component_types[str(component.get("type", "<missing>"))] += 1
            message_id = data.get("messageId")
            if message_id:
                self.event_message_references += 1
                self.event_message_ids[str(message_id)] += 1
            text_size = len(data.get("content")) if isinstance(data.get("content"), str) else 0
            event_content = bool(data.get("content"))
            event_output = bool(data.get("output"))
        elif source == "computer_use_turns":
            action = row.get("agent_action")
            action_type = action.get("action") if isinstance(action, dict) else None
            self.action_types[str(action_type or "<missing>")] += 1
            text_size = len(action.get("text")) if isinstance(action, dict) and isinstance(action.get("text"), str) else 0
        elif source == "chat_messages":
            content = row.get("content")
            length = len(content) if isinstance(content, str) else 0
            self.message_lengths.append(length)
            text_size = length
            message_id = row.get("id")
            if message_id:
                self.chat_ids.add(str(message_id))
            if agent_id:
                self.chat_agent_counts[agent_id] += 1
            if room_id:
                self.chat_room_counts[room_id] += 1
            if any(key in row and row[key] is not None for key in ("recipient_id", "recipient_ids", "recipient", "to")):
                self.recipient_field_rows += 1
        timestamp = parse_timestamp(row.get("created_at"))
        if timestamp is None:
            return
        hour = hour_number(timestamp)
        day = timestamp.date().isoformat()
        if agent_id:
            self.agent_days[agent_id].add(day)
            self.daily_agents[day][agent_id] += 1
            self.hour_agents[hour].add(agent_id)
            self.hour_agent_counts[source][hour][agent_id] += 1
            self.hour_agent_text[source][hour][agent_id] += text_size
            self.agent_first[agent_id] = min(self.agent_first.get(agent_id, timestamp), timestamp)
            self.agent_last[agent_id] = max(self.agent_last.get(agent_id, timestamp), timestamp)
        if session_id:
            self.hour_sessions[hour].add(session_id)
            if agent_id:
                self.hour_agent_sessions[hour][agent_id].add(session_id)
        if room_id:
            self.hour_rooms[hour].add(room_id)
        self.hour_text[hour] += text_size
        if source == "events":
            self.hour_event_content[hour] += event_content
            self.hour_event_output[hour] += event_output
        if room_id and agent_id:
            self.hour_agent_rooms[hour][agent_id].add(room_id)

    def _process_special(self, source: str, row: dict[str, Any]) -> None:
        if source == "computer_use_sessions" and row.get("id") and row.get("agent_id"):
            self.session_agents[str(row["id"])] = str(row["agent_id"])
        elif source == "agents" and row.get("id"):
            self.agent_names[str(row["id"])] = str(row.get("name") or row["id"])
        elif source == "agent_goals" and row.get("agent_id"):
            self.goal_intervals[str(row["agent_id"])].append((parse_timestamp(row.get("start_time")), parse_timestamp(row.get("end_time"))))
        elif source == "village_goals":
            self.village_goal_intervals.append((parse_timestamp(row.get("start_time")), parse_timestamp(row.get("end_time"))))

    def scan_source(self, source: str) -> None:
        stats = self._inventory_start(source)
        path = self.dataset_dir / SOURCE_FILES[source]
        print(f"[{source}] scanning {path.name}", flush=True)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    stats["blank_lines"] += 1
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("record root is not an object")
                except (json.JSONDecodeError, ValueError) as error:
                    stats["malformed_records"] += 1
                    self.malformed_examples.setdefault(source, f"line {line_number}: {error}")
                    continue
                self._record_inventory(source, row, stats)
                record_id = row.get("id")
                if record_id is not None:
                    self.connection.execute("INSERT OR IGNORE INTO ids VALUES (?, ?)", (source, str(record_id)))
                self._process_special(source, row)
                if source in ACTIVITY_SOURCES:
                    self._record_activity(source, row)
                source_timestamp = parse_timestamp(row.get(TIME_FIELDS[source]))
                if source_timestamp is not None:
                    self.source_hour[source][hour_number(source_timestamp)] += 1
                    self.source_day[source][source_timestamp.date().isoformat()] += 1
        distinct_ids = self.connection.execute("SELECT COUNT(*) FROM ids WHERE source = ?", (source,)).fetchone()[0]
        stats["distinct_ids"] = distinct_ids
        self.inventory[source] = stats
        print(f"[{source}] {stats['records']:,} records, {stats['malformed_records']} malformed", flush=True)

    def scan(self) -> None:
        self.validate()
        # Session mappings are loaded before computer-use turns are attributed.
        order = ("computer_use_sessions", "agents", "chat_messages", "events", "computer_use_turns", "chat_rooms", "agent_goals", "village_goals")
        for source in order:
            self.scan_source(source)
        self._field_coverage()
        for message_id in self.event_message_ids:
            self.linked_event_messages += message_id in self.chat_ids
        print("[audit] source scans complete", flush=True)

    def _window_stats(self, start_hour: int, duration_hours: int) -> dict[str, Any]:
        end_hour = start_hour + duration_hours
        counts = {source: 0 for source in ACTIVITY_SOURCES}
        agents: set[str] = set()
        sessions: set[str] = set()
        rooms: set[str] = set()
        hourly_overlap = 0
        text_chars = 0
        event_content_records = 0
        event_output_records = 0
        per_agent: dict[str, Counter[str]] = defaultdict(Counter)
        per_agent_text: Counter[str] = Counter()
        per_agent_sessions: dict[str, set[str]] = defaultdict(set)
        per_agent_rooms: dict[str, set[str]] = defaultdict(set)
        for hour in range(start_hour, end_hour):
            active = self.hour_agents.get(hour, set())
            agents.update(active)
            hourly_overlap += len(active) >= 2
            sessions.update(self.hour_sessions.get(hour, set()))
            rooms.update(self.hour_rooms.get(hour, set()))
            for agent_id, values in self.hour_agent_sessions.get(hour, {}).items():
                per_agent_sessions[agent_id].update(values)
            for agent_id, values in self.hour_agent_rooms.get(hour, {}).items():
                per_agent_rooms[agent_id].update(values)
            text_chars += self.hour_text.get(hour, 0)
            event_content_records += self.hour_event_content.get(hour, 0)
            event_output_records += self.hour_event_output.get(hour, 0)
            for source in ACTIVITY_SOURCES:
                counts[source] += self.source_hour[source].get(hour, 0)
                per_agent_counts = self.hour_agent_counts[source].get(hour)
                if per_agent_counts:
                    for agent_id, count in per_agent_counts.items():
                        per_agent[agent_id][source] += count
                        per_agent_text[agent_id] += self.hour_agent_text[source][hour].get(agent_id, 0)
        total = sum(counts.values())
        event_evidence = 1 + (event_content_records + event_output_records) / max(counts["events"] * 2, 1)
        score = len(agents) * math.log1p(total) * (1 + math.log1p(counts["chat_messages"])) * (1 + hourly_overlap / max(duration_hours, 1)) * event_evidence
        return {
            "start": hour_datetime(start_hour),
            "end": hour_datetime(end_hour),
            "duration_hours": duration_hours,
            "counts": counts,
            "active_agents": sorted(agents),
            "active_agent_count": len(agents),
            "sessions": sorted(sessions),
            "rooms": sorted(rooms),
            "session_count": len(sessions),
            "room_count": len(rooms),
            "interaction_records_per_active_agent": total / len(agents) if agents else 0,
            "text_characters": text_chars,
            "event_content_records": event_content_records,
            "event_output_records": event_output_records,
            "overlapping_agent_hours": hourly_overlap,
            "score": score,
            "per_agent_counts": {agent: dict(per_agent[agent]) for agent in per_agent},
            "per_agent_text": dict(per_agent_text),
            "per_agent_sessions": {agent: sorted(values) for agent, values in per_agent_sessions.items()},
            "per_agent_rooms": {agent: sorted(values) for agent, values in per_agent_rooms.items()},
        }

    def candidate_windows(self, duration_hours: int) -> list[dict[str, Any]]:
        activity_hours = [hour for source in ACTIVITY_SOURCES for hour in self.source_hour[source]]
        if not activity_hours:
            return []
        first, last = min(activity_hours), max(activity_hours)
        step = 1 if duration_hours == 6 else (6 if duration_hours == 24 else 24)
        windows = []
        start = first
        while start <= last:
            windows.append(self._window_stats(start, duration_hours))
            start += step
        return windows

    @staticmethod
    def public_window(window: dict[str, Any], include_agents: bool = True) -> dict[str, Any]:
        result = {
            "start": format_timestamp(window["start"]),
            "end": format_timestamp(window["end"]),
            "counts": window["counts"],
            "active_agent_count": window["active_agent_count"],
            "session_count": window["session_count"],
            "room_count": window["room_count"],
            "interaction_records_per_active_agent": round(window["interaction_records_per_active_agent"], 2),
            "text_characters": window["text_characters"],
            "event_content_records": window["event_content_records"],
            "event_output_records": window["event_output_records"],
            "overlapping_agent_hours": window["overlapping_agent_hours"],
        }
        if include_agents:
            result["agent_ids"] = window["active_agents"]
        return result

    def window_summary(self, windows: list[dict[str, Any]]) -> dict[str, Any]:
        fields = ("events", "computer_use_turns", "chat_messages", "active_agent_count", "session_count", "room_count", "text_characters")
        summaries = {}
        for field in fields:
            values = [sum(w["counts"].values()) if field == "interactions" else w["counts"].get(field, 0) if field in w["counts"] else w.get(field, 0) for w in windows]
            summaries[field] = {"median": statistics.median(values) if values else 0, "minimum": min(values, default=0), "maximum": max(values, default=0)}
        ranked = sorted(windows, key=lambda w: (-w["score"], w["start"]))
        return {
            "candidate_window_count": len(windows),
            "distribution": summaries,
            "most_promising": [self.public_window(window) for window in ranked[:5]],
            "least_active_example": self.public_window(min(windows, key=lambda w: (sum(w["counts"].values()), w["start"]))) if windows else None,
        }

    def agent_summary(self) -> dict[str, Any]:
        rows = []
        for agent_id, source_counts in self.agent_counts.items():
            counts = {source: source_counts.get(source, 0) for source in ACTIVITY_SOURCES}
            total = sum(counts.values())
            rows.append({
                "agent_id": agent_id,
                "name": self.agent_names.get(agent_id),
                "counts": counts,
                "total_interactions": total,
                "active_days": len(self.agent_days[agent_id]),
                "first_activity": format_timestamp(self.agent_first.get(agent_id)),
                "last_activity": format_timestamp(self.agent_last.get(agent_id)),
            })
        rows.sort(key=lambda row: (-row["total_interactions"], row["agent_id"]))
        totals = [row["total_interactions"] for row in rows]
        return {
            "registered_agent_count": len(self.agent_names),
            "agents_with_observable_activity": len(rows),
            "agents_with_computer_use": sum(row["counts"]["computer_use_turns"] > 0 for row in rows),
            "agents_with_chat": sum(row["counts"]["chat_messages"] > 0 for row in rows),
            "agents_with_events": sum(row["counts"]["events"] > 0 for row in rows),
            "activity_distribution": {
                "mean": statistics.mean(totals) if totals else 0,
                "median": statistics.median(totals) if totals else 0,
                "q1": percentile(totals, 0.25),
                "q3": percentile(totals, 0.75),
                "maximum": max(totals, default=0),
            },
            "ranked_agents": rows,
        }

    def agent_subset_counts(self, ranked_agents: list[dict[str, Any]]) -> dict[str, Any]:
        result = {}
        for amount in (2, 5, 10, 20, 50, "all"):
            selected = ranked_agents if amount == "all" else ranked_agents[:amount]
            result[str(amount)] = {
                "agent_count": len(selected),
                "agent_ids": [row["agent_id"] for row in selected],
                "records": {source: sum(row["counts"][source] for row in selected) for source in ACTIVITY_SOURCES},
            }
        return result

    def overlap_summary(self) -> dict[str, Any]:
        hour_counts = Counter(len(agents) for agents in self.hour_agents.values())
        active_hours = sum(hour_counts.values())
        daily_counts = Counter(len(agents) for agents in self.daily_agents.values())
        return {
            "activity_hours_with_multiple_agents": sum(count for agents, count in hour_counts.items() if agents >= 2),
            "activity_hours": active_hours,
            "fraction_activity_hours_with_multiple_agents": sum(count for agents, count in hour_counts.items() if agents >= 2) / active_hours if active_hours else 0,
            "maximum_agents_in_same_hour": max(hour_counts, default=0),
            "hours_by_simultaneous_agent_count": {str(key): value for key, value in sorted(hour_counts.items())},
            "days_with_multiple_active_agents": sum(count for agents, count in daily_counts.items() if agents >= 2),
            "active_days": len(daily_counts),
            "maximum_agents_per_day": max(daily_counts, default=0),
            "daily_active_agent_count_distribution": {str(key): value for key, value in sorted(daily_counts.items())},
        }

    def observability(self) -> list[dict[str, Any]]:
        inventory = self.inventory
        def coverage(source: str, field: str) -> float | None:
            entry = inventory[source]["fields"].get(field)
            return entry["present"] / inventory[source]["records"] if entry and inventory[source]["records"] else None
        return [
            {"capability": "Agent identity", "source": "events", "coverage": f"{self.activity_agent_attributed['events'] / self.activity_row_counts['events']:.4f}", "observed_or_joined": "direct, restricted to registered agents", "limitation": "HUMAN events and IDs absent from agents.jsonl are not attributed to agents.", "importance": "Required for agent-level reconstruction."},
            {"capability": "Agent identity", "source": "computer_use_turns", "coverage": f"{self.turns_attributed / self.inventory['computer_use_turns']['records']:.4f}" if self.inventory['computer_use_turns']['records'] else "0", "observed_or_joined": "recovered through computer_use_sessions.id and validated against agents.jsonl", "limitation": "Unmatched session references and unknown agent IDs cannot be attributed.", "importance": "Required for agent-level reconstruction."},
            {"capability": "Agent identity", "source": "chat_messages", "coverage": f"{self.activity_agent_attributed['chat_messages'] / self.activity_row_counts['chat_messages']:.4f}", "observed_or_joined": "direct, restricted to registered agent senders", "limitation": "User/non-agent speakers are not attributed to agents.", "importance": "Required for sender attribution."},
            {"capability": "Timestamp", "source": "events, computer_use_turns, chat_messages", "coverage": "per-source timestamp inventory", "observed_or_joined": "direct", "limitation": "Naive timestamps have no declared timezone; audit buckets assume UTC.", "importance": "Required for temporal ordering."},
            {"capability": "Action and arguments", "source": "computer_use_turns", "coverage": "agent_action structure", "observed_or_joined": "direct", "limitation": "Action arguments vary; a requested action does not guarantee completion.", "importance": "High for action traces."},
            {"capability": "Result/output", "source": "events, computer_use_turns", "coverage": "events.data.output; turn output/error", "observed_or_joined": "direct", "limitation": "Output components may be summaries or tool traces; null turn output is not proof of failure.", "importance": "High, but outcomes require careful interpretation."},
            {"capability": "Resource/tool identifiers", "source": "events, computer_use_turns", "coverage": "event component IDs and nested action fields", "observed_or_joined": "direct when present", "limitation": "Identifiers are heterogeneous and may not name durable shared artifacts.", "importance": "Potential evidence for artifact continuity."},
            {"capability": "Goal/task context", "source": "agent_goals, village_goals, sessions", "coverage": "goal records and session_goal", "observed_or_joined": "direct context; goal intervals use timestamps", "limitation": "Shared context can explain similar actions without information transfer.", "importance": "Needed as alternative explanation."},
            {"capability": "Communication", "source": "chat_messages, chat_rooms", "coverage": "sender and room IDs", "observed_or_joined": "direct", "limitation": "No explicit recipient field observed; room membership does not prove message readership.", "importance": "Strong evidence of communication opportunity, not proof of receipt or use."},
            {"capability": "Cross-source linkage", "source": "events to chat_messages", "coverage": "messageId to chat id", "observed_or_joined": "exact ID comparison", "limitation": "Matching event/message records are not necessarily unique interactions; other similarity is not information-transfer evidence.", "importance": "Useful only for explicit linkage, not causal inference."},
            {"capability": "Information transfer", "source": "all activity sources", "coverage": "not directly observable", "observed_or_joined": "inference only", "limitation": "Shared context, independent convergence, and actual transfer are not separable from text similarity alone.", "importance": "Central limitation for TIDE claims."},
        ]

    def temporal_data(self, windows_by_duration: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        daily = {source: dict(sorted(counter.items())) for source, counter in self.source_day.items()}
        daily_active = {day: len(agents) for day, agents in sorted(self.daily_agents.items())}
        active_hours = sorted(set(hour for counts in self.source_hour.values() for hour in counts))
        return {
            "daily_records_by_source": daily,
            "daily_active_agents": daily_active,
            "hourly_records_by_source": {source: {str(hour): count for hour, count in sorted(counter.items())} for source, counter in self.source_hour.items()},
            "active_days_by_source": {source: len(days) for source, days in ((source, set(counter)) for source, counter in self.source_day.items())},
            "hours_with_data_by_source": {source: len(counter) for source, counter in self.source_hour.items()},
            "activity_hour_range": [format_timestamp(hour_datetime(active_hours[0])), format_timestamp(hour_datetime(active_hours[-1]))] if active_hours else None,
            "window_analysis": {label: self.window_summary(windows) for label, windows in windows_by_duration.items()},
        }

    def _best_by_duration(self, windows: list[dict[str, Any]], duration: int) -> dict[str, Any] | None:
        options = [window for window in windows if window["duration_hours"] == duration]
        return max(options, key=lambda item: (item["score"], -hour_number(item["start"]))) if options else None

    def _sample_specs(self, windows_by_duration: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        specs = []
        occupied: list[tuple[datetime, datetime]] = []
        for label, duration in (("one_day_top10", 24), ("three_day_top10", 72), ("one_week_top20", 168)):
            candidates = sorted(windows_by_duration[str(duration)], key=lambda item: (-item["score"], item["start"]))
            chosen = next((window for window in candidates if all(window["end"] <= start or window["start"] >= end for start, end in occupied)), None)
            if chosen is None:
                continue
            occupied.append((chosen["start"], chosen["end"]))
            ranked = sorted(chosen["per_agent_counts"].items(), key=lambda item: (-sum(item[1].values()), item[0]))
            limit = 10 if duration < 168 else 20
            chosen_agents = [agent for agent, _ in ranked[:limit]]
            specs.append({
                "name": label,
                "selection": chosen,
                "agent_ids": chosen_agents,
                "requested_agent_limit": limit,
                "session_ids": sorted({session for agent in chosen_agents for session in chosen["per_agent_sessions"].get(agent, [])}),
                "room_ids": sorted({room for agent in chosen_agents for room in chosen["per_agent_rooms"].get(agent, [])}),
                "text_characters": sum(chosen["per_agent_text"].get(agent, 0) for agent in chosen_agents),
            })
        return specs

    def _matches_sample(self, source: str, row: dict[str, Any], spec: dict[str, Any]) -> bool:
        window = spec["selection"]
        start, end = window["start"], window["end"]
        agents = set(spec["agent_ids"])
        if source in ACTIVITY_SOURCES:
            timestamp = parse_timestamp(row.get("created_at"))
            if timestamp is None or not start <= timestamp < end:
                return False
            agent_id, _, _ = self._activity_identity(source, row)
            return bool(agent_id and agent_id in agents)
        if source == "computer_use_sessions":
            return row.get("id") in spec["session_ids"] and row.get("agent_id") in agents
        if source == "chat_rooms":
            return row.get("id") in spec["room_ids"]
        if source == "agents":
            return row.get("id") in agents
        if source == "agent_goals":
            return row.get("agent_id") in agents
        if source == "village_goals":
            goal_start = parse_timestamp(row.get("start_time"))
            goal_end = parse_timestamp(row.get("end_time"))
            return bool(goal_start and goal_end and goal_start < end and goal_end > start)
        return False

    def export_sample(self, spec: dict[str, Any]) -> dict[str, Any]:
        sample_dir = self.output_dir / "samples" / spec["name"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        counts = {}
        for source, filename in SOURCE_FILES.items():
            output_file = sample_dir / filename
            count = 0
            with (self.dataset_dir / filename).open("r", encoding="utf-8") as reader, output_file.open("w", encoding="utf-8", newline="\n") as writer:
                for line in reader:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and self._matches_sample(source, row, spec):
                        writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        count += 1
            counts[source] = count
        window = spec["selection"]
        manifest = {
            "sample": spec["name"],
            "dataset": "main-village",
            "source_files": SOURCE_FILES,
            "selection": {
                "start_inclusive": format_timestamp(window["start"]),
                "end_exclusive": format_timestamp(window["end"]),
                "agent_ids": spec["agent_ids"],
                "agent_selection": f"Top {spec['requested_agent_limit']} agents by interaction count within the selected temporal window, ties by agent ID.",
                "deterministic": True,
                "record_filter": "Original rows retained; activity rows selected by source timestamp and attributed agent; context rows selected by referenced IDs or overlapping goal interval.",
            },
            "record_counts": counts,
            "activity_counts": {source: counts[source] for source in ACTIVITY_SOURCES},
            "session_ids": spec["session_ids"],
            "room_ids": spec["room_ids"],
            "text_characters": spec["text_characters"],
            "provenance": "Rows are copied without field transformation from the extracted source JSONL files listed above.",
        }
        (sample_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest

    def _chat_length_summary(self) -> dict[str, Any]:
        values = self.message_lengths
        return {
            "message_count_with_string_content": len(values),
            "mean_characters": statistics.mean(values) if values else 0,
            "median_characters": statistics.median(values) if values else 0,
            "q1_characters": percentile(values, 0.25),
            "q3_characters": percentile(values, 0.75),
            "maximum_characters": max(values, default=0),
            "histogram_characters": {
                "0": sum(value == 0 for value in values),
                "1-49": sum(1 <= value < 50 for value in values),
                "50-199": sum(50 <= value < 200 for value in values),
                "200-999": sum(200 <= value < 1000 for value in values),
                "1000+": sum(value >= 1000 for value in values),
            },
        }

    def build_summary(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        windows_by_duration = {str(hours): self.candidate_windows(hours) for hours in (6, 24, 72, 168, 720)}
        agents = self.agent_summary()
        active_agents = agents["ranked_agents"]
        session_total = self.inventory["computer_use_turns"]["records"]
        turn_records = self.inventory["computer_use_turns"]["records"]
        turn_agent_records = self.turns_attributed
        activity_totals = {source: self.inventory[source]["records"] for source in ACTIVITY_SOURCES}
        active_hour_bins = sorted(self.hour_agents)
        sample_specs = self._sample_specs(windows_by_duration)
        report_windows = {}
        for label, duration in (("best_6_hours", 6), ("best_1_day", 24), ("best_3_days", 72), ("best_1_week", 168), ("best_1_month", 720)):
            report_windows[label] = self.public_window(self._best_by_duration(windows_by_duration[str(duration)], duration)) if self._best_by_duration(windows_by_duration[str(duration)], duration) else None
        combined = {}
        for label, duration, amount in (("top10_1_day", 24, 10), ("top10_3_days", 72, 10), ("top10_1_week", 168, 10), ("top10_1_month", 720, 10)):
            selected = self._best_by_duration(windows_by_duration[str(duration)], duration)
            if selected is None:
                combined[label] = None
                continue
            ranked = sorted(selected["per_agent_counts"].items(), key=lambda item: (-sum(item[1].values()), item[0]))[:amount]
            ids = {agent_id for agent_id, _ in ranked}
            counts = {source: sum(selected["per_agent_counts"].get(agent_id, {}).get(source, 0) for agent_id in ids) for source in ACTIVITY_SOURCES}
            combined[label] = {
                "window": self.public_window(selected),
                "agent_ids": sorted(ids),
                "counts": counts,
                "active_agent_count": len(ids),
                "text_characters": sum(selected["per_agent_text"].get(agent_id, 0) for agent_id in ids),
            }
        source_links = {
            "event_message_id_references": self.event_message_references,
            "distinct_event_message_ids_matching_chat_message_ids": self.linked_event_messages,
            "event_message_id_match_method": "Exact equality of events.data.messageId and chat_messages.id; no timestamp/text similarity treated as a reliable link.",
            "computer_use_turns": turn_records,
            "computer_use_turns_with_session_id": self.turns_with_session_id,
            "computer_use_turn_records_attributed_to_agents": turn_agent_records,
            "session_table_ids_with_agent_mapping": len(self.session_agents),
            "turn_session_record_counts": len(self.records_per_session),
            "explicit_chat_recipient_rows": self.recipient_field_rows,
            "chat_room_membership_available": False,
        }
        summary = {
            "audit": {
                "dataset_directory": str(self.dataset_dir),
                "dataset_name": "main-village",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "timestamp_policy": "Timezone-aware timestamps normalized to UTC; timezone-naive timestamps are treated as UTC for analysis. Original timestamps are preserved in sample rows.",
                "scope": list(SOURCE_FILES),
                "excluded_sources": ["claude_code_messages", "claude_code_sessions", "agent_memories", "summaries"],
                "interaction_definition": "One source row per event, computer-use turn, or chat message. Counts are separate and are not summed as unique interactions.",
                "malformed_record_examples": self.malformed_examples,
            },
            "inventory": self.inventory,
            "activity": {
                "records_by_source": activity_totals,
                "events": activity_totals["events"],
                "computer_use_interactions": activity_totals["computer_use_turns"],
                "communication_interactions": activity_totals["chat_messages"],
                "computer_use_turn_join_coverage": {
                    "total_turns": turn_records,
                    "attributed_turns": turn_agent_records,
                    "unattributed_turns": max(0, turn_records - turn_agent_records),
                    "coverage": turn_agent_records / turn_records if turn_records else 0,
                },
                "registered_agent_identity_coverage": {
                    source: {
                        "rows": self.activity_row_counts[source],
                        "attributed_to_registered_agent": self.activity_agent_attributed[source],
                        "unattributed": self.activity_row_counts[source] - self.activity_agent_attributed[source],
                        "coverage": self.activity_agent_attributed[source] / self.activity_row_counts[source] if self.activity_row_counts[source] else 0,
                    }
                    for source in ACTIVITY_SOURCES
                },
                "per_session_computer_use_turn_count": {
                    "distinct_sessions_with_turns": len(self.records_per_session),
                    "mean": statistics.mean(self.records_per_session.values()) if self.records_per_session else 0,
                    "median": statistics.median(self.records_per_session.values()) if self.records_per_session else 0,
                    "maximum": max(self.records_per_session.values(), default=0),
                },
                "records_per_room": {source: dict(counter.most_common()) for source, counter in self.records_per_room.items()},
                "event_action_types": dict(self.event_action_types.most_common()),
                "computer_use_action_types": dict(self.action_types.most_common()),
                "event_output_component_types": dict(self.event_component_types.most_common()),
                "message_lengths": self._chat_length_summary(),
                "links": source_links,
                "chat": {
                    "distinct_senders": len(self.chat_agent_counts),
                    "distinct_message_rooms": len(self.chat_room_counts),
                    "messages_per_room": dict(self.chat_room_counts.most_common()),
                    "messages_per_agent": dict(self.chat_agent_counts.most_common()),
                    "room_records_in_table": self.inventory["chat_rooms"]["records"],
                    "room_participation_reconstructed_from_messages_only": True,
                    "room_participation_caveat": "Observed senders can be linked to rooms; membership and readership cannot be inferred from room presence.",
                },
                "goals": {
                    "individual_goal_records": self.inventory["agent_goals"]["records"],
                    "agents_with_goal_records": len(self.goal_intervals),
                    "village_goal_records": self.inventory["village_goals"]["records"],
                    "shared_task_context_available": self.inventory["village_goals"]["records"] > 0,
                },
            },
            "agents": agents,
            "agent_subset_counts": self.agent_subset_counts(active_agents),
            "overlap": self.overlap_summary(),
            "temporal": self.temporal_data(windows_by_duration),
            "best_windows": report_windows,
            "combined_temporal_agent_subsets": combined,
            "observability": self.observability(),
            "sampling": {"candidate_exports": [spec["name"] for spec in sample_specs]},
        }
        return summary, sample_specs

    def create_figures(self, summary: dict[str, Any]) -> list[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as error:
            raise RuntimeError("Matplotlib and NumPy are required; install the declared project dependencies with `uv sync`.") from error
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        def save(fig: Any, name: str) -> None:
            fig.tight_layout()
            fig.savefig(figures_dir / name, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(f"figures/{name}")

        daily = self.source_day
        day_keys = sorted(set(day for counts in daily.values() for day in counts))
        dates = [datetime.fromisoformat(day) for day in day_keys]
        fig, ax = plt.subplots(figsize=(13, 5))
        for source, label in (("events", "Events"), ("computer_use_turns", "Computer-use turns"), ("chat_messages", "Chat messages")):
            ax.plot(dates, [daily[source].get(day, 0) for day in day_keys], label=label, linewidth=1.1)
        ax.set(title="Daily observable activity", ylabel="Records", xlabel="Date")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        save(fig, "daily_activity.png")

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(dates, [len(self.daily_agents.get(day, {})) for day in day_keys], color="#287a68", linewidth=1.2)
        ax.set(title="Daily active agents", ylabel="Distinct agents", xlabel="Date")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid(axis="y", alpha=0.25)
        save(fig, "daily_active_agents.png")

        ranked = summary["agents"]["ranked_agents"]
        top = ranked[:20]
        labels = [row["name"] or row["agent_id"] for row in top]
        values = [row["total_interactions"] for row in top]
        other = sum(row["total_interactions"] for row in ranked[20:])
        if other:
            labels.append("Other")
            values.append(other)
        fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.27)))
        ax.barh(labels[::-1], values[::-1], color="#287a68")
        ax.set(title="Agent activity ranking", xlabel="Event, computer-use, and chat records")
        ax.grid(axis="x", alpha=0.25)
        save(fig, "agent_activity_ranking.png")

        top_ids = [row["agent_id"] for row in top]
        weeks = sorted({datetime.fromisoformat(day).strftime("%G-W%V") for day in day_keys})
        week_dates = {week: [] for week in weeks}
        for day in day_keys:
            week_dates[datetime.fromisoformat(day).strftime("%G-W%V")].append(day)
        matrix = np.zeros((len(top_ids), len(weeks)), dtype=int)
        for agent_index, agent_id in enumerate(top_ids):
            for week_index, week in enumerate(weeks):
                matrix[agent_index, week_index] = sum(self.daily_agents[day].get(agent_id, 0) for day in week_dates[week])
        fig_width = min(18, max(9, len(weeks) * 0.16))
        fig, ax = plt.subplots(figsize=(fig_width, max(5, len(top_ids) * 0.28)))
        if len(weeks) and len(top_ids):
            image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="YlGnBu")
            ax.set_yticks(range(len(top_ids)), [self.agent_names.get(agent, agent) for agent in top_ids])
            stride = max(1, len(weeks) // 12)
            ax.set_xticks(range(0, len(weeks), stride), [weeks[index] for index in range(0, len(weeks), stride)], rotation=45, ha="right")
            fig.colorbar(image, ax=ax, label="Interactions")
        ax.set(title="Top-agent activity by week", xlabel="ISO week")
        save(fig, "agent_week_heatmap.png")

        self._category_chart(plt, self.chat_room_counts, "Messages per room", "Messages", "messages_per_room.png", figures_dir, saved, 20)
        self._category_chart(plt, self.chat_agent_counts, "Messages per agent", "Messages", "messages_per_agent.png", figures_dir, saved, 20, self.agent_names)
        self._category_chart(plt, self.action_types, "Computer-use action types", "Turns", "computer_use_action_types.png", figures_dir, saved, 20)
        self._category_chart(plt, self.event_component_types, "Event output component types", "Components across event records", "event_output_component_types.png", figures_dir, saved, 20)

        source_ranges = []
        for source, stats in self.inventory.items():
            if stats["earliest_timestamp"] and stats["latest_timestamp"]:
                source_ranges.append((source, parse_timestamp(stats["earliest_timestamp"]), parse_timestamp(stats["latest_timestamp"])))
        fig, ax = plt.subplots(figsize=(12, max(4, len(source_ranges) * 0.42)))
        for index, (source, start, end) in enumerate(source_ranges):
            ax.plot([start, end], [index, index], linewidth=7, solid_capstyle="butt")
        ax.set_yticks(range(len(source_ranges)), [row[0] for row in source_ranges])
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.set(title="Temporal coverage by source", xlabel="Timestamp")
        ax.grid(axis="x", alpha=0.25)
        save(fig, "temporal_coverage.png")

        window_labels = ["6h", "1d", "3d", "1w", "30d"]
        durations = ["6", "24", "72", "168", "720"]
        maxima = [summary["temporal"]["window_analysis"][duration]["distribution"] for duration in durations]
        x = np.arange(len(window_labels))
        width = 0.24
        fig, ax = plt.subplots(figsize=(10, 5))
        for index, (key, label) in enumerate((("events", "Events"), ("computer_use_turns", "Computer-use turns"), ("chat_messages", "Chat messages"))):
            ax.bar(x + (index - 1) * width, [item[key]["maximum"] for item in maxima], width, label=label)
        ax.set_xticks(x, window_labels)
        ax.set(title="Maximum records in candidate temporal windows", ylabel="Records")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        save(fig, "temporal_window_comparison.png")

        subset_summary = summary["agent_subset_counts"]
        subset_labels = ["2", "5", "10", "20", "50", "all"]
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.24
        for index, (source, label) in enumerate((("events", "Events"), ("computer_use_turns", "Computer-use turns"), ("chat_messages", "Chat messages"))):
            ax.bar(np.arange(len(subset_labels)) + (index - 1) * width, [subset_summary[key]["records"][source] for key in subset_labels], width, label=label)
        ax.set_xticks(np.arange(len(subset_labels)), subset_labels)
        ax.set(title="Records retained by global agent-activity subset", xlabel="Selected agents", ylabel="Records")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        save(fig, "agent_subset_comparison.png")
        return saved

    @staticmethod
    def _category_chart(plt: Any, counter: Counter[str], title: str, ylabel: str, filename: str, figures_dir: Path, saved: list[str], limit: int, labels: dict[str, str] | None = None) -> None:
        items = counter.most_common(limit)
        other = sum(counter.values()) - sum(value for _, value in items)
        names = [(labels.get(key, key) if labels else key) for key, _ in items]
        values = [value for _, value in items]
        if other:
            names.append("Other")
            values.append(other)
        fig, ax = plt.subplots(figsize=(11, max(4, len(names) * 0.27)))
        ax.barh(names[::-1], values[::-1], color="#c56d3a")
        ax.set(title=title, xlabel=ylabel)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(f"figures/{filename}")

    def write_report(self, summary: dict[str, Any], manifests: list[dict[str, Any]], figures: list[str]) -> None:
        summary["sampling"]["candidate_exports"] = manifests
        (self.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        activity = summary["activity"]
        agents = summary["agents"]
        overlap = summary["overlap"]
        best_day = summary["best_windows"].get("best_1_day")
        top_agents = agents["ranked_agents"][:20]
        figure_lines = "\n".join(f"![{Path(item).stem.replace('_', ' ').title()}]({item})" for item in figures)
        table_rows = []
        for key, label in (("best_1_day", "1 day"), ("best_3_days", "3 days"), ("best_1_week", "1 week"), ("best_1_month", "30 days")):
            window = summary["best_windows"].get(key)
            if window:
                counts = window["counts"]
                table_rows.append(f"| {label} | {window['start']} to {window['end']} | {window['active_agent_count']} | {counts['events']} | {counts['computer_use_turns']} | {counts['chat_messages']} | {window['session_count']} | {window['text_characters']:,} |")
        comparison_table = "\n".join(table_rows)
        agent_lines = "\n".join(f"| {row['agent_id']} | {row['name'] or ''} | {row['total_interactions']} | {row['counts']['events']} | {row['counts']['computer_use_turns']} | {row['counts']['chat_messages']} |" for row in top_agents)
        sample_lines = "\n".join(f"| {manifest['sample']} | {manifest['selection']['start_inclusive']} to {manifest['selection']['end_exclusive']} | {len(manifest['selection']['agent_ids'])} | {manifest['activity_counts']['events']} | {manifest['activity_counts']['computer_use_turns']} | {manifest['activity_counts']['chat_messages']} | `samples/{manifest['sample']}/manifest.json` |" for manifest in manifests)
        join = activity["computer_use_turn_join_coverage"]
        link = activity["links"]
        identities = activity["registered_agent_identity_coverage"]
        report = f"""# AI Village Main-Village Data Audit

## Scope and method

This audit reads only extracted JSONL files under `{self.dataset_dir.as_posix()}`. It excludes Claude Code records, memories, and summaries. Source row counts are reported independently and are not treated as deduplicated interaction totals. Processing is streaming; samples preserve original nested records and source-file provenance.

Timezone-naive source timestamps are treated as UTC for bucketing because the dataset does not declare a timezone. Their original strings remain unchanged in sample exports. Malformed JSON rows are counted and skipped. No dataset content is sent to external services.

## Headline findings

- Event records: **{activity['events']:,}**; computer-use turns: **{activity['computer_use_interactions']:,}**; chat messages: **{activity['communication_interactions']:,}**. These are separate source-row counts, not a unique-interaction sum.
- Rows attributed to registered agents: events **{identities['events']['attributed_to_registered_agent']:,}/{identities['events']['rows']:,}**, computer-use turns **{identities['computer_use_turns']['attributed_to_registered_agent']:,}/{identities['computer_use_turns']['rows']:,}**, chat messages **{identities['chat_messages']['attributed_to_registered_agent']:,}/{identities['chat_messages']['rows']:,}**. Source totals also include records without an identifiable registered agent.
- Registered agents: **{agents['registered_agent_count']}**; agents with observable event, computer-use, or chat activity: **{agents['agents_with_observable_activity']}**.
- Agents with events: **{agents['agents_with_events']}**; computer-use: **{agents['agents_with_computer_use']}**; chat: **{agents['agents_with_chat']}**.
- Agents have activity in the same hour in **{overlap['activity_hours_with_multiple_agents']:,} of {overlap['activity_hours']:,} hours containing activity** ({overlap['fraction_activity_hours_with_multiple_agents']:.1%}); peak same-hour concurrency is **{overlap['maximum_agents_in_same_hour']}**. This measures temporal co-activity, not interaction.
- Computer-use turns attributed to agents via session join: **{join['attributed_turns']:,}/{join['total_turns']:,} ({join['coverage']:.2%})**.
- Exact event-to-chat ID linkage: **{link['distinct_event_message_ids_matching_chat_message_ids']:,} distinct event message IDs** match chat message IDs, out of {link['event_message_id_references']:,} event rows with a message ID. Other source pairs cannot be reliably deduplicated from timestamps or text.
- Chat rows with a populated explicit-recipient field: **{link['explicit_chat_recipient_rows']:,}**. Rooms expose observed senders/rooms, not membership or readership.

## Inventory

See `summary.json` for per-file record counts, unique IDs, field/type availability, null and missing coverage, timestamp range/order, malformed rows, and daily/hourly counts. For sampling, extracted files with the requested source names were used directly; no compressed archives were decompressed.

## Temporal windows

Fixed windows are evaluated at one-hour steps for 6-hour windows, six-hour steps for one-day windows, and daily steps for 3-day, 7-day, and 30-day windows. The JSON summary includes median/minimum/maximum counts across candidate windows and the five highest-scoring examples for each duration. Ranking favors multi-agent activity, total records, chat volume, and same-hour overlap; it is a reproducible screening heuristic, not an inference result.

| Window | Best start/end | Active agents | Events | Computer-use turns | Chat messages | Sessions | Text characters |
|---|---|---:|---:|---:|---:|---:|---:|
{comparison_table}

One-day recommendation for manual inspection: **{best_day['start'] if best_day else 'no timed activity'} to {best_day['end'] if best_day else 'no timed activity'}**, with {best_day['active_agent_count'] if best_day else 0} agents and {best_day['counts']['events'] if best_day else 0} events, {best_day['counts']['computer_use_turns'] if best_day else 0} computer-use turns, and {best_day['counts']['chat_messages'] if best_day else 0} chat messages. The candidate export manifest identifies the selected agent IDs and exact session/room IDs.

## Agent activity and sampling

The agent activity distribution (all source rows per observable agent) is mean {agents['activity_distribution']['mean']:.2f}, median {agents['activity_distribution']['median']:.2f}, Q1 {agents['activity_distribution']['q1']}, Q3 {agents['activity_distribution']['q3']}, maximum {agents['activity_distribution']['maximum']}. Top agents are ranked by total observed records; subset counts in `summary.json` show the actual effect of selecting 2, 5, 10, 20, 50, or all active agents.

| Agent ID | Name | All records | Events | Computer-use turns | Chat messages |
|---|---|---:|---:|---:|---:|
{agent_lines}

| Sample | Time window | Agents | Events | Computer-use turns | Chat messages | Manifest |
|---|---|---:|---:|---:|---:|---|
{sample_lines}

Exports select the top 10 agents for one-day and three-day candidates, and top 20 for the week candidate, using activity within each selected window (ties by ID). Candidate windows are chosen to avoid temporal overlap between exported activity rows. Context records are included by linked session/room/agent IDs or overlapping village-goal intervals. Every manifest lists source counts and criteria.

## Communication and context

Chat provides explicit message content, sender, timestamp, and room. It does not expose a populated recipient field in this dataset, and room IDs do not prove which agents read a message. Individual and village goals provide task context, but shared goals/instructions and common environmental exposure are alternative explanations for similar actions. An agent's message is not itself evidence that another agent received or used its information.

Events retain each outer event as one record. Output-component chart counts nested `events.data.output[*].type` values without expanding records. Computer-use action types come from `agent_action.action`. Content similarity is not treated as evidence of information transfer.

## TIDE observability assessment

| Capability | Source | Coverage / method | Limitation | TIDE importance |
|---|---|---|---|---|
""" + "\n".join(f"| {row['capability']} | {row['source']} | {row['coverage']} | {row['limitation']} | {row['importance']} |" for row in summary["observability"]) + f"""

## Figures

{figure_lines}

## Recommendation and limitations

Start manual inspection with the **one-day top-10-agent sample** in `samples/one_day_top10/`. Verify individual event rows against the joined computer-use turns, then inspect explicit chat links and goal intervals. Include events, computer-use turns/sessions, chat messages/rooms, agent identities, and applicable individual/village goals in a first observational dataset. Treat goals as shared-context controls, not evidence of transfer.

The data supports activity timing and explicit communication opportunity. It does not directly show what another agent read, which shared environment state it observed, or whether a later action used information from a prior action. Event outputs may contain model/tool outputs rather than verified environmental outcomes. Temporal co-activity and exact message-ID links improve observability but do not validate TIDE or establish causal dependencies.
"""
        (self.output_dir / "report.md").write_text(report, encoding="utf-8")

    def run(self) -> None:
        self.scan()
        summary, sample_specs = self.build_summary()
        self.output_dir.mkdir(parents=True)
        manifests = []
        for spec in sample_specs:
            print(f"[sample] exporting {spec['name']}", flush=True)
            manifests.append(self.export_sample(spec))
        figures = self.create_figures(summary)
        summary["figures"] = figures
        self.write_report(summary, manifests, figures)
        print(f"[audit] report: {self.output_dir / 'report.md'}", flush=True)
        print(f"[audit] summary: {self.output_dir / 'summary.json'}", flush=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit extracted AI Village main-village activity for TIDE sampling.")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Path to extracted main-village JSONL files")
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory; existing paths are never overwritten")
    args = parser.parse_args(argv)
    try:
        Audit(args.dataset_dir, args.output_dir).run()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"audit failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
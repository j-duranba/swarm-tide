from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


WINDOWS = {
    "Oct 24-31, 2026": ("2026-10-24 05:00:00", "2026-10-31 05:00:00"),
    "Apr 3-10, 2025": ("2025-04-03 17:00:00", "2025-04-10 17:00:00"),
}
SAMPLE_NAMES = ("one_day_top10", "three_day_top10", "one_week_top20")
ACTION_TARGET_KEYS = {"target", "url", "path", "file", "filename", "selector", "coordinate", "coordinates", "text", "key", "keys", "command", "query", "button"}


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON in {path} line {line_number}: {error}") from error
            if isinstance(row, dict):
                yield row


def timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        return None


def month_key(value: datetime) -> str:
    return value.strftime("%Y-%m")


def hour_key(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H")


def digest_text(value: Any) -> tuple[str | None, int]:
    if not isinstance(value, str):
        return None, 0
    return hashlib.sha256(value.encode("utf-8")).hexdigest(), len(value)


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def short_id(value: str | None) -> str:
    return value[:8] if value else "-"


def state(value: Any, present: bool = True) -> str:
    if not present:
        return "missing"
    if value is None:
        return "null"
    if value == "" or value == [] or value == {}:
        return "empty"
    return "non-empty"


class ForensicAudit:
    def __init__(self, dataset_dir: Path, samples_dir: Path, output_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.samples_dir = samples_dir
        self.output_dir = output_dir
        self.temp = tempfile.TemporaryDirectory(prefix="tide-ai-village-forensics-")
        self.db = sqlite3.connect(str(Path(self.temp.name) / "forensic.sqlite"))
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA temp_store=FILE")
        self.agent_names: dict[str, str] = {}
        self.session_info: dict[str, tuple[str | None, str | None]] = {}
        self.speaker_type_rows: Counter[str] = Counter()
        self.speaker_type_ids: dict[str, set[str]] = defaultdict(set)
        self.event_speakers: set[str] = set()
        self.chat_agent_ids: set[str] = set()
        self.chat_user_ids: set[str] = set()
        self.event_speaker_rows = 0
        self.event_speaker_missing = 0
        self.event_speaker_null = 0
        self.event_rows_by_source_type: Counter[str] = Counter()
        self.event_ids_registered: Counter[str] = Counter()
        self.chat_ids_registered: Counter[str] = Counter()
        self.event_identity_by_source_type: dict[str, set[str]] = defaultdict(set)
        self.turn_actions: Counter[str] = Counter()
        self.turn_states: dict[str, Counter[str]] = defaultdict(Counter)
        self.action_field_rows: Counter[str] = Counter()
        self.action_field_types: dict[str, Counter[str]] = defaultdict(Counter)
        self.action_field_keys: Counter[str] = Counter()
        self.target_field_rows: Counter[str] = Counter()
        self.tool_call_names: Counter[str] = Counter()
        self.action_arg_rows = 0
        self.turn_count = 0
        self.turn_timestamp_invalid = 0
        self.daily: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
        self.monthly: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
        self.daily_agents: dict[str, set[str]] = defaultdict(set)
        self.monthly_agents: dict[str, set[str]] = defaultdict(set)
        self.hour_sources: dict[str, set[str]] = defaultdict(set)
        self.hour_agents: dict[str, set[str]] = defaultdict(set)
        self.windows = {
            name: {
                "start": timestamp(start), "end": timestamp(end), "duration_hours": (timestamp(end) - timestamp(start)).total_seconds() / 3600,
                "counts": Counter(), "agents": set(), "sessions": set(), "rooms": set(), "hour_agents": defaultdict(set), "hour_sources": defaultdict(set),
                "per_source_agents": defaultdict(set), "hour_validated_sources": defaultdict(set),
                "validated_counts": Counter(), "per_agent_counts": defaultdict(Counter),
                "per_agent_sessions": defaultdict(set), "per_agent_active_hours": defaultdict(set),
            }
            for name, (start, end) in WINDOWS.items()
        }
        self.turn_session_counts: Counter[str] = Counter()
        self.turn_interarrival: list[float] = []
        self.event_total = 0
        self.chat_total = 0
        self.event_action_types: Counter[str] = Counter()
        self.output_component_types: Counter[str] = Counter()
        self.link_summary: dict[str, Any] = {}
        self.sample_summaries: list[dict[str, Any]] = []
        self.session_examples: list[dict[str, Any]] = []

    def validate(self) -> None:
        if self.dataset_dir.name != "main-village" or not self.dataset_dir.is_dir():
            raise ValueError(f"--dataset-dir must point to extracted main-village: {self.dataset_dir}")
        for name in ("agents", "events", "computer_use_sessions", "computer_use_turns", "chat_messages"):
            if not (self.dataset_dir / f"{name}.jsonl").is_file():
                raise FileNotFoundError(self.dataset_dir / f"{name}.jsonl")
        report_file = self.output_dir / "forensic_validation.md"
        if report_file.exists():
            raise FileExistsError(f"Refusing to overwrite {report_file}")

    def setup_db(self) -> None:
        self.db.executescript("""
            CREATE TABLE chat (
                row_id INTEGER PRIMARY KEY, message_id TEXT, created_at TEXT, speaker_id TEXT,
                user_speaker_id TEXT, speaker_type TEXT, room_id TEXT, content_hash TEXT, content_chars INTEGER
            );
            CREATE INDEX chat_message_id_idx ON chat(message_id);
            CREATE TABLE event_message (
                row_id INTEGER PRIMARY KEY, message_id TEXT, event_id TEXT, created_at TEXT,
                speaker_id TEXT, speaker_type TEXT, action_type TEXT, room_id TEXT,
                content_hash TEXT, content_chars INTEGER, output_types TEXT, event_index INTEGER, village_id TEXT
            );
            CREATE INDEX event_message_id_idx ON event_message(message_id);
            CREATE TABLE turn (
                row_id INTEGER PRIMARY KEY, session_id TEXT, created_at TEXT, agent_id TEXT,
                action TEXT, action_keys TEXT, result_hint TEXT
            );
            CREATE INDEX turn_session_time_idx ON turn(session_id, created_at);
        """)

    def load_identities(self) -> None:
        for row in rows(self.dataset_dir / "agents.jsonl"):
            if row.get("id"):
                self.agent_names[str(row["id"])] = str(row.get("name") or row["id"])
        for row in rows(self.dataset_dir / "computer_use_sessions.jsonl"):
            if row.get("id"):
                agent_id = str(row["agent_id"]) if row.get("agent_id") else None
                if agent_id not in self.agent_names:
                    agent_id = None
                self.session_info[str(row["id"])] = (agent_id, row.get("created_at"))

    def add_time_record(self, source: str, dt: datetime | None, agent_id: str | None, session_id: str | None = None, room_id: str | None = None) -> None:
        if dt is None:
            return
        day, month, hour = dt.date().isoformat(), month_key(dt), hour_key(dt)
        self.daily[day][source] += 1
        self.monthly[month][source] += 1
        self.hour_sources[hour].add(source)
        if agent_id:
            self.daily_agents[day].add(agent_id)
            self.monthly_agents[month].add(agent_id)
            self.hour_agents[hour].add(agent_id)
        for window in self.windows.values():
            if window["start"] <= dt < window["end"]:
                window["counts"][source] += 1
                window["hour_sources"][hour].add(source)
                if agent_id:
                    window["agents"].add(agent_id)
                    window["hour_agents"][hour].add(agent_id)
                    window["hour_validated_sources"][hour].add(source)
                    window["per_source_agents"][source].add(agent_id)
                    window["validated_counts"][source] += 1
                    window["per_agent_counts"][agent_id][source] += 1
                    window["per_agent_active_hours"][agent_id].add(hour)
                if session_id:
                    window["sessions"].add(session_id)
                    if agent_id:
                        window["per_agent_sessions"][agent_id].add(session_id)
                if room_id:
                    window["rooms"].add(room_id)

    def scan_chat(self) -> None:
        print("[forensics] indexing chat IDs and identities", flush=True)
        batch = []
        for row in rows(self.dataset_dir / "chat_messages.jsonl"):
            self.chat_total += 1
            speaker = str(row["agent_speaker_id"]) if row.get("agent_speaker_id") else None
            user_speaker = str(row["user_speaker_id"]) if row.get("user_speaker_id") else None
            if speaker:
                self.chat_agent_ids.add(speaker)
                if row.get("speaker_type") == "agent" and speaker in self.agent_names:
                    self.chat_ids_registered["rows"] += 1
            if user_speaker:
                self.chat_user_ids.add(user_speaker)
            text_hash, chars = digest_text(row.get("content"))
            batch.append((row.get("id"), row.get("created_at"), speaker, user_speaker, row.get("speaker_type"), row.get("room_id"), text_hash, chars))
            dt = timestamp(row.get("created_at"))
            valid_speaker = speaker if row.get("speaker_type") == "agent" and speaker in self.agent_names else None
            self.add_time_record("chat", dt, valid_speaker, room_id=row.get("room_id"))
            if len(batch) >= 5000:
                self.db.executemany("INSERT INTO chat(message_id,created_at,speaker_id,user_speaker_id,speaker_type,room_id,content_hash,content_chars) VALUES (?,?,?,?,?,?,?,?)", batch)
                batch.clear()
        if batch:
            self.db.executemany("INSERT INTO chat(message_id,created_at,speaker_id,user_speaker_id,speaker_type,room_id,content_hash,content_chars) VALUES (?,?,?,?,?,?,?,?)", batch)
        self.db.commit()

    def scan_events(self) -> None:
        print("[forensics] classifying event speakers and indexing message references", flush=True)
        batch = []
        for row in rows(self.dataset_dir / "events.jsonl"):
            self.event_total += 1
            data = row.get("data") if isinstance(row.get("data"), dict) else {}
            speaker = str(data["speakerId"]) if data.get("speakerId") else None
            speaker_type = str(data.get("speakerType") or "<missing-or-null>")
            self.event_rows_by_source_type[speaker_type] += 1
            if speaker is None:
                self.event_speaker_missing += "speakerId" not in data
                self.event_speaker_null += "speakerId" in data and data.get("speakerId") is None
            else:
                self.event_speaker_rows += 1
                self.event_speakers.add(speaker)
                self.speaker_type_ids[speaker_type].add(speaker)
                if speaker in self.agent_names:
                    self.event_ids_registered[speaker_type] += 1
                    self.event_identity_by_source_type[speaker_type].add(speaker)
            action = str(data.get("actionType") or "<missing-or-null>")
            self.event_action_types[action] += 1
            outputs = data.get("output") if isinstance(data.get("output"), list) else []
            output_types = [str(item.get("type", "<missing>")) for item in outputs if isinstance(item, dict)]
            self.output_component_types.update(output_types)
            text_hash, chars = digest_text(data.get("content"))
            message_id = data.get("messageId")
            if message_id:
                batch.append((str(message_id), row.get("id"), row.get("created_at"), speaker, speaker_type, action, data.get("roomId"), text_hash, chars, json.dumps(output_types), row.get("event_index"), row.get("village_id")))
            valid_agent = speaker if speaker in self.agent_names else None
            self.add_time_record("events", timestamp(row.get("created_at")), valid_agent, room_id=data.get("roomId"))
        if batch:
            self.db.executemany("INSERT INTO event_message(message_id,event_id,created_at,speaker_id,speaker_type,action_type,room_id,content_hash,content_chars,output_types,event_index,village_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        self.db.commit()

    @staticmethod
    def turn_result_hint(row: dict[str, Any]) -> str:
        agent_messages = row.get("agent_messages")
        calls = []
        if isinstance(agent_messages, dict) and isinstance(agent_messages.get("content"), list):
            for item in agent_messages["content"]:
                if isinstance(item, dict):
                    name = item.get("name")
                    if name:
                        calls.append(str(name))
        availability = ",".join(f"{key}:{state(row.get(key), key in row)}" for key in ("output", "error", "system"))
        call_part = "calls=" + "+".join(calls[:3]) if calls else "calls=none"
        return f"{availability}; {call_part}"

    def scan_turns(self) -> None:
        print("[forensics] summarizing computer-use turns", flush=True)
        batch = []
        path = self.dataset_dir / "computer_use_turns.jsonl"
        for row in rows(path):
            self.turn_count += 1
            session_id = str(row["session_id"]) if row.get("session_id") else None
            agent_id = self.session_info.get(session_id, (None, None))[0] if session_id else None
            action_value = row.get("agent_action")
            action = str(action_value.get("action") or "<missing-or-null>") if isinstance(action_value, dict) else "<missing-action>" if "agent_action" not in row else "<null-action>"
            self.turn_actions[action] += 1
            for key in ("agent_action", "output", "error", "system", "agent_messages"):
                self.turn_states[key][state(row.get(key), key in row)] += 1
            if isinstance(action_value, dict):
                self.action_arg_rows += bool(set(action_value) - {"action"})
                for key, value in action_value.items():
                    self.action_field_rows[key] += 1
                    self.action_field_types[key][type(value).__name__] += 1
                    if key.lower() in ACTION_TARGET_KEYS and value not in (None, "", [], {}):
                        self.target_field_rows[key] += 1
                self.action_field_keys.update(action_value.keys())
            messages = row.get("agent_messages")
            if isinstance(messages, dict) and isinstance(messages.get("content"), list):
                for item in messages["content"]:
                    if isinstance(item, dict) and item.get("name"):
                        self.tool_call_names[str(item["name"])] += 1
            created = row.get("created_at")
            parsed = timestamp(created)
            if parsed is None:
                self.turn_timestamp_invalid += 1
            else:
                self.add_time_record("computer_use", parsed, agent_id, session_id=session_id)
            if session_id:
                self.turn_session_counts[session_id] += 1
            batch.append((session_id, created, agent_id, action, ",".join(sorted(action_value.keys())) if isinstance(action_value, dict) else "", self.turn_result_hint(row)))
            if len(batch) >= 5000:
                self.db.executemany("INSERT INTO turn(session_id,created_at,agent_id,action,action_keys,result_hint) VALUES (?,?,?,?,?,?)", batch)
                batch.clear()
        if batch:
            self.db.executemany("INSERT INTO turn(session_id,created_at,agent_id,action,action_keys,result_hint) VALUES (?,?,?,?,?,?)", batch)
        self.db.commit()

    def analyze_links(self) -> None:
        query = """
            SELECT e.message_id,e.event_id,e.created_at,e.speaker_id,e.speaker_type,e.action_type,e.room_id,e.content_hash,e.content_chars,e.output_types,
                   c.message_id,c.created_at,c.speaker_id,c.user_speaker_id,c.speaker_type,c.room_id,c.content_hash,c.content_chars
            FROM event_message e LEFT JOIN chat c ON c.message_id=e.message_id
            ORDER BY e.message_id,e.row_id
        """
        matched_refs = matched_distinct = same_content = same_agent = same_sender = same_room = exact_time = within_one_second = 0
        delta_ms: list[float] = []
        rows_per_message: Counter[str] = Counter()
        examples = []
        event_extra_action: Counter[str] = Counter()
        linked_output_types: Counter[str] = Counter()
        total_refs = self.db.execute("SELECT COUNT(*) FROM event_message").fetchone()[0]
        distinct_refs = self.db.execute("SELECT COUNT(DISTINCT message_id) FROM event_message").fetchone()[0]
        matched_message_ids: set[str] = set()
        for item in self.db.execute(query):
            (message_id,event_id,event_time,event_speaker,event_speaker_type,event_action,event_room,event_hash,event_chars,event_outputs,
             chat_id,chat_time,chat_speaker,chat_user,chat_speaker_type,chat_room,chat_hash,chat_chars) = item
            rows_per_message[message_id] += 1
            if chat_id is None:
                continue
            matched_refs += 1
            matched_message_ids.add(message_id)
            same_content += event_hash == chat_hash
            same_agent += event_speaker is not None and event_speaker == chat_speaker and chat_speaker_type == "agent"
            same_sender += event_speaker == (chat_speaker if chat_speaker_type == "agent" else chat_user)
            same_room += event_room == chat_room
            event_dt, chat_dt = timestamp(event_time), timestamp(chat_time)
            if event_dt is not None and chat_dt is not None:
                delta = (event_dt - chat_dt).total_seconds() * 1000
                delta_ms.append(delta)
                exact_time += delta == 0
                within_one_second += abs(delta) <= 1000
            event_extra_action[event_action or "<null>"] += 1
            linked_output_types.update(json.loads(event_outputs or "[]"))
            if len(examples) < 6 and rows_per_message[message_id] == 1:
                examples.append({
                    "event_id": short_id(event_id), "message_id": short_id(message_id),
                    "time_delta_ms": round(delta_ms[-1], 3) if delta_ms else None,
                    "same_registered_sender": bool(event_speaker and event_speaker == chat_speaker and chat_speaker_type == "agent"),
                    "same_room": event_room == chat_room, "content_hash_equal": event_hash == chat_hash,
                    "event_chars": event_chars, "chat_chars": chat_chars,
                    "event_action_type": event_action, "event_output_types": json.loads(event_outputs or "[]"),
                })
        multi_event_ids = sum(count > 1 for count in rows_per_message.values())
        extra_rows = self.db.execute("""
            SELECT COUNT(*) FROM event_message e JOIN chat c ON c.message_id=e.message_id
        """).fetchone()[0]
        total_chat_matched_ids = self.db.execute("""
            SELECT COUNT(DISTINCT c.message_id) FROM chat c JOIN event_message e ON e.message_id=c.message_id
        """).fetchone()[0]
        self.link_summary = {
            "event_rows_with_message_id": total_refs,
            "distinct_event_message_ids": distinct_refs,
            "event_rows_linked_to_chat": matched_refs,
            "distinct_message_ids_linked": len(matched_message_ids),
            "unmatched_event_rows": total_refs - matched_refs,
            "chat_messages_linked": total_chat_matched_ids,
            "message_ids_with_multiple_event_rows": multi_event_ids,
            "maximum_event_rows_per_message_id": max(rows_per_message.values(), default=0),
            "linked_rows_same_exact_timestamp": exact_time,
            "linked_rows_within_one_second": within_one_second,
            "timestamp_delta_ms_median": statistics.median(delta_ms) if delta_ms else None,
            "timestamp_delta_ms_p05": quantile(delta_ms, 0.05),
            "timestamp_delta_ms_p95": quantile(delta_ms, 0.95),
            "linked_rows_same_registered_sender": same_agent,
            "linked_rows_same_source_sender_id": same_sender,
            "linked_rows_same_room": same_room,
            "linked_rows_identical_content_sha256": same_content,
            "linked_event_action_types": dict(event_extra_action.most_common()),
            "linked_event_output_component_types": dict(linked_output_types.most_common()),
            "examples_without_content": examples,
        }

    def analyze_turn_order(self) -> dict[str, Any]:
        counts = sorted(self.turn_session_counts.values())
        gap_count = 0
        gap_sum = 0.0
        gaps: list[float] = []
        previous_session = None
        previous_time = None
        for session_id, created_at in self.db.execute("SELECT session_id,created_at FROM turn WHERE session_id IS NOT NULL ORDER BY session_id,created_at,row_id"):
            current = timestamp(created_at)
            if current is None:
                continue
            if session_id == previous_session and previous_time is not None:
                gap = (current - previous_time).total_seconds()
                if gap >= 0:
                    gaps.append(gap)
                    gap_sum += gap
                    gap_count += 1
            previous_session, previous_time = session_id, current
        self.turn_interarrival = gaps
        values = list(self.turn_session_counts.values())
        turn_types = {"output": dict(self.turn_states["output"]), "error": dict(self.turn_states["error"]), "system": dict(self.turn_states["system"]), "agent_messages": dict(self.turn_states["agent_messages"]), "agent_action": dict(self.turn_states["agent_action"])}
        return {
            "turn_count": self.turn_count,
            "action_type_counts": dict(self.turn_actions.most_common()),
            "action_type_percentages": {key: 100 * value / self.turn_count for key, value in self.turn_actions.items()} if self.turn_count else {},
            "missing_null_action_rows": self.turn_actions.get("<missing-action>", 0) + self.turn_actions.get("<null-action>", 0),
            "turn_field_availability": turn_types,
            "agent_action_field_rows": dict(self.action_field_rows.most_common()),
            "agent_action_field_types": {key: dict(value) for key, value in self.action_field_types.items()},
            "turns_with_non_action_arguments": self.action_arg_rows,
            "target_argument_rows_by_key": dict(self.target_field_rows.most_common()),
            "agent_message_tool_call_names": dict(self.tool_call_names.most_common(30)),
            "timestamp_invalid_rows": self.turn_timestamp_invalid,
            "sessions_with_turns": len(values),
            "turns_per_session": {"minimum": min(values, default=0), "median": statistics.median(values) if values else 0, "p90": quantile(values, 0.90), "p99": quantile(values, 0.99), "maximum": max(values, default=0)},
            "interarrival_seconds_within_sessions": {"count": gap_count, "median": statistics.median(gaps) if gaps else None, "p90": quantile(gaps, 0.90), "p95": quantile(gaps, 0.95), "maximum": max(gaps, default=None), "mean": gap_sum / gap_count if gap_count else None, "under_1s": sum(gap < 1 for gap in gaps), "over_1m": sum(gap > 60 for gap in gaps), "over_10m": sum(gap > 600 for gap in gaps)},
        }

    def sample_session_ids(self, counts: dict[str, int], bucket: str, target: int = 10) -> list[str]:
        values = sorted(counts.items(), key=lambda item: (item[1], item[0]))
        median_count = statistics.median(counts.values()) if counts else 0
        p90_count = quantile(list(counts.values()), 0.90) or 0
        if bucket == "short":
            candidates = [(sid, count) for sid, count in values if count <= 5]
        elif bucket == "median":
            candidates = sorted(values, key=lambda item: (abs(item[1] - median_count), item[0]))
        else:
            candidates = [(sid, count) for sid, count in values if count >= p90_count]
            candidates.reverse()
        picked: list[str] = []
        seen_agent_month: set[tuple[str | None, str | None]] = set()
        deferred = []
        for session_id, _ in candidates:
            agent_id, created = self.session_info.get(session_id, (None, None))
            dt = timestamp(created)
            diversity_key = (agent_id, month_key(dt) if dt else None)
            if diversity_key not in seen_agent_month or len(picked) + len(deferred) >= max(0, target - 3):
                picked.append(session_id)
                seen_agent_month.add(diversity_key)
            else:
                deferred.append(session_id)
            if len(picked) >= target:
                break
        for session_id in deferred:
            if len(picked) >= target:
                break
            if session_id not in picked:
                picked.append(session_id)
        return picked[:target]

    def build_session_examples(self) -> None:
        counts = dict(self.turn_session_counts)
        selected: list[tuple[str, str]] = []
        for bucket in ("short", "median", "long"):
            selected.extend((bucket, session_id) for session_id in self.sample_session_ids(counts, bucket))
        for bucket, session_id in selected:
            sequence = list(self.db.execute("SELECT created_at,action,action_keys,result_hint FROM turn WHERE session_id=? ORDER BY created_at,row_id", (session_id,)))
            turn_count = len(sequence)
            if turn_count <= 8:
                view = sequence
            else:
                view = sequence[:4] + [(None, "...", "", f"{turn_count - 8} turns omitted")]+ sequence[-4:]
            agent_id, session_start = self.session_info.get(session_id, (None, None))
            self.session_examples.append({
                "stratum": bucket, "session_id": short_id(session_id), "agent_id": short_id(agent_id),
                "agent_name": self.agent_names.get(agent_id, "unmatched") if agent_id else "unmatched",
                "session_start": session_start, "turn_count": turn_count,
                "sequence": [" — ".join(part for part in (action, f"keys={keys}" if keys else "", hint) if part) for _, action, keys, hint in view],
            })

    def temporal_summary(self) -> dict[str, Any]:
        result = {}
        for name, window in self.windows.items():
            hourly = window["hour_sources"]
            concurrency = [len(value) for value in window["hour_agents"].values()]
            overlap_counts = {}
            source_pairs = (("events", "chat"), ("events", "computer_use"), ("computer_use", "chat"), ("events", "computer_use", "chat"))
            for pair in source_pairs:
                overlap_counts[" + ".join(pair)] = sum(set(pair).issubset(sources) for sources in hourly.values())
            duration = window["duration_hours"]
            per_agent_total = {
                agent_id: sum(window["per_agent_counts"].get(agent_id, {}).values())
                for agent_id in self.agent_names
            }
            active_yields = [count for count in per_agent_total.values() if count]
            ranked_yields = sorted(per_agent_total.items(), key=lambda item: (-item[1], item[0]))
            validated_overlap = window["hour_validated_sources"]
            result[name] = {
                "start_inclusive": window["start"].isoformat() + "Z",
                "end_exclusive": window["end"].isoformat() + "Z",
                "duration_hours": duration,
                "records": {source: window["counts"].get(source, 0) for source in ("events", "computer_use", "chat")},
                "records_per_hour": {source: window["counts"].get(source, 0) / duration for source in ("events", "computer_use", "chat")},
                "registered_identity_records": {source: window["validated_counts"].get(source, 0) for source in ("events", "computer_use", "chat")},
                "registered_identity_records_per_hour": {source: window["validated_counts"].get(source, 0) / duration for source in ("events", "computer_use", "chat")},
                "validated_active_agents": len(window["agents"]),
                "active_sessions": len(window["sessions"]),
                "active_rooms": len(window["rooms"]),
                "same_hour_agent_concurrency": {"hours_with_two_or_more": sum(count >= 2 for count in concurrency), "maximum": max(concurrency, default=0), "mean_active_hour": statistics.mean(concurrency) if concurrency else 0},
                "hours_by_source_overlap": overlap_counts,
                "hours_by_validated_agent_source_overlap": {
                    "events + chat": sum({"events", "chat"}.issubset(sources) for sources in validated_overlap.values()),
                    "events + computer_use": sum({"events", "computer_use"}.issubset(sources) for sources in validated_overlap.values()),
                    "computer_use + chat": sum({"computer_use", "chat"}.issubset(sources) for sources in validated_overlap.values()),
                    "all three": sum({"events", "computer_use", "chat"}.issubset(sources) for sources in validated_overlap.values()),
                },
                "agents_by_source": {source: len(window["per_source_agents"].get(source, set())) for source in ("events", "computer_use", "chat")},
                "agent_yield_distribution_all_registered": {
                    "registered_agents": len(per_agent_total),
                    "active_agents": len(active_yields),
                    "zero_records": sum(count == 0 for count in per_agent_total.values()),
                    "fewer_than_10_records": sum(0 < count < 10 for count in per_agent_total.values()),
                    "median_including_zero": statistics.median(per_agent_total.values()) if per_agent_total else 0,
                    "median_active_only": statistics.median(active_yields) if active_yields else 0,
                    "q1_active_only": quantile(active_yields, 0.25),
                    "q3_active_only": quantile(active_yields, 0.75),
                    "maximum": max(per_agent_total.values(), default=0),
                },
                "top_active_agent_yields": [
                    {
                        "agent_id": agent_id,
                        "name": self.agent_names[agent_id],
                        "records": count,
                        "by_source": dict(window["per_agent_counts"].get(agent_id, {})),
                        "sessions": len(window["per_agent_sessions"].get(agent_id, set())),
                        "active_hours": len(window["per_agent_active_hours"].get(agent_id, set())),
                    }
                    for agent_id, count in ranked_yields[:12] if count
                ],
            }
        monthly = []
        for month in sorted(self.monthly):
            monthly.append({"month": month, "events": self.monthly[month].get("events", 0), "computer_use_turns": self.monthly[month].get("computer_use", 0), "chat_messages": self.monthly[month].get("chat", 0), "validated_agents": len(self.monthly_agents[month]), "active_days": sum(self.daily[day].get("events", 0) + self.daily[day].get("computer_use", 0) + self.daily[day].get("chat", 0) > 0 for day in self.daily if day.startswith(month))})
        result["monthly_timeline"] = monthly
        result["all_timeline"] = {
            "first_month": monthly[0]["month"] if monthly else None,
            "last_month": monthly[-1]["month"] if monthly else None,
            "active_days": len(self.daily),
            "days_with_multiple_validated_agents": sum(len(values) >= 2 for values in self.daily_agents.values()),
            "maximum_daily_validated_agents": max((len(values) for values in self.daily_agents.values()), default=0),
        }
        return result

    def inspect_sample_exports(self) -> list[dict[str, Any]]:
        for sample_name in SAMPLE_NAMES:
            directory = self.samples_dir / sample_name
            record_counts = {}
            paths = {name: directory / f"{name}.jsonl" for name in ("agents", "events", "computer_use_sessions", "computer_use_turns", "chat_messages", "chat_rooms")}
            missing = [name for name, path in paths.items() if not path.is_file()]
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
            sample_agents = {str(row["id"]) for row in rows(paths["agents"])} if paths["agents"].is_file() else set()
            event_speakers: set[str] = set()
            event_refs: Counter[str] = Counter()
            event_link_fields = {}
            for row in rows(paths["events"]) if paths["events"].is_file() else []:
                record_counts["events"] = record_counts.get("events", 0) + 1
                data = row.get("data") if isinstance(row.get("data"), dict) else {}
                if data.get("speakerId"):
                    event_speakers.add(str(data["speakerId"]))
                if data.get("messageId"):
                    event_refs[str(data["messageId"])] += 1
            chat_ids: set[str] = set()
            chat_senders: dict[str, set[str]] = defaultdict(set)
            chat_rooms: dict[str, set[str]] = defaultdict(set)
            sample_chat_rows = 0
            for row in rows(paths["chat_messages"]) if paths["chat_messages"].is_file() else []:
                sample_chat_rows += 1
                if row.get("id"):
                    chat_ids.add(str(row["id"]))
                sender = row.get("agent_speaker_id")
                if row.get("speaker_type") == "agent" and sender:
                    chat_senders[str(row.get("room_id"))].add(str(sender))
                if row.get("room_id"):
                    chat_rooms[str(row["room_id"])].add(str(sender)) if sender else None
            sample_sessions: dict[str, str | None] = {}
            if paths["computer_use_sessions"].is_file():
                for row in rows(paths["computer_use_sessions"]):
                    if row.get("id"):
                        sample_sessions[str(row["id"])] = str(row["agent_id"]) if row.get("agent_id") else None
            turn_count = 0
            matched_turns = 0
            interpretable_actions = Counter()
            if paths["computer_use_turns"].is_file():
                for row in rows(paths["computer_use_turns"]):
                    turn_count += 1
                    session = str(row.get("session_id")) if row.get("session_id") else None
                    matched_turns += bool(session and sample_sessions.get(session) in self.agent_names)
                    action = row.get("agent_action")
                    interpretable_actions[str(action.get("action") or "<missing>") if isinstance(action, dict) else "<null>"] += 1
            event_registered = event_speakers & set(self.agent_names)
            event_unregistered = event_speakers - set(self.agent_names)
            room_multi_sender = {room: len(senders) for room, senders in chat_senders.items() if len(senders) >= 2}
            linked_refs = sum(count for message_id, count in event_refs.items() if message_id in chat_ids)
            duplicate_ids = sum(count > 1 for count in event_refs.values())
            summary = {
                "name": sample_name,
                "exists": directory.is_dir(),
                "files_missing": missing,
                "manifest_available": manifest is not None,
                "manifest_declared_agent_count": len((manifest or {}).get("selection", {}).get("agent_ids", [])),
                "manifest_declared_agents_all_registered": all(agent_id in self.agent_names for agent_id in (manifest or {}).get("selection", {}).get("agent_ids", [])),
                "agents_file_count": len(sample_agents),
                "agents_file_all_registered": sample_agents.issubset(self.agent_names),
                "event_record_count": record_counts.get("events", 0),
                "distinct_event_speakers": len(event_speakers),
                "registered_event_speakers": len(event_registered),
                "unregistered_event_speakers": len(event_unregistered),
                "chat_message_count": sample_chat_rows,
                "chat_distinct_senders": len(set().union(*chat_senders.values())) if chat_senders else 0,
                "event_message_id_refs": sum(event_refs.values()),
                "event_message_id_refs_matching_sample_chat": linked_refs,
                "event_message_ids_referenced_multiple_times": duplicate_ids,
                "computer_use_turn_count": turn_count,
                "turns_joined_to_registered_agent": matched_turns,
                "turn_join_coverage": matched_turns / turn_count if turn_count else None,
                "computer_use_action_types": dict(interpretable_actions.most_common(12)),
                "shared_chat_rooms_with_multiple_registered_senders": room_multi_sender,
                "sample_agent_ids": sorted(sample_agents),
            }
            self.sample_summaries.append(summary)
        return self.sample_summaries

    def build_summary(self) -> dict[str, Any]:
        speaker_matches_registered = self.event_speakers & set(self.agent_names)
        speaker_matches_chat_agents = self.event_speakers & self.chat_agent_ids
        speaker_matches_chat_users = self.event_speakers & self.chat_user_ids
        unknown = self.event_speakers - set(self.agent_names) - self.chat_user_ids - self.chat_agent_ids
        identity = {
            "registered_agent_records": len(self.agent_names),
            "distinct_event_speaker_ids": len(self.event_speakers),
            "distinct_event_speaker_ids_matching_agents_table": len(speaker_matches_registered),
            "distinct_event_speaker_ids_matching_chat_agent_senders": len(speaker_matches_chat_agents),
            "distinct_event_speaker_ids_matching_chat_user_speakers": len(speaker_matches_chat_users),
            "distinct_event_speaker_ids_unresolved_after_chat_comparison": len(unknown),
            "distinct_event_speaker_ids_matching_both_agent_and_user_tables": len(speaker_matches_registered & self.chat_user_ids),
            "event_rows_with_speaker_id": self.event_speaker_rows,
            "event_rows_missing_speaker_id": self.event_speaker_missing,
            "event_rows_null_speaker_id": self.event_speaker_null,
            "event_rows_by_speaker_type": dict(self.event_rows_by_source_type),
            "speaker_type_distinct_ids": {key: len(value) for key, value in self.speaker_type_ids.items()},
            "registered_agent_id_rows_by_speaker_type": dict(self.event_ids_registered),
            "event_sender_ids_in_chat_agent_speakers": len(speaker_matches_chat_agents),
            "event_sender_ids_in_chat_user_speakers": len(speaker_matches_chat_users),
            "unresolved_event_speaker_ids_examples": sorted(unknown)[:13],
        }
        turn_summary = self.analyze_turn_order()
        self.build_session_examples()
        result = {
            "dataset_scope": str(self.dataset_dir),
            "identity": identity,
            "event_chat_linkage": self.link_summary,
            "computer_use": turn_summary,
            "temporal": self.temporal_summary(),
            "samples": self.inspect_sample_exports(),
            "sampled_sessions": self.session_examples,
            "source_record_counts": {"events": self.event_total, "computer_use_turns": self.turn_count, "chat_messages": self.chat_total},
            "event_action_types": dict(self.event_action_types.most_common(30)),
            "event_output_component_types": dict(self.output_component_types.most_common(30)),
        }
        return result

    def write_figures(self, summary: dict[str, Any]) -> list[str]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import numpy as np

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        output = []
        months = summary["temporal"]["monthly_timeline"]
        x = [datetime.strptime(row["month"], "%Y-%m") for row in months]
        series = (
            ("events", "Event rows", "#376f9f"),
            ("computer_use_turns", "Computer-use turns", "#c56d3a"),
            ("chat_messages", "Chat messages", "#287a68"),
        )
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for ax, (key, label, color) in zip(axes, series):
            per_day = [row[key] / max(1, __import__("calendar").monthrange(int(row["month"][:4]), int(row["month"][5:]))[1]) for row in months]
            ax.plot(x, per_day, marker="o", markersize=3, linewidth=1.5, color=color)
            ax.set_ylabel("Rows / day")
            ax.set_title(label, loc="left", fontsize=10)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Monthly activity by source (independent y-scales)", x=0.08, ha="left", fontsize=12)
        axes[-1].set_xlabel("Month")
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[-1].tick_params(axis="x", rotation=35)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(figures_dir / "monthly_activity_regimes.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        output.append("figures/monthly_activity_regimes.png")

        windows = list(WINDOWS)
        source_series = (
            ("events", "Events", "#376f9f"),
            ("computer_use", "Computer-use turns", "#c56d3a"),
            ("chat", "Chat messages", "#287a68"),
        )
        x_positions = np.arange(len(windows))
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for ax, (source, label, color) in zip(axes, source_series):
            values = [summary["temporal"][name]["records_per_hour"][source] for name in windows]
            bars = ax.bar(x_positions, values, width=0.55, color=color)
            ax.set_ylim(0, max(values, default=0) * 1.22 or 1)
            ax.set_ylabel("Rows / hour")
            ax.set_title(label, loc="left", fontsize=10)
            ax.grid(axis="y", alpha=0.25)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
        fig.suptitle("Weekly activity rates by source (independent y-scales)", x=0.08, ha="left", fontsize=12)
        axes[-1].set_xticks(x_positions, windows)
        axes[-1].tick_params(axis="x", rotation=0)
        axes[-1].set_xlabel("Seven-day window")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(figures_dir / "recommended_window_rates.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        output.append("figures/recommended_window_rates.png")

        actions = self.turn_actions.most_common(14)
        other = self.turn_count - sum(value for _, value in actions)
        if other:
            actions.append(("Other", other))
        fig, ax = plt.subplots(figsize=(10, max(4, len(actions) * 0.32)))
        ax.barh([name for name, _ in actions][::-1], [value for _, value in actions][::-1], color="#c56d3a")
        ax.set(title="Computer-use action types", xlabel="Turn rows")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "computer_use_action_distribution.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        output.append("figures/computer_use_action_distribution.png")
        return output

    def write_report(self, summary: dict[str, Any], figures: list[str]) -> Path:
        identity = summary["identity"]
        links = summary["event_chat_linkage"]
        turns = summary["computer_use"]
        temporal = summary["temporal"]
        report = self.output_dir / "forensic_validation.md"
        identity_rows = []
        for kind, count in sorted(identity["event_rows_by_speaker_type"].items(), key=lambda item: -item[1]):
            identity_rows.append(f"| {kind} | {count:,} | {identity['speaker_type_distinct_ids'].get(kind, 0):,} | {identity['registered_agent_id_rows_by_speaker_type'].get(kind, 0):,} |")
        win_rows = []
        for name in WINDOWS:
            row = temporal[name]
            rates = row["records_per_hour"]
            validated_rates = row["registered_identity_records_per_hour"]
            yield_stats = row["agent_yield_distribution_all_registered"]
            validated_overlap = row["hours_by_validated_agent_source_overlap"]
            win_rows.append(
                f"| {name} | {row['validated_active_agents']}/58 | {row['active_sessions']} | "
                f"{row['records']['events']:,} ({validated_rates['events']:.1f} validated/hr) | "
                f"{row['records']['computer_use']:,} ({rates['computer_use']:.1f}/hr) | "
                f"{row['records']['chat']:,} ({rates['chat']:.1f}/hr) | {yield_stats['zero_records']} | "
                f"{yield_stats['median_active_only']:.1f} | "
                f"{row['same_hour_agent_concurrency']['hours_with_two_or_more']}/{int(row['duration_hours'])} "
                f"(peak {row['same_hour_agent_concurrency']['maximum']}) | {validated_overlap['all three']} |"
            )
        agent_yield_rows = []
        for window_name, window in temporal.items():
            if not isinstance(window, dict) or "top_active_agent_yields" not in window:
                continue
            for item in window["top_active_agent_yields"]:
                counts = item["by_source"]
                agent_yield_rows.append(
                    f"| {window_name} | {item['name']} (`{item['agent_id']}`) | {counts.get('events', 0):,} | "
                    f"{counts.get('computer_use', 0):,} | {counts.get('chat', 0):,} | {item['sessions']:,} | {item['active_hours']:,} |"
                )
        action_rows = []
        for action, count in list(turns["action_type_counts"].items())[:18]:
            action_rows.append(f"| {action} | {count:,} | {turns['action_type_percentages'][action]:.2f}% |")
        state_rows = []
        for field in ("output", "error", "system", "agent_messages", "agent_action"):
            states = turns["turn_field_availability"][field]
            state_rows.append(f"| {field} | {states.get('non-empty', 0):,} | {states.get('empty', 0):,} | {states.get('null', 0):,} | {states.get('missing', 0):,} |")
        sample_rows = []
        for sample in summary["samples"]:
            sample_rows.append(f"| {sample['name']} | {sample['manifest_available']} | {sample['files_missing'] or 'none'} | {sample['agents_file_count']} | {sample['registered_event_speakers']}/{sample['distinct_event_speakers']} | {sample['event_message_id_refs_matching_sample_chat']}/{sample['event_message_id_refs']} | {sample['turns_joined_to_registered_agent']}/{sample['computer_use_turn_count']} | {sum(sample['shared_chat_rooms_with_multiple_registered_senders'].values())} |")
        session_rows = []
        for example in summary["sampled_sessions"]:
            seq = "<br>".join(example["sequence"])
            session_rows.append(f"| {example['stratum']} | {example['session_id']} | {example['agent_name']} ({example['agent_id']}) | {example['session_start'] or 'unknown'} | {example['turn_count']} | {seq} |")
        monthly_rows = []
        for row in temporal["monthly_timeline"]:
            if row["events"] or row["computer_use_turns"] or row["chat_messages"]:
                monthly_rows.append(f"| {row['month']} | {row['events']:,} | {row['computer_use_turns']:,} | {row['chat_messages']:,} | {row['validated_agents']} | {row['active_days']} |")
        figure_md = "\n".join(f"![{Path(path).stem.replace('_', ' ').title()}]({path})" for path in figures)
        text = f"""# TIDE AI Village Forensic Validation

## Scope and evidence

This targeted check uses only `agents`, `events`, `computer_use_sessions`, `computer_use_turns`, and `chat_messages` in `{self.dataset_dir.as_posix()}` plus the surviving existing exports under `{self.samples_dir.as_posix()}`. It streams JSONL; SQLite indexes only compact message-link and turn-order fields. It does not rerun or replace the broad audit and does not modify the TIDE pipeline or regenerate sample exports. Text is hashed for equality checks and omitted from examples.

**Known** means directly present in records or exact ID joins. **Inferred** means a conservative interpretation of those fields. **Unresolved** means the available records cannot establish the claim.

## 1. Identity: resolving 58 versus 673

`agents.jsonl` has {identity['registered_agent_records']} unique registered IDs. The 673 reported previously were distinct values of `events.data.speakerId`, not validated agents. Of these, {identity['distinct_event_speaker_ids_matching_agents_table']} match `agents.id`, {identity['distinct_event_speaker_ids_matching_chat_user_speakers']} match `chat_messages.user_speaker_id`, and {identity['distinct_event_speaker_ids_unresolved_after_chat_comparison']} match neither. {identity['distinct_event_speaker_ids_matching_chat_agent_senders']} also match `chat_messages.agent_speaker_id`. Thus the prior “673 agents” figure incorrectly treated user and unresolved identifiers as agents. Event `speakerType` is useful but imperfect: registered IDs also occur with missing/null `speakerType`; the authoritative criterion here is exact match to `agents.id`.

| Event speakerType | Event rows | Distinct speakerId values | Rows whose speakerId matches agents.id |
|---|---:|---:|---:|
{chr(10).join(identity_rows)}

Event rows with speakerId: {identity['event_rows_with_speaker_id']:,}; missing key: {identity['event_rows_missing_speaker_id']:,}; explicit null: {identity['event_rows_null_speaker_id']:,}. Event identities overlap {identity['event_sender_ids_in_chat_agent_speakers']} chat `agent_speaker_id` IDs and {identity['event_sender_ids_in_chat_user_speakers']} chat `user_speaker_id` IDs. The remaining unresolved IDs are not classified as agents or users.

## 2. Events linked to chat messages

Exact `events.data.messageId -> chat_messages.id` matching links {links['event_rows_linked_to_chat']:,} event rows ({links['distinct_message_ids_linked']:,} message IDs) to chat rows. {links['message_ids_with_multiple_event_rows']:,} message IDs have multiple event rows; maximum rows per ID is {links['maximum_event_rows_per_message_id']}. Timestamp strings are not identical for {links['linked_rows_same_exact_timestamp']:,}/{links['event_rows_linked_to_chat']:,} linked rows; signed event-minus-chat delta has median {links['timestamp_delta_ms_median']:.2f} ms and 5th/95th percentiles {links['timestamp_delta_ms_p05']:.2f}/{links['timestamp_delta_ms_p95']:.2f} ms. Sender IDs match on {links['linked_rows_same_source_sender_id']:,} linked rows; registered agent sender matches on {links['linked_rows_same_registered_sender']:,}; room matches on {links['linked_rows_same_room']:,}; SHA-256 content hashes match on {links['linked_rows_identical_content_sha256']:,}.

| Example | eventId | messageId | Event minus chat (ms) | Same agent sender | Same room | Identical content | Event action / outputs |
|---|---|---|---:|---|---|---|---|
{chr(10).join(f"| {i + 1} | {row['event_id']} | {row['message_id']} | {row['time_delta_ms']} | {row['same_registered_sender']} | {row['same_room']} | {row['content_hash_equal']} ({row['event_chars']} / {row['chat_chars']} chars) | {row['event_action_type']} / {', '.join(row['event_output_types']) or 'none'} |" for i, row in enumerate(links['examples_without_content']))}

**Conclusion (inference):** these are best modeled as distinct event records explicitly linked to chat-message records, with substantial duplicate content in the `AGENT_TALK` case, not blindly collapsed. Event rows carry event IDs/index, action type, cost/token metadata, room/speaker fields, and structured output components not present in the chat row. Retain the link and source provenance; for content-level analyses, deduplicate identical text only when exact message linkage and hash equality are established. Preserve multiplicity where several event rows reference one message.

## 3. What a computer-use turn represents

The {turns['turn_count']:,} turns contain {turns['sessions_with_turns']:,} sessions with turns. Action values are from `agent_action.action`; `agent_messages` tool-call names are not substituted for the action label.

| Action type | Turns | Percent |
|---|---:|---:|
{chr(10).join(action_rows)}

Null/missing action rows: {turns['missing_null_action_rows']:,}. Non-action argument fields appear on {turns['turns_with_non_action_arguments']:,} turns. Argument keys include {', '.join(f"`{key}` ({count:,})" for key, count in list(turns['agent_action_field_rows'].items())[:14])}. Non-empty target-like argument coverage by key: {', '.join(f"{key}={count:,}" for key, count in turns['target_argument_rows_by_key'].items()) or 'none'}. The table below counts field states; “non-empty” means present with a non-null/non-empty value, not that it is a useful result.

| Field | Non-empty | Empty | Null | Missing |
|---|---:|---:|---:|---:|
{chr(10).join(state_rows)}

Turns per session: min {turns['turns_per_session']['minimum']}, median {turns['turns_per_session']['median']}, p90 {turns['turns_per_session']['p90']}, p99 {turns['turns_per_session']['p99']}, max {turns['turns_per_session']['maximum']}. Within-session inter-arrival seconds: median {turns['interarrival_seconds_within_sessions']['median']}, p90 {turns['interarrival_seconds_within_sessions']['p90']}, p95 {turns['interarrival_seconds_within_sessions']['p95']}; {turns['interarrival_seconds_within_sessions']['under_1s']:,} gaps are <1s and {turns['interarrival_seconds_within_sessions']['over_1m']:,} are >1min. Tool-call names in `agent_messages.content`: {', '.join(f"{name} ({count:,})" for name, count in list(turns['agent_message_tool_call_names'].items())[:12]) or 'none observed'}.

The deterministic session sample below takes 10 short (<=5-turn where available), 10 near-median, and 10 long (>=p90 where available) sessions, diversifying agent and start month. For long sessions only the first and last four turns are shown. Result hints report field availability/tool-call names, not hidden output text.

| Stratum | Session | Validated agent | Session start | Turns | Chronological action/result synopsis |
|---|---|---|---|---:|---|
{chr(10).join(session_rows)}

**Recommendation (inference):** retain every source turn as a low-level observation, but do not equate every mouse movement/keystroke with a separate TIDE decision. Use a short, consecutive action sequence ending at an observable tool/result boundary as the primary candidate event; retain constituent turns and session ID as provenance. Do not select only turns with non-null `output`: in this dataset output/error/system availability is sparse, and missing output is not evidence that no action occurred. Validate any grouping rule against the sampled sequences before inference.

## 4. One-week, all-agent baseline and temporal regimes

The first candidate is a full seven-day window with all registered identities eligible and no top-N agent filter. The October week contains the previously observed high-activity day (30 October). The April week is retained only as a comparison because the earlier audit suggested a separate activity regime. Monthly totals give broader context without introducing more sample slices.

| Candidate week | Active IDs | Active sessions | Event rows (validated-event rate) | Turns (rate) | Chat rows (rate) | Registered IDs with zero rows | Median rows per active ID | Hours with >=2 IDs / 168 (peak) | Hours with all 3 validated sources |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(win_rows)}

Rates in parentheses are per-hour averages across seven days; event totals include rows without a validated registered identity, while the parenthetical event rate counts only rows whose `speakerId` matches `agents.id`. “Zero rows” is out of 58 registered identities; per-source participation and validated-agent source overlap are calculated separately. Hour co-presence is not evidence of communication.

The all-registered yield distribution includes zeroes; active-only median/Q1/Q3 and the top-yield table show whether a broad week is too sparse or dominated by a few participants. These figures are the decision point for extending the continuous window or narrowing to agents with actual continuity/communication links, not total-row rank alone.

| Candidate week | Registered identity | Events | Computer-use turns | Chat | Sessions | Active hours |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(agent_yield_rows)}

**Initial sampling rule:** inspect the October all-registered week first if validated event/chat overlap and per-identity yields are workable. If most active identities have only a handful of interpretable sequences, extend the same continuous period. If the week is too dense, narrow participants using continuity and explicit communication/resource evidence, not total-row rank.

| Month | Events | Turns | Chat | Validated IDs | Active days |
|---|---:|---:|---:|---:|---:|
{chr(10).join(monthly_rows)}

{figure_md}

**Interpretation:** compare per-hour rates and validated-source overlap, not just totals. The candidate weeks are contrasting activity regimes, not proof of an instrumentation change. The records alone cannot identify the cause; an absent source may mean no activity or different logging coverage. Do not treat the largest week as automatically the best sample.

## 5. Existing sample exports

These are inspected read-only from `{self.samples_dir.as_posix()}`. Missing files and missing manifests are reported rather than reconstructed. “Event/chat links” count event references whose message ID exists in that sample's chat file; they remain separate rows unless exact content equality is established.

| Sample | Manifest | Missing files | Agent metadata rows | Registered event speakers / distinct speakers | Linked event refs / refs | Turns joined / turns | Shared rooms with >=2 senders |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(sample_rows)}

The existing exports are not all complete. The one-week folder is marked incomplete when files or its manifest are absent; its current rows are insufficient to validate the declared candidate. For each available sample, check `agents_file_all_registered`, `manifest_declared_agents_all_registered`, the actual manifest IDs, and turn join coverage before reuse. Common chat rooms show communication opportunity only; these logs do not establish which room members read a message. The preserved sample rows are not regenerated by this investigation.

## 6. Recommendation and unresolved points

For a first manual episode, start with the October seven-day all-registered-identity slice if its validated activity and per-identity yields support it; otherwise extend that same continuous period before selecting a handful of IDs. Keep the April week separate as a regime comparison, not pooled into the first slice. Restrict participant identity to IDs matched against `agents.id`; retain user records separately. Keep linked event and chat rows as linked source observations, not independent evidence of two communications.

Use a grouped computer-use action sequence with its raw turns retained, and define an explicit result boundary before building TIDE events. Exact transfer of information between agents remains unresolved: matching chat messages prove a logged message and sender, not receipt/use; same-room or same-hour activity proves co-presence only; event output and computer-use result fields are not necessarily verified environment state. Similar text is not evidence of information transfer.

This validation does not establish that the dataset validates TIDE. It establishes identity coverage, exact source links, interaction granularity and temporal differences that should constrain any first experiment.
"""
        report.write_text(text, encoding="utf-8")
        return report

    def run(self) -> Path:
        self.validate()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_db()
        self.load_identities()
        self.scan_chat()
        self.scan_events()
        self.scan_turns()
        self.analyze_links()
        summary = self.build_summary()
        figures = self.write_figures(summary)
        report = self.write_report(summary, figures)
        print(f"[forensics] report: {report}", flush=True)
        return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Targeted AI Village identity/link/action forensic validation.")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    audit = ForensicAudit(args.dataset_dir, args.samples_dir, args.output_dir)
    try:
        audit.run()
    finally:
        audit.db.close()
        audit.temp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
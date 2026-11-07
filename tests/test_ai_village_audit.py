from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_ai_village import Audit


class AiVillageAuditTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        dataset = root / "main-village"
        dataset.mkdir()
        rows = {
            "computer_use_sessions": [
                {"id": "s1", "agent_id": "a1", "created_at": "2026-01-01 00:00:00"},
                {"id": "s2", "agent_id": "a2", "created_at": "2026-01-01 00:00:00"},
            ],
            "agents": [{"id": "a1", "name": "Agent 1"}, {"id": "a2", "name": "Agent 2"}],
            "chat_messages": [
                {"id": "m1", "agent_speaker_id": "a1", "speaker_type": "agent", "room_id": "r1", "content": "hello", "created_at": "2026-01-01 00:10:00"},
                {"id": "m2", "agent_speaker_id": "a2", "speaker_type": "agent", "room_id": "r1", "content": "world!", "created_at": "2026-01-01 00:20:00"},
                {"id": "m3", "agent_speaker_id": "unknown", "speaker_type": "agent", "room_id": "r1", "content": "unregistered", "created_at": "2026-01-01 00:30:00"},
            ],
            "events": [
                {"id": "e1", "data": {"speakerId": "a1", "roomId": "r1", "messageId": "m1", "actionType": "AGENT_TALK", "output": [{"type": "message"}]}, "created_at": "2026-01-01 00:15:00"},
                {"id": "e2", "data": {"speakerId": "external", "speakerType": "HUMAN", "roomId": "r1", "actionType": "tool", "output": []}, "created_at": "2026-01-01 00:25:00"},
            ],
            "computer_use_turns": [
                {"id": "t1", "session_id": "s1", "agent_action": {"action": "click", "text": "button"}, "created_at": "2026-01-01 00:30:00"},
                {"id": "t2", "session_id": "missing", "agent_action": {"action": "type"}, "created_at": "2026-01-01 00:35:00"},
            ],
            "chat_rooms": [{"id": "r1", "name": "shared"}],
            "agent_goals": [{"id": "g1", "agent_id": "a1", "start_time": "2025-12-31 00:00:00", "end_time": None}],
            "village_goals": [{"id": "vg1", "start_time": "2025-12-31 00:00:00", "end_time": "2026-01-02 00:00:00"}],
        }
        for source, records in rows.items():
            (dataset / f"{source}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        with (dataset / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("not json\n")
        return dataset

    def test_scans_inventory_joins_links_and_overlapping_agents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.make_dataset(root)
            audit = Audit(dataset, root / "report")
            audit.scan()
            summary, _ = audit.build_summary()
            self.assertEqual(summary["activity"]["events"], 2)
            self.assertEqual(summary["inventory"]["events"]["malformed_records"], 1)
            self.assertEqual(summary["activity"]["computer_use_turn_join_coverage"]["attributed_turns"], 1)
            self.assertEqual(summary["activity"]["links"]["distinct_event_message_ids_matching_chat_message_ids"], 1)
            self.assertEqual(summary["overlap"]["maximum_agents_in_same_hour"], 2)
            self.assertEqual(summary["agents"]["agents_with_observable_activity"], 2)
            self.assertEqual(summary["activity"]["registered_agent_identity_coverage"]["events"]["attributed_to_registered_agent"], 1)
            self.assertEqual(summary["activity"]["registered_agent_identity_coverage"]["chat_messages"]["attributed_to_registered_agent"], 2)

    def test_rejects_non_main_village_and_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = root / "other-village"
            wrong.mkdir()
            with self.assertRaises(ValueError):
                Audit(wrong, root / "out").validate()
            dataset = self.make_dataset(root)
            output = root / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                Audit(dataset, output).validate()

    def test_exports_original_rows_and_manifest_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.make_dataset(root)
            audit = Audit(dataset, root / "report")
            audit.scan()
            _, specs = audit.build_summary()
            audit.output_dir.mkdir()
            spec = specs[0]
            manifest = audit.export_sample(spec)
            self.assertIn("events", manifest["record_counts"])
            exported = [json.loads(line) for line in (audit.output_dir / "samples" / spec["name"] / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all("data" in row for row in exported))


if __name__ == "__main__":
    unittest.main()
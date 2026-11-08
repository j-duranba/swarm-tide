import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from tide.cli import build_parser
from tide.pass2 import EMBEDDING_REVISION, event_representation, generate_candidates, request_rerank, run_pass2, validate_dependencies


class Pass2Tests(unittest.TestCase):
    def test_representation_keeps_content_and_reasoning_distinct(self):
        content, reasoning, combined = event_representation({
            "content": "private message",
            "output": [{"type": "message", "content": [
                {"type": "thinking", "thinking": "considered a plan"},
                {"type": "text", "text": "published result"},
            ]}],
        })
        self.assertIn("private message", content)
        self.assertIn("published result", content)
        self.assertEqual(reasoning, "considered a plan")
        self.assertIn("PRIVATE CONTENT:", combined)
        self.assertIn("THINKING / REASONING TRACE:", combined)

    def test_candidates_are_temporal_deduplicated_and_keep_sources(self):
        base = datetime(2026, 10, 24, 12, tzinfo=timezone.utc)

        def event(event_id, hours, author, action="AGENT_TALK", annotation=None):
            timestamp = base - timedelta(hours=hours)
            return {
                "event_id": event_id, "author_id": author, "author_name": {"a": "A", "b": "B"}.get(author),
                "timestamp": timestamp.isoformat(), "timestamp_dt": timestamp, "action_type": action,
                "representation": "event text", "private_content": "event text", "reasoning_trace": "",
                "pass1_annotation": annotation,
            }

        target = event("target", 0, "a", annotation={"from_agent": ["B"], "to_agent": []})
        history = event("history", 1, "a")
        overlap = event("overlap", 2, "b", annotation={"to_agent": ["A"]})
        other = event("other", 3, "c")
        future = event("future", -1, "b")
        stale = event("stale", 13, "b")
        embeddings = {
            row["event_id"]: {"query_embedding": [1.0, 0.0], "document_embedding": [1.0, 0.0]}
            for row in (target, history, overlap, other, future, stale)
        }
        candidates = generate_candidates(target, [history, overlap, other, future, stale], embeddings, {"B": "b"})
        by_id = {row["source_event_id"]: row for row in candidates}
        self.assertEqual(set(by_id), {"history", "overlap", "other"})
        self.assertEqual(by_id["history"]["retrieval_sources"], ["same_agent_history"])
        self.assertEqual(by_id["overlap"]["retrieval_sources"], ["identified_information_source", "incoming_communication", "semantic"])
        self.assertEqual(by_id["other"]["retrieval_sources"], ["semantic"])
        self.assertEqual(by_id["overlap"]["similarity_score"], 1.0)

    def test_dependency_response_requires_all_candidate_ids_and_bounded_scores(self):
        parsed = validate_dependencies({"dependencies": [{
            "source_event_id": "source", "dependency_score": 0.75, "reason": "Used its result."
        }]}, {"source"})
        self.assertEqual(parsed["dependencies"]["source"]["dependency_score"], 0.75)
        with self.assertRaises(ValueError):
            validate_dependencies({"dependencies": []}, {"source"})
        with self.assertRaises(ValueError):
            validate_dependencies({"dependencies": [{
                "source_event_id": "source", "dependency_score": 1.1, "reason": "reason"
            }]}, {"source"})

    def test_rerank_request_contains_full_private_components(self):
        client = MagicMock()
        client.responses.create.return_value.status = "completed"
        client.responses.create.return_value.output_text = json.dumps({"dependencies": [{
            "source_event_id": "source", "dependency_score": 0.5, "reason": "The target uses its result."
        }]})
        target = {"event_id": "target", "timestamp": "t", "author_id": "a", "private_content": "target text", "reasoning_trace": "target thought"}
        candidate = {"source_event_id": "source", "timestamp": "s", "author_id": "b", "private_content": "source text", "reasoning_trace": "source thought"}
        result = request_rerank(client, "deepseek", target, [candidate])
        sent = client.responses.create.call_args.kwargs
        self.assertIn("target thought", sent["input"])
        self.assertIn("source thought", sent["input"])
        self.assertEqual(result["dependencies"]["source"]["dependency_score"], 0.5)

    def test_pass2_cli_options(self):
        args = build_parser().parse_args([
            "pass2", "--dataset-dir", "data", "--pass1-dir", "pass1", "--output-dir", "pass2", "--limit", "2",
        ])
        self.assertEqual(args.limit, 2)
        self.assertEqual(args.embedding_model, "jinaai/jina-embeddings-v5-text-nano")
        self.assertEqual(args.embedding_revision, EMBEDDING_REVISION)
        self.assertFalse(args.overwrite)

    def test_run_writes_separate_outputs_and_resumes_without_reinference(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            pass1_dir = root / "pass1"
            output = root / "pass2"
            dataset.mkdir()
            pass1_dir.mkdir()
            (dataset / "agents.jsonl").write_text(
                '{"id":"a","name":"A"}\n{"id":"b","name":"B"}\n', encoding="utf-8"
            )
            rows = []
            for event_id, timestamp, author_id, text in (
                ("source", "2026-10-24T11:00:00Z", "b", "source result"),
                ("target", "2026-10-24T12:00:00Z", "a", "target action"),
            ):
                rows.append({
                    "timestamp_utc": timestamp,
                    "identity": {"registered_agent_id": author_id},
                    "source_event": {"source_record": {
                        "id": event_id, "created_at": timestamp,
                        "data": {"actionType": "AGENT_TALK", "output": [{"type": "text", "text": text}]},
                    }},
                })
            source_bytes = ("".join(json.dumps(row) + "\n" for row in rows)).encode()
            (dataset / "events.jsonl").write_bytes(source_bytes)
            annotation_rows = [
                {"event_id": "source", "summary": "source", "to_agent": ["A"], "from_agent": []},
                {"event_id": "target", "summary": "target", "to_agent": [], "from_agent": ["B"]},
            ]
            annotation_bytes = ("".join(json.dumps(row) + "\n" for row in annotation_rows)).encode()
            (pass1_dir / "annotations.jsonl").write_bytes(annotation_bytes)
            fake_client = MagicMock()
            fake_client.base_url = "https://api.example.test"

            class FakeEmbeddingModel:
                def encode(self, texts, task, prompt_name):
                    self.last_task = (task, prompt_name)
                    return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

            with patch("tide.pass2.Adaption", return_value=fake_client), \
                    patch("tide.pass2._load_embedding_model", return_value=FakeEmbeddingModel()) as embedder, \
                    patch("tide.pass2.request_rerank", return_value={"dependencies": {
                        "source": {"dependency_score": 0.75, "reason": "Target uses the earlier result."}
                    }}) as reranker:
                result = run_pass2(dataset, pass1_dir, output, "key")
            self.assertEqual((result["targets"], result["completed_targets"], result["unresolved_targets"]), (2, 2, 0))
            self.assertEqual(embedder.call_count, 1)
            self.assertEqual(reranker.call_count, 1)
            dependencies = [
                json.loads(line)
                for line in (output / "dependencies.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            dependency = next(row for row in dependencies if row["event_id"] == "target")
            candidate = dependency["candidates"][0]
            self.assertEqual(candidate["source_event_id"], "source")
            self.assertEqual(candidate["dependency_score"], 0.75)
            self.assertIn("identified_information_source", candidate["retrieval_sources"])
            self.assertNotIn("private_content", candidate)
            self.assertEqual((dataset / "events.jsonl").read_bytes(), source_bytes)
            self.assertEqual((pass1_dir / "annotations.jsonl").read_bytes(), annotation_bytes)
            with patch("tide.pass2.Adaption", return_value=fake_client), \
                    patch("tide.pass2._load_embedding_model") as embedder, \
                    patch("tide.pass2.request_rerank") as reranker:
                resumed = run_pass2(dataset, pass1_dir, output, "key")
            embedder.assert_not_called()
            reranker.assert_not_called()
            self.assertEqual(resumed["completed_targets"], 2)


if __name__ == "__main__":
    unittest.main()
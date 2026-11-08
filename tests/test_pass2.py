import json
import threading
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from tide.cli import build_parser
from tide.pass2 import EMBEDDING_REVISION, _load_embedding_model, event_representation, generate_candidates, request_rerank, validate_dependencies
from tide.pass2_stages import (
    _rerank_target,
    filter_candidates_for_rerank,
    run_pass2_all,
    run_pass2_embed,
    run_pass2_rerank,
    run_pass2_retrieve,
)


class Pass2Tests(unittest.TestCase):
    def test_representation_keeps_content_and_reasoning_distinct(self):
        content, reasoning, combined = event_representation({
            "content": "private message",
            "output": [{"type": "message", "content": [
                {"type": "thinking", "thinking": "considered a plan"},
                {"type": "text", "text": "published result"},
            ]}],
        }, "Description: Help customers publish chapters.")
        self.assertIn("private message", content)
        self.assertIn("published result", content)
        self.assertEqual(reasoning, "considered a plan")
        self.assertIn("PRIVATE CONTENT:", combined)
        self.assertIn("THINKING / REASONING TRACE:", combined)
        self.assertIn("AGENT GOAL:\nDescription: Help customers publish chapters.", combined)

    def test_candidates_are_temporal_deduplicated_and_keep_sources(self):
        base = datetime(2026, 10, 24, 12, tzinfo=timezone.utc)

        def event(event_id, hours, author, action="AGENT_TALK", annotation=None):
            timestamp = base - timedelta(hours=hours)
            return {
                "event_id": event_id, "author_id": author, "author_name": {"a": "A", "b": "B"}.get(author),
                "timestamp": timestamp.isoformat(), "timestamp_dt": timestamp, "action_type": action,
                "representation": "event text", "private_content": "event text", "reasoning_trace": "",
                "agent_goal": "Example goal", "pass1_relations": annotation or {},
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
        self.assertEqual(by_id["overlap"]["retrieval_sources"], [
            "identified_information_source", "incoming_communication", "semantic",
        ])
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
        target = {"event_id": "target", "timestamp": "t", "author_id": "a", "agent_goal": "target goal", "private_content": "target text", "reasoning_trace": "target thought"}
        candidate = {"source_event_id": "source", "timestamp": "s", "author_id": "b", "agent_goal": "source goal", "private_content": "source text", "reasoning_trace": "source thought"}
        result = request_rerank(client, "deepseek", target, [candidate])
        sent = client.responses.create.call_args.kwargs
        self.assertIn("target thought", sent["input"])
        self.assertIn("source thought", sent["input"])
        self.assertIn("target goal", sent["input"])
        self.assertIn("source goal", sent["input"])
        self.assertIn("Semantic similarity, temporal proximity", sent["instructions"])
        private_payload = json.loads(sent["input"].split("\n", 1)[1])
        for item in [private_payload["target"], *private_payload["candidates"]]:
            self.assertNotIn("similarity_score", item)
            self.assertNotIn("retrieval_sources", item)
        self.assertEqual(result["dependencies"]["source"]["dependency_score"], 0.5)

    def test_pass2_cli_options(self):
        args = build_parser().parse_args([
            "pass2", "--dataset-dir", "data", "--pass1-dir", "pass1", "--output-dir", "pass2", "--limit", "2",
        ])
        self.assertEqual(args.limit, 2)
        self.assertEqual(args.embedding_model, "jinaai/jina-embeddings-v5-text-nano")
        self.assertEqual(args.embedding_revision, EMBEDDING_REVISION)
        self.assertEqual(args.precision, "auto")
        self.assertFalse(args.overwrite)

        embed_args = build_parser().parse_args([
            "pass2-embed", "--dataset-dir", "data", "--output-dir", "cache",
        ])
        retrieve_args = build_parser().parse_args([
            "pass2-retrieve", "--dataset-dir", "data", "--pass1-dir", "pass1", "--output-dir", "cache",
        ])
        rerank_args = build_parser().parse_args([
            "pass2-rerank", "--dataset-dir", "data", "--output-dir", "cache",
        ])
        filtered_rerank_args = build_parser().parse_args([
            "pass2-rerank", "--dataset-dir", "data", "--output-dir", "cache",
            "--candidate-filter", "proposed", "--limit", "2",
        ])
        top10_rerank_args = build_parser().parse_args([
            "pass2-public-rerank", "--dataset-dir", "data", "--pass1-dir", "pass1",
            "--output-dir", "public-cache", "--candidate-filter", "top10", "--limit", "2",
        ])
        public_embed_args = build_parser().parse_args([
            "pass2-public-embed", "--dataset-dir", "data", "--pass1-dir", "pass1", "--output-dir", "public-cache",
        ])
        public_retrieve_args = build_parser().parse_args([
            "pass2-public-retrieve", "--dataset-dir", "data", "--pass1-dir", "pass1", "--output-dir", "public-cache",
        ])
        public_rerank_args = build_parser().parse_args([
            "pass2-public-rerank", "--dataset-dir", "data", "--pass1-dir", "pass1", "--output-dir", "public-cache",
        ])
        self.assertEqual(embed_args.embedding_revision, EMBEDDING_REVISION)
        self.assertEqual(embed_args.precision, "auto")
        self.assertIsNone(retrieve_args.limit)
        self.assertEqual(rerank_args.timeout, 120)
        self.assertEqual(rerank_args.concurrency, 1)
        self.assertEqual(rerank_args.candidate_group_size, 4)
        self.assertEqual(rerank_args.candidate_filter, "none")
        self.assertEqual(filtered_rerank_args.candidate_filter, "proposed")
        self.assertEqual(filtered_rerank_args.limit, 2)
        self.assertEqual(top10_rerank_args.candidate_filter, "top10")
        self.assertEqual(top10_rerank_args.limit, 2)
        self.assertEqual(public_embed_args.output_dir, "public-cache")
        self.assertEqual(public_retrieve_args.pass1_dir, "pass1")
        self.assertEqual(public_rerank_args.candidate_group_size, 4)

    def test_rerank_target_groups_candidates_and_merges_scores(self):
        target = {"event_id": "target"}
        retrieval = {
            "event_id": "target", "source_index": 1, "timestamp": "timestamp",
            "author_id": "author", "agent_goal": "goal",
            "candidate_event_ids": [f"source-{index}" for index in range(10)],
        }
        candidates = [{
            "source_event_id": f"source-{index}",
            "author_id": "author",
            "author_name": "Agent",
            "timestamp": "timestamp",
            "agent_goal": "goal",
            "retrieval_sources": ["semantic"],
            "similarity_score": 0.9,
        } for index in range(10)]

        def score_group(client, model, passed_target, group, *, public_only=False):
            self.assertIs(passed_target, target)
            return {"dependencies": {
                candidate["source_event_id"]: {
                    "dependency_score": 0.5,
                    "reason": "Plausible contribution.",
                }
                for candidate in group
            }}

        with patch("tide.pass2_stages.request_rerank", side_effect=score_group) as request:
            result = _rerank_target(MagicMock(), "model", retrieval, target, candidates, 4)
        self.assertEqual([len(call.args[3]) for call in request.call_args_list], [4, 4, 2])
        self.assertEqual(result["candidate_event_ids"], retrieval["candidate_event_ids"])
        self.assertEqual([candidate["source_event_id"] for candidate in result["candidates"]], retrieval["candidate_event_ids"])

    def test_proposed_candidate_filter_only_filters_cached_candidates(self):
        target_time = datetime(2026, 10, 24, 12, tzinfo=timezone.utc)

        def candidate(event_id, minutes, author, sources, score=0.1):
            timestamp = target_time - timedelta(minutes=minutes)
            return {
                "source_event_id": event_id,
                "author_id": author,
                "timestamp": timestamp.isoformat(),
                "retrieval_sources": sources,
                "similarity_score": score,
            }

        cached_candidates = [
            candidate("same-1", 90, "target-agent", ["same_agent_history"]),
            candidate("same-2", 60, "target-agent", ["same_agent_history"]),
            candidate("same-3", 30, "target-agent", ["same_agent_history"]),
            candidate("source-b-old", 110, "source-b", ["identified_information_source"]),
            candidate("source-b-new", 50, "source-b", ["identified_information_source", "semantic"], 0.99),
            candidate("source-c", 40, "source-c", ["identified_information_source"]),
            candidate("incoming-after", 20, "incoming-agent", ["incoming_communication"]),
            candidate("incoming-before", 35, "incoming-agent", ["incoming_communication"]),
            candidate("semantic-1", 25, "semantic-agent", ["semantic"], 0.98),
            candidate("semantic-2", 24, "semantic-agent", ["semantic"], 0.97),
            candidate("semantic-3", 23, "semantic-agent", ["semantic"], 0.96),
            candidate("semantic-4", 22, "semantic-agent", ["semantic"], 0.95),
            candidate("semantic-5", 21, "semantic-agent", ["semantic"], 0.94),
            candidate("semantic-6", 20, "semantic-agent", ["semantic"], 0.93),
        ]
        retrieval = {
            "timestamp": target_time.isoformat(),
            "candidate_event_ids": [candidate_row["source_event_id"] for candidate_row in cached_candidates],
            "candidates": cached_candidates,
        }
        original_ids = list(retrieval["candidate_event_ids"])
        filtered = filter_candidates_for_rerank(retrieval, "proposed")
        filtered_ids = {candidate_row["source_event_id"] for candidate_row in filtered}
        self.assertEqual(filtered_ids, {
            "same-1", "same-2", "same-3", "source-b-new", "source-c", "incoming-after",
            "semantic-1", "semantic-2", "semantic-3", "semantic-4", "semantic-5",
        })
        self.assertEqual(retrieval["candidate_event_ids"], original_ids)
        self.assertEqual(len(retrieval["candidates"]), 14)
        self.assertEqual(len(filtered), 11)
        top10 = filter_candidates_for_rerank(retrieval, "top10")
        expected_top10 = sorted(
            retrieval["candidates"], key=lambda candidate_row: candidate_row["similarity_score"], reverse=True
        )[:10]
        self.assertEqual(top10, expected_top10)
        self.assertLessEqual(len(top10), 10)
        self.assertEqual(retrieval["candidate_event_ids"], original_ids)

        public_candidates = [
            {
                "source_event_id": f"same-{index}",
                "retrieval_sources": ["same_agent_history"],
                "similarity_score": 0.01,
            }
            for index in range(3)
        ] + [
            {
                "source_event_id": f"semantic-{index}",
                "retrieval_sources": ["semantic"],
                "similarity_score": 1.0 - index / 100,
            }
            for index in range(11)
        ]
        public_retrieval = {
            "candidate_event_ids": [candidate_row["source_event_id"] for candidate_row in public_candidates],
            "candidates": public_candidates,
        }
        public_top10 = filter_candidates_for_rerank(public_retrieval, "top10", context_mode="public")
        private_top10 = filter_candidates_for_rerank(public_retrieval, "top10", context_mode="private")
        public_ids = {candidate_row["source_event_id"] for candidate_row in public_top10}
        private_ids = {candidate_row["source_event_id"] for candidate_row in private_top10}
        self.assertEqual(len(public_top10), 10)
        self.assertTrue({"same-0", "same-1", "same-2"}.issubset(public_ids))
        self.assertEqual(sum(event_id.startswith("semantic-") for event_id in public_ids), 7)
        self.assertFalse(any(event_id.startswith("same-") for event_id in private_ids))

    def test_public_context_uses_only_pass1_summaries(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            pass1_dir = root / "pass1"
            output = root / "public-pass2"
            dataset.mkdir()
            pass1_dir.mkdir()
            (dataset / "agents.jsonl").write_text(
                '{"id":"a","name":"A"}\n{"id":"b","name":"B"}\n', encoding="utf-8"
            )
            events = []
            for event_id, timestamp, author_id, private_message in (
                ("source", "2026-10-24T11:00:00Z", "b", "PRIVATE_MESSAGE_SOURCE"),
                ("target", "2026-10-24T12:00:00Z", "a", "PRIVATE_MESSAGE_TARGET"),
            ):
                events.append({
                    "timestamp_utc": timestamp,
                    "identity": {"registered_agent_id": author_id},
                    "metadata": {"agent_goals_active_at_event": [{
                        "name": "PRIVATE_GOAL", "description": "PRIVATE_GOAL_DESCRIPTION",
                    }]},
                    "source_event": {"source_record": {
                        "id": event_id, "created_at": timestamp,
                        "data": {
                            "actionType": "AGENT_TALK",
                            "content": private_message,
                            "output": [
                                {"type": "thinking", "thinking": "PRIVATE_REASONING"},
                                {"type": "text", "text": private_message},
                            ],
                        },
                    }},
                })
            (dataset / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            summaries = {
                "source": "PUBLIC_SUMMARY_SOURCE",
                "target": "PUBLIC_SUMMARY_TARGET",
            }
            annotations = [
                {"event_id": "source", "summary": summaries["source"], "from_agent": [], "to_agent": ["A"]},
                {"event_id": "target", "summary": summaries["target"], "from_agent": ["B"], "to_agent": []},
            ]
            (pass1_dir / "annotations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in annotations), encoding="utf-8"
            )

            class SummaryEmbeddingModel:
                def __init__(self):
                    self.encoded_texts = []

                def encode(self, texts, task, prompt_name):
                    self.encoded_texts.extend(texts)
                    return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

            embedding_model = SummaryEmbeddingModel()
            with patch("tide.pass2_stages._load_embedding_model", return_value=embedding_model):
                embedded = run_pass2_embed(
                    dataset, output, context_mode="public", pass1_dir=pass1_dir
                )
            self.assertEqual(embedded["embedded"], 2)
            self.assertEqual(set(embedding_model.encoded_texts), set(summaries.values()))
            embedding_text = (output / "event_embeddings.jsonl").read_text(encoding="utf-8")
            embedding_config = json.loads((output / "pass2_embed_config.json").read_text(encoding="utf-8"))
            self.assertEqual(embedding_config["context_mode"], "public")
            for private_value in ("PRIVATE_MESSAGE", "PRIVATE_REASONING", "PRIVATE_GOAL"):
                self.assertNotIn(private_value, embedding_text)
            self.assertNotIn("agent_goal", embedding_text)

            retrieved = run_pass2_retrieve(dataset, pass1_dir, output, context_mode="public")
            self.assertEqual(retrieved["retrieved"], 2)
            retrieval_text = (output / "retrieval_candidates.jsonl").read_text(encoding="utf-8")
            retrieval_config = json.loads((output / "pass2_retrieve_config.json").read_text(encoding="utf-8"))
            self.assertEqual(retrieval_config["context_mode"], "public")
            self.assertIn("semantic", retrieval_text)
            self.assertNotIn("identified_information_source", retrieval_text)
            self.assertNotIn("incoming_communication", retrieval_text)
            for private_value in ("PRIVATE_MESSAGE", "PRIVATE_REASONING", "PRIVATE_GOAL", "from_agent", "to_agent"):
                self.assertNotIn(private_value, retrieval_text)

            sent_requests = []
            fake_client = MagicMock()
            fake_client.base_url = "https://api.example.test"

            def create_response(**kwargs):
                payload = json.loads(kwargs["input"].split("\n", 1)[1])
                sent_requests.append((kwargs["instructions"], payload))
                response = MagicMock(status="completed")
                response.output_text = json.dumps({"dependencies": [
                    {
                        "source_event_id": candidate["source_event_id"],
                        "dependency_score": 0.25,
                        "reason": "Weak public contextual relevance.",
                    }
                    for candidate in payload["candidates"]
                ]})
                return response

            fake_client.responses.create.side_effect = create_response
            with patch("tide.pass2_stages.Adaption", return_value=fake_client):
                reranked = run_pass2_rerank(
                    dataset, output, "key", context_mode="public", pass1_dir=pass1_dir
                )
            self.assertEqual(reranked["completed_targets"], 2)
            for instructions, payload in sent_requests:
                serialized = json.dumps(payload)
                self.assertIn("PUBLIC_SUMMARY_", serialized)
                self.assertIn("public summary", instructions.lower())
                self.assertNotIn("semantic similarity", instructions.lower())
                self.assertNotIn("temporal proximity", instructions.lower())
                self.assertEqual(set(payload["target"]), {"event_id", "public_summary"})
                self.assertTrue(all(
                    set(candidate) == {"source_event_id", "public_summary"}
                    for candidate in payload["candidates"]
                ))
                self.assertNotIn("timestamp", instructions.lower())
                self.assertNotIn("author", instructions.lower())
                self.assertNotIn("agent", instructions.lower())
                self.assertNotIn("similarity_score", serialized)
                self.assertNotIn("retrieval_sources", serialized)
                for private_value in ("PRIVATE_MESSAGE", "PRIVATE_REASONING", "PRIVATE_GOAL", "from_agent", "to_agent"):
                    self.assertNotIn(private_value, serialized)
                self.assertNotIn("agent_goal", serialized)
            dependency_text = (output / "dependencies.jsonl").read_text(encoding="utf-8")
            rerank_config = json.loads((output / "pass2_rerank_config.json").read_text(encoding="utf-8"))
            self.assertEqual(rerank_config["context_mode"], "public")
            self.assertIn("PUBLIC_SUMMARY_TARGET", dependency_text)
            for private_value in ("PRIVATE_MESSAGE", "PRIVATE_REASONING", "PRIVATE_GOAL", "agent_goal"):
                self.assertNotIn(private_value, dependency_text)
            with patch("tide.pass2_stages.Adaption", return_value=fake_client), \
                    patch("tide.pass2_stages.request_rerank") as reranker:
                limited = run_pass2_rerank(
                    dataset, output, "key", context_mode="public", pass1_dir=pass1_dir, limit=1
                )
            reranker.assert_not_called()
            self.assertEqual((limited["targets"], limited["completed_targets"], limited["unresolved_targets"]), (1, 1, 0))
            self.assertEqual(len((output / "dependencies.jsonl").read_text(encoding="utf-8").splitlines()), 2)
            with self.assertRaisesRegex(ValueError, "separate output directories"):
                run_pass2_embed(dataset, output, overwrite=True)

    def test_auto_embedding_precision_uses_fp16_on_cuda_and_fp32_on_cpu(self):
        import torch

        for cuda_available, expected_dtype, expected_device in (
            (True, torch.float16, "cuda"),
            (False, torch.float32, "cpu"),
        ):
            model = MagicMock()
            with patch("torch.cuda.is_available", return_value=cuda_available), \
                    patch("transformers.AutoModel.from_pretrained", return_value=model):
                result = _load_embedding_model("model", "revision")
            self.assertIs(result, model.to.return_value)
            model.to.assert_called_once_with(device=torch.device(expected_device), dtype=expected_dtype)

    def test_all_in_one_runs_each_cached_stage(self):
        embed_result = {"events": 12, "skipped_consolidations": 2, "embedded": 10, "failures": [], "elapsed_seconds": 1, "output": "embeddings.jsonl"}
        retrieve_result = {"events": 10, "skipped_consolidations": 2, "targets": 3, "retrieved": 3, "candidate_count": 20, "elapsed_seconds": 2, "output": "retrieval.jsonl"}
        rerank_result = {"targets": 3, "completed_targets": 3, "unresolved_targets": 0, "failures": [], "elapsed_seconds": 3, "output": "dependencies.jsonl"}
        with patch("tide.pass2_stages.run_pass2_embed", return_value=embed_result) as embed, \
                patch("tide.pass2_stages.run_pass2_retrieve", return_value=retrieve_result) as retrieve, \
                patch("tide.pass2_stages.run_pass2_rerank", return_value=rerank_result) as rerank:
            result = run_pass2_all(
                "dataset", "pass1", "cache", "key", limit=3,
                concurrency=4, candidate_group_size=4, candidate_filter="proposed",
            )
        embed.assert_called_once()
        retrieve.assert_called_once()
        rerank.assert_called_once()
        self.assertEqual(rerank.call_args.kwargs["concurrency"], 4)
        self.assertEqual(rerank.call_args.kwargs["candidate_group_size"], 4)
        self.assertEqual(rerank.call_args.kwargs["candidate_filter"], "proposed")
        self.assertEqual(result["candidate_count"], 20)
        self.assertEqual(result["output"], "dependencies.jsonl")

    def test_stages_write_separate_outputs_and_resume_without_reinference(self):
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
            for event_id, timestamp, author_id, text, action in (
                ("source", "2026-10-24T11:00:00Z", "b", "source result", "AGENT_TALK"),
                ("consolidate", "2026-10-24T11:30:00Z", "b", "private consolidation", "CONSOLIDATE"),
                ("target-later", "2026-10-24T12:30:00Z", "a", "later target action", "AGENT_TALK"),
                ("target", "2026-10-24T12:00:00Z", "a", "target action", "AGENT_TALK"),
            ):
                rows.append({
                    "timestamp_utc": timestamp,
                    "identity": {"registered_agent_id": author_id},
                    "metadata": {"agent_goals_active_at_event": [{
                        "name": "Publish stories", "short_name": "Publishing",
                        "description": f"Goal for {author_id}",
                    }]},
                    "source_event": {"source_record": {
                        "id": event_id, "created_at": timestamp,
                        "data": {"actionType": action, "output": [{"type": "text", "text": text}]},
                    }},
                })
            source_bytes = ("".join(json.dumps(row) + "\n" for row in rows)).encode()
            (dataset / "events.jsonl").write_bytes(source_bytes)
            annotation_rows = [
                {"event_id": "source", "summary": "source", "to_agent": ["A"], "from_agent": []},
                {"event_id": "consolidate", "summary": "consolidate", "to_agent": [], "from_agent": ["B"]},
                {"event_id": "target", "summary": "target", "to_agent": [], "from_agent": ["B"]},
                {"event_id": "target-later", "summary": "later target", "to_agent": [], "from_agent": ["B"]},
            ]
            annotation_bytes = ("".join(json.dumps(row) + "\n" for row in annotation_rows)).encode()
            (pass1_dir / "annotations.jsonl").write_bytes(annotation_bytes)
            fake_client = MagicMock()
            fake_client.base_url = "https://api.example.test"

            class FakeEmbeddingModel:
                def encode(self, texts, task, prompt_name):
                    self.last_task = (task, prompt_name)
                    return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

            rerank_barrier = threading.Barrier(2)

            def rerank_concurrently(client, model, target, candidates, *, public_only=False):
                rerank_barrier.wait(timeout=10)
                return {"dependencies": {
                    candidate["source_event_id"]: {
                        "dependency_score": 0.75,
                        "reason": "Target uses the earlier result.",
                    }
                    for candidate in candidates
                }}

            with patch("tide.pass2_stages.Adaption", return_value=fake_client), \
                    patch("tide.pass2_stages._load_embedding_model", return_value=FakeEmbeddingModel()) as embedder, \
                    patch("tide.pass2_stages.request_rerank", side_effect=rerank_concurrently) as reranker:
                embed_result = run_pass2_embed(dataset, output)
                retrieve_result = run_pass2_retrieve(dataset, pass1_dir, output)
                rerank_result = run_pass2_rerank(dataset, output, "key", concurrency=2)
            self.assertEqual(embed_result["embedded"], 3)
            self.assertEqual(embed_result["skipped_consolidations"], 1)
            self.assertEqual((retrieve_result["targets"], retrieve_result["retrieved"]), (3, 3))
            self.assertEqual(rerank_result["unresolved_targets"], 0)
            self.assertEqual((rerank_result["targets"], rerank_result["completed_targets"]), (3, 3))
            self.assertEqual(embedder.call_count, 1)
            self.assertEqual(reranker.call_count, 2)
            embedding_rows = [
                json.loads(line)
                for line in (output / "event_embeddings.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            dependencies = [
                json.loads(line)
                for line in (output / "dependencies.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            dependency = next(row for row in dependencies if row["event_id"] == "target")
            candidate = dependency["candidates"][0]
            self.assertEqual(candidate["source_event_id"], "source")
            self.assertEqual(candidate["dependency_score"], 0.75)
            self.assertIn("semantic", candidate["retrieval_sources"])
            self.assertIn("identified_information_source", candidate["retrieval_sources"])
            self.assertIn("incoming_communication", candidate["retrieval_sources"])
            self.assertIn("Goal for a", dependency["agent_goal"])
            self.assertNotIn("private_content", candidate)
            self.assertNotIn("consolidate", {row["event_id"] for row in embedding_rows})
            self.assertNotIn("consolidate", {row["event_id"] for row in dependencies})
            self.assertNotIn("from_agent", json.dumps(embedding_rows + dependencies))
            self.assertNotIn("to_agent", json.dumps(embedding_rows + dependencies))
            self.assertTrue((output / "retrieval_candidates.jsonl").exists())
            self.assertTrue((output / "pass2_embed_config.json").exists())
            self.assertTrue((output / "pass2_retrieve_config.json").exists())
            self.assertTrue((output / "pass2_rerank_config.json").exists())
            self.assertEqual((dataset / "events.jsonl").read_bytes(), source_bytes)
            self.assertEqual((pass1_dir / "annotations.jsonl").read_bytes(), annotation_bytes)
            with patch("tide.pass2_stages.Adaption", return_value=fake_client), \
                    patch("tide.pass2_stages._load_embedding_model") as embedder, \
                    patch("tide.pass2_stages.request_rerank") as reranker:
                run_pass2_embed(dataset, output)
                run_pass2_retrieve(dataset, pass1_dir, output)
                resumed = run_pass2_rerank(dataset, output, "key", concurrency=2)
            embedder.assert_not_called()
            reranker.assert_not_called()
            self.assertEqual(resumed["completed_targets"], 3)

            checkpoint = output / "rerank_checkpoint.jsonl"
            checkpoint_rows = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
            checkpoint.write_text("".join(
                json.dumps(row) + "\n" for row in checkpoint_rows if row["event_id"] != "target"
            ), encoding="utf-8")
            with patch("tide.pass2_stages.Adaption", return_value=fake_client), \
                    patch("tide.pass2_stages.request_rerank", return_value={"dependencies": {
                        "source": {"dependency_score": 0.75, "reason": "Earlier result."}
                    }}) as reranker:
                limited = run_pass2_rerank(dataset, output, "key", concurrency=2, limit=2)
            self.assertEqual((limited["targets"], limited["completed_targets"], limited["unresolved_targets"]), (2, 2, 0))
            self.assertEqual(reranker.call_count, 1)
            self.assertEqual(reranker.call_args.args[2]["event_id"], "target")
            saved_ids = {
                json.loads(line)["event_id"]
                for line in checkpoint.read_text(encoding="utf-8").splitlines()
            }
            self.assertEqual(saved_ids, {"source", "target", "target-later"})
            with patch("tide.pass2_stages.Adaption", return_value=fake_client), \
                    patch("tide.pass2_stages.request_rerank") as reranker:
                resumed = run_pass2_rerank(dataset, output, "key", concurrency=2)
            reranker.assert_not_called()
            self.assertEqual(resumed["completed_targets"], 3)


if __name__ == "__main__":
    unittest.main()
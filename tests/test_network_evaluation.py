import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tide.network_evaluation import aggregate_max_scores, build_community_graph, build_event_mention_reference, cohort_agent_ids, compare_edge_sets, community_comparison, evaluate_event_links, precision_recall, reference_mentions, run_network_evaluation
from tide.pass2 import file_sha256


class NetworkEvaluationTests(unittest.TestCase):
    def test_reference_relative_edge_comparison(self):
        result = compare_edge_sets({("a", "b"), ("b", "c")}, {("a", "b"), ("c", "a")})
        self.assertEqual((result["shared"], result["reference_only"], result["tide_only"]), (1, 1, 1))
        self.assertEqual((result["precision"], result["recall"], result["f1"]), (0.5, 0.5, 0.5))
        self.assertAlmostEqual(result["jaccard"], 1 / 3)

    def test_empty_denominators_are_not_given_false_certainty(self):
        result = compare_edge_sets({("a", "b")}, set())
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0)

    def test_explicit_mentions_and_max_agent_scores(self):
        agents = {"a": {"name": "Agent A", "model_string": "vendor/agent-a"}, "b": {"name": "Agent B"}}
        self.assertEqual(reference_mentions("Hi @Agent A, @agent-b; email person@agent-a.com", agents), {"a", "b"})
        authors = {"source": "a", "target": "b", "other": "a"}
        rows = [{"source_event_id": "source", "target_event_id": "target", "score": 0.3},
                {"source_event_id": "other", "target_event_id": "target", "score": 0.8},
                {"source_event_id": "source", "target_event_id": "other", "score": 0.9}]
        self.assertEqual(aggregate_max_scores(rows, authors, {"a", "b"}, {"target", "other"}, "score"), {("a", "b"): 0.8})

    def test_pr_includes_unscored_reference_edges(self):
        reference = {("a", "b"), ("b", "c")}
        result = precision_recall(reference, {("a", "b"): 0.8, ("c", "a"): 0.5})
        self.assertEqual(result["unscored_reference_edges"], 1)
        self.assertAlmostEqual(result["average_precision"], 0.5 + 0.5 * (2 / 3))
        self.assertEqual(community_comparison([0, 0, 1], [1, 1, 0])["ari"], 1.0)

    def test_leiden_keeps_registered_isolates(self):
        graph, membership, modularity = build_community_graph(["a", "b", "c"], {("a", "b"): 0.8})
        self.assertEqual(graph.vcount(), 3)
        self.assertEqual(graph.ecount(), 1)
        self.assertEqual(len(membership), 3)
        self.assertNotEqual(membership[2], membership[0])
        self.assertIsNotNone(modularity)

    def test_node_universe_contains_only_registered_cohort_authors(self):
        agents = {"a": {"name": "A"}, "b": {"name": "B"}, "c": {"name": "C"}}
        events = [{"author_id": "a"}, {"author_id": "b"}, {"author_id": "a"}, {"author_id": None}, {"author_id": "unknown"}]
        self.assertEqual(cohort_agent_ids(events, agents), ["a", "b"])

    def test_explicit_mention_uses_closest_strict_predecessor_and_reports_coverage(self):
        from datetime import datetime, timedelta, timezone

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        agents = {"a": {"name": "A"}, "b": {"name": "B"}, "c": {"name": "C"}}
        events = [
            {"event_id": "b0", "author_id": "b", "timestamp_dt": base, "source_index": 0},
            {"event_id": "b1", "author_id": "b", "timestamp_dt": base + timedelta(minutes=2), "source_index": 1},
            {"event_id": "a1", "author_id": "a", "timestamp_dt": base + timedelta(minutes=2), "source_index": 2,
             "chat_content": "@B @C @A"},
            {"event_id": "a2", "author_id": "a", "timestamp_dt": base + timedelta(minutes=3), "source_index": 3,
             "chat_content": "@B"},
        ]
        edges, counts, tokens = build_event_mention_reference(events, agents, {"a1", "a2"})
        self.assertEqual(edges, {("b0", "a1"), ("b1", "a2")})
        self.assertEqual(counts["explicit_mention_agent_pairs"], 4)
        self.assertEqual(counts["distinct_mentioned_agents"], 3)
        self.assertEqual(counts["mentions_with_eligible_predecessor"], 2)
        self.assertEqual(counts["mentions_without_eligible_predecessor"], 1)
        self.assertEqual(counts["self_mentions_excluded"], 1)
        self.assertEqual(counts["targets_with_at_least_one_reference_link"], 2)
        self.assertEqual(tokens[("b0", "a1")], ["@B"])

    def test_event_metrics_do_not_count_unresolvable_mentions_as_false_negatives(self):
        from datetime import datetime, timedelta, timezone

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [{"event_id": "s", "timestamp_dt": base},
                  {"event_id": "t", "timestamp_dt": base + timedelta(hours=1)},
                  {"event_id": "u", "timestamp_dt": base + timedelta(hours=2)}]
        reference = {("s", "t")}
        result = evaluate_event_links(reference, {("s", "t"): 0.8}, events, {"t", "u"}, 0.5)
        self.assertEqual(result["eligible_event_pairs"], 3)
        self.assertEqual((result["shared"], result["reference_only"], result["tide_only"]), (1, 0, 0))
        self.assertFalse(result["unobserved_pairs_are_verified_negatives"])

    def test_end_to_end_excludes_nonauthor_mentions_from_all_graphs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()

            def rows(path, records):
                path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

            agents = [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}, {"id": "c", "name": "C"}]
            rows(dataset / "agents.jsonl", agents)
            events = [
                {"timestamp_utc": f"2026-01-01T0{index}:00:00Z", "identity": {"registered_agent_id": author},
                 "source_event": {"source_record": {"id": f"e{index}", "data": {"actionType": "AGENT_TALK", "content": content}}}}
                for index, (author, content) in enumerate([("b", "start"), ("a", "@B @C"), ("b", "@A")])
            ]
            rows(dataset / "events.jsonl", events)
            retrieval = [{"event_id": f"e{index}", "source_index": index + 1, "candidates": []} for index in range(3)]
            checkpoint = [{"event_id": f"e{index}", "source_index": index + 1, "candidates": [] if index == 0 else [
                {"source_event_id": f"e{index - 1}", "dependency_score": 0.8}]} for index in range(3)]
            configs = {}
            for mode in ("public", "private"):
                stage = root / mode
                stage.mkdir()
                retrieval_path = stage / "retrieval_candidates.jsonl"
                rows(retrieval_path, retrieval)
                config = {"context_mode": mode, "events_sha256": file_sha256(dataset / "events.jsonl"),
                          "agents_sha256": file_sha256(dataset / "agents.jsonl"), "pass1_sha256": "same"}
                configs[mode] = config
                (stage / "pass2_retrieve_config.json").write_text(json.dumps(config), encoding="utf-8")
                (stage / "pass2_rerank_config.json").write_text(json.dumps({
                    **config, "retrieval_sha256": file_sha256(retrieval_path), "candidate_filter": "none",
                }), encoding="utf-8")
                rows(stage / "rerank_checkpoint.jsonl", checkpoint)
            calibration = root / "calibration"
            calibration.mkdir()
            metadata = {"cohort": {"targets": 3, "training_targets": 1,
                                   "event_ids_sha256": hashlib.sha256(b"e0\ne1\ne2").hexdigest()},
                        "input_hashes": {**configs["public"],
                                         "public_retrieval_sha256": file_sha256(root / "public" / "retrieval_candidates.jsonl"),
                                         "public_rerank_snapshot_sha256": file_sha256(root / "public" / "rerank_checkpoint.jsonl"),
                                         "private_rerank_snapshot_sha256": file_sha256(root / "private" / "rerank_checkpoint.jsonl")}}
            (calibration / "calibration.json").write_text(json.dumps(metadata), encoding="utf-8")
            rows(calibration / "matched_pairs.jsonl", [])
            rows(calibration / "public_decayed_edges.jsonl", [
                {"source_event_id": f"e{index - 1}", "target_event_id": f"e{index}",
                 "public_score": 0.8, "calibrated_public_score": 0.7, "sigmoid_decay_public_score": 0.7,
                  "ranking_calibrated_public_score": 0.65} for index in (1, 2)
            ])
            output = root / "evaluation"
            result = run_network_evaluation(dataset, dataset / "agents.jsonl", root / "public", root / "private", calibration, output)
            self.assertEqual(result["scope"]["registered_agents"], 2)
            self.assertEqual(result["scope"]["full_registry_agents"], 3)
            self.assertEqual({item["nodes"] for item in result["structures"].values()}, {2})
            self.assertIn("Public TIDE", result["structures"])
            self.assertIn("Public TIDE (no temporal)", result["structures"])
            self.assertEqual(result["structures"]["Explicit mentions"]["edges"], 2)
            self.assertEqual(result["comparisons"]["Private inference"]["variants"]["sigmoid_decay"]["at_threshold"]["shared"], 2)
            self.assertEqual(result["comparisons"]["Private inference"]["variants"]["ranking_only"]["at_threshold"]["shared"], 2)
            self.assertEqual(result["event_level_explicit_reference"]["reference_counts"]["reference_event_links"], 2)
            self.assertEqual(result["event_level_explicit_reference"]["reference_counts"]["mentions_without_eligible_predecessor"], 1)
            self.assertEqual(result["event_level_explicit_reference"]["metrics"]["f1"], 1.0)
            self.assertEqual(result["event_level_explicit_reference"]["metrics"]["eligible_event_pairs"], 3)
            self.assertTrue(all((output / filename).exists() for filename in
                                ("metrics.json", "report.md", "precision_recall.png", "precision_recall.pdf",
                                 "communities.png", "communities.pdf", "community-overlap.png", "community-overlap.pdf",
                                 "event_level_explicit_reference.json",
                                 "event_level_explicit_reference_edges.jsonl", "event_level_tide_event_edges.jsonl")))
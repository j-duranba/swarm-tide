import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from tide.graph_analysis import aggregate_agents, direct_reference_graph, graph_statistics, induced_event_edges, pass1_edges, pass2_edges, read_snapshot, run_graph_analysis
from tide.pass2 import file_sha256


class GraphAnalysisTests(unittest.TestCase):
    def test_pass1_references_use_strictly_previous_and_next_events(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            {"event_id": name, "author_id": author, "timestamp_dt": base + timedelta(minutes=minute),
             "source_index": index, "pass1_relations": relations}
            for index, (name, author, minute, relations) in enumerate([
                ("before", "b", 0, {}),
                ("equal", "b", 1, {}),
                ("target", "a", 1, {"from_agent": ["B", "unknown"], "to_agent": ["B"]}),
                ("after", "b", 2, {}),
            ])
        ]
        edges, counts = pass1_edges(events, {"B": "b"})
        self.assertEqual([(edge["source_event_id"], edge["target_event_id"]) for edge in edges],
                         [("before", "target"), ("target", "after")])
        self.assertEqual(counts["unknown_agent"], 1)

    def test_pass2_uses_inclusive_threshold_and_source_to_target_direction(self):
        rows = [{"event_id": "target", "candidates": [
            {"source_event_id": "source", "similarity_score": 0.5},
            {"source_event_id": "other", "similarity_score": 0.4},
        ]}]
        self.assertEqual(pass2_edges(rows, "similarity_score", 0.5, {"target", "source", "other"}), [
            {"source_event_id": "source", "target_event_id": "target", "similarity_score": 0.5},
        ])

    def test_induced_graph_requires_both_endpoints_in_cohort(self):
        edges = [
            {"source_event_id": "inside", "target_event_id": "also-inside"},
            {"source_event_id": "outside", "target_event_id": "inside"},
            {"source_event_id": "inside", "target_event_id": "outside"},
        ]
        self.assertEqual(induced_event_edges(edges, {"inside", "also-inside"}), edges[:1])

    def test_direct_reference_graph_counts_direction_and_references_to_nonauthors(self):
        agents = {"a": {"name": "A"}, "b": {"name": "B"}, "c": {"name": "C"}}
        events = [
            {"event_id": "one", "author_id": "a", "pass1_relations": {"from_agent": ["B"], "to_agent": ["C"]}},
            {"event_id": "two", "author_id": "b", "pass1_relations": {"to_agent": ["A"], "from_agent": ["unknown"]}},
            {"event_id": "three", "author_id": None, "pass1_relations": {"to_agent": ["C"]}},
        ]
        nodes, edges, counts = direct_reference_graph(events, agents, {"A": "a", "B": "b", "C": "c"}, 2)
        self.assertEqual(edges, [{"source_agent_id": "b", "target_agent_id": "a", "from_agent_count": 1,
                                  "to_agent_count": 1, "weight": 2,
                                  "references": [{"event_id": "one", "type": "from_agent"},
                                                 {"event_id": "two", "type": "to_agent"}]}])
        self.assertFalse(nodes[2]["has_cohort_event"])
        self.assertEqual(counts, {"references": 5, "unknown_agent": 1, "missing_author": 1, "pairs_before_threshold": 2})

    def test_partial_checkpoint_is_read_without_modification(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            contents = b'{"event_id":"one"}\n{"event_id":"partial"'
            path.write_bytes(contents)
            rows, digest = read_snapshot(path)
            self.assertEqual(rows, [{"event_id": "one"}])
            self.assertEqual(len(digest), 64)
            self.assertEqual(path.read_bytes(), contents)

    def test_aggregation_and_statistics_include_loops_and_isolates(self):
        events = [{"event_id": event_id, "author_id": author, "author_name": author} for event_id, author in [
            ("e1", "a"), ("e2", "a"), ("e3", "b"), ("e4", "c"), ("e5", None),
        ]]
        edges = [
            {"source_event_id": "e1", "target_event_id": "e2", "weight": 0.5},
            {"source_event_id": "e1", "target_event_id": "e3", "weight": 0.7},
            {"source_event_id": "e2", "target_event_id": "e3", "weight": 0.2},
            {"source_event_id": "e5", "target_event_id": "e3", "weight": 0.8},
        ]
        nodes, agent_edges, omitted = aggregate_agents(events, edges)
        self.assertEqual(omitted, 1)
        self.assertEqual(agent_edges[1]["event_edge_count"], 2)
        self.assertAlmostEqual(agent_edges[1]["weight"], 0.9)
        stats = graph_statistics(nodes, agent_edges, level="agent", weighted=True, seed=3)
        self.assertEqual(stats["weak_component_sizes"], [2, 1])
        self.assertEqual(stats["self_loops"], 1)
        self.assertEqual(stats["nodes"][0]["out_degree"], 2)
        self.assertAlmostEqual(stats["nodes"][0]["out_strength"], 1.4)
        self.assertEqual(len(stats["nodes"]), 3)

    def test_empty_and_zero_weight_graphs(self):
        self.assertEqual(graph_statistics([], [], level="event", weighted=True, seed=0)["community_count"], 0)
        nodes = [{"id": "a"}, {"id": "b"}]
        edges = [{"source_event_id": "a", "target_event_id": "b", "weight": 0.0}]
        stats = graph_statistics(nodes, edges, level="event", weighted=True, seed=0)
        self.assertEqual(stats["edge_count"], 1)
        self.assertEqual(stats["community_count"], 2)

    def test_analysis_supports_private_and_public_snapshots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            pass1 = root / "pass1"
            dataset.mkdir()
            pass1.mkdir()

            def write_rows(path, rows):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            write_rows(dataset / "agents.jsonl", [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}])
            write_rows(dataset / "events.jsonl", [
                {"timestamp_utc": f"2026-01-01T00:0{index}:00Z", "identity": {"registered_agent_id": author},
                 "source_event": {"source_record": {"id": event_id, "data": {"actionType": "AGENT_TALK"}}}}
                for index, (event_id, author) in enumerate([("source", "b"), ("target", "a")])
            ])
            write_rows(pass1 / "annotations.jsonl", [
                {"event_id": "source", "summary": "source", "from_agent": [], "to_agent": []},
                {"event_id": "target", "summary": "target", "from_agent": ["B"], "to_agent": []},
            ])
            retrieval = [{"event_id": "target", "source_index": 2, "candidates": [{
                "source_event_id": "source", "similarity_score": 0.6, "retrieval_sources": ["semantic"],
            }]}]
            for mode in ("private", "public"):
                pass2 = root / mode
                pass2.mkdir()
                retrieval_path = pass2 / "retrieval_candidates.jsonl"
                write_rows(retrieval_path, retrieval)
                config = {"context_mode": mode, "events_sha256": file_sha256(dataset / "events.jsonl"),
                          "agents_sha256": file_sha256(dataset / "agents.jsonl"),
                          "pass1_sha256": file_sha256(pass1 / "annotations.jsonl")}
                (pass2 / "pass2_retrieve_config.json").write_text(json.dumps(config), encoding="utf-8")
                (pass2 / "pass2_rerank_config.json").write_text(json.dumps({
                    "context_mode": mode, "candidate_filter": "none", "retrieval_sha256": file_sha256(retrieval_path),
                }), encoding="utf-8")
                write_rows(pass2 / "rerank_checkpoint.jsonl", [{"event_id": "target", "candidates": [{
                    **retrieval[0]["candidates"][0], "dependency_score": 0.8,
                }]}])
                output = root / f"analysis-{mode}"
                summary = run_graph_analysis(dataset, pass1, pass2, output, similarity_threshold=0.5, dependency_threshold=0.7)
                self.assertEqual(summary["context_mode"], mode)
                self.assertEqual(summary["graphs"]["pass1"]["event_edges"], 1)
                self.assertEqual(summary["graphs"]["pass1_direct"]["agent_edges"], 1)
                self.assertEqual(summary["graphs"]["pass2_retrieval_matched"]["event_edges"], 1)
                self.assertEqual(summary["graphs"]["pass2_reranked"]["event_edges"], 1)
                stats = json.loads((output / "pass2_reranked_agent_statistics.json").read_text(encoding="utf-8"))
                self.assertAlmostEqual(stats["nodes"][1]["out_strength"], 0.8)
                cohort_output = root / f"cohort-{mode}"
                cohort_summary = run_graph_analysis(
                    dataset, pass1, pass2, cohort_output, similarity_threshold=0.5, dependency_threshold=0.7,
                    cohort_retrieval=retrieval_path, cohort_limit=1,
                )
                self.assertEqual(cohort_summary["events"], 1)
                self.assertEqual(cohort_summary["retrieval_targets"], 1)
                self.assertEqual(cohort_summary["reranked_targets"], 1)
                for graph in ("pass1", "pass2_retrieval", "pass2_retrieval_matched", "pass2_reranked"):
                    self.assertEqual(cohort_summary["graphs"][graph]["event_edges"], 0)
                    self.assertEqual(cohort_summary["graphs"][graph]["edges_outside_cohort"], 1)
                self.assertEqual(cohort_summary["pass1_references"]["no_matching_event"], 0)
                self.assertEqual(cohort_summary["graphs"]["pass1_direct"]["agent_nodes"], 2)
                self.assertEqual(cohort_summary["graphs"]["pass1_direct"]["agent_edges"], 1)
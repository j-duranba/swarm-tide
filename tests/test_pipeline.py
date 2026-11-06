from __future__ import annotations

import tempfile
import unittest

from tide.graphing import aggregate_agent_matrix, build_event_graph
from tide.pipeline import run_synthetic_pipeline
from tide.retrieval import RetrievalConfig, retrieve_candidates, strictly_precedes
from tide.schema import build_event
from tide.support import score_candidates
from tide.synthetic import generate_synthetic_trace


class TidePipelineTests(unittest.TestCase):
    def test_retrieval_never_uses_future_events(self) -> None:
        bundle = generate_synthetic_trace(seed=7)
        events = list(bundle.events)
        by_id = {e.event_id: e for e in events}
        candidates = retrieve_candidates(events, RetrievalConfig())
        for target_id, cands in candidates.items():
            target = by_id[target_id]
            for c in cands:
                source = by_id[c.source_event_id]
                self.assertTrue(strictly_precedes(source, target))

    def test_detects_explicit_message_support(self) -> None:
        bundle = generate_synthetic_trace(seed=7)
        events = list(bundle.events)
        by_id = {e.event_id: e for e in events}
        candidates = retrieve_candidates(events, RetrievalConfig())
        support_edges = []
        for cands in candidates.values():
            support_edges.extend(score_candidates(by_id, cands))
        lookup = {(e.source_event_id, e.target_event_id): e for e in support_edges}
        self.assertIn(("e9", "e10"), lookup)
        self.assertGreaterEqual(lookup[("e9", "e10")].support_score, 0.9)

    def test_agent_matrix_direction_convention(self) -> None:
        bundle = generate_synthetic_trace(seed=7)
        events = list(bundle.events)
        graph = build_event_graph(events, list(bundle.ground_truth_edges), min_score=0.5)
        matrix = aggregate_agent_matrix(events, graph)
        self.assertGreater(matrix["raw_sum"]["B"]["A"], 0.0)
        self.assertLess(matrix["raw_sum"]["A"]["B"], matrix["raw_sum"]["B"]["A"])

    def test_events_with_unknown_order_not_forced_as_predecessors(self) -> None:
        e1 = build_event(
            {
                "event_id": "x1",
                "episode_id": "x",
                "agent_id": "A",
                "timestamp": None,
                "event_index": None,
                "action_type": "output",
                "action_summary": "first",
            },
            source_table="t",
            source_record_id="1",
        )
        e2 = build_event(
            {
                "event_id": "x2",
                "episode_id": "x",
                "agent_id": "B",
                "timestamp": None,
                "event_index": None,
                "action_type": "output",
                "action_summary": "second",
            },
            source_table="t",
            source_record_id="2",
        )
        cands = retrieve_candidates([e1, e2], RetrievalConfig())
        self.assertEqual(cands["x1"], [])
        self.assertEqual(cands["x2"], [])

    def test_cli_pipeline_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outputs = run_synthetic_pipeline(output_dir=td, seed=7, min_edge_score=0.4)
            self.assertEqual(outputs.summary["mode"], "synthetic")
            self.assertGreater(outputs.summary["event_count"], 0)
            self.assertGreater(outputs.summary["retrieval_pair_count"], 0)

    def test_hard_scenarios_present(self) -> None:
        bundle = generate_synthetic_trace(seed=7)
        scenarios = {e.episode_id for e in bundle.events}
        self.assertIn("hardneg_timing_offset", scenarios)
        self.assertIn("hardpos_misleading_relay", scenarios)

    def test_message_recipient_mismatch_not_direct(self) -> None:
        bundle = generate_synthetic_trace(seed=7)
        events = list(bundle.events)
        by_id = {e.event_id: e for e in events}
        scored = []
        cands = retrieve_candidates(events, RetrievalConfig())
        for cs in cands.values():
            scored.extend(score_candidates(by_id, cs))
        lookup = {(e.source_event_id, e.target_event_id): e for e in scored}
        self.assertIn(("e24", "e27"), lookup)
        self.assertLess(lookup[("e24", "e27")].support_score, 0.5)


if __name__ == "__main__":
    unittest.main()

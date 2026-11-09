import math
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tide.calibration import fit_sigmoid_decay, fit_sigmoid_ranking, fit_temporal_decay, mean_squared_error, run_calibration, sigmoid_decay_score, sigmoid_ranking_score
from tide.pass2 import file_sha256


class CalibrationTests(unittest.TestCase):
    def test_grid_recovers_known_decay_and_keeps_zero_baseline(self):
        pairs = [{"public_score": 0.8, "private_score": 0.8 * math.exp(-0.2 * hours), "elapsed_hours": hours}
                 for hours in (0.5, 1, 2)]
        fit = fit_temporal_decay(pairs, lambda_max=1, steps=100)
        self.assertAlmostEqual(fit["lambda_per_hour"], 0.2)
        self.assertAlmostEqual(fit["decayed_mse"], 0)
        unchanged = [{"public_score": 0.8, "private_score": 0.8, "elapsed_hours": 2}]
        self.assertEqual(fit_temporal_decay(unchanged)["lambda_per_hour"], 0)

    def test_no_pairs_is_not_a_zero_error(self):
        self.assertIsNone(mean_squared_error([], 0))
        with self.assertRaises(ValueError):
            fit_temporal_decay([])

    def test_joint_fit_recovers_synthetic_sigmoid_and_decay(self):
        pairs = [{"public_score": score, "elapsed_hours": hours,
                  "private_score": sigmoid_decay_score(score, hours, 3.0, -1.2, 0.3)}
                 for score in (0.1, 0.4, 0.8, 1.0) for hours in (0.1, 1.0, 3.0)]
        fit = fit_sigmoid_decay(pairs)
        self.assertAlmostEqual(fit["alpha"], 3.0, places=4)
        self.assertAlmostEqual(fit["intercept"], -1.2, places=4)
        self.assertAlmostEqual(fit["lambda_per_hour"], 0.3, places=4)
        self.assertLess(fit["training_mse"], 1e-12)

    def test_ranking_only_fit_recovers_sigmoid_without_time_input(self):
        pairs = [{"public_score": score, "elapsed_hours": hours,
                  "private_score": sigmoid_ranking_score(score, 2.5, -0.8)}
                 for score in (0.0, 0.2, 0.5, 0.8, 1.0) for hours in (0.1, 2.0, 10.0)]
        fit = fit_sigmoid_ranking(pairs)
        self.assertAlmostEqual(fit["alpha"], 2.5, places=4)
        self.assertAlmostEqual(fit["intercept"], -0.8, places=4)
        self.assertLess(fit["training_mse"], 1e-12)

    def test_chronological_split_missing_scores_and_public_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()

            def rows(path, records):
                path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

            rows(dataset / "agents.jsonl", [{"id": "a", "name": "A"}])
            rows(dataset / "events.jsonl", [
                {"timestamp_utc": f"2026-01-01T0{index}:00:00Z", "identity": {"registered_agent_id": "a"},
                 "source_event": {"source_record": {"id": f"e{index}", "data": {"actionType": "AGENT_TALK"}}}}
                for index in range(4)
            ])
            retrieval = [{"event_id": f"e{index}", "source_index": index + 1, "candidates": []} for index in range(4)]
            scores = {
                "public": [
                    {"event_id": "e1", "source_index": 2, "candidates": [{"source_event_id": "e0", "dependency_score": 1.0}]},
                    {"event_id": "e2", "source_index": 3, "candidates": [
                        {"source_event_id": "e1", "dependency_score": 1.0}, {"source_event_id": "e0", "dependency_score": 0.3}]},
                    {"event_id": "e3", "source_index": 4, "candidates": [{"source_event_id": "e2", "dependency_score": 1.0}]},
                ],
                "private": [
                    {"event_id": "e1", "source_index": 2, "candidates": [{"source_event_id": "e0", "dependency_score": math.exp(-0.2)}]},
                    {"event_id": "e2", "source_index": 3, "candidates": [{"source_event_id": "e1", "dependency_score": math.exp(-0.2)}]},
                ],
            }
            for mode in ("public", "private"):
                stage = root / mode
                stage.mkdir()
                retrieval_path = stage / "retrieval_candidates.jsonl"
                rows(retrieval_path, retrieval)
                config = {"context_mode": mode, "events_sha256": file_sha256(dataset / "events.jsonl"),
                          "agents_sha256": file_sha256(dataset / "agents.jsonl"), "pass1_sha256": "same"}
                (stage / "pass2_retrieve_config.json").write_text(json.dumps(config), encoding="utf-8")
                (stage / "pass2_rerank_config.json").write_text(json.dumps({
                    **config, "retrieval_sha256": file_sha256(retrieval_path), "candidate_filter": "none",
                }), encoding="utf-8")
                rows(stage / "rerank_checkpoint.jsonl", scores[mode])
            output = root / "calibration"
            result = run_calibration(dataset, root / "public", root / "private", output,
                                     train_targets=2, cohort_limit=4, lambda_max=1, grid_steps=100, model="decay-only")
            self.assertAlmostEqual(result["fit"]["lambda_per_hour"], 0.2)
            self.assertEqual(result["coverage"]["train"]["matched_pairs"], 1)
            self.assertEqual(result["coverage"]["holdout"]["matched_pairs"], 1)
            self.assertEqual(result["coverage"]["holdout"]["public_only_pairs"], 1)
            self.assertEqual(result["coverage"]["holdout"]["private_completed_targets"], 1)
            self.assertAlmostEqual(result["metrics"]["holdout"]["decayed"]["mse"], 0)
            public_edges = [json.loads(line) for line in (output / "public_decayed_edges.jsonl").read_text().splitlines()]
            self.assertEqual(len(public_edges), 4)
            self.assertEqual(public_edges[-1]["target_event_id"], "e3")
            self.assertAlmostEqual(public_edges[-1]["decayed_public_score"], math.exp(-0.2))
            joint_output = root / "joint-calibration"
            joint = run_calibration(dataset, root / "public", root / "private", joint_output,
                                    train_targets=2, cohort_limit=4, lambda_max=1, grid_steps=100)
            joint_edges = [json.loads(line) for line in (joint_output / "public_decayed_edges.jsonl").read_text().splitlines()]
            self.assertEqual(joint["model"], "sigmoid-decay")
            self.assertAlmostEqual(joint_edges[-1]["calibrated_public_score"], sigmoid_decay_score(
                1.0, 1.0, joint["fit"]["alpha"], joint["fit"]["intercept"], joint["fit"]["lambda_per_hour"]))
            self.assertAlmostEqual(joint["metrics"]["holdout"]["sigmoid_decay"]["mse"],
                                   (joint_edges[1]["calibrated_public_score"] - math.exp(-0.2)) ** 2)
            self.assertAlmostEqual(joint_edges[-1]["ranking_calibrated_public_score"], sigmoid_ranking_score(
                1.0, joint["ranking_only_fit"]["alpha"], joint["ranking_only_fit"]["intercept"]))
            self.assertIn("ranking_only", joint["metrics"]["holdout"])
            self.assertIn("sigmoid_decay", joint["metrics"]["holdout"])
            self.assertEqual(joint["metrics"]["holdout"]["ranking_only"]["pairs"],
                             joint["metrics"]["holdout"]["sigmoid_decay"]["pairs"])
            self.assertEqual(joint["ranking_only_fit"]["optimizer"], "scipy.optimize.least_squares")
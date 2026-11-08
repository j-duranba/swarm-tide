import hashlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from adaption import Adaption

from tide.cli import build_parser
from tide.pass1 import build_event_prompt, build_system_prompt, inspect_event_source, load_agent_registry, load_checkpoint, prepare_event, request_annotation, run_pass1, validate_annotation


class Pass1ValidationTests(unittest.TestCase):
    def test_concurrent_results_are_ordered_after_out_of_order_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "master"
            dataset.mkdir()
            output = Path(directory) / "output"
            (dataset / "agents.jsonl").write_text('{"id":"a1","name":"Example"}\n', encoding="utf-8")
            events = [{"source_event": {"source_record": {"id": f"event-{index}", "data": {"actionType": "WAIT"}}}}
                      for index in range(2)]
            (dataset / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            first_written = threading.Event()
            finished = []

            def annotate(client, model, prompt, event, registered):
                if event["event_id"] == "event-0" and not first_written.wait(5):
                    raise RuntimeError("Timed out waiting for the later event")
                finished.append(event["event_id"])
                return {"event_id": event["event_id"], "summary": "Waited.", "to_agent": [], "from_agent": []}

            with patch("tide.pass1.request_annotation", side_effect=annotate), patch("tide.pass1.os.fsync", side_effect=lambda _: first_written.set()):
                result = run_pass1(dataset, output, "model", "key", concurrency=2)
            self.assertEqual(result["failures"], [])
            self.assertEqual(finished, ["event-1", "event-0"])
            checkpoint = [json.loads(line) for line in (output / "pass1_checkpoint.jsonl").read_text(encoding="utf-8").splitlines()]
            annotations = [json.loads(line) for line in (output / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["source_index"] for row in checkpoint], [1, 2])
            self.assertEqual([row["event_id"] for row in annotations], ["event-0", "event-1"])

    def test_errors_only_does_not_process_unattempted_events(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "master"
            dataset.mkdir()
            output = Path(directory) / "output"
            (dataset / "agents.jsonl").write_text('{"id":"a1","name":"Example"}\n', encoding="utf-8")
            events = [{"source_event": {"source_record": {"id": f"event-{index}", "data": {"actionType": "WAIT"}}}}
                      for index in range(3)]
            (dataset / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

            def first_attempt(client, model, prompt, event, registered):
                if event["event_id"] == "event-0":
                    raise ValueError("Invalid JSON annotation")
                return {"event_id": event["event_id"], "summary": "Waited.", "to_agent": [], "from_agent": []}

            with patch("tide.pass1.request_annotation", side_effect=first_attempt):
                first = run_pass1(dataset, output, "model", "key", limit=2)
            self.assertEqual((first["selected"], first["unresolved_total"]), (2, 1))
            annotation = {"event_id": "event-0", "summary": "Waited.", "to_agent": [], "from_agent": []}
            with patch("tide.pass1.request_annotation", return_value=annotation) as mocked:
                retried = run_pass1(dataset, output, "model", "key", retry_errors_only=True)
            mocked.assert_called_once()
            self.assertEqual((retried["selected"], retried["completed_total"], retried["unresolved_total"]), (1, 2, 0))
            recorded = [json.loads(line)["event_id"] for line in (output / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(recorded, ["event-0", "event-1"])

    def test_errors_do_not_stop_later_events_and_resume_clears_error_file(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "master"
            dataset.mkdir()
            output = Path(directory) / "output"
            (dataset / "agents.jsonl").write_text('{"id":"a1","name":"Example"}\n', encoding="utf-8")
            events = [{"source_event": {"source_record": {"id": f"event-{index}", "data": {"actionType": "WAIT"}}}}
                      for index in range(3)]
            (dataset / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

            def annotate(client, model, prompt, event, registered):
                if event["event_id"] == "event-0":
                    raise ValueError("Invalid JSON annotation")
                return {"event_id": event["event_id"], "summary": "Waited.", "to_agent": [], "from_agent": []}

            with patch("tide.pass1.request_annotation", side_effect=annotate) as mocked:
                first = run_pass1(dataset, output, "model", "key")
            self.assertEqual(mocked.call_count, 3)
            self.assertEqual((first["selected"], first["completed_total"], first["unresolved_total"]), (3, 2, 1))
            errors_path = output / "pass1_errors.jsonl"
            self.assertEqual(json.loads(errors_path.read_text(encoding="utf-8")), {"event_id": "event-0", "source_index": 1, "error": "Invalid JSON annotation"})
            annotation = {"event_id": "event-0", "summary": "Waited.", "to_agent": [], "from_agent": []}
            with patch("tide.pass1.request_annotation", return_value=annotation) as mocked:
                second = run_pass1(dataset, output, "model", "key", retry_errors_only=True)
            mocked.assert_called_once()
            self.assertEqual((second["submitted"], second["completed_total"], second["unresolved_total"]), (1, 3, 0))
            self.assertEqual(errors_path.read_text(encoding="utf-8"), "")
            annotations = [json.loads(line) for line in (output / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["event_id"] for row in annotations], ["event-0", "event-1", "event-2"])
            checkpoint = [json.loads(line) for line in (output / "pass1_checkpoint.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["source_index"] for row in checkpoint], [1, 2, 3])
            self.assertEqual([row["annotation"] for row in checkpoint], annotations)
            errors_path.write_text(json.dumps({"event_id": "event-0", "source_index": 1, "error": "stale"}) + "\n", encoding="utf-8")
            with patch("tide.pass1.request_annotation") as mocked:
                reconciled = run_pass1(dataset, output, "model", "key")
            mocked.assert_not_called()
            self.assertEqual(reconciled["unresolved_total"], 0)
            self.assertEqual(errors_path.read_text(encoding="utf-8"), "")
            with self.assertRaises(ValueError):
                run_pass1(dataset, output, "model", "key", overwrite=True, retry_errors_only=True)

    def test_invalid_annotations_are_retried_with_correction_feedback(self):
        client = MagicMock()
        invalid_json = MagicMock(status="completed", output_text="not JSON")
        unknown_agent = MagicMock(status="completed", output_text=json.dumps({"summary": "Sent.", "to_agent": [], "from_agent": ["unknown"]}))
        valid = MagicMock(status="completed", output_text=json.dumps({"summary": "Sent.", "to_agent": [], "from_agent": []}))
        client.responses.create.side_effect = [invalid_json, unknown_agent, valid]
        result = request_annotation(client, "model", "system", {"event_id": "event-1"}, {"Example"})
        self.assertEqual(result["event_id"], "event-1")
        self.assertEqual(client.responses.create.call_count, 3)
        self.assertIn("Invalid JSON annotation", client.responses.create.call_args_list[1].kwargs["input"])
        self.assertIn("from_agent contains duplicate or unregistered agent names", client.responses.create.call_args_list[2].kwargs["input"])

    def test_invalid_annotations_stop_after_configured_retries(self):
        client = MagicMock()
        client.responses.create.return_value = MagicMock(status="completed", output_text="not JSON")
        with self.assertRaisesRegex(ValueError, "Invalid JSON annotation"):
            request_annotation(client, "model", "system", {"event_id": "event-1"}, set(), validation_retries=1)
        self.assertEqual(client.responses.create.call_count, 2)

    def test_overwrite_restarts_run_and_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "master"
            dataset.mkdir()
            output = Path(directory) / "output"
            (dataset / "agents.jsonl").write_text('{"id":"a1","name":"Example"}\n', encoding="utf-8")
            (dataset / "events.jsonl").write_text('{"source_event":{"source_record":{"id":"event-1","data":{"actionType":"WAIT"}}}}\n', encoding="utf-8")

            def annotate(client, model, prompt, event, registered):
                return {"event_id": event["event_id"], "summary": model, "to_agent": [], "from_agent": []}

            with patch("tide.pass1.request_annotation", side_effect=annotate) as mocked:
                run_pass1(dataset, output, "first-model", "key")
                (output / "unrelated.txt").write_text("keep", encoding="utf-8")
                with self.assertRaises(ValueError):
                    run_pass1(dataset, output, "second-model", "key")
                result = run_pass1(dataset, output, "second-model", "key", overwrite=True)
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(result["submitted"], 1)
            self.assertEqual(json.loads((output / "annotations.jsonl").read_text(encoding="utf-8"))["summary"], "second-model")
            self.assertEqual(json.loads((output / "pass1_config.json").read_text(encoding="utf-8"))["model"], "second-model")
            self.assertEqual((output / "unrelated.txt").read_text(encoding="utf-8"), "keep")
            (output / "pass1.lock").write_text("active", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                run_pass1(dataset, output, "third-model", "key", overwrite=True)
            self.assertEqual(json.loads((output / "pass1_config.json").read_text(encoding="utf-8"))["model"], "second-model")
            self.assertEqual((output / "pass1.lock").read_text(encoding="utf-8"), "active")
            with self.assertRaises(ValueError):
                run_pass1(dataset, dataset, "third-model", "key", overwrite=True)
            self.assertFalse(build_parser().parse_args(["pass1", "--output-dir", "out"]).overwrite)
            self.assertTrue(build_parser().parse_args(["pass1", "--output-dir", "out", "--overwrite"]).overwrite)

    def test_event_inspection_hash_and_eligible_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            actions = ["CONSOLIDATE", "AGENT_TALK", "CONSOLIDATE", "SEARCH_HISTORY"]
            rows = [{"source_event": {"source_record": {"id": f"event-{index}", "data": {"actionType": action}}}}
                    for index, action in enumerate(actions, 1)]
            path.write_text("\n" + "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(inspect_event_source(path, None), (expected_hash, [2, 4]))
            self.assertEqual(inspect_event_source(path, 1), (expected_hash, [2]))

    def test_progress_reports_total_without_faking_success_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "master"
            dataset.mkdir()
            (dataset / "agents.jsonl").write_text('{"id":"a1","name":"Example"}\n', encoding="utf-8")
            (dataset / "events.jsonl").write_text('{"source_event":{"source_record":{"id":"event-1","data":{"actionType":"WAIT"}}}}\n', encoding="utf-8")
            display = io.StringIO()
            with patch("tide.pass1.sys.stderr", display), patch("tide.pass1.request_annotation", side_effect=RuntimeError("failure")):
                result = run_pass1(dataset, Path(directory) / "output", "model", "key")
            self.assertIn("0/1", display.getvalue())
            self.assertIn("failed=1", display.getvalue())
            self.assertEqual(result["completed_total"], 0)
            self.assertGreaterEqual(result["elapsed_seconds"], 0)

    def test_skip_consolidations_limit_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "master"
            dataset.mkdir()
            output = Path(directory) / "output"
            (dataset / "agents.jsonl").write_text('{"id":"a1","name":"Example"}\n', encoding="utf-8")
            actions = ["CONSOLIDATE", "AGENT_TALK", "CONSOLIDATE", "SEARCH_HISTORY", "AGENT_TALK"]
            events = [{"source_event": {"source_record": {"id": f"event-{index}", "data": {"actionType": action}}}}
                      for index, action in enumerate(actions, 1)]
            (dataset / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

            def annotate(client, model, prompt, event, registered):
                return {"event_id": event["event_id"], "summary": "Observed an event.", "to_agent": [], "from_agent": []}

            with patch("tide.pass1.request_annotation", side_effect=annotate) as mocked, patch("tide.pass1.tqdm") as progress:
                first = run_pass1(dataset, output, "model", "key", limit=2)
                resumed = run_pass1(dataset, output, "model", "key")
            self.assertEqual([(call.kwargs["total"], call.kwargs["initial"]) for call in progress.call_args_list], [(2, 0), (3, 2)])
            self.assertEqual(progress.return_value.__enter__.return_value.update.call_count, 3)
            self.assertEqual((first["selected"], first["skipped_consolidations"], first["submitted"]), (2, 2, 2))
            self.assertEqual((resumed["selected"], resumed["submitted"], resumed["completed_total"]), (3, 1, 3))
            self.assertEqual(mocked.call_count, 3)
            checkpoint = [json.loads(line) for line in (output / "pass1_checkpoint.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["source_index"] for row in checkpoint], [2, 4, 5])
            self.assertEqual([row["annotation"]["event_id"] for row in checkpoint], ["event-2", "event-4", "event-5"])
            config_path = output / "pass1_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["version"] = 2
            config.pop("excluded_action_types")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_pass1(dataset, output, "model", "key")

    def test_only_consolidations_makes_no_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "master"
            dataset.mkdir()
            (dataset / "agents.jsonl").write_text('{"id":"a1","name":"Example"}\n', encoding="utf-8")
            (dataset / "events.jsonl").write_text('{"source_event":{"source_record":{"id":"event-1","data":{"actionType":"CONSOLIDATE"}}}}\n', encoding="utf-8")
            with patch("tide.pass1.request_annotation") as mocked:
                result = run_pass1(dataset, Path(directory) / "output", "model", "key", limit=10)
            mocked.assert_not_called()
            self.assertEqual((result["selected"], result["skipped_consolidations"], result["completed_total"]), (0, 1, 0))

    def test_cli_default_model_and_override(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["pass1", "--output-dir", "out"]).model, "deepseek-v4-flash")
        self.assertEqual(parser.parse_args(["pass1", "--output-dir", "out", "--model", "qwen3.5-397b"]).model, "qwen3.5-397b")

    def test_checkpoint_recovers_incomplete_final_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            annotation = {"event_id": "event-1", "summary": "Waited.", "to_agent": [], "from_agent": []}
            complete = json.dumps({"source_index": 1, "annotation": annotation}) + "\n"
            path.write_text(complete + '{"source_index":2', encoding="utf-8")
            self.assertEqual(load_checkpoint(path, {"Example"}), {"event-1": (1, annotation)})
            self.assertEqual(path.read_text(encoding="utf-8"), complete)

    def test_duplicate_names_are_rejected(self):
        with self.assertRaises(ValueError):
            build_system_prompt({"a": {"name": "Example"}, "b": {"name": "Example"}})

    def test_failure_is_resumable_with_mock_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "master"
            dataset.mkdir()
            output = Path(directory) / "output"
            (dataset / "agents.jsonl").write_text('{"id":"a1","name":"Example"}\n', encoding="utf-8")
            (dataset / "events.jsonl").write_text('{"source_event":{"source_record":{"id":"event-1","data":{"actionType":"WAIT"}}}}\n', encoding="utf-8")
            with patch("tide.pass1.request_annotation", side_effect=RuntimeError("HTTP 429")):
                result = run_pass1(dataset, output, "model", "key")
            self.assertEqual(result["failures"][0]["event_id"], "event-1")
            annotation = {"event_id": "event-1", "summary": "Waited.", "to_agent": [], "from_agent": []}
            with patch("tide.pass1.request_annotation", return_value=annotation):
                result = run_pass1(dataset, output, "model", "key")
            self.assertEqual(result["completed_total"], 1)
            self.assertEqual(result["failures"], [])

    def test_resume_and_master_protection_with_mock_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "master"
            dataset.mkdir()
            output = Path(directory) / "output"
            (dataset / "agents.jsonl").write_text('{"id":"a1","name":"Example"}\n', encoding="utf-8")
            events = [{"source_event": {"source_record": {"id": f"event-{index}", "data": {"actionType": "WAIT"}}}} for index in range(2)]
            (dataset / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

            def annotate(client, model, prompt, event, registered):
                return {"event_id": event["event_id"], "summary": "Waited.", "to_agent": [], "from_agent": []}

            with patch("tide.pass1.request_annotation", side_effect=annotate) as mocked:
                first = run_pass1(dataset, output, "model", "key", limit=1)
                second = run_pass1(dataset, output, "model", "key")
                self.assertEqual(mocked.call_count, 2)
            self.assertEqual(first["completed_total"], 1)
            self.assertEqual(second["completed_total"], 2)
            annotations = [json.loads(line) for line in (output / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["event_id"] for row in annotations], ["event-0", "event-1"])
            with self.assertRaises(ValueError):
                run_pass1(dataset, dataset / "output", "model", "key")
            with self.assertRaises(ValueError):
                run_pass1(dataset, output, "other-model", "key")

    def test_request_uses_adaption_sdk_and_validates_result(self):
        annotation = {"summary": "Sent a message.", "to_agent": [], "from_agent": []}
        client = MagicMock()
        client.responses.create.return_value.status = "completed"
        client.responses.create.return_value.output_text = json.dumps(annotation)
        result = request_annotation(client, "model", "system", {"event_id": "event-1"}, set())
        sent = client.responses.create.call_args.kwargs
        self.assertEqual(result, {"event_id": "event-1", **annotation})
        self.assertEqual(sent["instructions"], "system")
        self.assertEqual(sent["model"], "model")
        self.assertFalse(sent["agentic_memory"])
        self.assertEqual(sent["tools"], [])
        self.assertFalse(sent["store"])

    def test_sdk_request_wire_format_without_network(self):
        calls = []

        def handle(request):
            calls.append(request)
            return httpx.Response(200, json={
                "id": "response-1", "object": "response", "status": "completed", "model": "model",
                "output": [], "output_text": '{"summary":"Waited.","to_agent":[],"from_agent":[]}',
            })

        with Adaption(api_key="test", base_url="https://example.test", http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            result = request_annotation(client, "model", "system", {"event_id": "event-1"}, set())
        self.assertEqual(result["event_id"], "event-1")
        self.assertEqual(calls[0].url.path, "/api/v1/responses")
        body = json.loads(calls[0].content)
        self.assertEqual(body["instructions"], "system")
        self.assertEqual(body["tools"], [])
        self.assertFalse(body["agentic_memory"])
        self.assertFalse(body["store"])

    def test_reject_incomplete_or_invalid_sdk_response(self):
        client = MagicMock()
        client.responses.create.return_value.status = "incomplete"
        with self.assertRaises(ValueError):
            request_annotation(client, "model", "system", {"event_id": "event-1"}, set())
        client.responses.create.return_value.status = "completed"
        client.responses.create.return_value.output_text = "not JSON"
        with self.assertRaises(ValueError):
            request_annotation(client, "model", "system", {"event_id": "event-1"}, set())

    def test_prompts_include_registry_and_exclude_private_components(self):
        registry = {"agent-1": {"name": "Example", "model_string": "example-model", "goal": "private"}}
        system = build_system_prompt(registry)
        self.assertIn('"model_string": "example-model"', system)
        self.assertNotIn('"goal"', system)
        self.assertNotIn('"id": "agent-1"', system)
        example = system.split("Response:\n", 1)[1].split("\nThe recipient", 1)[0]
        self.assertEqual(json.loads(example)["to_agent"], ["gpt-luna"])
        row = {
            "source_event": {"source_record": {"id": "event-1", "data": {
                "actionType": "AGENT_TALK", "content": "Hello Example", "output": [{"type": "thinking", "text": "SECRET"}]
            }}},
            "identity": {"registered_agent_id": "agent-1"},
            "metadata": {"private": "SECRET"},
        }
        event = prepare_event(row, registry)
        self.assertEqual(event["public_message"], "Hello Example")
        self.assertEqual(event["actor"], "Example")
        self.assertNotIn("event-1", build_event_prompt(event))
        self.assertNotIn("SECRET", build_event_prompt(event))
        row["source_event"]["source_record"]["data"]["actionType"] = "CONSOLIDATE"
        self.assertNotIn("public_message", prepare_event(row))

    def test_load_registry_and_reject_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agents.jsonl"
            row = {"id": "agent-1", "name": "Example", "model_string": "example-model"}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(load_agent_registry(path), {"agent-1": row})
            path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_agent_registry(path)

    def test_validate_annotation_preserves_distinct_reference_lists(self):
        annotation = {"summary": "Sent a message.", "to_agent": ["Example"], "from_agent": []}
        self.assertEqual(validate_annotation(annotation, "event-1", {"Example"}), {"event_id": "event-1", **annotation})

    def test_reject_invalid_annotations(self):
        valid = {"summary": "Sent a message.", "to_agent": [], "from_agent": []}
        invalid = [
            {**valid, "event_id": "other"},
            {**valid, "summary": "  "},
            {**valid, "to_agent": ["unknown"]},
            {**valid, "from_agent": ["Example", "Example"]},
            {**valid, "from_agent": "Example"},
            {**valid, "extra": True},
        ]
        for annotation in invalid:
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                validate_annotation(annotation, "event-1", {"Example"})


if __name__ == "__main__":
    unittest.main()
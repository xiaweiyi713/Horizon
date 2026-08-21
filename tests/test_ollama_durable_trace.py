"""Unit tests for the local Ollama durable-trace experiment launcher."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import ollama_durable_trace as launcher  # noqa: E402
from benchmarks.horizonbench.adapters import JsonlTaskAdapter  # noqa: E402


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class OllamaDurableTraceLauncherTests(unittest.TestCase):
    def test_ollama_root_requires_an_openai_compatible_v1_endpoint(self) -> None:
        self.assertEqual(
            launcher._ollama_root("http://127.0.0.1:11434/v1/"),
            "http://127.0.0.1:11434",
        )
        with self.assertRaisesRegex(launcher.OllamaExperimentError, "end with /v1"):
            launcher._ollama_root("http://127.0.0.1:11434/api")

    def test_digest_is_taken_from_the_exact_installed_model(self) -> None:
        digest = "a" * 64
        payload = {
            "models": [
                {"name": "other:1b", "digest": "b" * 64},
                {"name": "qwen2.5:7b", "digest": digest},
            ]
        }
        with patch.object(launcher, "urlopen", return_value=_Response(payload)) as request:
            observed = launcher.ollama_model_digest("http://127.0.0.1:11434/v1", "qwen2.5:7b")

        self.assertEqual(observed, digest)
        self.assertEqual(request.call_args.args[0].full_url, "http://127.0.0.1:11434/api/tags")

    def test_conditions_and_seeds_are_strictly_frozen(self) -> None:
        conditions = launcher.load_conditions(launcher.DEFAULT_CONDITIONS)
        self.assertEqual(
            [condition.condition_id for condition in conditions],
            ["Horizon_anchor_off", "Horizon_anchor_always", "Horizon_runtime_heuristic"],
        )
        self.assertEqual(launcher._parse_seeds("17,23"), [17, 23])
        with self.assertRaisesRegex(launcher.OllamaExperimentError, "duplicates"):
            launcher._parse_seeds("17,17")

    def test_default_mechanism_tasks_are_held_out_and_have_objective_trace_checks(self) -> None:
        tasks = JsonlTaskAdapter("ollama-mechanism", "v3").load(launcher.DEFAULT_TASKS)
        self.assertEqual(launcher.DEFAULT_TASKS.name, "ollama_durable_trace_tasks_v3.jsonl")
        self.assertEqual(
            tasks.task_ids,
            ("ollama-trace-evidence-01", "ollama-trace-failure-02", "ollama-trace-decision-03"),
        )
        self.assertTrue(
            all("horizon_trace_expectation" in task.metadata for task in tasks.tasks)
        )
        evidence_task = next(task for task in tasks.tasks if task.task_id == "ollama-trace-evidence-01")
        self.assertEqual(
            evidence_task.metadata["horizon_trace_expectation"]["required_evidence_terms"],
            ("The verified checksum matches the durable prefix",),
        )


if __name__ == "__main__":
    unittest.main()

"""Regression tests for no-execution HorizonBench matrix preflight."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "benchmarks" / "horizonbench" / "fixtures"
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.adapters import JsonlTaskAdapter  # noqa: E402
from benchmarks.horizonbench.execution import ExecutionValidationError  # noqa: E402
from benchmarks.horizonbench.preflight_matrix import (  # noqa: E402
    PREFLIGHT_PROTOCOL,
    preflight_matrix,
)
from benchmarks.horizonbench.protocol import (  # noqa: E402
    EvaluationCondition,
    ModelProfile,
    PromptArtifact,
    TaskSetIdentity,
    build_cross_model_matrix,
    write_manifest_jsonl,
)


class HorizonBenchMatrixPreflightTests(unittest.TestCase):
    def _task_source(self, directory: Path) -> Path:
        path = directory / "tasks.jsonl"
        records = [
            {
                "id": "heldout-01",
                "category": "goal_retention",
                "goal": "Preserve the original objective.",
                "constraint": "Do not alter the source data.",
                "split": "heldout",
            },
            {
                "id": "heldout-02",
                "category": "fault_recovery",
                "goal": "Recover after an injected restart.",
                "constraint": "Reuse the recorded checkpoint.",
                "split": "heldout",
            },
        ]
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return path

    def _matrix(self, directory: Path, tasks_path: Path) -> Path:
        task_set = JsonlTaskAdapter("local-fixture", "fixture-v1").load(tasks_path, split="heldout")
        manifests = build_cross_model_matrix(
            models=[ModelProfile("fixture", "local", "fixture-model", "v1", {"temperature": 0})],
            conditions=[EvaluationCondition("H", "git:test", {"state_anchor": True})],
            task_set=TaskSetIdentity.from_task_set(task_set),
            prompt=PromptArtifact("prompt-v1", "Use durable state."),
            seeds=[7],
            runtime_revision="git:test",
            checkpoint_cadence=1,
            fault_schedule=[],
        )
        path = directory / "matrix.jsonl"
        write_manifest_jsonl(path, manifests)
        return path

    def test_preflight_never_invokes_episode_executor_or_writes_artifacts(self) -> None:
        calls: list[tuple[str, str]] = []

        def execute(context: Any) -> dict[str, Any]:
            calls.append(("execute", context.task.task_id))
            raise AssertionError("preflight must not call the episode executor")

        def preflight(context: Any) -> MappingProxyType:
            calls.append(("preflight", context.task.task_id))
            return MappingProxyType(
                {
                    "ready": True,
                    "frozen": (context.manifest.run_id, MappingProxyType({"task": context.task.task_id})),
                }
            )

        execute.preflight = preflight  # type: ignore[attr-defined]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tasks = self._task_source(directory)
            matrix = self._matrix(directory, tasks)
            report = preflight_matrix(
                matrix,
                tasks_path=tasks,
                source_name="local-fixture",
                source_revision="fixture-v1",
                split="heldout",
                executor=execute,
                executor_spec="tests:preflight-fixture",
            )
            artifact_paths = list(directory.glob("*.receipt.json")) + list(directory.glob("*.jsonl"))

        self.assertEqual(report["protocol"], PREFLIGHT_PROTOCOL)
        self.assertEqual(report["run_count"], 1)
        self.assertEqual(report["task_count"], 2)
        self.assertEqual(calls, [("preflight", "heldout-01"), ("preflight", "heldout-02")])
        first = report["runs"][0]["checks"][0]["preflight"]
        self.assertTrue(first["ready"])
        self.assertEqual(first["frozen"][1], {"task": "heldout-01"})
        self.assertEqual(len(artifact_paths), 2)  # Only the input task/matrix JSONL files exist.

    def test_preflight_requires_an_explicit_static_hook(self) -> None:
        def execute(_context: Any) -> dict[str, Any]:
            return {}

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tasks = self._task_source(directory)
            matrix = self._matrix(directory, tasks)
            with self.assertRaisesRegex(ExecutionValidationError, "preflight"):
                preflight_matrix(
                    matrix,
                    tasks_path=tasks,
                    source_name="local-fixture",
                    source_revision="fixture-v1",
                    split="heldout",
                    executor=execute,
                    executor_spec="tests:no-preflight",
                )

    def test_cli_checks_scripted_trace_fixture_without_a_runtime_or_results_directory(self) -> None:
        tasks = FIXTURES / "durable_trace_tasks.jsonl"
        task_set = JsonlTaskAdapter("horizon-durable-trace-fixture", "fixture-v1").load(
            tasks, split="heldout"
        )
        manifests = build_cross_model_matrix(
            models=[
                ModelProfile(
                    "scripted-durable-trace-fixture",
                    "scripted-fixture",
                    "scripted-durable-trace",
                    "fixture-v1",
                    {},
                )
            ],
            conditions=[
                EvaluationCondition(
                    "H-trace-fixture",
                    "fixture-v1",
                    {"agent": {"max_steps": 6, "semantic_memory_limit": 0}},
                )
            ],
            task_set=TaskSetIdentity.from_task_set(task_set),
            prompt=PromptArtifact("fixture-v1", (FIXTURES / "durable_trace_prompt.txt").read_text()),
            seeds=[7],
            runtime_revision="fixture-v1",
            checkpoint_cadence=1,
            fault_schedule=[],
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            matrix = directory / "matrix.jsonl"
            write_manifest_jsonl(matrix, manifests)
            environment = dict(os.environ)
            environment["PYTHONPATH"] = "{}:{}".format(ROOT / "python", ROOT)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "horizonbench" / "preflight_matrix.py"),
                    "--matrix",
                    str(matrix),
                    "--tasks",
                    str(tasks),
                    "--source-name",
                    "horizon-durable-trace-fixture",
                    "--source-revision",
                    "fixture-v1",
                    "--executor",
                    "benchmarks.horizonbench.scripted_durable_trace_executor:execute",
                ],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)
            result_artifacts = list(directory.glob("*.receipt.json")) + list(directory.glob("*.results*"))

        self.assertEqual(report["protocol"], PREFLIGHT_PROTOCOL)
        self.assertEqual(report["task_count"], 3)
        self.assertFalse(report["runs"][0]["checks"][0]["preflight"]["model_call"])
        self.assertEqual(result_artifacts, [])


if __name__ == "__main__":
    unittest.main()

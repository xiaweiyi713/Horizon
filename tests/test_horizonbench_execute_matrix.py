"""Regression tests for receipt-bound HorizonBench matrix execution."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.adapters import JsonlTaskAdapter  # noqa: E402
from benchmarks.horizonbench.execute_matrix import (  # noqa: E402
    EXECUTION_PROTOCOL,
    MatrixExecutionError,
    execute_matrix,
)
from benchmarks.horizonbench.protocol import (  # noqa: E402
    EvaluationCondition,
    ModelProfile,
    PromptArtifact,
    TaskSetIdentity,
    build_cross_model_matrix,
    write_manifest_jsonl,
)
from benchmarks.horizonbench.score import EpisodeResult, read_jsonl, write_jsonl  # noqa: E402


class HorizonBenchMatrixExecutionTests(unittest.TestCase):
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
            {
                "id": "development-01",
                "category": "goal_retention",
                "goal": "Development-only task.",
                "constraint": "Do not score this row.",
                "split": "development",
            },
        ]
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return path

    def _matrix(self, directory: Path, tasks_path: Path) -> tuple[Path, Any]:
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
        return path, manifests[0]

    @staticmethod
    def make_outcome(context: Any) -> EpisodeResult:
        return EpisodeResult(
            task_id=context.task.task_id,
            category=context.task.category,
            success=True,
            goal_retained=True,
            constraint_violations=0,
            state_consistent=True,
            tokens=11,
            anchors_injected=1,
        )

    def _execute(
        self,
        matrix: Path,
        tasks: Path,
        results: Path,
        executor: Any,
        *,
        executor_spec: str = "tests:objective-fixture",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return execute_matrix(
            matrix,
            tasks_path=tasks,
            source_name="local-fixture",
            source_revision="fixture-v1",
            split="heldout",
            executor=executor,
            executor_spec=executor_spec,
            results_dir=results,
            **kwargs,
        )

    def test_persists_manifest_bound_results_receipt_and_idempotent_resume(self) -> None:
        calls: list[tuple[str, int]] = []

        def executor(context: Any) -> EpisodeResult:
            calls.append((context.task.task_id, context.attempt))
            return self.make_outcome(context)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tasks = self._task_source(directory)
            matrix, manifest = self._matrix(directory, tasks)
            results = directory / "results"
            report = self._execute(matrix, tasks, results, executor)
            result_path = results / (manifest.run_id + ".jsonl")
            receipt_path = results / (manifest.run_id + ".receipt.json")
            rows = read_jsonl(result_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            resumed = self._execute(matrix, tasks, results, executor)

        self.assertEqual(report["protocol"], EXECUTION_PROTOCOL)
        self.assertEqual(report["runs"][0]["executed_task_ids"], ["heldout-01", "heldout-02"])
        self.assertEqual([row.task_id for row in rows], ["heldout-01", "heldout-02"])
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["manifest_sha256"], manifest.manifest_sha256)
        self.assertEqual(receipt["completed_task_ids"], ["heldout-01", "heldout-02"])
        self.assertEqual(calls, [("heldout-01", 1), ("heldout-02", 1)])
        self.assertEqual(resumed["runs"][0]["executed_task_ids"], [])

    def test_failure_is_recorded_and_resume_only_retries_missing_task(self) -> None:
        calls: list[tuple[str, int]] = []
        fail_second = True

        def executor(context: Any) -> EpisodeResult:
            nonlocal fail_second
            calls.append((context.task.task_id, context.attempt))
            if context.task.task_id == "heldout-02" and fail_second:
                raise RuntimeError("injected task-environment outage")
            return self.make_outcome(context)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tasks = self._task_source(directory)
            matrix, manifest = self._matrix(directory, tasks)
            results = directory / "results"
            with self.assertRaisesRegex(MatrixExecutionError, "heldout-02"):
                self._execute(matrix, tasks, results, executor)
            result_path = results / (manifest.run_id + ".jsonl")
            receipt_path = results / (manifest.run_id + ".receipt.json")
            first_pass = read_jsonl(result_path)
            failed_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            fail_second = False
            report = self._execute(matrix, tasks, results, executor)
            completed_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

        self.assertEqual([row.task_id for row in first_pass], ["heldout-01"])
        self.assertEqual(failed_receipt["status"], "failed")
        self.assertIn("injected task-environment outage", failed_receipt["error"])
        self.assertEqual(calls, [("heldout-01", 1), ("heldout-02", 1), ("heldout-02", 2)])
        self.assertEqual(report["runs"][0]["executed_task_ids"], ["heldout-02"])
        self.assertEqual(completed_receipt["status"], "completed")
        self.assertEqual(len(completed_receipt["attempts"]), 3)

    def test_rejects_wrong_source_identity_and_unreceipted_result_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tasks = self._task_source(directory)
            matrix, manifest = self._matrix(directory, tasks)
            results = directory / "results"
            with self.assertRaisesRegex(ValueError, "does not match"):
                execute_matrix(
                    matrix,
                    tasks_path=tasks,
                    source_name="local-fixture",
                    source_revision="wrong-revision",
                    split="heldout",
                    executor=self.make_outcome,
                    executor_spec="tests:objective-fixture",
                    results_dir=results,
                )
            results.mkdir()
            write_jsonl(
                results / (manifest.run_id + ".jsonl"),
                [
                    EpisodeResult(
                        task_id="heldout-01",
                        category="goal_retention",
                        success=True,
                        goal_retained=True,
                        constraint_violations=0,
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "without a receipt"):
                self._execute(matrix, tasks, results, self.make_outcome)
            result_path = results / (manifest.run_id + ".jsonl")
            result_path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "without a receipt"):
                self._execute(matrix, tasks, results, self.make_outcome)

    def test_rejects_resume_with_a_different_executor_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tasks = self._task_source(directory)
            matrix, _ = self._matrix(directory, tasks)
            results = directory / "results"
            self._execute(matrix, tasks, results, self.make_outcome)
            with self.assertRaisesRegex(ValueError, "executor does not match"):
                self._execute(
                    matrix,
                    tasks,
                    results,
                    self.make_outcome,
                    executor_spec="tests:another-objective-fixture",
                )

    def test_cli_loads_fixture_executor_and_writes_completed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tasks = self._task_source(directory)
            matrix, manifest = self._matrix(directory, tasks)
            results = directory / "results"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "horizonbench" / "execute_matrix.py"),
                    "--matrix",
                    str(matrix),
                    "--tasks",
                    str(tasks),
                    "--source-name",
                    "local-fixture",
                    "--source-revision",
                    "fixture-v1",
                    "--executor",
                    "benchmarks.horizonbench.synthetic_fixture_executor:execute",
                    "--results-dir",
                    str(results),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)
            receipt = json.loads(
                (results / (manifest.run_id + ".receipt.json")).read_text(encoding="utf-8")
            )

        self.assertEqual(report["runs"][0]["status"], "completed")
        self.assertEqual(receipt["executor"], "benchmarks.horizonbench.synthetic_fixture_executor:execute")


if __name__ == "__main__":
    unittest.main()

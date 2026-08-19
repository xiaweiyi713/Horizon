"""Integration tests for cross-domain workflow scoring and reporting."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.cross_domain_score import score_cross_domain  # noqa: E402
from benchmarks.horizonbench.protocol import (  # noqa: E402
    EvaluationCondition,
    ManifestValidationError,
    ModelProfile,
    PromptArtifact,
    TaskSetIdentity,
    build_cross_model_matrix,
    read_manifest_jsonl,
    write_manifest_jsonl,
)
from benchmarks.horizonbench.score import EpisodeResult, write_jsonl  # noqa: E402
from benchmarks.horizonbench.workflows import load_cross_domain_workflows  # noqa: E402
from benchmarks.horizonbench.render_report import render_technical_report  # noqa: E402


class CrossDomainEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.suite = load_cross_domain_workflows()
        self.model = ModelProfile(
            "fixture-model", "fixture-provider", "fixture-v1", "release-1", {"temperature": 0}
        )
        self.condition = EvaluationCondition(
            "H", "git:0576cad", {"semantic_memory_limit": 4, "state_anchor": True}
        )
        self.prompt = PromptArtifact("system-v1", "Operate from durable state.")

    def _manifests(self):
        return build_cross_model_matrix(
            models=[self.model],
            conditions=[self.condition],
            task_set=TaskSetIdentity.from_task_set(self.suite.task_set),
            prompt=self.prompt,
            seeds=[1, 2],
            runtime_revision="git:0576cad",
            checkpoint_cadence=1,
            fault_schedule=[],
        )

    def _results_for(self, seed: int) -> list[EpisodeResult]:
        failed_dependency = self.suite.workflows[0].tasks[1].task_id
        dependent = self.suite.workflows[0].tasks[2].task_id
        rows: list[EpisodeResult] = []
        for workflow_task in self.suite.workflow_tasks:
            # Seed two makes a predecessor fail but lets its dependent task
            # succeed, exercising workflow prerequisite-integrity accounting.
            success = not (seed == 2 and workflow_task.task_id == failed_dependency)
            if seed == 2 and workflow_task.task_id == dependent:
                success = True
            rows.append(
                EpisodeResult(
                    task_id=workflow_task.task_id,
                    category=workflow_task.task.category,
                    success=success,
                    goal_retained=success,
                    constraint_violations=int(not success),
                    recovery_attempted=workflow_task.recovery_boundary,
                    recovery_succeeded=success if workflow_task.recovery_boundary else False,
                    recovery_distance=1 if workflow_task.recovery_boundary and success else None,
                    state_consistent=success,
                    tokens=10,
                )
            )
        return rows

    def test_scores_domains_workflows_and_cross_seed_summaries(self) -> None:
        manifests = self._manifests()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            matrix = directory / "matrix.jsonl"
            results_dir = directory / "results"
            write_manifest_jsonl(matrix, manifests)
            for manifest in manifests:
                write_jsonl(results_dir / (manifest.run_id + ".jsonl"), self._results_for(manifest.seed or 0))
            report = score_cross_domain(matrix, results_dir)

        self.assertEqual(report["protocol"], "horizonbench_cross_domain_score_v1")
        self.assertEqual(report["suite"]["task_set_sha256"], self.suite.task_set.task_set_sha256)
        self.assertEqual(report["matrix"]["run_count"], 2)
        self.assertEqual(len(report["runs"]), 2)
        self.assertEqual(len(report["summaries"]), 1)
        for run in report["runs"]:
            self.assertEqual(len(run["domains"]), 4)
            self.assertEqual(len(run["workflows"]), 4)
            self.assertTrue(all(not key.startswith("_") for workflow in run["workflows"] for key in workflow))
        damaged_run = next(run for run in report["runs"] if run["manifest"]["seed"] == 2)
        damaged_domain = next(
            domain for domain in damaged_run["domains"] if domain["domain"] == self.suite.workflows[0].domain
        )
        self.assertEqual(damaged_domain["workflow_success_rate"], 0.0)
        self.assertEqual(damaged_domain["prerequisite_integrity_rate"], 0.0)
        aggregate_domain = next(
            domain
            for domain in report["summaries"][0]["domains"]
            if domain["domain"] == self.suite.workflows[0].domain
        )
        self.assertEqual(aggregate_domain["workflow_success_rate"]["mean"], 0.5)
        self.assertEqual(aggregate_domain["prerequisite_integrity_rate"]["mean"], 0.5)
        rendered = render_technical_report(report)
        self.assertIn("Cross-domain workflow outcomes", rendered)
        self.assertIn(self.suite.workflows[0].workflow_id, rendered)
        self.assertIn("does not by itself establish", rendered)

    def test_rejects_workflow_suite_that_does_not_match_frozen_manifest(self) -> None:
        manifests = self._manifests()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            matrix = directory / "matrix.jsonl"
            results_dir = directory / "results"
            write_manifest_jsonl(matrix, manifests)
            for manifest in manifests:
                write_jsonl(results_dir / (manifest.run_id + ".jsonl"), self._results_for(manifest.seed or 0))
            with self.assertRaisesRegex(ManifestValidationError, "does not match frozen manifest"):
                score_cross_domain(matrix, results_dir, source_revision="another-fixture-revision")

    def test_direct_cross_domain_cli_writes_a_json_report(self) -> None:
        manifests = self._manifests()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            matrix = directory / "matrix.jsonl"
            results_dir = directory / "results"
            output = directory / "cross-domain.json"
            write_manifest_jsonl(matrix, manifests)
            for manifest in manifests:
                write_jsonl(results_dir / (manifest.run_id + ".jsonl"), self._results_for(manifest.seed or 0))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "horizonbench" / "cross_domain_score.py"),
                    "--matrix",
                    str(matrix),
                    "--results-dir",
                    str(results_dir),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertIn("horizonbench_cross_domain_score_v1", completed.stdout)
        self.assertEqual(report["matrix"]["run_count"], 2)

    def test_full_plan_score_and_render_cli_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            models_path = directory / "models.json"
            conditions_path = directory / "conditions.json"
            prompt_path = directory / "prompt.txt"
            matrix_path = directory / "matrix.jsonl"
            results_dir = directory / "results"
            score_path = directory / "score.json"
            report_path = directory / "report.md"
            models_path.write_text(json.dumps([self.model.as_dict()]), encoding="utf-8")
            conditions_path.write_text(json.dumps([self.condition.as_dict()]), encoding="utf-8")
            prompt_path.write_text(self.prompt.text, encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "horizonbench" / "plan_matrix.py"),
                    "--tasks",
                    str(ROOT / "benchmarks" / "horizonbench" / "cross_domain_tasks.jsonl"),
                    "--source-name",
                    "horizon-cross-domain-fixture",
                    "--source-revision",
                    "v1",
                    "--models",
                    str(models_path),
                    "--conditions",
                    str(conditions_path),
                    "--prompt",
                    str(prompt_path),
                    "--prompt-revision",
                    "system-v1",
                    "--runtime-revision",
                    "git:0576cad",
                    "--seeds",
                    "1",
                    "--output",
                    str(matrix_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifests = read_manifest_jsonl(matrix_path)
            self.assertEqual(len(manifests), 1)
            executed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "horizonbench" / "execute_matrix.py"),
                    "--matrix",
                    str(matrix_path),
                    "--tasks",
                    str(ROOT / "benchmarks" / "horizonbench" / "cross_domain_tasks.jsonl"),
                    "--source-name",
                    "horizon-cross-domain-fixture",
                    "--source-revision",
                    "v1",
                    "--executor",
                    "benchmarks.horizonbench.synthetic_fixture_executor:execute",
                    "--results-dir",
                    str(results_dir),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "horizonbench" / "cross_domain_score.py"),
                    "--matrix",
                    str(matrix_path),
                    "--results-dir",
                    str(results_dir),
                    "--output",
                    str(score_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            rendered = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "horizonbench" / "render_report.py"),
                    str(score_path),
                    "--output",
                    str(report_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            report = report_path.read_text(encoding="utf-8")
        self.assertIn("horizonbench_matrix_execution_v1", executed.stdout)
        self.assertIn("wrote", rendered.stdout)
        self.assertIn("Cross-domain workflow outcomes", report)
        self.assertIn("software_engineering", report)


if __name__ == "__main__":
    unittest.main()

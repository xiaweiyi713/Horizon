"""Tests for cross-domain technical-report rendering."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.render_report import (  # noqa: E402
    ReportValidationError,
    load_cross_domain_report,
    render_technical_report,
)


def _summary(mean: float | None) -> dict[str, object]:
    return {"observations": 1 if mean is not None else 0, "mean": mean, "sample_stddev": 0.0}


def _report() -> dict[str, object]:
    metrics = {
        "task_success_rate": _summary(0.75),
        "goal_retention_rate": _summary(1.0),
        "constraint_violation_rate": _summary(0.25),
        "tokens_per_successful_task": _summary(128.5),
    }
    domain = {
        "domain": "operations",
        "run_count": 1,
        "metrics": metrics,
        "workflow_success_rate": _summary(0.0),
        "prerequisite_integrity_rate": _summary(1.0),
        "evidence_task_success_rate": _summary(1.0),
        "recovery_boundary_task_success_rate": _summary(0.5),
    }
    return {
        "protocol": "horizonbench_cross_domain_score_v1",
        "suite": {"task_set_sha256": "b" * 64},
        "matrix": {
            "run_count": 1,
            "prompt_sha256": "c" * 64,
            "runtime_revision": "git:fac0bd8",
            "task_set": {
                "source_name": "fixture|source",
                "source_revision": "v1",
                "split": "heldout",
                "task_set_sha256": "a" * 64,
                "task_ids": ["task-1", "task-2"],
            },
        },
        "runs": [
            {
                "manifest": {
                    "run_id": "run-1",
                    "model_id": "model-a",
                    "condition_id": "H",
                    "seed": 7,
                },
                "workflows": [
                    {
                        "domain": "operations",
                        "workflow_id": "ops|recovery",
                        "completed_steps": 2,
                        "task_count": 3,
                        "workflow_success": False,
                        "prerequisite_integrity": True,
                    }
                ],
            }
        ],
        "summaries": [
            {
                "model_id": "model-a",
                "condition_id": "H",
                "run_count": 1,
                "overall_metrics": metrics,
                "domains": [domain],
            }
        ],
    }


class CrossDomainReportTests(unittest.TestCase):
    def test_renders_auditable_markdown_without_claiming_causality(self) -> None:
        rendered = render_technical_report(_report(), title="Report | title")
        self.assertIn("# Report | title", rendered)
        self.assertIn("does not by itself establish", rendered)
        self.assertIn("fixture\\|source", rendered)
        self.assertIn("75.0%", rendered)
        self.assertIn("ops\\|recovery", rendered)
        self.assertIn("2/3", rendered)
        self.assertIn("Interpretation constraints", rendered)

    def test_loader_rejects_non_cross_domain_or_incomplete_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps({"protocol": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ReportValidationError, "unsupported protocol"):
                load_cross_domain_report(path)
            path.write_text(json.dumps(_report()), encoding="utf-8")
            self.assertEqual(load_cross_domain_report(path)["protocol"], "horizonbench_cross_domain_score_v1")

    def test_direct_script_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "score.json"
            output_path = directory / "report.md"
            input_path.write_text(json.dumps(_report()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "horizonbench" / "render_report.py"),
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--title",
                    "CLI report",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            rendered = output_path.read_text(encoding="utf-8")
        self.assertIn("wrote", result.stdout)
        self.assertTrue(rendered.startswith("# CLI report"))


if __name__ == "__main__":
    unittest.main()

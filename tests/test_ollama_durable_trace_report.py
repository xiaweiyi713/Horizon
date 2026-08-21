"""Tests for the audited Markdown report of local Ollama smoke artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.execute_matrix import (  # noqa: E402
    EXECUTION_PROTOCOL,
    RECEIPT_SCHEMA_VERSION,
)
from benchmarks.horizonbench.preflight_matrix import PREFLIGHT_PROTOCOL  # noqa: E402
from benchmarks.horizonbench.protocol import (  # noqa: E402
    EvaluationCondition,
    ModelProfile,
    PromptArtifact,
    TaskSetIdentity,
    build_cross_model_matrix,
    write_manifest_jsonl,
)
from benchmarks.horizonbench.score import EpisodeResult, write_jsonl  # noqa: E402
from benchmarks.horizonbench.score_matrix import score_matrix  # noqa: E402
from experiments.render_ollama_durable_trace_report import (  # noqa: E402
    CLAIM_BOUNDARY,
    EXPERIMENT_KIND,
    SmokeReportValidationError,
    load_ollama_trace_artifacts,
    render_ollama_trace_report,
)


_EXECUTOR = "example.fixture_executor:execute"


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_artifacts(directory: Path) -> None:
    """Build a complete, no-network artifact directory with two model profiles."""

    task_set = TaskSetIdentity(
        source_name="fixture-ollama-trace",
        source_revision="v1",
        split="heldout",
        source_sha256="c" * 64,
        task_set_sha256="d" * 64,
        task_ids=("task-a", "task-b"),
    )
    model_rows = [
        ("alpha:1b", "a" * 64),
        ("beta:2b", "b" * 64),
    ]
    models = [
        ModelProfile(
            model_id="ollama-{}-{}".format(name.split(":")[0], digest[:12]),
            provider="openai-compatible",
            model=name,
            model_revision="ollama-digest:" + digest,
            decoding={"temperature": 0, "top_p": 1, "max_tokens": 128},
        )
        for name, digest in model_rows
    ]
    conditions = [
        EvaluationCondition(
            condition_id="Horizon_anchor_off",
            policy_revision="fixed-anchor-control-v1",
            configuration={"agent": {"anchor_strategy": "disabled", "max_steps": 4}},
        ),
        EvaluationCondition(
            condition_id="Horizon_anchor_always",
            policy_revision="fixed-anchor-control-v1",
            configuration={"agent": {"anchor_strategy": "always", "max_steps": 4}},
        ),
    ]
    manifests = build_cross_model_matrix(
        models=models,
        conditions=conditions,
        task_set=task_set,
        prompt=PromptArtifact("fixture-prompt-v1", "Return one action."),
        seeds=(7, 11),
        runtime_revision="git:fixture",
        checkpoint_cadence=1,
        metadata={"experiment_kind": EXPERIMENT_KIND, "claim_boundary": CLAIM_BOUNDARY},
    )
    matrix_path = directory / "matrix.jsonl"
    write_manifest_jsonl(matrix_path, manifests)
    results_dir = directory / "results"
    results_dir.mkdir()
    execution_runs = []
    preflight_runs = []
    for manifest in manifests:
        anchors = 2 if manifest.condition.condition_id == "Horizon_anchor_always" else 0
        first_success = manifest.model.model == "alpha:1b" or manifest.seed == 7
        results = [
            EpisodeResult(
                task_id="task-a",
                category="goal_retention",
                success=True,
                goal_retained=True,
                constraint_violations=0,
                state_consistent=True,
                tokens=100,
                anchors_injected=anchors,
            ),
            EpisodeResult(
                task_id="task-b",
                category="constraint_retention",
                success=first_success,
                goal_retained=True,
                constraint_violations=0,
                state_consistent=first_success,
                tokens=120,
                anchors_injected=anchors,
                diagnostic=None if first_success else "agent reached max_steps",
            ),
        ]
        result_path = results_dir / (manifest.run_id + ".jsonl")
        write_jsonl(result_path, results)
        receipt_path = results_dir / (manifest.run_id + ".receipt.json")
        _json(
            receipt_path,
            {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "protocol": EXECUTION_PROTOCOL,
                "run_id": manifest.run_id,
                "manifest_sha256": manifest.manifest_sha256,
                "task_set": task_set.as_dict(),
                "executor": _EXECUTOR,
                "status": "completed",
                "started_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:01Z",
                "completed_task_ids": list(task_set.task_ids),
                "attempts": [
                    {
                        "task_id": task_id,
                        "attempt": 1,
                        "started_at": "2026-01-01T00:00:00Z",
                        "finished_at": "2026-01-01T00:00:01Z",
                        "duration_ms": 1,
                        "status": "succeeded",
                        "error": None,
                    }
                    for task_id in task_set.task_ids
                ],
                "error": None,
            },
        )
        execution_runs.append(
            {
                "run_id": manifest.run_id,
                "status": "completed",
                "result_path": str(result_path),
                "receipt_path": str(receipt_path),
                "task_count": len(task_set.task_ids),
                "executed_task_ids": list(task_set.task_ids),
            }
        )
        preflight_runs.append(
            {
                "run_id": manifest.run_id,
                "manifest_sha256": manifest.manifest_sha256,
                "task_count": len(task_set.task_ids),
                "checks": [
                    {"task_id": task_id, "preflight": {"ready": True}}
                    for task_id in task_set.task_ids
                ],
            }
        )

    _json(
        directory / "preflight.json",
        {
            "protocol": PREFLIGHT_PROTOCOL,
            "matrix_path": str(matrix_path),
            "executor": _EXECUTOR,
            "task_set": task_set.as_dict(),
            "run_count": len(manifests),
            "task_count": len(manifests) * len(task_set.task_ids),
            "runs": preflight_runs,
        },
    )
    _json(
        directory / "execution.json",
        {
            "protocol": EXECUTION_PROTOCOL,
            "matrix_path": str(matrix_path),
            "results_dir": str(results_dir),
            "executor": _EXECUTOR,
            "task_set": task_set.as_dict(),
            "run_count": len(manifests),
            "runs": execution_runs,
        },
    )
    _json(directory / "score.json", score_matrix(matrix_path, results_dir))
    _json(
        directory / "experiment.json",
        {
            "experiment_kind": EXPERIMENT_KIND,
            "claim_boundary": CLAIM_BOUNDARY,
            "output_dir": str(directory),
            "models": [{"model": name, "digest": digest} for name, digest in model_rows],
            "model_count": len(model_rows),
            "matrix_runs": len(manifests),
            "task_count": len(task_set.task_ids),
            "conditions": [condition.condition_id for condition in conditions],
            "preflight": "preflight.json",
            "execution": "execution.json",
            "score": "score.json",
            "report": "report.md",
        },
    )


class OllamaDurableTraceReportTests(unittest.TestCase):
    def test_renders_only_after_recomputing_a_complete_frozen_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _make_artifacts(directory)
            artifacts = load_ollama_trace_artifacts(directory)
            rendered = render_ollama_trace_report(artifacts, title="Fixture smoke report")

        self.assertIn("# Fixture smoke report", rendered)
        self.assertIn("2 × 2 × 2", rendered)
        self.assertIn("已从上述原始结果重新计算，并逐字段匹配", rendered)
        self.assertIn("agent reached max_steps", rendered)
        self.assertIn("不能据此声称 Horizon 提升某个模型", rendered)

    def test_rejects_a_score_that_does_not_match_recomputed_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _make_artifacts(directory)
            score_path = directory / "score.json"
            score = json.loads(score_path.read_text(encoding="utf-8"))
            score["run_count"] = 1
            _json(score_path, score)

            with self.assertRaisesRegex(SmokeReportValidationError, "score.run_count"):
                load_ollama_trace_artifacts(directory)

    def test_suppresses_unrecognized_raw_diagnostic_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _make_artifacts(directory)
            result_path = next((directory / "results").glob("*.jsonl"))
            rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["diagnostic"] = "provider body secret=do-not-publish"
            result_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
            )
            rendered = render_ollama_trace_report(load_ollama_trace_artifacts(directory))

        self.assertNotIn("secret=do-not-publish", rendered)
        self.assertIn("未识别诊断（已抑制原始内容）", rendered)

    def test_direct_script_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _make_artifacts(directory)
            output = directory / "report.md"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "experiments" / "render_ollama_durable_trace_report.py"),
                    str(directory),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            rendered = output.read_text(encoding="utf-8")

        self.assertIn("wrote", result.stdout)
        self.assertTrue(rendered.startswith("# Horizon 本地 Ollama"))


if __name__ == "__main__":
    unittest.main()

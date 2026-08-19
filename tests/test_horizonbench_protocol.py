"""Tests for reproducible cross-model HorizonBench run manifests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.protocol import (  # noqa: E402
    EvaluationCondition,
    ManifestValidationError,
    ModelProfile,
    PromptArtifact,
    RunManifest,
    TaskSetIdentity,
    build_cross_model_matrix,
    load_manifest,
    manifest_summary,
    read_manifest_jsonl,
    validate_comparable_matrix,
    validate_result_task_ids,
    write_manifest_jsonl,
)
from benchmarks.horizonbench.score import EpisodeResult, write_jsonl  # noqa: E402
from benchmarks.horizonbench.score_runs import score_run  # noqa: E402
from benchmarks.horizonbench.plan_matrix import make_parser, plan_matrix  # noqa: E402
from benchmarks.horizonbench.score_matrix import score_matrix  # noqa: E402


class HorizonBenchProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task_set = TaskSetIdentity(
            source_name="public-long-horizon",
            source_revision="2026-08",
            split="heldout",
            source_sha256="a" * 64,
            task_set_sha256="b" * 64,
            task_ids=("task-03", "task-01", "task-02"),
        )
        self.model = ModelProfile(
            model_id="demo-model",
            provider="demo-provider",
            model="demo-1",
            model_revision="provider-release-1",
            decoding={"max_tokens": 512, "temperature": 0},
        )
        self.condition = EvaluationCondition(
            condition_id="H",
            policy_revision="git:fac0bd8",
            configuration={"semantic_memory_limit": 4, "state_anchor": True},
        )
        self.prompt = PromptArtifact("system-v1", "You are a durable task agent.")

    def _manifest(self) -> RunManifest:
        return RunManifest(
            run_id="demo__h__seed-7__bbbbbbbbbbbb",
            model=self.model,
            condition=self.condition,
            task_set=self.task_set,
            prompt=self.prompt,
            seed=7,
            runtime_revision="git:fac0bd8",
            checkpoint_cadence=2,
            fault_schedule=[{"after_boundary": 3, "kind": "restart"}],
            metadata={"memory_catalog_sha256": "c" * 64},
        )

    def test_manifest_hash_is_canonical_and_self_validating(self) -> None:
        first = self._manifest()
        equivalent = RunManifest(
            run_id=first.run_id,
            model=ModelProfile(
                "demo-model",
                "demo-provider",
                "demo-1",
                "provider-release-1",
                {"temperature": 0, "max_tokens": 512},
            ),
            condition=EvaluationCondition(
                "H",
                "git:fac0bd8",
                {"state_anchor": True, "semantic_memory_limit": 4},
            ),
            task_set=TaskSetIdentity(
                "public-long-horizon",
                "2026-08",
                "heldout",
                "a" * 64,
                "b" * 64,
                ("task-02", "task-01", "task-03"),
            ),
            prompt=self.prompt,
            seed=7,
            runtime_revision="git:fac0bd8",
            checkpoint_cadence=2,
            fault_schedule=[{"kind": "restart", "after_boundary": 3}],
            metadata={"memory_catalog_sha256": "c" * 64},
        )
        self.assertEqual(first.manifest_sha256, equivalent.manifest_sha256)
        self.assertEqual(first.task_set.task_ids, ("task-01", "task-02", "task-03"))

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(first.as_dict()), encoding="utf-8")
            self.assertEqual(load_manifest(path).manifest_sha256, first.manifest_sha256)

            tampered = first.as_dict()
            tampered["condition"]["configuration"]["semantic_memory_limit"] = 0
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ManifestValidationError, "does not match"):
                load_manifest(path)

    def test_manifest_json_controls_are_immutably_frozen(self) -> None:
        manifest = self._manifest()
        with self.assertRaises(TypeError):
            self.model.decoding["temperature"] = 1  # type: ignore[index]
        with self.assertRaises(TypeError):
            manifest.metadata["new"] = "value"  # type: ignore[index]
        with self.assertRaises(TypeError):
            manifest.fault_schedule[0]["kind"] = "different"  # type: ignore[index]
        self.assertEqual(manifest.manifest_sha256, self._manifest().manifest_sha256)

    def test_matrix_covers_every_model_condition_and_seed_once(self) -> None:
        second_model = ModelProfile(
            "second-model", "second-provider", "second", "revision-2", {"temperature": 0.2}
        )
        control = EvaluationCondition("S-off", "git:fac0bd8", {"semantic_memory_limit": 0})
        manifests = build_cross_model_matrix(
            models=[second_model, self.model],
            conditions=[control, self.condition],
            task_set=self.task_set,
            prompt=self.prompt,
            seeds=[11, 7],
            runtime_revision="git:fac0bd8",
            checkpoint_cadence=3,
            fault_schedule=[],
            metadata={"dataset_license": "research"},
        )
        self.assertEqual(len(manifests), 8)
        self.assertEqual(len({item.run_id for item in manifests}), 8)
        self.assertTrue(all(item.task_set == self.task_set for item in manifests))
        self.assertEqual(
            [(item.model.model_id, item.condition.condition_id, item.seed) for item in manifests],
            sorted(
                (item.model.model_id, item.condition.condition_id, item.seed)
                for item in manifests
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.jsonl"
            write_manifest_jsonl(path, manifests)
            loaded = read_manifest_jsonl(path)
        self.assertEqual([item.as_dict() for item in loaded], [item.as_dict() for item in manifests])

    def test_matrix_rejects_duplicate_dimensions(self) -> None:
        with self.assertRaisesRegex(ManifestValidationError, "model_id values must be unique"):
            build_cross_model_matrix(
                models=[self.model, self.model],
                conditions=[self.condition],
                task_set=self.task_set,
                prompt=self.prompt,
                seeds=[1],
                runtime_revision="git:fac0bd8",
            )

    def test_generated_run_id_changes_when_frozen_controls_change(self) -> None:
        baseline = build_cross_model_matrix(
            models=[self.model],
            conditions=[self.condition],
            task_set=self.task_set,
            prompt=self.prompt,
            seeds=[1],
            runtime_revision="git:fac0bd8",
        )[0]
        changed_prompt = build_cross_model_matrix(
            models=[self.model],
            conditions=[self.condition],
            task_set=self.task_set,
            prompt=PromptArtifact("system-v2", "Changed prompt"),
            seeds=[1],
            runtime_revision="git:fac0bd8",
        )[0]
        changed_condition = build_cross_model_matrix(
            models=[self.model],
            conditions=[
                EvaluationCondition("H", "git:fac0bd8", {"semantic_memory_limit": 0})
            ],
            task_set=self.task_set,
            prompt=self.prompt,
            seeds=[1],
            runtime_revision="git:fac0bd8",
        )[0]
        self.assertNotEqual(baseline.run_id, changed_prompt.run_id)
        self.assertNotEqual(baseline.run_id, changed_condition.run_id)

    def test_comparable_matrix_rejects_mixed_frozen_controls(self) -> None:
        first = self._manifest()
        same_controls = RunManifest(
            "other-run",
            ModelProfile("other", "provider", "other-model", "v1", {"temperature": 0}),
            self.condition,
            self.task_set,
            self.prompt,
            8,
            "git:fac0bd8",
            2,
            [{"after_boundary": 3, "kind": "restart"}],
            {},
        )
        validate_comparable_matrix([first, same_controls])
        changed_prompt = RunManifest(
            "changed-prompt",
            same_controls.model,
            self.condition,
            self.task_set,
            PromptArtifact("system-v2", "A different system prompt."),
            9,
            "git:fac0bd8",
            2,
            [{"after_boundary": 3, "kind": "restart"}],
            {},
        )
        with self.assertRaisesRegex(ManifestValidationError, "same prompt content"):
            validate_comparable_matrix([first, changed_prompt])
        with self.assertRaisesRegex(ManifestValidationError, "seeds must be unique"):
            build_cross_model_matrix(
                models=[self.model],
                conditions=[self.condition],
                task_set=self.task_set,
                prompt=self.prompt,
                seeds=[1, 1],
                runtime_revision="git:fac0bd8",
            )

    def test_result_task_validation_requires_exact_heldout_set(self) -> None:
        results = [
            EpisodeResult("task-01", "goal", True, True, 0),
            EpisodeResult("task-02", "goal", True, True, 0),
            EpisodeResult("task-03", "goal", True, True, 0),
        ]
        validate_result_task_ids(results, self._manifest())

        with self.assertRaisesRegex(ManifestValidationError, "missing task-03; unexpected task-04"):
            validate_result_task_ids(
                results[:-1] + [EpisodeResult("task-04", "goal", True, True, 0)], self._manifest()
            )
        with self.assertRaisesRegex(ManifestValidationError, "duplicate task ids"):
            validate_result_task_ids(results + [results[0]], self._manifest())

    def test_score_run_binds_metrics_to_single_manifest_or_matrix_row(self) -> None:
        results = [
            EpisodeResult("task-01", "goal", True, True, 0, tokens=10),
            EpisodeResult("task-02", "goal", True, True, 0, tokens=10),
            EpisodeResult("task-03", "goal", True, True, 0, tokens=10),
        ]
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_path = directory / "outcomes.jsonl"
            manifest_path = directory / "manifest.json"
            matrix_path = directory / "matrix.jsonl"
            write_jsonl(result_path, results)
            manifest_path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
            write_manifest_jsonl(matrix_path, [manifest])

            standalone = score_run(result_path, manifest_path=manifest_path)
            from_matrix = score_run(
                result_path, manifest_matrix_path=matrix_path, run_id=manifest.run_id
            )

        self.assertEqual(standalone, from_matrix)
        self.assertEqual(standalone["manifest"]["run_id"], manifest.run_id)
        self.assertEqual(standalone["metrics"]["task_count"], 3)
        self.assertEqual(standalone["metrics"]["token_consumption"], 30)

    def test_score_run_without_manifest_preserves_legacy_metric_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "outcomes.jsonl"
            write_jsonl(result_path, [EpisodeResult("legacy", "goal", True, True, 0, tokens=4)])
            report = score_run(result_path)
        self.assertEqual(report["task_count"], 1)
        self.assertEqual(report["token_consumption"], 4)
        self.assertNotIn("metrics", report)

    def test_score_run_rejects_manifest_and_result_task_set_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_path = directory / "outcomes.jsonl"
            manifest_path = directory / "manifest.json"
            write_jsonl(result_path, [EpisodeResult("not-frozen", "goal", True, True, 0)])
            manifest_path.write_text(json.dumps(self._manifest().as_dict()), encoding="utf-8")
            with self.assertRaisesRegex(ManifestValidationError, "result task ids do not match"):
                score_run(result_path, manifest_path=manifest_path)

    def test_plan_matrix_freezes_external_heldout_tasks_and_run_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tasks = directory / "public-tasks.jsonl"
            models = directory / "models.json"
            conditions = directory / "conditions.json"
            prompt = directory / "system.txt"
            output = directory / "matrix.jsonl"
            tasks.write_text(
                "\n".join(
                    (
                        json.dumps(
                            {
                                "id": "development-01",
                                "category": "goal",
                                "goal": "development only",
                                "constraint": "do not score",
                                "split": "development",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "heldout-02",
                                "category": "recovery",
                                "goal": "resume safely",
                                "constraint": "keep identifier",
                                "split": "heldout",
                                "source_url": "https://example.invalid/tasks",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "heldout-01",
                                "category": "goal",
                                "goal": "retain goal",
                                "constraint": "keep constraints",
                                "split": "heldout",
                            }
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            models.write_text(json.dumps({"models": [self.model.as_dict()]}), encoding="utf-8")
            conditions.write_text(
                json.dumps({"conditions": [self.condition.as_dict()]}), encoding="utf-8"
            )
            prompt.write_text(self.prompt.text, encoding="utf-8")
            args = make_parser().parse_args(
                [
                    "--tasks",
                    str(tasks),
                    "--source-name",
                    "public-long-horizon",
                    "--source-revision",
                    "release-42",
                    "--models",
                    str(models),
                    "--conditions",
                    str(conditions),
                    "--prompt",
                    str(prompt),
                    "--prompt-revision",
                    "system-v1",
                    "--runtime-revision",
                    "git:fac0bd8",
                    "--seeds",
                    "3,9",
                    "--checkpoint-cadence",
                    "2",
                    "--output",
                    str(output),
                ]
            )
            manifests = plan_matrix(args)
            loaded = read_manifest_jsonl(output)

            self.assertEqual(len(manifests), 2)
            self.assertEqual([item.as_dict() for item in loaded], [item.as_dict() for item in manifests])
            self.assertEqual(loaded[0].task_set.split, "heldout")
            self.assertEqual(loaded[0].task_set.task_ids, ("heldout-01", "heldout-02"))
            self.assertEqual(loaded[0].prompt.text, self.prompt.text)
            self.assertEqual({item.seed for item in loaded}, {3, 9})
            with self.assertRaisesRegex(ManifestValidationError, "refusing to overwrite"):
                plan_matrix(args)

    def test_score_matrix_scores_all_frozen_rows_and_aggregates_seeds(self) -> None:
        other_model = ModelProfile("other-model", "provider", "other", "v1", {"temperature": 0})
        manifests = build_cross_model_matrix(
            models=[self.model, other_model],
            conditions=[self.condition],
            task_set=self.task_set,
            prompt=self.prompt,
            seeds=[1, 2],
            runtime_revision="git:fac0bd8",
            checkpoint_cadence=1,
            fault_schedule=[],
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            matrix_path = directory / "matrix.jsonl"
            results_dir = directory / "results"
            write_manifest_jsonl(matrix_path, manifests)
            for manifest in manifests:
                success = manifest.seed == 1
                write_jsonl(
                    results_dir / (manifest.run_id + ".jsonl"),
                    [
                        EpisodeResult(task_id, "goal", success, True, 0, tokens=10)
                        for task_id in manifest.task_set.task_ids
                    ],
                )
            report = score_matrix(matrix_path, results_dir)

            self.assertEqual(report["run_count"], 4)
            self.assertEqual(len(report["runs"]), 4)
            self.assertEqual(len(report["summaries"]), 2)
            for summary in report["summaries"]:
                self.assertEqual(summary["run_count"], 2)
                task_success = summary["metrics"]["task_success_rate"]
                self.assertEqual(task_success["observations"], 2)
                self.assertEqual(task_success["mean"], 0.5)
                self.assertGreater(task_success["sample_stddev"], 0)

            (results_dir / (manifests[0].run_id + ".jsonl")).unlink()
            with self.assertRaisesRegex(ManifestValidationError, "missing result JSONL"):
                score_matrix(matrix_path, results_dir)

    def test_direct_script_cli_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tasks = directory / "tasks.jsonl"
            models = directory / "models.json"
            conditions = directory / "conditions.json"
            prompt = directory / "prompt.txt"
            matrix = directory / "matrix.jsonl"
            results_dir = directory / "results"
            tasks.write_text(
                json.dumps(
                    {
                        "id": "heldout-01",
                        "category": "goal_retention",
                        "goal": "keep goal",
                        "constraint": "keep constraint",
                        "split": "heldout",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            models.write_text(json.dumps([self.model.as_dict()]), encoding="utf-8")
            conditions.write_text(json.dumps([self.condition.as_dict()]), encoding="utf-8")
            prompt.write_text(self.prompt.text, encoding="utf-8")
            plan = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "horizonbench" / "plan_matrix.py"),
                    "--tasks",
                    str(tasks),
                    "--source-name",
                    "fixture-source",
                    "--source-revision",
                    "r1",
                    "--models",
                    str(models),
                    "--conditions",
                    str(conditions),
                    "--prompt",
                    str(prompt),
                    "--prompt-revision",
                    "system-v1",
                    "--runtime-revision",
                    "git:fac0bd8",
                    "--output",
                    str(matrix),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(plan.stdout)["manifest_count"], 1)
            manifest = read_manifest_jsonl(matrix)[0]
            write_jsonl(
                results_dir / (manifest.run_id + ".jsonl"),
                [EpisodeResult("heldout-01", "goal_retention", True, True, 0, tokens=7)],
            )
            scored = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "horizonbench" / "score_matrix.py"),
                    "--matrix",
                    str(matrix),
                    "--results-dir",
                    str(results_dir),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        report = json.loads(scored.stdout)
        self.assertEqual(report["run_count"], 1)
        self.assertEqual(report["runs"][0]["metrics"]["token_consumption"], 7)

    def test_protocol_rejects_nonportable_values_and_bad_hashes(self) -> None:
        with self.assertRaisesRegex(ManifestValidationError, "lowercase SHA-256"):
            TaskSetIdentity("source", "v1", "heldout", "not-a-hash", "b" * 64, ("task",))
        with self.assertRaisesRegex(ManifestValidationError, "JSON values"):
            ModelProfile("model", "provider", "name", "v1", {"bad": {1, 2}})
        with self.assertRaisesRegex(ManifestValidationError, "fault_schedule must be a JSON list"):
            RunManifest(
                "run",
                self.model,
                self.condition,
                self.task_set,
                self.prompt,
                None,
                "git:fac0bd8",
                1,
                {"not": "a list"},
                {},
            )

    def test_protocol_rejects_unhashed_top_level_configuration(self) -> None:
        model = self.model.as_dict()
        model["provider_retry_policy"] = "hidden"
        with self.assertRaisesRegex(ManifestValidationError, "unrecognized fields"):
            ModelProfile.from_mapping(model)

        manifest = self._manifest().as_dict()
        manifest["untracked_execution_setting"] = True
        with self.assertRaisesRegex(ManifestValidationError, "unrecognized fields"):
            RunManifest.from_mapping(manifest)

    def test_summary_does_not_repeat_prompt_or_policy_configuration(self) -> None:
        summary = manifest_summary(self._manifest())
        self.assertEqual(summary["task_count"], 3)
        self.assertEqual(summary["condition_id"], "H")
        self.assertNotIn("prompt", summary)
        self.assertNotIn("configuration", summary)


if __name__ == "__main__":
    unittest.main()

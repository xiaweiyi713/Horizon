"""Narrow unit tests for the HorizonBench episode executor contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.adapters import BenchmarkTask  # noqa: E402
from benchmarks.horizonbench.execution import (  # noqa: E402
    EpisodeExecutor,
    ExecutionContext,
    ExecutionValidationError,
    load_executor,
    normalize_episode_result,
)
from benchmarks.horizonbench.protocol import (  # noqa: E402
    EvaluationCondition,
    ModelProfile,
    PromptArtifact,
    RunManifest,
    TaskSetIdentity,
)
from benchmarks.horizonbench.score import EpisodeResult  # noqa: E402


def _result_mapping(task: BenchmarkTask, **overrides: Any) -> dict[str, Any]:
    payload = {
        "task_id": task.task_id,
        "category": task.category,
        "success": True,
        "goal_retained": True,
        "constraint_violations": 0,
    }
    payload.update(overrides)
    return payload


def _ok_executor(context: ExecutionContext) -> EpisodeResult:
    return EpisodeResult(
        task_id=context.task.task_id,
        category=context.task.category,
        success=True,
        goal_retained=True,
        constraint_violations=0,
    )


def _mapping_executor(context: ExecutionContext) -> Mapping[str, Any]:
    return MappingProxyType(_result_mapping(context.task))


class _ExecutorNamespace:
    @staticmethod
    def run(context: ExecutionContext) -> EpisodeResult:
        return _ok_executor(context)


NOT_CALLABLE = "a value"


class HorizonBenchExecutionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = BenchmarkTask(
            task_id="task-01",
            category="goal_retention",
            goal="Keep the original goal",
            constraint="Do not invent a new constraint",
            split="heldout",
            metadata={"source": "fixture"},
        )
        self.manifest = RunManifest(
            run_id="demo__h__seed-7__bbbbbbbbbbbb",
            model=ModelProfile(
                "demo-model",
                "demo-provider",
                "demo-1",
                "provider-release-1",
                {"max_tokens": 512, "temperature": 0},
            ),
            condition=EvaluationCondition(
                "H",
                "git:fac0bd8",
                {"semantic_memory_limit": 4, "state_anchor": True},
            ),
            task_set=TaskSetIdentity(
                "public-long-horizon",
                "2026-08",
                "heldout",
                "a" * 64,
                "b" * 64,
                ("task-01", "task-02"),
            ),
            prompt=PromptArtifact("system-v1", "You are a durable task agent."),
            seed=7,
            runtime_revision="git:fac0bd8",
            checkpoint_cadence=1,
            fault_schedule=[],
            metadata={},
        )

    def _context(self, **overrides: Any) -> ExecutionContext:
        values = {"manifest": self.manifest, "task": self.task, "attempt": 1}
        values.update(overrides)
        return ExecutionContext(**values)

    def test_error_type_and_immutable_matching_context(self) -> None:
        self.assertTrue(issubclass(ExecutionValidationError, ValueError))
        context = ExecutionContext(manifest=self.manifest, task=self.task)
        self.assertEqual(context.attempt, 1)
        self.assertIs(context.manifest, self.manifest)
        self.assertIs(context.task, self.task)
        with self.assertRaises(Exception):
            context.attempt = 2  # type: ignore[misc]

        observed = _ok_executor(context)
        self.assertEqual(normalize_episode_result(observed, context), observed)
        from_mapping = normalize_episode_result(_result_mapping(self.task), context)
        self.assertEqual(from_mapping.task_id, self.task.task_id)
        self.assertEqual(from_mapping.category, self.task.category)

        loaded = load_executor("{}:_ok_executor".format(__name__))
        self.assertIs(loaded, _ok_executor)
        nested = load_executor("{}:_ExecutorNamespace.run".format(__name__))
        self.assertEqual(normalize_episode_result(nested(context), context), observed)
        self.assertEqual(
            normalize_episode_result(_mapping_executor(context), context).task_id,
            self.task.task_id,
        )
        self.assertTrue(callable(EpisodeExecutor))

    def test_normalize_rejects_wrong_output_types_and_identity(self) -> None:
        context = self._context()
        for value in (None, "episode", 1, ["task-01"], object()):
            with self.assertRaisesRegex(ExecutionValidationError, "EpisodeResult or a mapping"):
                normalize_episode_result(value, context)

        with self.assertRaisesRegex(ExecutionValidationError, "not a valid EpisodeResult"):
            normalize_episode_result({"task_id": self.task.task_id}, context)

        with self.assertRaisesRegex(ExecutionValidationError, "task_id"):
            normalize_episode_result(
                EpisodeResult("task-02", self.task.category, True, True, 0),
                context,
            )
        with self.assertRaisesRegex(ExecutionValidationError, "category"):
            normalize_episode_result(
                _result_mapping(self.task, category="constraint_retention"),
                context,
            )

    def test_context_rejects_bad_attempt_identity_and_split(self) -> None:
        for attempt in (True, False, 0, -1, 1.5, "1", None):
            with self.assertRaisesRegex(ExecutionValidationError, "positive integer"):
                self._context(attempt=attempt)

        foreign = BenchmarkTask(
            task_id="task-99",
            category=self.task.category,
            goal=self.task.goal,
            constraint=self.task.constraint,
            split="heldout",
            metadata={},
        )
        with self.assertRaisesRegex(ExecutionValidationError, "not in manifest.task_set.task_ids"):
            self._context(task=foreign)

        wrong_split = BenchmarkTask(
            task_id=self.task.task_id,
            category=self.task.category,
            goal=self.task.goal,
            constraint=self.task.constraint,
            split="public",
            metadata={},
        )
        with self.assertRaisesRegex(ExecutionValidationError, "does not match manifest.task_set.split"):
            self._context(task=wrong_split)

        with self.assertRaisesRegex(ExecutionValidationError, "manifest must be a RunManifest"):
            ExecutionContext(manifest="manifest", task=self.task)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ExecutionValidationError, "task must be a BenchmarkTask"):
            ExecutionContext(manifest=self.manifest, task="task")  # type: ignore[arg-type]

    def test_load_executor_rejects_bad_specs_and_import_errors(self) -> None:
        for spec in (None, 1, "", "json.dumps", "json:", ":dumps", "json:dumps:extra", "json::dumps"):
            with self.assertRaisesRegex(ExecutionValidationError, "module:attribute|dotted identifier"):
                load_executor(spec)

        with self.assertRaisesRegex(ExecutionValidationError, "cannot import executor module"):
            load_executor("horizonbench_missing_executor_module:factory")
        with self.assertRaisesRegex(ExecutionValidationError, "cannot resolve executor attribute"):
            load_executor("json:not_a_real_executor_attribute")
        with self.assertRaisesRegex(ExecutionValidationError, "cannot resolve executor attribute"):
            load_executor("json:dumps.not_a_real_attribute")
        with self.assertRaisesRegex(ExecutionValidationError, "is not callable"):
            load_executor("{}:NOT_CALLABLE".format(__name__))
        with self.assertRaisesRegex(ExecutionValidationError, "is not callable"):
            load_executor("json:__name__")


if __name__ == "__main__":
    unittest.main()

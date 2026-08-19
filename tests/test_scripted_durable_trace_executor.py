"""Unit tests for the deterministic durable-trace fixture executor."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.execution import load_executor  # noqa: E402
from benchmarks.horizonbench import scripted_durable_trace_executor as plugin  # noqa: E402
from horizon_agent import BenchmarkExecutionError  # noqa: E402


class ScriptedDurableTraceExecutorTests(unittest.TestCase):
    def test_provider_deeply_detaches_frozen_fixture_actions(self) -> None:
        actions = (
            MappingProxyType(
                {
                    "action": MappingProxyType(
                        {"type": "finish", "data": MappingProxyType({"nested": ("value",)})}
                    )
                }
            ),
        )
        context = SimpleNamespace(task=SimpleNamespace(metadata=MappingProxyType({"fixture_actions": actions})))

        provider = plugin._provider(context)
        response = provider.complete(())

        self.assertEqual(
            json.loads(response.content),
            {"action": {"type": "finish", "data": {"nested": ["value"]}}},
        )

    def test_provider_rejects_missing_or_invalid_actions_before_runtime_use(self) -> None:
        for metadata in ({}, {"fixture_actions": []}, {"fixture_actions": ["not-an-action"]}):
            context = SimpleNamespace(task=SimpleNamespace(metadata=metadata))
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(BenchmarkExecutionError, "fixture_actions|fixture action"):
                    plugin._provider(context)
        with self.assertRaisesRegex(BenchmarkExecutionError, "NaN or infinity"):
            plugin._detach_json(float("nan"), "fixture action")

    def test_preflight_validates_fixture_actions_and_trace_without_starting_runtime(self) -> None:
        context = SimpleNamespace(
            task=SimpleNamespace(
                task_id="trace-01",
                category="goal_retention",
                goal="Preserve the durable objective.",
                constraint="Do not replace the operation ID.",
                metadata={
                    "initial_plan": ["finish fixture"],
                    "horizon_trace_expectation": {"required_event_types": ["checkpoint_created"]},
                    "fixture_actions": [{"action": {"type": "finish", "data": {}}}],
                },
            ),
            manifest=SimpleNamespace(
                condition=SimpleNamespace(configuration={"agent": {"max_steps": 2}}),
                checkpoint_cadence=1,
                fault_schedule=[],
                prompt=SimpleNamespace(text="Return one durable action."),
            ),
        )

        report = plugin.preflight(context)

        self.assertEqual(report["executor"], "scripted_durable_trace_fixture")
        self.assertFalse(report["model_call"])
        self.assertEqual(report["fixture_action_count"], 1)
        self.assertEqual(report["trace_expectation"]["required_event_types"], ["checkpoint_created"])

    def test_plugin_uses_the_standard_matrix_executor_import_shape(self) -> None:
        self.assertIs(
            load_executor("benchmarks.horizonbench.scripted_durable_trace_executor:execute"),
            plugin.execute,
        )
        self.assertIs(plugin.execute.preflight, plugin.preflight)


if __name__ == "__main__":
    unittest.main()

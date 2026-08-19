"""Tests for the opt-in manifest-bound OpenAI-compatible trace executor."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench import openai_compatible_trace_executor as plugin  # noqa: E402
from benchmarks.horizonbench.execution import load_executor  # noqa: E402
from horizon_agent import BenchmarkExecutionError  # noqa: E402


def _context(*, provider: str = "openai-compatible", decoding=None, seed=17):
    return SimpleNamespace(
        manifest=SimpleNamespace(
            model=SimpleNamespace(
                provider=provider,
                model="fixture-model",
                decoding={"temperature": 0.2, "stop": ("END",)} if decoding is None else decoding,
            ),
            seed=seed,
        )
    )


def _trace_context(*, provider: str = "openai-compatible", decoding=None, seed=17):
    return SimpleNamespace(
        task=SimpleNamespace(
            task_id="trace-01",
            category="goal_retention",
            goal="Preserve the frozen durable objective.",
            constraint="Do not replace the recorded operation ID.",
            metadata={
                "initial_plan": ["inspect durable state"],
                "horizon_trace_expectation": {
                    "required_event_types": ["checkpoint_created"],
                    "require_checkpoint": True,
                },
            },
        ),
        manifest=SimpleNamespace(
            model=SimpleNamespace(
                provider=provider,
                model="fixture-model",
                decoding={"temperature": 0.2, "stop": ("END",)} if decoding is None else decoding,
            ),
            seed=seed,
            condition=SimpleNamespace(configuration={"agent": {"max_steps": 3}}),
            checkpoint_cadence=1,
            fault_schedule=[],
            prompt=SimpleNamespace(text="Return one valid durable action."),
        ),
    )


class OpenAICompatibleTraceExecutorTests(unittest.TestCase):
    def test_effective_decoding_uses_the_frozen_manifest_seed_once(self) -> None:
        context = _context()
        self.assertEqual(
            plugin._effective_decoding(context),
            {"temperature": 0.2, "stop": ("END",), "seed": 17},
        )

        with self.assertRaisesRegex(BenchmarkExecutionError, "conflicts"):
            plugin._effective_decoding(_context(decoding={"seed": 9}, seed=17))
        with self.assertRaisesRegex(BenchmarkExecutionError, "integer or null"):
            plugin._effective_decoding(_context(seed=True))

    def test_provider_accepts_only_declared_openai_compatible_profiles(self) -> None:
        with patch.dict(os.environ, {"HORIZON_BENCH_API_KEY": "fixture-key"}, clear=False):
            provider = plugin._provider(_context(provider="openai_compatible"))
        self.assertEqual(provider.model, "fixture-model")
        self.assertEqual(dict(provider.decoding or {}), {"temperature": 0.2, "stop": ("END",), "seed": 17})
        with self.assertRaisesRegex(BenchmarkExecutionError, "model.provider"):
            plugin._provider(_context(provider="other"))

    def test_runtime_url_is_configurable_without_connecting(self) -> None:
        with patch.dict(os.environ, {"HORIZON_BENCH_RUNTIME_URL": "http://127.0.0.1:9876/"}, clear=False):
            client = plugin._runtime_client(_context())
        self.assertEqual(client.base_url, "http://127.0.0.1:9876")

    def test_preflight_checks_static_trace_and_credential_config_without_connecting(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HORIZON_BENCH_API_KEY": "fixture-key",
                "HORIZON_BENCH_RUNTIME_URL": "http://127.0.0.1:9876/",
                "HORIZON_BENCH_API_BASE_URL": "http://127.0.0.1:9999/v1/",
            },
            clear=True,
        ):
            report = plugin.preflight(_trace_context())

        self.assertEqual(report["executor"], "openai_compatible_durable_trace")
        self.assertEqual(report["credential_source"], "HORIZON_BENCH_API_KEY")
        self.assertTrue(report["runtime_url_configured"])
        self.assertTrue(report["api_base_url_configured"])
        self.assertEqual(report["decoding_keys"], ["seed", "stop", "temperature"])
        self.assertEqual(report["trace_expectation"]["required_event_types"], ["checkpoint_created"])

    def test_preflight_rejects_missing_model_credential_before_execution(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(BenchmarkExecutionError, "API_KEY"):
                plugin.preflight(_trace_context())

    def test_plugin_uses_the_standard_matrix_executor_import_shape(self) -> None:
        self.assertIs(
            load_executor("benchmarks.horizonbench.openai_compatible_trace_executor:execute"),
            plugin.execute,
        )
        self.assertIs(plugin.execute.preflight, plugin.preflight)


if __name__ == "__main__":
    unittest.main()

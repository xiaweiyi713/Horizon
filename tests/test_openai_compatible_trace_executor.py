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

    def test_plugin_uses_the_standard_matrix_executor_import_shape(self) -> None:
        self.assertIs(
            load_executor("benchmarks.horizonbench.openai_compatible_trace_executor:execute"),
            plugin.execute,
        )


if __name__ == "__main__":
    unittest.main()

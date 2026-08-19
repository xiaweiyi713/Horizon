"""Opt-in real-model executor for HorizonBench durable-trace tasks.

The module is deliberately inert on import. ``execute_matrix.py`` calls
``execute`` once per frozen task; only then does it construct an OpenAI-
compatible provider and contact the configured local Horizon runtime. It is
appropriate only for tasks that declare ``horizon_trace_expectation`` and have
an empty fault schedule. Public artifact/domain tasks need their own objective
environment executor instead.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from horizon_agent import (
    BenchmarkExecutionError,
    DurableAgentEpisodeExecutor,
    DurableTraceJudge,
    HorizonClient,
)
from horizon_agent.providers import OpenAICompatibleProvider


_SUPPORTED_PROVIDER_IDS = frozenset(("openai-compatible", "openai_compatible"))
_DEFAULT_RUNTIME_URL = "http://127.0.0.1:8787"
_DEFAULT_API_BASE_URL = "https://api.openai.com/v1"


def _runtime_client(_context: Any) -> HorizonClient:
    return HorizonClient(os.getenv("HORIZON_BENCH_RUNTIME_URL", _DEFAULT_RUNTIME_URL))


def _provider(context: Any) -> OpenAICompatibleProvider:
    manifest = getattr(context, "manifest", None)
    model = getattr(manifest, "model", None)
    provider_id = getattr(model, "provider", None)
    if not isinstance(provider_id, str) or provider_id.casefold() not in _SUPPORTED_PROVIDER_IDS:
        raise BenchmarkExecutionError(
            "openai_compatible_trace_executor requires model.provider to be "
            "'openai-compatible' or 'openai_compatible'"
        )
    model_name = getattr(model, "model", None)
    if not isinstance(model_name, str) or not model_name.strip():
        raise BenchmarkExecutionError("execution context must expose a non-empty manifest.model.model")
    return OpenAICompatibleProvider(
        model_name,
        api_key=os.getenv("HORIZON_BENCH_API_KEY"),
        base_url=os.getenv("HORIZON_BENCH_API_BASE_URL", _DEFAULT_API_BASE_URL),
        decoding=_effective_decoding(context),
    )


def _effective_decoding(context: Any) -> dict[str, Any]:
    """Merge the manifest's separate seed field into provider decoding controls."""

    manifest = getattr(context, "manifest", None)
    model = getattr(manifest, "model", None)
    decoding = getattr(model, "decoding", None)
    if not isinstance(decoding, Mapping):
        raise BenchmarkExecutionError("execution context must expose manifest.model.decoding")
    effective = dict(decoding)
    seed = getattr(manifest, "seed", None)
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise BenchmarkExecutionError("manifest.seed must be an integer or null")
        configured_seed = effective.get("seed")
        if configured_seed is not None and configured_seed != seed:
            raise BenchmarkExecutionError(
                "manifest.seed conflicts with manifest.model.decoding.seed; freeze the value once"
            )
        effective["seed"] = seed
    return effective


_EXECUTOR = DurableAgentEpisodeExecutor(
    client_factory=_runtime_client,
    provider_factory=_provider,
    judge=DurableTraceJudge(),
)


def execute(context: Any) -> Any:
    """Execute one frozen durable-trace episode and return an observed result."""

    return _EXECUTOR(context)


__all__ = ["execute"]

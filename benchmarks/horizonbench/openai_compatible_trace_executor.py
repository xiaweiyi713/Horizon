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
    return HorizonClient(_configured_url("HORIZON_BENCH_RUNTIME_URL", _DEFAULT_RUNTIME_URL))


def _model_profile(context: Any) -> tuple[str, str]:
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
    return provider_id, model_name


def _provider(context: Any) -> OpenAICompatibleProvider:
    _provider_id, model_name = _model_profile(context)
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


def _configured_api_key() -> tuple[str, str]:
    """Resolve the same credential fallback the provider will use, without exposing it."""

    value = os.getenv("HORIZON_BENCH_API_KEY")
    source = "HORIZON_BENCH_API_KEY"
    if value is None:
        value = os.getenv("OPENAI_API_KEY")
        source = "OPENAI_API_KEY"
    if not value:
        raise BenchmarkExecutionError(
            "configure HORIZON_BENCH_API_KEY or OPENAI_API_KEY before executing a model-backed matrix"
        )
    return value, source


def _configured_url(name: str, default: str) -> str:
    value = os.getenv(name, default)
    if not value.strip():
        raise BenchmarkExecutionError("{} must be a non-empty URL".format(name))
    return value.rstrip("/")


_JUDGE = DurableTraceJudge()
_EXECUTOR = DurableAgentEpisodeExecutor(
    client_factory=_runtime_client,
    provider_factory=_provider,
    judge=_JUDGE,
)


def execute(context: Any) -> Any:
    """Execute one frozen durable-trace episode and return an observed result."""

    return _EXECUTOR(context)


def preflight(context: Any) -> dict[str, Any]:
    """Validate a real-model trace row without connecting to any endpoint."""

    report = _EXECUTOR.preflight(context)
    trace_report = _JUDGE.preflight(context)
    provider_id, model_name = _model_profile(context)
    decoding = _effective_decoding(context)
    api_key, credential_source = _configured_api_key()
    api_base_url = _configured_url("HORIZON_BENCH_API_BASE_URL", _DEFAULT_API_BASE_URL)
    runtime_url = _configured_url("HORIZON_BENCH_RUNTIME_URL", _DEFAULT_RUNTIME_URL)
    # Constructor-only validation freezes the provider boundary but never calls
    # ``complete`` or opens a runtime connection.
    OpenAICompatibleProvider(
        model_name,
        api_key=api_key,
        base_url=api_base_url,
        decoding=decoding,
    )
    return {
        **report,
        "executor": "openai_compatible_durable_trace",
        "provider": provider_id,
        "model": model_name,
        "credential_source": credential_source,
        # URLs can themselves contain credentials, so record only that their
        # validated configuration exists; never print them in a preflight log.
        "api_base_url_configured": bool(api_base_url),
        "runtime_url_configured": bool(runtime_url),
        "decoding_keys": sorted(decoding),
        "trace_expectation": trace_report,
    }


# ``load_executor`` returns the callable itself, so attach the optional
# protocol hook to the same object rather than asking every runner to import
# private plugin state.
execute.preflight = preflight  # type: ignore[attr-defined]


__all__ = ["execute", "preflight"]

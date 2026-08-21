"""Shared static configuration for OpenAI-compatible HorizonBench executors.

Keeping this boundary separate avoids two different task executors silently
freezing different provider/seed semantics.  The helpers construct clients and
providers only when called; importing this module opens no network connection.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from horizon_agent import BenchmarkExecutionError, HorizonClient
from horizon_agent.providers import OpenAICompatibleProvider


SUPPORTED_PROVIDER_IDS = frozenset(("openai-compatible", "openai_compatible"))
DEFAULT_RUNTIME_URL = "http://127.0.0.1:8787"
DEFAULT_API_BASE_URL = "https://api.openai.com/v1"


def model_profile(context: Any) -> tuple[str, str]:
    manifest = getattr(context, "manifest", None)
    model = getattr(manifest, "model", None)
    provider_id = getattr(model, "provider", None)
    if not isinstance(provider_id, str) or provider_id.casefold() not in SUPPORTED_PROVIDER_IDS:
        raise BenchmarkExecutionError(
            "OpenAI-compatible executor requires model.provider to be "
            "'openai-compatible' or 'openai_compatible'"
        )
    model_name = getattr(model, "model", None)
    if not isinstance(model_name, str) or not model_name.strip():
        raise BenchmarkExecutionError("execution context must expose a non-empty manifest.model.model")
    return provider_id, model_name


def effective_decoding(context: Any) -> dict[str, Any]:
    """Merge the frozen separate seed into provider decoding exactly once."""

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


def configured_api_key() -> tuple[str, str]:
    """Resolve a credential source without exposing its value."""

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


def configured_url(name: str, default: str) -> str:
    value = os.getenv(name, default)
    if not value.strip():
        raise BenchmarkExecutionError("{} must be a non-empty URL".format(name))
    return value.rstrip("/")


def runtime_client(_context: Any) -> HorizonClient:
    return HorizonClient(configured_url("HORIZON_BENCH_RUNTIME_URL", DEFAULT_RUNTIME_URL))


def provider(context: Any) -> OpenAICompatibleProvider:
    _provider_id, model_name = model_profile(context)
    return OpenAICompatibleProvider(
        model_name,
        api_key=os.getenv("HORIZON_BENCH_API_KEY"),
        base_url=os.getenv("HORIZON_BENCH_API_BASE_URL", DEFAULT_API_BASE_URL),
        decoding=effective_decoding(context),
    )


def preflight_provider(context: Any) -> dict[str, Any]:
    """Validate provider configuration without contacting either endpoint."""

    provider_id, model_name = model_profile(context)
    decoding = effective_decoding(context)
    api_key, credential_source = configured_api_key()
    api_base_url = configured_url("HORIZON_BENCH_API_BASE_URL", DEFAULT_API_BASE_URL)
    runtime_url = configured_url("HORIZON_BENCH_RUNTIME_URL", DEFAULT_RUNTIME_URL)
    OpenAICompatibleProvider(
        model_name,
        api_key=api_key,
        base_url=api_base_url,
        decoding=decoding,
    )
    return {
        "provider": provider_id,
        "model": model_name,
        "credential_source": credential_source,
        # These URLs can carry credentials; record only configuration state.
        "api_base_url_configured": bool(api_base_url),
        "runtime_url_configured": bool(runtime_url),
        "decoding_keys": sorted(decoding),
    }


__all__ = [
    "DEFAULT_API_BASE_URL",
    "DEFAULT_RUNTIME_URL",
    "SUPPORTED_PROVIDER_IDS",
    "configured_api_key",
    "configured_url",
    "effective_decoding",
    "model_profile",
    "preflight_provider",
    "provider",
    "runtime_client",
]

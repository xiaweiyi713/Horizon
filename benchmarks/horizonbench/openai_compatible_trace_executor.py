"""Opt-in real-model executor for HorizonBench durable-trace tasks.

The module is deliberately inert on import. ``execute_matrix.py`` calls
``execute`` once per frozen task; only then does it construct an OpenAI-
compatible provider and contact the configured local Horizon runtime. It is
appropriate only for tasks that declare ``horizon_trace_expectation`` and have
an empty fault schedule. Public artifact/domain tasks need their own objective
environment executor instead.
"""

from __future__ import annotations

from typing import Any

from horizon_agent import (
    DurableAgentEpisodeExecutor,
    DurableTraceJudge,
)
from .openai_compatible_support import (
    effective_decoding as _effective_decoding,
    preflight_provider as _preflight_provider,
    provider as _provider,
    runtime_client as _runtime_client,
)


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
    return {
        **report,
        "executor": "openai_compatible_durable_trace",
        **_preflight_provider(context),
        "trace_expectation": trace_report,
    }


# ``load_executor`` returns the callable itself, so attach the optional
# protocol hook to the same object rather than asking every runner to import
# private plugin state.
execute.preflight = preflight  # type: ignore[attr-defined]


__all__ = ["execute", "preflight"]

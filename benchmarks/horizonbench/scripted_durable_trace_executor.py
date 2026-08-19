"""No-network HorizonBench executor for the durable-trace integration fixture.

This executor is intentionally deterministic and must not be reported as an
empirical model result. It still exercises the real Python ``DurableAgent`` and
Rust HTTP runtime: each task supplies its fixture action sequence in immutable
task metadata, then ``DurableTraceJudge`` checks the resulting durable trace.
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping, Sequence

from horizon_agent import (
    BenchmarkExecutionError,
    DurableAgentEpisodeExecutor,
    DurableTraceJudge,
    HorizonClient,
)
from horizon_agent.providers import ScriptedProvider


_DEFAULT_RUNTIME_URL = "http://127.0.0.1:8787"
_FIXTURE_ACTIONS_KEY = "fixture_actions"


def _runtime_client(_context: Any) -> HorizonClient:
    return HorizonClient(os.getenv("HORIZON_BENCH_RUNTIME_URL", _DEFAULT_RUNTIME_URL))


def _provider(context: Any) -> ScriptedProvider:
    task = getattr(context, "task", None)
    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise BenchmarkExecutionError("benchmark task metadata must be an object")
    raw_actions = metadata.get(_FIXTURE_ACTIONS_KEY)
    if isinstance(raw_actions, (str, bytes)) or not isinstance(raw_actions, Sequence) or not raw_actions:
        raise BenchmarkExecutionError(
            "benchmark task metadata.fixture_actions must be a non-empty array of action objects"
        )
    actions: list[dict[str, Any]] = []
    for index, action in enumerate(raw_actions, 1):
        if not isinstance(action, Mapping):
            raise BenchmarkExecutionError("fixture action {} must be an object".format(index))
        # Task metadata is deliberately frozen into MappingProxy/tuple values.
        # ``ScriptedProvider`` serializes each response, so make a JSON-shaped
        # deep copy rather than leaking those immutable implementation types.
        detached = _detach_json(action, "fixture action {}".format(index))
        if not isinstance(detached, dict):  # Keeps the provider boundary narrow.
            raise BenchmarkExecutionError("fixture action {} must be an object".format(index))
        actions.append(detached)
    return ScriptedProvider(actions)


def _detach_json(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BenchmarkExecutionError("{} must not contain NaN or infinity".format(label))
        return value
    if isinstance(value, Mapping):
        detached: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BenchmarkExecutionError("{} has a non-string key".format(label))
            detached[key] = _detach_json(item, "{}.{}".format(label, key))
        return detached
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_detach_json(item, "{}[]".format(label)) for item in value]
    raise BenchmarkExecutionError("{} is not JSON-safe".format(label))


_EXECUTOR = DurableAgentEpisodeExecutor(
    client_factory=_runtime_client,
    provider_factory=_provider,
    judge=DurableTraceJudge(),
)


def execute(context: Any) -> Any:
    """Run one deterministic fixture trajectory against the real runtime."""

    return _EXECUTOR(context)


__all__ = ["execute"]

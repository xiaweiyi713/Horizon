"""No-network executor for artifact-workspace integration coverage only.

Fixture action sequences live in immutable task metadata.  They exercise the
real DurableAgent, durable AdapterRegistry boundaries, and artifact verifier,
but they are not a model result and must never be reported as one.
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping, Sequence

from horizon_agent import (
    BenchmarkExecutionError,
    DurableArtifactWorkspaceExecutor,
    HorizonClient,
)
from horizon_agent.providers import ScriptedProvider

from .artifact_workspace_support import artifact_root, workspace_factory


_DEFAULT_RUNTIME_URL = "http://127.0.0.1:8787"
_FIXTURE_ACTIONS_KEY = "fixture_actions"
_FIXTURE_RECOVERY_ACTIONS_KEY = "fixture_recovery_actions"


def _configured_runtime_url() -> str:
    value = os.getenv("HORIZON_BENCH_RUNTIME_URL", _DEFAULT_RUNTIME_URL)
    if not value.strip():
        raise BenchmarkExecutionError("HORIZON_BENCH_RUNTIME_URL must be a non-empty URL")
    return value.rstrip("/")


def _runtime_client(_context: Any) -> HorizonClient:
    return HorizonClient(_configured_runtime_url())


def _fixture_actions(context: Any, *, recovery: bool = False) -> list[dict[str, Any]]:
    task = getattr(context, "task", None)
    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise BenchmarkExecutionError("benchmark task metadata must be an object")
    key = _FIXTURE_RECOVERY_ACTIONS_KEY if recovery else _FIXTURE_ACTIONS_KEY
    raw = metadata.get(key)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise BenchmarkExecutionError(
            "benchmark task metadata.{} must be a non-empty array of action objects".format(key)
        )
    actions: list[dict[str, Any]] = []
    for index, action in enumerate(raw, 1):
        if not isinstance(action, Mapping):
            raise BenchmarkExecutionError("fixture action {} must be an object".format(index))
        detached = _detach_json(action, "fixture action {}".format(index))
        if not isinstance(detached, dict):
            raise BenchmarkExecutionError("fixture action {} must be an object".format(index))
        actions.append(detached)
    return actions


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


def _provider(context: Any) -> ScriptedProvider:
    return ScriptedProvider(_fixture_actions(context))


def _recovery_provider(context: Any) -> ScriptedProvider:
    return ScriptedProvider(_fixture_actions(context, recovery=True))


_EXECUTOR = DurableArtifactWorkspaceExecutor(
    client_factory=_runtime_client,
    provider_factory=_provider,
    recovery_provider_factory=_recovery_provider,
    workspace_factory=workspace_factory,
)


def execute(context: Any) -> Any:
    """Run one deterministic fixture trajectory against the real runtime."""

    return _EXECUTOR(context)


def preflight(context: Any) -> dict[str, Any]:
    """Validate fixture actions and task environment without running either."""

    report = _EXECUTOR.preflight(context)
    actions = _fixture_actions(context)
    recovery_actions = (
        _fixture_actions(context, recovery=True) if report["fault_plan"] is not None else []
    )
    runtime_url = _configured_runtime_url()
    artifact_root()
    return {
        **report,
        "executor": "scripted_artifact_workspace_fixture",
        "model_call": False,
        "runtime_url_configured": bool(runtime_url),
        "artifact_root_configured": True,
        "fixture_action_count": len(actions),
        "fixture_recovery_action_count": len(recovery_actions),
    }


execute.preflight = preflight  # type: ignore[attr-defined]


__all__ = ["execute", "preflight"]

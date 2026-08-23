"""Real-model executor for bounded, objectively verified artifact tasks.

The selected model interacts with a task-owned workspace only through
``invoke_adapter``.  A successful model self-report is insufficient: the
workspace must contain the verified artifact and the adapter must have recorded
its durable terminal result.  It can enact the bounded ``policy_restart``
schedule defined by ``DurableArtifactWorkspaceExecutor``; runtime-server crash
tasks still need their own environment-specific executor.
"""

from __future__ import annotations

from typing import Any

from horizon_agent import DurableArtifactWorkspaceExecutor

from .artifact_workspace_support import artifact_root, workspace_factory
from .openai_compatible_support import preflight_provider, provider, runtime_client


_EXECUTOR = DurableArtifactWorkspaceExecutor(
    client_factory=runtime_client,
    provider_factory=provider,
    workspace_factory=workspace_factory,
)


def execute(context: Any) -> Any:
    """Execute one frozen artifact task against the configured model/runtime."""

    return _EXECUTOR(context)


def preflight(context: Any) -> dict[str, Any]:
    """Validate task, provider and artifact-root configuration without side effects."""

    report = _EXECUTOR.preflight(context)
    artifact_root()  # Parse only; do not create or inspect the directory during preflight.
    return {
        **report,
        "executor": "openai_compatible_artifact_workspace",
        **preflight_provider(context),
        "artifact_root_configured": True,
    }


execute.preflight = preflight  # type: ignore[attr-defined]


__all__ = ["execute", "preflight"]

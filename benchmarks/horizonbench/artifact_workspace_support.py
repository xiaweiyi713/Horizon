"""Shared local-root binding for artifact-workspace benchmark executors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from horizon_agent import BenchmarkExecutionError, workspace_from_root


ARTIFACT_ROOT_ENV = "HORIZON_BENCH_ARTIFACT_ROOT"


def artifact_root() -> Path:
    """Read a caller-owned artifact root without creating or inspecting it."""

    value = os.getenv(ARTIFACT_ROOT_ENV)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise BenchmarkExecutionError(
            "configure {} to a non-empty local artifact directory before execution".format(
                ARTIFACT_ROOT_ENV
            )
        )
    return Path(value).expanduser()


def workspace_factory(context: Any, spec: Any) -> Any:
    return workspace_from_root(artifact_root())(context, spec)


__all__ = ["ARTIFACT_ROOT_ENV", "artifact_root", "workspace_factory"]

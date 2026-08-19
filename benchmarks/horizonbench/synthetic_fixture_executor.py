"""A deterministic executor for testing matrix-execution plumbing only.

It intentionally makes every supplied task succeed without calling a model or
task environment.  Its output is a smoke-test fixture, not an empirical LLM
result and must never be reported as one.
"""

from __future__ import annotations

try:
    from .execution import ExecutionContext
except ImportError:  # pragma: no cover - direct-script compatibility
    from execution import ExecutionContext  # type: ignore[no-redef]


def execute(context: ExecutionContext) -> dict[str, object]:
    """Return a deliberately synthetic observation for runner smoke tests."""

    return {
        "task_id": context.task.task_id,
        "category": context.task.category,
        "success": True,
        "goal_retained": True,
        "constraint_violations": 0,
        "state_consistent": True,
        "tokens": 0,
        "anchors_injected": 0,
    }

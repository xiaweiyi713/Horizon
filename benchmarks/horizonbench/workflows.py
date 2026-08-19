"""Strict loader for cross-domain long-horizon HorizonBench workflows.

The loader is dependency-free. It reuses JsonlTaskAdapter to select a local
JSONL split, then validates workflow metadata into immutable objects. The
bundled fixture is a deterministic evaluation source only and is not an
empirical model result.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Union

try:
    from .adapters import BenchmarkTask, BenchmarkTaskSet, JsonlTaskAdapter
except ImportError:  # pragma: no cover - direct-script compatibility
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from adapters import BenchmarkTask, BenchmarkTaskSet, JsonlTaskAdapter


PathLike = Union[str, Path]
DEFAULT_CROSS_DOMAIN_TASKS = Path(__file__).resolve().with_name("cross_domain_tasks.jsonl")

_REQUIRED_WORKFLOW_FIELDS = (
    "workflow_id",
    "workflow_title",
    "domain",
    "step_index",
    "expected_boundaries",
    "depends_on",
    "recovery_boundary",
    "evidence_required",
)


class WorkflowValidationError(ValueError):
    """Raised when a selected task cannot form a valid long-horizon workflow."""


@dataclass(frozen=True)
class WorkflowTask:
    """One validated workflow step backed by a normalized BenchmarkTask."""

    task: BenchmarkTask
    task_id: str
    workflow_id: str
    workflow_title: str
    domain: str
    step_index: int
    expected_boundaries: int
    depends_on: tuple[str, ...]
    recovery_boundary: bool
    evidence_required: bool

    def __post_init__(self) -> None:
        if not isinstance(self.task, BenchmarkTask):
            raise WorkflowValidationError("task must be a BenchmarkTask")
        object.__setattr__(self, "task_id", _require_nonempty_string(self.task_id, "task_id", task_id=self.task.task_id))
        if self.task_id != self.task.task_id:
            raise WorkflowValidationError(
                "task {!r} field task_id does not match the underlying benchmark task".format(self.task.task_id)
            )
        object.__setattr__(
            self,
            "workflow_id",
            _require_nonempty_string(self.workflow_id, "workflow_id", task_id=self.task_id),
        )
        object.__setattr__(
            self,
            "workflow_title",
            _require_nonempty_string(
                self.workflow_title,
                "workflow_title",
                task_id=self.task_id,
                workflow_id=self.workflow_id,
            ),
        )
        object.__setattr__(
            self,
            "domain",
            _require_nonempty_string(self.domain, "domain", task_id=self.task_id, workflow_id=self.workflow_id),
        )
        object.__setattr__(
            self,
            "step_index",
            _require_positive_int(self.step_index, "step_index", task_id=self.task_id, workflow_id=self.workflow_id),
        )
        object.__setattr__(
            self,
            "expected_boundaries",
            _require_positive_int(
                self.expected_boundaries,
                "expected_boundaries",
                task_id=self.task_id,
                workflow_id=self.workflow_id,
            ),
        )
        object.__setattr__(
            self,
            "depends_on",
            _normalize_depends_on(self.depends_on, task_id=self.task_id, workflow_id=self.workflow_id),
        )
        object.__setattr__(
            self,
            "recovery_boundary",
            _require_bool(self.recovery_boundary, "recovery_boundary", task_id=self.task_id, workflow_id=self.workflow_id),
        )
        object.__setattr__(
            self,
            "evidence_required",
            _require_bool(self.evidence_required, "evidence_required", task_id=self.task_id, workflow_id=self.workflow_id),
        )
        _ensure_json_safe(self.task.metadata, _locate("task metadata", task_id=self.task_id, workflow_id=self.workflow_id))

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "workflow_title": self.workflow_title,
            "domain": self.domain,
            "step_index": self.step_index,
            "expected_boundaries": self.expected_boundaries,
            "depends_on": list(self.depends_on),
            "recovery_boundary": self.recovery_boundary,
            "evidence_required": self.evidence_required,
            "task": self.task.as_dict(),
        }


@dataclass(frozen=True)
class WorkflowDefinition:
    """Ordered long-horizon workflow with a shared domain and boundary budget."""

    workflow_id: str
    workflow_title: str
    domain: str
    expected_boundaries: int
    tasks: tuple[WorkflowTask, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _require_nonempty_string(self.workflow_id, "workflow_id"))
        object.__setattr__(
            self,
            "workflow_title",
            _require_nonempty_string(self.workflow_title, "workflow_title", workflow_id=self.workflow_id),
        )
        object.__setattr__(self, "domain", _require_nonempty_string(self.domain, "domain", workflow_id=self.workflow_id))
        object.__setattr__(
            self,
            "expected_boundaries",
            _require_positive_int(self.expected_boundaries, "expected_boundaries", workflow_id=self.workflow_id),
        )
        tasks = tuple(self.tasks)
        if not all(isinstance(task, WorkflowTask) for task in tasks):
            raise WorkflowValidationError(
                "workflow {!r} field tasks must contain WorkflowTask values".format(self.workflow_id)
            )
        if len(tasks) < 3:
            raise WorkflowValidationError(
                "workflow {!r} must have at least 3 steps".format(self.workflow_id)
            )
        identifiers = [task.task_id for task in tasks]
        if len(set(identifiers)) != len(identifiers):
            raise WorkflowValidationError(
                "workflow {!r} field task_id must be unique within the workflow".format(self.workflow_id)
            )
        for task in tasks:
            _assert_task_agrees_with_workflow(task, self.workflow_id, self.workflow_title, self.domain, self.expected_boundaries)
        ordered = tuple(sorted(tasks, key=lambda item: (item.step_index, item.task_id)))
        expected_steps = tuple(range(1, len(ordered) + 1))
        actual_steps = tuple(task.step_index for task in ordered)
        if actual_steps != expected_steps:
            raise WorkflowValidationError(
                "workflow {!r} field step_index must be consecutive integers starting at 1".format(self.workflow_id)
            )
        if self.expected_boundaries < len(ordered):
            raise WorkflowValidationError(
                "workflow {!r} field expected_boundaries must be at least the number of tasks".format(self.workflow_id)
            )
        if not any(task.recovery_boundary for task in ordered):
            raise WorkflowValidationError(
                "workflow {!r} field recovery_boundary must be true for at least one task".format(self.workflow_id)
            )
        _assert_dependencies(self.workflow_id, ordered)
        object.__setattr__(self, "tasks", ordered)

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_title": self.workflow_title,
            "domain": self.domain,
            "expected_boundaries": self.expected_boundaries,
            "task_ids": [task.task_id for task in self.tasks],
            "tasks": [task.as_dict() for task in self.tasks],
        }


@dataclass(frozen=True)
class CrossDomainWorkflowSuite:
    """Validated workflow selection plus the underlying BenchmarkTaskSet."""

    task_set: BenchmarkTaskSet
    workflow_tasks: tuple[WorkflowTask, ...]
    workflows: tuple[WorkflowDefinition, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task_set, BenchmarkTaskSet):
            raise WorkflowValidationError("task_set must be a BenchmarkTaskSet")
        workflow_tasks = tuple(self.workflow_tasks)
        workflows = tuple(self.workflows)
        if not workflow_tasks:
            raise WorkflowValidationError("workflow_tasks must not be empty")
        if not workflows:
            raise WorkflowValidationError("workflows must not be empty")
        if not all(isinstance(task, WorkflowTask) for task in workflow_tasks):
            raise WorkflowValidationError("workflow_tasks must contain WorkflowTask values")
        if not all(isinstance(workflow, WorkflowDefinition) for workflow in workflows):
            raise WorkflowValidationError("workflows must contain WorkflowDefinition values")

        task_ids = [task.task_id for task in workflow_tasks]
        if len(set(task_ids)) != len(task_ids):
            raise WorkflowValidationError("task IDs must be globally unique")
        if tuple(task_ids) != self.task_set.task_ids:
            raise WorkflowValidationError("workflow_tasks must follow task_set order and identifiers")
        for workflow_task, benchmark_task in zip(workflow_tasks, self.task_set.tasks):
            if workflow_task.task != benchmark_task:
                raise WorkflowValidationError(
                    "task {!r} does not match the corresponding BenchmarkTask".format(workflow_task.task_id)
                )

        workflow_ids = [workflow.workflow_id for workflow in workflows]
        if len(set(workflow_ids)) != len(workflow_ids):
            raise WorkflowValidationError("workflow IDs must be unique")

        assigned = [task.task_id for workflow in workflows for task in workflow.tasks]
        if sorted(assigned) != sorted(task_ids):
            raise WorkflowValidationError("workflows must cover every selected task exactly once")

        object.__setattr__(self, "workflow_tasks", workflow_tasks)
        object.__setattr__(self, "workflows", workflows)
        object.__setattr__(
            self,
            "_task_by_id",
            MappingProxyType({task.task_id: task for task in workflow_tasks}),
        )
        object.__setattr__(
            self,
            "_workflow_by_id",
            MappingProxyType({workflow.workflow_id: workflow for workflow in workflows}),
        )

    @property
    def task_by_id(self) -> Mapping[str, WorkflowTask]:
        return self._task_by_id

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({workflow.domain for workflow in self.workflows}))

    @property
    def workflow_by_id(self) -> Mapping[str, WorkflowDefinition]:
        return self._workflow_by_id

    def identity(self) -> dict[str, Any]:
        """Compact JSON-safe identity suitable for an evaluation audit."""

        payload = self.task_set.identity()
        payload["workflow_ids"] = [workflow.workflow_id for workflow in self.workflows]
        payload["domains"] = list(self.domains)
        return payload

    def as_dict(self) -> dict[str, Any]:
        payload = self.identity()
        payload["task_set"] = self.task_set.as_dict()
        payload["workflow_tasks"] = [task.as_dict() for task in self.workflow_tasks]
        payload["workflows"] = [workflow.as_dict() for workflow in self.workflows]
        return payload


def load_cross_domain_workflows(
    path: PathLike = DEFAULT_CROSS_DOMAIN_TASKS,
    *,
    source_name: str = "horizon-cross-domain-fixture",
    source_revision: str = "v1",
    split: str = "heldout",
) -> CrossDomainWorkflowSuite:
    """Load a JSONL split and validate it as a cross-domain workflow suite."""

    adapter = JsonlTaskAdapter(source_name, source_revision)
    try:
        task_set = adapter.load(path, split=split)
    except ValueError as error:
        raise WorkflowValidationError(str(error)) from error

    workflow_tasks = tuple(_workflow_task_from_benchmark(task) for task in task_set.tasks)
    grouped: dict[str, list[WorkflowTask]] = {}
    workflow_order: list[str] = []
    for task in workflow_tasks:
        if task.workflow_id not in grouped:
            grouped[task.workflow_id] = []
            workflow_order.append(task.workflow_id)
        grouped[task.workflow_id].append(task)

    workflows = tuple(
        WorkflowDefinition(
            workflow_id=workflow_id,
            workflow_title=members[0].workflow_title,
            domain=members[0].domain,
            expected_boundaries=members[0].expected_boundaries,
            tasks=tuple(members),
        )
        for workflow_id, members in ((workflow_id, grouped[workflow_id]) for workflow_id in workflow_order)
    )
    return CrossDomainWorkflowSuite(task_set=task_set, workflow_tasks=workflow_tasks, workflows=workflows)


def _workflow_task_from_benchmark(task: BenchmarkTask) -> WorkflowTask:
    metadata = task.metadata
    if not isinstance(metadata, Mapping):
        raise WorkflowValidationError("task {!r} field metadata must be an object".format(task.task_id))
    missing = [field for field in _REQUIRED_WORKFLOW_FIELDS if field not in metadata]
    if missing:
        raise WorkflowValidationError(
            "task {!r} is missing required field {}".format(task.task_id, missing[0])
        )
    return WorkflowTask(
        task=task,
        task_id=task.task_id,
        workflow_id=metadata["workflow_id"],
        workflow_title=metadata["workflow_title"],
        domain=metadata["domain"],
        step_index=metadata["step_index"],
        expected_boundaries=metadata["expected_boundaries"],
        depends_on=metadata["depends_on"],
        recovery_boundary=metadata["recovery_boundary"],
        evidence_required=metadata["evidence_required"],
    )


def _assert_task_agrees_with_workflow(
    task: WorkflowTask,
    workflow_id: str,
    workflow_title: str,
    domain: str,
    expected_boundaries: int,
) -> None:
    if task.workflow_id != workflow_id:
        raise WorkflowValidationError(
            "workflow {!r} task {!r} field workflow_id is inconsistent".format(workflow_id, task.task_id)
        )
    if task.workflow_title != workflow_title:
        raise WorkflowValidationError(
            "workflow {!r} task {!r} field workflow_title is inconsistent".format(workflow_id, task.task_id)
        )
    if task.domain != domain:
        raise WorkflowValidationError(
            "workflow {!r} task {!r} field domain is inconsistent".format(workflow_id, task.task_id)
        )
    if task.expected_boundaries != expected_boundaries:
        raise WorkflowValidationError(
            "workflow {!r} task {!r} field expected_boundaries is inconsistent".format(workflow_id, task.task_id)
        )


def _assert_dependencies(workflow_id: str, tasks: Sequence[WorkflowTask]) -> None:
    by_id = {task.task_id: task for task in tasks}
    graph: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        predecessors: list[str] = []
        for dependency in task.depends_on:
            predecessor = by_id.get(dependency)
            if predecessor is None:
                raise WorkflowValidationError(
                    "workflow {!r} task {!r} field depends_on refers to {!r}, which is not an earlier task in the same workflow".format(
                        workflow_id,
                        task.task_id,
                        dependency,
                    )
                )
            if predecessor.step_index >= task.step_index:
                raise WorkflowValidationError(
                    "workflow {!r} task {!r} field depends_on refers to later or current task {!r}".format(
                        workflow_id,
                        task.task_id,
                        dependency,
                    )
                )
            predecessors.append(dependency)
        graph[task.task_id] = tuple(predecessors)
    _assert_acyclic(workflow_id, graph)


def _assert_acyclic(workflow_id: str, graph: Mapping[str, Sequence[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise WorkflowValidationError(
                "workflow {!r} task {!r} field depends_on forms a cycle".format(workflow_id, node)
            )
        visiting.add(node)
        for predecessor in graph.get(node, ()):
            if predecessor in graph:
                visit(predecessor)
        visiting.remove(node)
        visited.add(node)

    for task_id in graph:
        visit(task_id)


def _normalize_depends_on(value: Any, *, task_id: str, workflow_id: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise WorkflowValidationError(
            "workflow {!r} task {!r} field depends_on must be a JSON array of task IDs".format(workflow_id, task_id)
        )
    dependencies: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        dependency = _require_nonempty_string(
            item,
            "depends_on[{}]".format(index),
            task_id=task_id,
            workflow_id=workflow_id,
        )
        if dependency in seen:
            raise WorkflowValidationError(
                "workflow {!r} task {!r} field depends_on contains duplicate task id {!r}".format(
                    workflow_id,
                    task_id,
                    dependency,
                )
            )
        seen.add(dependency)
        dependencies.append(dependency)
    return tuple(dependencies)


def _require_nonempty_string(
    value: Any,
    field: str,
    *,
    task_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowValidationError(
            "{} must be a nonempty string".format(_locate(field, task_id=task_id, workflow_id=workflow_id))
        )
    return value


def _require_positive_int(
    value: Any,
    field: str,
    *,
    task_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorkflowValidationError(
            "{} must be a positive integer".format(_locate(field, task_id=task_id, workflow_id=workflow_id))
        )
    return value


def _require_bool(
    value: Any,
    field: str,
    *,
    task_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> bool:
    if not isinstance(value, bool):
        raise WorkflowValidationError(
            "{} must be a boolean".format(_locate(field, task_id=task_id, workflow_id=workflow_id))
        )
    return value


def _locate(field: str, *, task_id: Optional[str] = None, workflow_id: Optional[str] = None) -> str:
    parts: list[str] = []
    if workflow_id:
        parts.append("workflow {!r}".format(workflow_id))
    if task_id:
        parts.append("task {!r}".format(task_id))
    parts.append("field {}".format(field))
    return " ".join(parts)


def _ensure_json_safe(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise WorkflowValidationError("{} has a non-string metadata key".format(where))
            _ensure_json_safe(item, where)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _ensure_json_safe(item, where)
        return
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorkflowValidationError("{} contains an unsupported non-finite number".format(where))
        return
    raise WorkflowValidationError("{} contains an unsupported metadata JSON value".format(where))


__all__ = [
    "DEFAULT_CROSS_DOMAIN_TASKS",
    "CrossDomainWorkflowSuite",
    "WorkflowDefinition",
    "WorkflowTask",
    "WorkflowValidationError",
    "load_cross_domain_workflows",
]

"""Executor contract for a future HorizonBench matrix runner.

The contract is dependency-free and does not invoke a model, provider, or
network. A later runner can load a callable, bind one frozen task to a run
manifest, and normalize the observed episode outcome.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, Union

try:  # Supports imports from the direct-script HorizonBench entry points.
    from .adapters import BenchmarkTask
    from .protocol import RunManifest
    from .score import EpisodeResult
except ImportError:  # pragma: no cover - exercised through direct-script callers.
    from adapters import BenchmarkTask  # type: ignore[no-redef]
    from protocol import RunManifest  # type: ignore[no-redef]
    from score import EpisodeResult  # type: ignore[no-redef]


class ExecutionValidationError(ValueError):
    """Raised when an executor spec, context, or episode result is invalid."""


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable binding of one task attempt to a frozen run manifest."""

    manifest: RunManifest
    task: BenchmarkTask
    attempt: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, RunManifest):
            raise ExecutionValidationError("manifest must be a RunManifest")
        if not isinstance(self.task, BenchmarkTask):
            raise ExecutionValidationError("task must be a BenchmarkTask")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ExecutionValidationError("attempt must be a positive integer")
        if self.task.task_id not in self.manifest.task_set.task_ids:
            raise ExecutionValidationError(
                "task_id {!r} is not in manifest.task_set.task_ids".format(self.task.task_id)
            )
        if self.task.split != self.manifest.task_set.split:
            raise ExecutionValidationError(
                "task.split {!r} does not match manifest.task_set.split {!r}".format(
                    self.task.split,
                    self.manifest.task_set.split,
                )
            )


class EpisodeExecutor(Protocol):
    """Callable that executes one frozen benchmark task."""

    def __call__(
        self, context: ExecutionContext
    ) -> Union[EpisodeResult, Mapping[str, Any]]:
        """Return an observed episode outcome for ``context``."""


class ExecutorPreflight(Protocol):
    """Optional static validation hook exposed by a matrix executor.

    The hook is intentionally separate from :class:`EpisodeExecutor`: legacy
    executors remain runnable, while an experimenter can require an explicit
    no-execution validation phase before using a provider or task environment.
    Implementations must return a JSON-safe, non-sensitive object and must not
    call their episode executor, model provider, runtime, or task environment.
    """

    def preflight(self, context: ExecutionContext) -> Mapping[str, Any]:
        """Validate one frozen manifest/task binding without executing it."""


def normalize_episode_result(value: Any, context: ExecutionContext) -> EpisodeResult:
    """Coerce an executor return value into a validated ``EpisodeResult``."""

    if not isinstance(context, ExecutionContext):
        raise ExecutionValidationError("context must be an ExecutionContext")
    if isinstance(value, EpisodeResult):
        result = value
    elif isinstance(value, Mapping):
        try:
            result = EpisodeResult.from_mapping(value)
        except (TypeError, ValueError) as error:
            raise ExecutionValidationError(
                "executor mapping is not a valid EpisodeResult: {}".format(error)
            ) from error
    else:
        raise ExecutionValidationError(
            "executor must return EpisodeResult or a mapping, not {}".format(type(value).__name__)
        )

    if result.task_id != context.task.task_id:
        raise ExecutionValidationError(
            "episode result task_id {!r} does not match task {!r}".format(
                result.task_id,
                context.task.task_id,
            )
        )
    if result.category != context.task.category:
        raise ExecutionValidationError(
            "episode result category {!r} does not match task {!r}".format(
                result.category,
                context.task.category,
            )
        )
    return result


def preflight_executor(executor: EpisodeExecutor, context: ExecutionContext) -> dict[str, Any]:
    """Run an executor's opt-in static preflight hook and normalize its report.

    This helper never invokes ``executor(context)``.  It makes preflight an
    explicit protocol capability so a matrix CLI can fail before writing an
    execution artifact or crossing a model/runtime boundary.
    """

    if not isinstance(context, ExecutionContext):
        raise ExecutionValidationError("context must be an ExecutionContext")
    if not callable(executor):
        raise ExecutionValidationError("executor must be callable")
    preflight = getattr(executor, "preflight", None)
    if not callable(preflight):
        raise ExecutionValidationError(
            "executor does not expose callable preflight(context); use an executor with a static preflight hook"
        )
    value = preflight(context)
    if not isinstance(value, Mapping):
        raise ExecutionValidationError("executor preflight must return a mapping")
    normalized = _json_safe_snapshot(value, "executor preflight")
    if not isinstance(normalized, dict):  # Defensive; Mapping above guarantees this today.
        raise ExecutionValidationError("executor preflight must return an object")
    return normalized


def load_executor(spec: Any) -> EpisodeExecutor:
    """Import ``module:attribute`` and return the resolved callable."""

    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ExecutionValidationError("executor spec must use exact module:attribute format")
    module_name, attribute = spec.split(":", 1)
    _require_dotted_identifier(module_name, "executor module")
    attribute_parts = _require_dotted_identifier(attribute, "executor attribute")

    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise ExecutionValidationError(
            "cannot import executor module {!r}: {}".format(module_name, error)
        ) from error

    target: Any = module
    resolved: list[str] = []
    for part in attribute_parts:
        resolved.append(part)
        try:
            target = getattr(target, part)
        except AttributeError as error:
            raise ExecutionValidationError(
                "cannot resolve executor attribute {!r} on module {!r}: {}".format(
                    ".".join(resolved),
                    module_name,
                    error,
                )
            ) from error

    if not callable(target):
        raise ExecutionValidationError("executor {!r} is not callable".format(spec))
    return target


def _require_dotted_identifier(value: str, field: str) -> tuple[str, ...]:
    parts = value.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        raise ExecutionValidationError("{} must be a dotted identifier".format(field))
    return tuple(parts)


def _json_safe_snapshot(value: Any, label: str, ancestors: frozenset[int] = frozenset()) -> Any:
    """Detach a plugin report into finite, JSON-compatible built-in values."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExecutionValidationError("{} must not contain NaN or infinity".format(label))
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ExecutionValidationError("{} must not contain cyclic values".format(label))
        next_ancestors = ancestors | frozenset((identity,))
        detached: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExecutionValidationError("{} has a non-string key".format(label))
            detached[key] = _json_safe_snapshot(item, "{}.{}".format(label, key), next_ancestors)
        return detached
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in ancestors:
            raise ExecutionValidationError("{} must not contain cyclic values".format(label))
        next_ancestors = ancestors | frozenset((identity,))
        return [
            _json_safe_snapshot(item, "{}[]".format(label), next_ancestors)
            for item in value
        ]
    raise ExecutionValidationError("{} is not JSON-safe".format(label))

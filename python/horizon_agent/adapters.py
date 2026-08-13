"""Durable adapters for Python-owned external tools and task systems.

Rust continues to own state transitions and persistence. Adapters execute in
the Python research layer, but their invocation and normalized terminal result
are durably recorded around each side effect using a stable operation ID.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Protocol, Tuple


Json = Any
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "timed_out", "cancelled"})


class AdapterError(RuntimeError):
    """Configuration or result-contract failure in an external adapter."""


@dataclass(frozen=True)
class AdapterRequest:
    """Stable context supplied to an external adapter invocation."""

    run_id: str
    operation_id: str
    input: Json


@dataclass(frozen=True)
class AdapterResult:
    """Normalized terminal result emitted by an external adapter."""

    status: str = "succeeded"
    output: Json = None
    error: Optional[str] = None
    metadata: Mapping[str, Json] = field(default_factory=dict)

    def validate(self) -> None:
        if self.status not in _TERMINAL_STATUSES:
            raise AdapterError(
                "adapter result status must be one of " + ", ".join(sorted(_TERMINAL_STATUSES))
            )
        if self.status != "succeeded" and not (self.error and self.error.strip()):
            raise AdapterError("non-successful adapter results must include a non-empty error")
        if not isinstance(self.metadata, Mapping):
            raise AdapterError("adapter result metadata must be a mapping")
        _ensure_json(self.output, "adapter result output")
        _ensure_json(dict(self.metadata), "adapter result metadata")


class ExternalTaskAdapter(Protocol):
    """Protocol for one Python-owned external task integration."""

    name: str

    def execute(self, request: AdapterRequest) -> AdapterResult:
        """Perform the side effect and return its normalized terminal result."""


class FunctionAdapter:
    """Small adapter wrapper for a callable, useful in experiments and tests."""

    def __init__(self, name: str, function: Callable[[AdapterRequest], AdapterResult]) -> None:
        self.name = name
        self._function = function

    def execute(self, request: AdapterRequest) -> AdapterResult:
        return self._function(request)


class AdapterRegistry:
    """Name-addressable adapter registry with durable before/after audit events.

    The invocation event is persisted before calling the adapter. If the process
    dies after the external side effect but before its result is stored, replay
    retains the same operation ID so the integration can deduplicate or query
    it. This is deliberately at-least-once semantics, not a claim of exactly
    once delivery.
    """

    def __init__(self, adapters: Iterable[ExternalTaskAdapter] = ()) -> None:
        self._adapters: Dict[str, ExternalTaskAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def register(self, adapter: ExternalTaskAdapter) -> None:
        name = getattr(adapter, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise AdapterError("an adapter must expose a non-empty string `name`")
        if not callable(getattr(adapter, "execute", None)):
            raise AdapterError("an adapter must expose an `execute(request)` method")
        if name in self._adapters:
            raise AdapterError("an adapter named `{}` is already registered".format(name))
        self._adapters[name] = adapter

    def execute(
        self,
        client: Any,
        run_id: str,
        adapter_name: str,
        operation_id: str,
        input: Json,
        *,
        task_id: Optional[str] = None,
    ) -> AdapterResult:
        """Run an adapter and durably store one rich terminal tool result.

        ``client`` is duck-typed to the two Horizon client methods used here,
        making this work with both ``HorizonClient`` and ``NativeHorizonRuntime``.
        A failure to persist the invocation prevents execution; a failure after
        the side effect is surfaced to the caller so it can recover using the
        operation ID rather than silently losing the audit boundary.
        """
        if not isinstance(run_id, str) or not run_id.strip():
            raise AdapterError("run_id cannot be empty")
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise AdapterError("operation_id cannot be empty")
        adapter = self._adapters.get(adapter_name)
        if adapter is None:
            raise AdapterError("no adapter named `{}` is registered".format(adapter_name))

        # Persist the causal before-boundary before allowing the external call.
        client.record_tool_invocation(
            run_id,
            operation_id,
            adapter_name,
            {"adapter": adapter_name, "input": input},
        )
        started = time.monotonic_ns()
        try:
            result = adapter.execute(AdapterRequest(run_id, operation_id, input))
            if not isinstance(result, AdapterResult):
                raise AdapterError("adapter `execute` must return AdapterResult")
            result.validate()
        except Exception as error:
            result = AdapterResult(
                status="failed",
                error="{}: {}".format(type(error).__name__, error),
                metadata={"exception_type": type(error).__name__},
            )

        duration_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
        metadata = dict(result.metadata)
        metadata["adapter"] = adapter_name
        if task_id is not None:
            metadata["task_id"] = task_id
        # This is intentionally the only terminal audit event emitted by the
        # registry: consumers do not need to infer outcomes from legacy strings.
        client.record_tool_result(
            run_id,
            operation_id,
            adapter_name,
            result.status,
            output=result.output,
            error=result.error,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        return AdapterResult(
            status=result.status,
            output=result.output,
            error=result.error,
            metadata=metadata,
        )


def _ensure_json(value: Json, description: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise AdapterError("{} must be JSON-serializable: {}".format(description, error)) from error

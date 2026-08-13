"""Optional direct bindings to Horizon's local Rust runtime.

The normal :class:`~horizon_agent.client.HorizonClient` continues to use the
localhost HTTP contract. ``NativeHorizonRuntime`` exposes the same useful
client methods but calls the PyO3 module directly when ``horizon-native`` has
been installed with Maturin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from .client import HorizonApiError, HorizonClient


Json = Any


class NativeRuntimeUnavailable(RuntimeError):
    """The optional ``horizon_native`` extension module is not installed."""


try:
    from horizon_native import Runtime as _NativeRuntime
except ImportError as error:  # pragma: no cover - host dependent
    _NativeRuntime = None
    _NATIVE_IMPORT_ERROR = error
else:
    _NATIVE_IMPORT_ERROR = None


class NativeHorizonRuntime(HorizonClient):
    """Synchronous local SQLite client backed by the PyO3 runtime module.

    It intentionally follows ``HorizonClient``'s command interface, so a
    ``DurableAgent`` can switch between HTTP and native mode without acquiring
    a second mutable state model. The native extension is optional; importing
    this class never changes the default HTTP workflow.
    """

    def __init__(
        self,
        database_path: Union[str, Path] = "horizon.db",
        *,
        checkpoint_every: int = 12,
        retain_checkpoints: int = 8,
        snapshot_encoding: str = "zstd_json",
    ) -> None:
        if _NativeRuntime is None:
            detail = "" if _NATIVE_IMPORT_ERROR is None else ": " + str(_NATIVE_IMPORT_ERROR)
            raise NativeRuntimeUnavailable(
                "native Horizon bindings are not installed; build them with "
                "`maturin develop --manifest-path crates/horizon-py/Cargo.toml`" + detail
            )
        self._runtime = _NativeRuntime(
            str(database_path),
            checkpoint_every=checkpoint_every,
            retain_checkpoints=retain_checkpoints,
            snapshot_encoding=snapshot_encoding,
        )

    def health(self) -> bool:
        """A constructed native runtime is immediately ready for requests."""
        return True

    def create_run(self, goal: str) -> Mapping[str, Json]:
        return self._mapping(self._call("create_run", goal))

    def list_runs(self) -> Sequence[Mapping[str, Json]]:
        value = self._call("list_runs")
        if not isinstance(value, list):
            raise HorizonApiError("native runtime returned a non-list run response")
        return value

    def get_run(self, run_id: str) -> Mapping[str, Json]:
        return self._mapping(self._call("projection", run_id))

    def events(self, run_id: str) -> Sequence[Mapping[str, Json]]:
        value = self._call("events", run_id)
        if not isinstance(value, list):
            raise HorizonApiError("native runtime returned a non-list event response")
        return value

    def state_anchor(self, run_id: str) -> Mapping[str, Json]:
        return self._mapping(self._call("state_anchor", run_id))

    def dispatch(self, run_id: str, command: Mapping[str, Json]) -> Mapping[str, Json]:
        return self._mapping(self._call("dispatch", run_id, json.dumps(dict(command))))

    def checkpoint(self, run_id: str) -> Mapping[str, Json]:
        return self._mapping(self._call("checkpoint", run_id))

    def recover(self, run_id: str) -> Mapping[str, Json]:
        return self._mapping(self._call("recover", run_id))

    def intervene_if_needed(
        self,
        run_id: str,
        *,
        context_pressure: float = 0.0,
        subgoal_switches: int = 0,
        recent_failures: int = 0,
        recovered_session: bool = False,
        steps_since_anchor: int = 0,
    ) -> Optional[Mapping[str, Json]]:
        value = self._call(
            "intervene",
            run_id,
            json.dumps(
                {
                    "steps_since_anchor": steps_since_anchor,
                    "context_pressure": context_pressure,
                    "subgoal_switches": subgoal_switches,
                    "recent_failures": recent_failures,
                    "recovered_session": recovered_session,
                }
            ),
        )
        if value is None:
            return None
        return self._mapping(value)

    def execute_ready_tasks(self, run_id: str) -> Sequence[Mapping[str, Json]]:
        value = self._call("execute_ready_tasks", run_id)
        if not isinstance(value, list):
            raise HorizonApiError("native runtime returned a non-list task response")
        return value

    def compact_checkpoints(self, run_id: str, retain_latest: int = 8) -> Mapping[str, Json]:
        return self._mapping(self._call("compact_checkpoints", run_id, retain_latest))

    def _call(self, method: str, *arguments: object) -> Json:
        try:
            payload = getattr(self._runtime, method)(*arguments)
            return json.loads(payload)
        except HorizonApiError:
            raise
        except Exception as error:
            raise HorizonApiError(str(error)) from error

    @staticmethod
    def _mapping(value: Json) -> Mapping[str, Json]:
        if not isinstance(value, Mapping):
            raise HorizonApiError("native runtime returned a non-object response")
        return value

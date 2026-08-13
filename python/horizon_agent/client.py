"""Small dependency-free client for Horizon's local JSON/HTTP runtime."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


Json = Any


class HorizonApiError(RuntimeError):
    """Raised when the runtime rejects a command or cannot be reached."""

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class HorizonClient:
    """Synchronous client suitable for policies, demos, and benchmark scripts.

    The runtime defaults to ``http://127.0.0.1:8787``. It intentionally exposes
    commands rather than an arbitrary state-update endpoint, preserving the
    Rust-side invariant: validate -> event -> persist -> apply.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8787", *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        return self._request("GET", "/health") == "ok"

    def create_run(self, goal: str) -> Mapping[str, Json]:
        return self._request("POST", "/v1/runs", {"goal": goal})

    def list_runs(self) -> Sequence[Mapping[str, Json]]:
        return self._request("GET", "/v1/runs")

    def get_run(self, run_id: str) -> Mapping[str, Json]:
        return self._request("GET", f"/v1/runs/{run_id}")

    def events(self, run_id: str) -> Sequence[Mapping[str, Json]]:
        return self._request("GET", f"/v1/runs/{run_id}/events")

    def state_anchor(self, run_id: str) -> Mapping[str, Json]:
        return self._request("GET", f"/v1/runs/{run_id}/anchor")

    def dispatch(self, run_id: str, command: Mapping[str, Json]) -> Mapping[str, Json]:
        return self._request("POST", f"/v1/runs/{run_id}/commands", dict(command))

    def command(self, run_id: str, command_type: str, data: Mapping[str, Json]) -> Mapping[str, Json]:
        return self.dispatch(run_id, {"type": command_type, "data": dict(data)})

    def transition(self, run_id: str, to: str) -> Mapping[str, Json]:
        return self.command(run_id, "transition", {"to": to})

    def add_constraint(
        self,
        run_id: str,
        content: str,
        *,
        constraint_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Mapping[str, Json]:
        return self.command(
            run_id,
            "add_constraint",
            {
                "constraint": {
                    "id": constraint_id or "constraint-" + uuid4().hex[:12],
                    "content": content,
                    "source": source,
                }
            },
        )

    def set_plan(self, run_id: str, steps: Sequence[str]) -> Mapping[str, Json]:
        return self.command(run_id, "set_plan", {"steps": list(steps)})

    def open_subgoal(
        self,
        run_id: str,
        content: str,
        *,
        subgoal_id: Optional[str] = None,
    ) -> Mapping[str, Json]:
        return self.command(
            run_id,
            "open_subgoal",
            {
                "subgoal": {
                    "id": subgoal_id or "subgoal-" + uuid4().hex[:12],
                    "content": content,
                    "completed": False,
                }
            },
        )

    def complete_subgoal(self, run_id: str, subgoal_id: str) -> Mapping[str, Json]:
        return self.command(run_id, "complete_subgoal", {"id": subgoal_id})

    def create_task(
        self,
        run_id: str,
        title: str,
        *,
        command: Optional[Sequence[str]] = None,
        dependencies: Optional[Sequence[str]] = None,
        priority: int = 0,
        max_retries: int = 1,
        timeout_ms: Optional[int] = None,
        working_dir: Optional[str] = None,
        task_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        operation_id: Optional[str] = None,
    ) -> Mapping[str, Json]:
        task = {
            "id": task_id or str(uuid4()),
            "title": title,
            "parent": parent_task_id,
            "operation_id": operation_id,
            "command": list(command) if command is not None else None,
            "working_dir": working_dir,
            "priority": priority,
            "dependencies": list(dependencies or ()),
            "max_retries": max_retries,
            "timeout_ms": timeout_ms,
        }
        return self.command(run_id, "create_task", {"task": task})

    def remember_failure(
        self,
        run_id: str,
        approach: str,
        reason: str,
        *,
        retryable: bool = False,
        operation_id: Optional[str] = None,
        failure_id: Optional[str] = None,
    ) -> Mapping[str, Json]:
        return self.command(
            run_id,
            "remember_failure",
            {
                "failure": {
                    "id": failure_id or "failure-" + uuid4().hex[:12],
                    "approach": approach,
                    "reason": reason,
                    "operation_id": operation_id,
                    "retryable": retryable,
                }
            },
        )

    def block_task(self, run_id: str, task_id: str, reason: str) -> Mapping[str, Json]:
        return self.command(run_id, "block_task", {"task_id": task_id, "reason": reason})

    def unblock_task(self, run_id: str, task_id: str) -> Mapping[str, Json]:
        return self.command(run_id, "unblock_task", {"task_id": task_id})

    def record_decision(
        self,
        run_id: str,
        decision: str,
        rationale: str,
        *,
        decision_id: Optional[str] = None,
    ) -> Mapping[str, Json]:
        return self.command(
            run_id,
            "record_decision",
            {
                "decision": {
                    "id": decision_id or "decision-" + uuid4().hex[:12],
                    "decision": decision,
                    "rationale": rationale,
                }
            },
        )

    def record_evidence(
        self,
        run_id: str,
        content: str,
        *,
        source: Optional[str] = None,
        evidence_id: Optional[str] = None,
    ) -> Mapping[str, Json]:
        return self.command(
            run_id,
            "record_evidence",
            {
                "evidence": {
                    "id": evidence_id or "evidence-" + uuid4().hex[:12],
                    "content": content,
                    "source": source,
                }
            },
        )

    def record_tool_invocation(
        self, run_id: str, operation_id: str, tool: str, input: Json
    ) -> Mapping[str, Json]:
        return self.command(
            run_id,
            "record_tool_invocation",
            {"operation_id": operation_id, "tool": tool, "input": input},
        )

    def record_tool_success(
        self, run_id: str, operation_id: str, output: Json
    ) -> Mapping[str, Json]:
        return self.command(
            run_id,
            "record_tool_success",
            {"operation_id": operation_id, "output": output},
        )

    def record_tool_failure(
        self, run_id: str, operation_id: str, error: str, *, retryable: bool
    ) -> Mapping[str, Json]:
        return self.command(
            run_id,
            "record_tool_failure",
            {"operation_id": operation_id, "error": error, "retryable": retryable},
        )

    def update_budget(
        self,
        run_id: str,
        *,
        tokens_used: int,
        token_budget: Optional[int] = None,
        wall_time_used_ms: int = 0,
        wall_time_budget_ms: Optional[int] = None,
    ) -> Mapping[str, Json]:
        return self.command(
            run_id,
            "update_budget",
            {
                "budget": {
                    "token_budget": token_budget,
                    "tokens_used": tokens_used,
                    "wall_time_budget_ms": wall_time_budget_ms,
                    "wall_time_used_ms": wall_time_used_ms,
                }
            },
        )

    def update_environment(self, run_id: str, environment: Json) -> Mapping[str, Json]:
        return self.command(run_id, "update_environment", {"environment": environment})

    def checkpoint(self, run_id: str) -> Mapping[str, Json]:
        return self._request("POST", f"/v1/runs/{run_id}/checkpoint", {})

    def recover(self, run_id: str) -> Mapping[str, Json]:
        return self._request("POST", f"/v1/runs/{run_id}/recover", {})

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
        return self._request(
            "POST",
            f"/v1/runs/{run_id}/interventions",
            {
                "steps_since_anchor": steps_since_anchor,
                "context_pressure": context_pressure,
                "subgoal_switches": subgoal_switches,
                "recent_failures": recent_failures,
                "recovered_session": recovered_session,
            },
        )

    def execute_ready_tasks(self, run_id: str) -> Sequence[Mapping[str, Json]]:
        return self._request("POST", f"/v1/runs/{run_id}/tasks/execute", {})

    def _request(self, method: str, path: str, payload: Optional[Json] = None) -> Json:
        url = self.base_url + path
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("error", body)
            except json.JSONDecodeError:
                detail = body
            raise HorizonApiError(str(detail), status=error.code) from error
        except URLError as error:
            raise HorizonApiError("could not reach Horizon runtime at {}: {}".format(url, error.reason)) from error
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

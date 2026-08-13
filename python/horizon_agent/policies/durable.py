"""A strict single-action protocol for durable long-horizon agent policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..providers.base import ChatMessage, LlmProvider, ProviderResponse


class ActionParseError(ValueError):
    """The provider response did not satisfy Horizon's one-action contract."""


@dataclass(frozen=True)
class AgentAction:
    """One policy decision applied at a durable execution boundary."""

    action_type: str
    data: Mapping[str, Any]
    thought: str = ""

    @classmethod
    def from_response(cls, response: str) -> "AgentAction":
        raw = _load_json(response)
        if "action" in raw:
            action = raw["action"]
            thought = str(raw.get("thought", ""))
        else:
            action = raw
            thought = str(raw.get("thought", ""))
        if not isinstance(action, Mapping):
            raise ActionParseError("`action` must be a JSON object")
        action_type = action.get("type")
        data = action.get("data", {})
        if not isinstance(action_type, str) or not action_type.strip():
            raise ActionParseError("action must contain a non-empty string `type`")
        if not isinstance(data, Mapping):
            raise ActionParseError("action `data` must be an object")
        return cls(action_type=action_type.strip(), data=dict(data), thought=thought)


class JsonActionPolicy:
    """Prompts an arbitrary chat provider for one valid Horizon action at a time."""

    SYSTEM_PROMPT = """You are a policy for a durable long-horizon agent runtime.
The Rust runtime, not you, owns state transitions, persistence, scheduling, and
side effects. Return exactly one JSON object and never markdown. Use this shape:
{"thought":"short rationale", "action":{"type":"ACTION", "data":{...}}}

Allowed actions:
- transition: {"to":"planning|executing|waiting|evaluating|recovering|suspended|completed|failed"}
- add_constraint: {"constraint":{"id":"...","content":"...","source":null}}
- set_plan: {"steps":["..."]}
- open_subgoal: {"subgoal":{"id":"...","content":"...","completed":false}}
- complete_subgoal: {"id":"..."}
- create_task: {"task":{"id":"uuid","title":"...","command":["program","arg"],"dependencies":[],"priority":0,"max_retries":1,"timeout_ms":null,"working_dir":null}}
- start_task, block_task, unblock_task, succeed_task, fail_task, retry_task, cancel_task (with the runtime command fields)
- remember_failure: {"failure":{"id":"...","approach":"...","reason":"...","operation_id":null,"retryable":false}}
- record_decision: {"decision":{"id":"...","decision":"...","rationale":"..."}}
- record_evidence: {"evidence":{"id":"...","content":"...","source":null}}
- record_tool_invocation, record_tool_success, record_tool_failure for external tool audit events
- update_environment: {"environment":{...}}
- checkpoint, execute_ready_tasks, recover, intervene, finish, suspend

Never repeat an approach listed under Failed Approaches unless the recorded
failure is retryable and you explicitly choose retry_task. Choose finish only
when the goal and all required constraints are satisfied."""

    def __init__(self, provider: LlmProvider) -> None:
        self.provider = provider

    def next_action(self, context: str) -> tuple[AgentAction, ProviderResponse]:
        response = self.provider.complete(
            [
                ChatMessage(role="system", content=self.SYSTEM_PROMPT),
                ChatMessage(role="user", content=context),
            ],
            json_mode=True,
        )
        return AgentAction.from_response(response.content), response


def _load_json(value: str) -> Mapping[str, Any]:
    candidate = value.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[:-3]
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ActionParseError("provider did not return valid JSON: {}".format(error)) from error
    if not isinstance(result, Mapping):
        raise ActionParseError("provider response must be a JSON object")
    return result

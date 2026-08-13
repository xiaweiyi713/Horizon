"""A durable policy loop that combines Python intelligence with Rust reliability."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

from .client import HorizonClient
from .memory import InterventionTracker, LearnedInterventionPolicy
from .policies import AgentAction, JsonActionPolicy
from .providers.base import LlmProvider


_RUNTIME_COMMANDS = frozenset(
    {
        "transition",
        "register_goal",
        "add_constraint",
        "set_plan",
        "open_subgoal",
        "complete_subgoal",
        "create_task",
        "start_task",
        "block_task",
        "unblock_task",
        "succeed_task",
        "fail_task",
        "retry_task",
        "cancel_task",
        "remember_failure",
        "record_decision",
        "record_evidence",
        "record_tool_invocation",
        "record_tool_success",
        "record_tool_failure",
        "record_tool_result",
        "update_budget",
        "update_environment",
        "apply_intervention",
        "suspend",
    }
)
_TERMINAL_STATES = frozenset(("completed", "failed"))


@dataclass
class AgentConfig:
    max_steps: int = 24
    token_budget: Optional[int] = None
    wall_time_budget_ms: Optional[int] = None
    checkpoint_on_finish: bool = True
    # Optional offline-trained state-decay policy. The default remains the Rust
    # heuristic so existing HTTP/native deployments retain their current
    # behavior until an experiment explicitly supplies a trained model.
    learned_intervention_policy: Optional[LearnedInterventionPolicy] = None


@dataclass
class RunResult:
    run_id: str
    state: str
    steps: int
    tokens_used: int
    anchors_injected: int
    error: Optional[str] = None


class DurableAgent:
    """Execute an LLM policy without making it the source of durable truth.

    Each model response chooses exactly one action. The action is validated by
    Rust, persisted as an event, and then materialized into the next state. A
    full State Anchor is provided only when the Rust heuristic emits a proactive
    intervention; ordinary turns receive a small current-state summary.
    """

    def __init__(
        self,
        client: HorizonClient,
        provider: LlmProvider,
        *,
        config: Optional[AgentConfig] = None,
    ) -> None:
        self.client = client
        self.policy = JsonActionPolicy(provider)
        self.config = config or AgentConfig()

    def start(
        self,
        goal: str,
        *,
        constraints: Sequence[str] = (),
        plan: Sequence[str] = (),
    ) -> str:
        created = self.client.create_run(goal)
        run_id = str(created["projection"]["run_id"])
        self.client.transition(run_id, "planning")
        for index, constraint in enumerate(constraints, 1):
            self.client.add_constraint(run_id, constraint, constraint_id="initial-constraint-{}".format(index))
        if plan:
            self.client.set_plan(run_id, plan)
        self.client.transition(run_id, "executing")
        return run_id

    def run(
        self,
        goal: str,
        *,
        constraints: Sequence[str] = (),
        plan: Sequence[str] = (),
    ) -> RunResult:
        return self.run_existing(self.start(goal, constraints=constraints, plan=plan))

    def run_existing(self, run_id: str, *, recovered_session: bool = False) -> RunResult:
        tracker = InterventionTracker(
            context_budget_tokens=self.config.token_budget,
            recovered_session=recovered_session,
        )
        started_at = time.monotonic()
        anchors_injected = 0
        for step in range(1, self.config.max_steps + 1):
            projection = self.client.get_run(run_id)
            state = str(projection["state"])
            if state in _TERMINAL_STATES:
                return RunResult(run_id, state, step - 1, tracker.tokens_used, anchors_injected)
            if self._budget_expired(tracker, started_at):
                self._suspend_for_budget(run_id, tracker, started_at)
                return RunResult(
                    run_id,
                    "suspended",
                    step - 1,
                    tracker.tokens_used,
                    anchors_injected,
                    "configured budget exhausted",
                )

            prompt, context_kind, injected_anchor = self._context_for_boundary(
                run_id, projection, tracker
            )
            anchors_injected += int(injected_anchor)

            operation_id = "llm:{}:{}".format(run_id, step)
            self.client.record_tool_invocation(
                run_id,
                operation_id,
                "llm_policy",
                {"step": step, "context_kind": context_kind, "context_chars": len(prompt)},
            )
            model_started_at = time.monotonic()
            try:
                action, response = self.policy.next_action(prompt)
            except Exception as error:
                self.client.record_tool_result(
                    run_id,
                    operation_id,
                    "llm_policy",
                    "failed",
                    error=str(error),
                    duration_ms=int((time.monotonic() - model_started_at) * 1000),
                    metadata={
                        "step": step,
                        "context_kind": context_kind,
                        "retryable": True,
                    },
                )
                return RunResult(
                    run_id,
                    str(self.client.get_run(run_id)["state"]),
                    step,
                    tracker.tokens_used,
                    anchors_injected,
                    "LLM policy call failed: {}".format(error),
                )
            self.client.record_tool_result(
                run_id,
                operation_id,
                "llm_policy",
                "succeeded",
                output={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "response_chars": len(response.content),
                },
                duration_ms=int((time.monotonic() - model_started_at) * 1000),
                metadata={"step": step, "context_kind": context_kind},
            )
            tracker.note_response(response.input_tokens, response.output_tokens)
            self._persist_budget(run_id, tracker, started_at)
            try:
                self._apply_action(run_id, action, tracker)
            except Exception as error:  # Surface provider policy mistakes without hiding the durable run.
                return RunResult(
                    run_id,
                    str(self.client.get_run(run_id)["state"]),
                    step,
                    tracker.tokens_used,
                    anchors_injected,
                    "action `{}` rejected: {}".format(action.action_type, error),
                )
        self.client.command(run_id, "suspend", {"reason": "agent reached max_steps"})
        return RunResult(
            run_id,
            "suspended",
            self.config.max_steps,
            tracker.tokens_used,
            anchors_injected,
            "agent reached max_steps",
        )

    def _apply_action(self, run_id: str, action: AgentAction, tracker: InterventionTracker) -> None:
        if action.action_type == "checkpoint":
            self.client.checkpoint(run_id)
            return
        if action.action_type == "recover":
            self.client.recover(run_id)
            tracker.recovered_session = True
            return
        if action.action_type == "intervene":
            self.client.intervene_if_needed(run_id, **tracker.consume_boundary())
            return
        if action.action_type == "execute_ready_tasks":
            outcomes = self.client.execute_ready_tasks(run_id)
            for outcome in outcomes:
                if outcome.get("output", {}).get("Err") or outcome.get("output", {}).get("err"):
                    tracker.note_failure()
            return
        if action.action_type == "finish":
            self._finish(run_id)
            return
        if action.action_type not in _RUNTIME_COMMANDS:
            raise ValueError("unknown policy action `{}`".format(action.action_type))
        data = dict(action.data)
        if action.action_type == "create_task":
            self._normalize_task(data)
        if action.action_type == "open_subgoal":
            self._normalize_subgoal(data)
            tracker.note_subgoal_switch()
        self.client.command(run_id, action.action_type, data)

    def _context_for_boundary(
        self,
        run_id: str,
        projection: Mapping[str, Any],
        tracker: InterventionTracker,
    ) -> tuple[str, str, bool]:
        """Choose compact vs State Anchor context at one durable boundary.

        The default path delegates the whole decision to Rust's explainable
        heuristic. A supplied learned policy runs in Python, then submits its
        full assessment to Rust so both injected and skipped decisions are
        durable, inspectable, and replayable.
        """
        compact_context = self._compact_context(projection)
        learned_policy = self.config.learned_intervention_policy
        if learned_policy is None:
            intervention = self.client.intervene_if_needed(run_id, **tracker.consume_boundary())
            if intervention is not None:
                context = self.client.state_anchor(run_id)["content"]
                return (
                    "A proactive State Anchor was injected. Use it as binding context:\n\n"
                    + str(context),
                    "state_anchor",
                    True,
                )
            return compact_context, "compact_projection", False

        sequence = int(projection.get("sequence") or 0)
        last_anchor_sequence = int(projection.get("last_anchor_sequence") or 0)
        signals = tracker.consume_boundary(
            steps_since_anchor=max(0, sequence - last_anchor_sequence)
        )
        immediate_context_pressure = min(
            1.0,
            learned_policy.controller.estimate_tokens(compact_context)
            / learned_policy.controller.context_window_tokens,
        )
        # A total run budget can be unset even when the immediate prompt is
        # becoming too large. Persist the stronger of the two pressures so the
        # learned predictor sees the same context risk its controller uses.
        signals["context_pressure"] = max(
            float(signals["context_pressure"]), immediate_context_pressure
        )
        # Rendering a candidate anchor is a read-only request. It lets the
        # controller account for the real token premium before deciding whether
        # its durable injection is worth the cost.
        anchor_context = str(self.client.state_anchor(run_id)["content"])
        decision = learned_policy.decide(
            signals,
            compact_context=compact_context,
            anchor_context=anchor_context,
            run_budget_pressure=tracker.run_budget_pressure(),
        )
        outcome = self.client.apply_intervention(
            run_id,
            decision.as_runtime_assessment(
                policy_id=learned_policy.policy_id,
                policy_version=learned_policy.predictor.model_version,
            ),
        )
        events = outcome.get("events", ())
        injected_anchor = any(
            isinstance(event, Mapping) and event.get("type") == "state_anchor_injected"
            for event in events
        )
        if injected_anchor != decision.inject_anchor:
            raise RuntimeError("runtime intervention outcome disagreed with the submitted policy action")
        if injected_anchor:
            return (
                "A learned, budget-aware State Anchor was injected. Use it as binding context:\n\n"
                + anchor_context,
                "learned_state_anchor",
                True,
            )
        return compact_context, "learned_compact_projection", False

    def _finish(self, run_id: str) -> None:
        state = str(self.client.get_run(run_id)["state"])
        if state == "executing":
            self.client.transition(run_id, "evaluating")
            state = "evaluating"
        if state == "evaluating":
            self.client.transition(run_id, "completed")
        elif state != "completed":
            raise ValueError("finish is only legal from executing/evaluating, got {}".format(state))
        if self.config.checkpoint_on_finish:
            self.client.checkpoint(run_id)

    def _persist_budget(self, run_id: str, tracker: InterventionTracker, started_at: float) -> None:
        self.client.update_budget(
            run_id,
            tokens_used=tracker.tokens_used,
            token_budget=self.config.token_budget,
            wall_time_used_ms=int((time.monotonic() - started_at) * 1000),
            wall_time_budget_ms=self.config.wall_time_budget_ms,
        )

    def _suspend_for_budget(self, run_id: str, tracker: InterventionTracker, started_at: float) -> None:
        self._persist_budget(run_id, tracker, started_at)
        state = str(self.client.get_run(run_id)["state"])
        if state not in _TERMINAL_STATES and state != "suspended":
            self.client.command(run_id, "suspend", {"reason": "configured agent budget exhausted"})

    def _budget_expired(self, tracker: InterventionTracker, started_at: float) -> bool:
        if self.config.token_budget is not None and tracker.tokens_used >= self.config.token_budget:
            return True
        if self.config.wall_time_budget_ms is not None:
            return (time.monotonic() - started_at) * 1000 >= self.config.wall_time_budget_ms
        return False

    @staticmethod
    def _normalize_task(data: dict[str, Any]) -> None:
        task = data.get("task")
        if not isinstance(task, dict):
            raise ValueError("create_task requires a task object")
        task.setdefault("id", str(uuid4()))
        task.setdefault("parent", None)
        task.setdefault("command", None)
        task.setdefault("working_dir", None)
        task.setdefault("priority", 0)
        task.setdefault("dependencies", [])
        task.setdefault("max_retries", 1)
        task.setdefault("timeout_ms", None)
        task.setdefault("resources", {})
        task.setdefault("executor", {"kind": "local"})

    @staticmethod
    def _normalize_subgoal(data: dict[str, Any]) -> None:
        subgoal = data.get("subgoal")
        if not isinstance(subgoal, dict):
            raise ValueError("open_subgoal requires a subgoal object")
        subgoal.setdefault("id", "subgoal-" + uuid4().hex[:12])
        subgoal.setdefault("completed", False)

    @staticmethod
    def _compact_context(projection: Mapping[str, Any]) -> str:
        cognitive = projection.get("cognitive", {})
        goal = cognitive.get("primary_goal") or "<unset>"
        open_subgoals = cognitive.get("open_subgoals") or []
        current_subgoal = open_subgoals[0].get("content") if open_subgoals else "<none>"
        return (
            "Current durable boundary (this is not a full memory injection):\n"
            "- Run state: {}\n"
            "- Primary goal: {}\n"
            "- Current subgoal: {}\n"
            "- Tasks: {}\n"
            "Choose one next action.".format(
                projection.get("state"),
                goal,
                current_subgoal,
                len(projection.get("tasks") or {}),
            )
        )

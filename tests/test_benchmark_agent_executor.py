"""Tests for the model-backed, objectively judged HorizonBench bridge."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from horizon_agent.agent import AgentConfig, DurableAgent, RunResult  # noqa: E402
from horizon_agent.benchmark import (  # noqa: E402
    BenchmarkExecutionError,
    DurableAgentEpisodeExecutor,
    DurableTraceJudge,
    _bounded_run_diagnostic,
    agent_config_from_condition,
    initial_plan_from_task,
)


def _context(
    *,
    metadata: dict[str, Any] | None = None,
    condition: dict[str, Any] | None = None,
    checkpoint_cadence: int = 1,
    fault_schedule: list[dict[str, Any]] | None = None,
    prompt: str = "Return one valid durable action.",
) -> Any:
    task = SimpleNamespace(
        task_id="durable-01",
        category="fault_recovery",
        goal="Restore the checkpointed durable run.",
        constraint="Reuse the recorded operation ID.",
        metadata=metadata or {},
    )
    return SimpleNamespace(
        task=task,
        manifest=SimpleNamespace(
            condition=SimpleNamespace(
                configuration=condition or {}, policy_revision="fixture-policy-v1"
            ),
            checkpoint_cadence=checkpoint_cadence,
            fault_schedule=fault_schedule or [],
            prompt=SimpleNamespace(text=prompt),
        ),
    )


def _projection(context: Any) -> dict[str, Any]:
    return {
        "state": "completed",
        "cognitive": {
            "primary_goal": context.task.goal,
            "constraints": [{"content": context.task.constraint}],
            "evidence": [{"content": "Audit checksum confirms the restored prefix."}],
            "failed_attempts": [{"approach": "restart from event zero"}],
        },
    }


class BenchmarkAgentExecutorTests(unittest.TestCase):
    def test_trace_judge_scores_durable_runtime_facts(self) -> None:
        context = _context(
            metadata={
                "horizon_trace_expectation": {
                    "required_event_types": ["evidence_recorded", "failure_remembered"],
                    "required_evidence_terms": ["checksum"],
                    "required_failure_approaches": ["event zero"],
                    "require_checkpoint": True,
                    "require_recovery": True,
                }
            }
        )
        events = [
            {"type": "run_created"},
            {"type": "evidence_recorded"},
            {"type": "failure_remembered"},
            {"type": "checkpoint_created"},
            {"type": "agent_recovered"},
            {"type": "state_transitioned"},
        ]
        outcome = DurableTraceJudge()(
            context,
            RunResult("run-1", "completed", 6, 42, 1, "agent reached max_steps"),
            _projection(context),
            events,
        )

        self.assertTrue(outcome["success"])
        self.assertTrue(outcome["goal_retained"])
        self.assertEqual(outcome["constraint_violations"], 0)
        self.assertTrue(outcome["recovery_attempted"])
        self.assertTrue(outcome["recovery_succeeded"])
        self.assertIsNone(outcome["recovery_distance"])
        self.assertEqual(outcome["tokens"], 42)
        self.assertEqual(outcome["diagnostic"], "agent reached max_steps")

    def test_trace_judge_diagnostic_uses_a_safe_operational_class(self) -> None:
        self.assertEqual(
            _bounded_run_diagnostic("LLM policy call failed: provider body with task transcript"),
            "LLM policy call failed",
        )
        self.assertEqual(
            _bounded_run_diagnostic("action `record_evidence` rejected: provider detail"),
            "policy action rejected",
        )
        self.assertEqual(
            _bounded_run_diagnostic("unexpected internal detail"),
            "agent stopped with an unclassified error",
        )

    def test_trace_judge_rejects_missing_expectations_and_forbidden_events(self) -> None:
        context = _context(
            metadata={
                "horizon_trace_expectation": {
                    "required_evidence_terms": ["missing required artifact"],
                    "forbidden_event_types": ["task_failed"],
                    "require_checkpoint": True,
                }
            }
        )
        projection = _projection(context)
        projection["cognitive"]["constraints"] = []
        outcome = DurableTraceJudge()(
            context,
            RunResult("run-2", "completed", 3, 4, 0),
            projection,
            [{"type": "task_failed"}],
        )

        self.assertFalse(outcome["success"])
        self.assertFalse(outcome["state_consistent"])
        self.assertEqual(outcome["constraint_violations"], 2)

    def test_agent_executor_runs_factories_then_hands_trace_to_judge(self) -> None:
        context = _context(
            metadata={"initial_plan": ["recover checkpoint", "verify operation ID"]},
            condition={"agent": {"max_steps": 7, "semantic_memory_limit": 0}},
        )
        calls: dict[str, Any] = {}

        class FakeClient:
            def get_run(self, run_id: str) -> dict[str, Any]:
                calls["get_run"] = run_id
                return _projection(context)

            def events(self, run_id: str) -> list[dict[str, Any]]:
                calls["events"] = run_id
                return [{"type": "checkpoint_created"}]

        class FakeAgent:
            def run(self, goal: str, *, constraints: tuple[str, ...], plan: tuple[str, ...]) -> RunResult:
                calls["run"] = (goal, constraints, plan)
                return RunResult("run-factory", "completed", 2, 8, 0)

        client = FakeClient()
        provider = object()

        def agent_factory(
            received_client: Any,
            received_provider: Any,
            config: AgentConfig,
            system_prompt: str,
        ) -> FakeAgent:
            calls["factory"] = (received_client, received_provider, config, system_prompt)
            return FakeAgent()

        def judge(received_context: Any, run: RunResult, projection: Any, events: Any) -> dict[str, Any]:
            calls["judge"] = (received_context, run, projection, events)
            return {
                "task_id": received_context.task.task_id,
                "category": received_context.task.category,
                "success": True,
                "goal_retained": True,
                "constraint_violations": 0,
            }

        executor = DurableAgentEpisodeExecutor(
            client_factory=lambda _context: client,
            provider_factory=lambda _context: provider,  # type: ignore[return-value]
            judge=judge,
            agent_factory=agent_factory,
        )
        outcome = executor(context)

        self.assertTrue(outcome["success"])
        self.assertEqual(calls["run"], (context.task.goal, (context.task.constraint,), tuple(context.task.metadata["initial_plan"])))
        self.assertIs(calls["factory"][0], client)
        self.assertIs(calls["factory"][1], provider)
        self.assertEqual(calls["factory"][2].max_steps, 7)
        self.assertEqual(calls["factory"][2].semantic_memory_limit, 0)
        self.assertEqual(calls["factory"][2].checkpoint_every_steps, 1)
        self.assertEqual(calls["factory"][3], context.manifest.prompt.text)
        self.assertEqual(calls["get_run"], "run-factory")
        self.assertEqual(calls["events"], "run-factory")

    def test_condition_and_task_factories_reject_ambiguous_inputs(self) -> None:
        self.assertEqual(agent_config_from_condition(_context(condition={"semantic_memory_limit": 0})).semantic_memory_limit, 0)
        fixed_control = agent_config_from_condition(
            _context(condition={"agent": {"anchor_strategy": "always"}})
        )
        self.assertEqual(fixed_control.anchor_strategy, "always")
        self.assertEqual(fixed_control.anchor_policy_revision, "fixture-policy-v1")
        self.assertEqual(initial_plan_from_task(_context(metadata={"initial_plan": []})), ())
        with self.assertRaisesRegex(BenchmarkExecutionError, "unrecognized"):
            agent_config_from_condition(_context(condition={"agent": {"unknown": 1}}))
        with self.assertRaisesRegex(BenchmarkExecutionError, "anchor_strategy"):
            agent_config_from_condition(_context(condition={"agent": {"anchor_strategy": 1}}))
        with self.assertRaisesRegex(BenchmarkExecutionError, "array of strings"):
            initial_plan_from_task(_context(metadata={"initial_plan": "one step"}))
        with self.assertRaisesRegex(BenchmarkExecutionError, "horizon_trace_expectation"):
            DurableTraceJudge()(_context(), RunResult("run", "completed", 1, 0, 0), {}, [])
        factory_calls: list[str] = []
        executor = DurableAgentEpisodeExecutor(
            client_factory=lambda _context: factory_calls.append("client") or object(),
            provider_factory=lambda _context: factory_calls.append("provider") or object(),  # type: ignore[return-value]
            judge=lambda *_args: {},
        )
        with self.assertRaisesRegex(BenchmarkExecutionError, "fault_schedule"):
            executor(_context(fault_schedule=[{"after_boundary": 3, "kind": "restart"}]))
        self.assertEqual(factory_calls, [])

    def test_agent_executor_preflight_validates_static_inputs_without_factories(self) -> None:
        context = _context(
            metadata={
                "initial_plan": ["recover checkpoint", "verify operation ID"],
                "horizon_trace_expectation": {"required_event_types": ["checkpoint_created"]},
            },
            condition={"agent": {"max_steps": 7, "semantic_memory_limit": 0}},
            checkpoint_cadence=2,
        )
        factory_calls: list[str] = []
        executor = DurableAgentEpisodeExecutor(
            client_factory=lambda _context: factory_calls.append("client") or object(),
            provider_factory=lambda _context: factory_calls.append("provider") or object(),  # type: ignore[return-value]
            judge=lambda *_args: {},
        )

        report = executor.preflight(context)

        self.assertEqual(factory_calls, [])
        self.assertEqual(report["task_id"], "durable-01")
        self.assertEqual(report["checkpoint_cadence"], 2)
        self.assertEqual(report["initial_plan_steps"], 2)
        self.assertEqual(report["agent"]["checkpoint_every_steps"], 2)
        self.assertEqual(report["agent"]["max_steps"], 7)
        self.assertEqual(report["agent"]["anchor_strategy"], "runtime_heuristic")

    def test_trace_judge_preflight_checks_immutable_expectation(self) -> None:
        context = _context(
            metadata={
                "horizon_trace_expectation": {
                    "required_event_types": ["checkpoint_created"],
                    "require_checkpoint": True,
                }
            }
        )

        report = DurableTraceJudge().preflight(context)

        self.assertEqual(report["required_event_types"], ["checkpoint_created"])
        self.assertTrue(report["require_checkpoint"])

    def test_compact_context_keeps_bounded_binding_facts_without_full_anchor(self) -> None:
        projection = {
            "state": "executing",
            "tasks": {"a": {}, "b": {}},
            "cognitive": {
                "primary_goal": "Preserve durable facts.",
                "open_subgoals": [{"content": "finish validation"}],
                "constraints": [{"content": "constraint {}".format(index)} for index in range(5)],
                "current_plan": ["step {}".format(index) for index in range(4)],
                "failed_attempts": [{"approach": "failed {}".format(index)} for index in range(4)],
                "evidence": [{"content": "full-anchor-only evidence"}],
                "decisions": [{"decision": "one bounded decision"}],
            },
        }
        compact = DurableAgent._compact_context(projection)

        self.assertIn("Binding constraints: constraint 0 | constraint 1 | constraint 2 | constraint 3", compact)
        self.assertNotIn("constraint 4", compact)
        self.assertIn("Approved plan: step 0 | step 1 | step 2", compact)
        self.assertNotIn("step 3", compact)
        self.assertIn("Known failed approaches: failed 0 | failed 1 | failed 2", compact)
        self.assertNotIn("failed 3", compact)
        self.assertNotIn("full-anchor-only evidence", compact)
        self.assertIn("Durable records: decisions=1, evidence=1, remembered failures=4", compact)


if __name__ == "__main__":
    unittest.main()

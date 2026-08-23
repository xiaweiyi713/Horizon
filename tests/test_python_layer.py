"""Unit tests for dependency-free Python policy helpers."""

from __future__ import annotations

import sys
import unittest
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from horizon_agent.adapters import AdapterRegistry, AdapterResult, FunctionAdapter  # noqa: E402
from horizon_agent.agent import AgentConfig, DurableAgent  # noqa: E402
from horizon_agent.memory import InterventionTracker  # noqa: E402
from horizon_agent.memory import (  # noqa: E402
    AdaptiveContextBudgetController,
    DecayFeatures,
    DecayTrainingExample,
    LearnedInterventionPolicy,
    LogisticDecayPredictor,
)
from horizon_agent.native import NativeHorizonRuntime  # noqa: E402
import horizon_agent.native as native_module  # noqa: E402
from horizon_agent.policies import AgentAction, JsonActionPolicy  # noqa: E402
from horizon_agent.providers import ScriptedProvider  # noqa: E402


class PythonLayerTests(unittest.TestCase):
    def test_action_policy_parses_single_json_action(self) -> None:
        provider = ScriptedProvider(
            [{"thought": "goal is complete", "action": {"type": "finish", "data": {}}}]
        )
        action, response = JsonActionPolicy(provider).next_action("Current durable boundary")
        self.assertEqual(action, AgentAction(action_type="finish", data={}, thought="goal is complete"))
        self.assertEqual(response.input_tokens, 0)
        self.assertEqual(len(provider.history), 1)

    def test_action_policy_uses_a_frozen_supplied_system_prompt(self) -> None:
        provider = ScriptedProvider([{"action": {"type": "finish", "data": {}}}])
        JsonActionPolicy(provider, system_prompt="Frozen benchmark prompt v1.").next_action("boundary")
        self.assertEqual(provider.history[0][0].content, "Frozen benchmark prompt v1.")
        with self.assertRaisesRegex(ValueError, "system_prompt"):
            JsonActionPolicy(ScriptedProvider([]), system_prompt="  ")

    def test_checkpoint_cadence_is_validated_and_skips_duplicate_boundary_checkpoint(self) -> None:
        class CheckpointClient:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def checkpoint(self, run_id: str) -> None:
                self.calls.append(run_id)

        with self.assertRaisesRegex(ValueError, "checkpoint_every_steps"):
            DurableAgent(CheckpointClient(), ScriptedProvider([]), config=AgentConfig(checkpoint_every_steps=0))
        client = CheckpointClient()
        agent = DurableAgent(
            client,
            ScriptedProvider([]),
            config=AgentConfig(checkpoint_every_steps=2),
        )
        agent._checkpoint_if_due("run-cadence", 1, checkpointed=False)
        agent._checkpoint_if_due("run-cadence", 2, checkpointed=False)
        agent._checkpoint_if_due("run-cadence", 4, checkpointed=True)
        self.assertEqual(client.calls, ["run-cadence"])

    def test_recovered_agent_allocates_new_llm_operation_ids_from_durable_results(self) -> None:
        projection = {
            "tool_results": [
                {"tool": "llm_policy", "operation_id": "llm:run-recovered:1"},
                {"tool": "llm_policy", "operation_id": "llm:run-recovered:3"},
                {"tool": "other_tool", "operation_id": "llm:run-recovered:99"},
                {"tool": "llm_policy", "operation_id": "llm:another-run:12"},
                {"tool": "llm_policy", "operation_id": "llm:run-recovered:not-a-number"},
            ]
        }
        self.assertEqual(DurableAgent._next_llm_policy_step(projection, "run-recovered"), 4)
        self.assertEqual(DurableAgent._next_llm_policy_step({}, "run-recovered"), 1)

    def test_fixed_anchor_strategies_are_durable_and_auditable(self) -> None:
        class FixedStrategyClient:
            def __init__(self) -> None:
                self.assessments = []
                self.anchor_reads = 0

            def apply_intervention(self, _run_id, assessment):
                self.assessments.append(assessment)
                events = [{"type": "state_decay_assessed", "data": {"assessment": assessment}}]
                if assessment["action"] == "inject_anchor":
                    events.append({"type": "state_anchor_injected", "data": {}})
                return {"events": events}

            def state_anchor(self, _run_id):
                self.anchor_reads += 1
                return {"content": "STATE ANCHOR\nPrimary Goal: preserve durable facts"}

        projection = {
            "sequence": 9,
            "last_anchor_sequence": 2,
            "state": "executing",
            "cognitive": {
                "primary_goal": "preserve durable facts",
                "constraints": [{"content": "do not mutate the source"}],
            },
            "tasks": {},
            "tool_results": [
                {
                    "tool": "artifact_workspace",
                    "status": "succeeded",
                    "operation_id": "fixture:read",
                    "metadata": {
                        "environment": "artifact_workspace_v1",
                        "model_context": {
                            "kind": "artifact_workspace_read_v1",
                            "files": [
                                {
                                    "path": "input/brief.txt",
                                    "content": "produce the durable artifact\n",
                                }
                            ],
                        },
                    },
                }
            ],
        }
        disabled_client = FixedStrategyClient()
        disabled = DurableAgent(
            disabled_client,
            ScriptedProvider([]),
            config=AgentConfig(anchor_strategy="disabled", anchor_policy_revision="fixture-v1"),
        )
        context, kind, injected = disabled._context_for_boundary(
            "run-disabled", projection, InterventionTracker()
        )
        self.assertFalse(injected)
        self.assertEqual(kind, "disabled_compact_projection")
        self.assertIn("Primary goal: preserve durable facts", context)
        self.assertIn("untrusted task input", context)
        self.assertIn("produce the durable artifact", context)
        self.assertEqual(disabled_client.anchor_reads, 0)
        self.assertEqual(disabled_client.assessments[0]["action"], "continue")
        self.assertEqual(disabled_client.assessments[0]["signals"]["steps_since_anchor"], 7)
        self.assertEqual(disabled_client.assessments[0]["metadata"]["anchor_strategy"], "disabled")

        always_client = FixedStrategyClient()
        always = DurableAgent(
            always_client,
            ScriptedProvider([]),
            config=AgentConfig(anchor_strategy="always", anchor_policy_revision="fixture-v1"),
        )
        context, kind, injected = always._context_for_boundary(
            "run-always", projection, InterventionTracker()
        )
        self.assertTrue(injected)
        self.assertEqual(kind, "always_state_anchor")
        self.assertIn("STATE ANCHOR", context)
        self.assertIn("untrusted task input", context)
        self.assertIn("produce the durable artifact", context)
        self.assertEqual(always_client.anchor_reads, 1)
        self.assertEqual(always_client.assessments[0]["action"], "inject_anchor")
        self.assertEqual(always_client.assessments[0]["policy_version"], "fixture-v1")

        with self.assertRaisesRegex(ValueError, "anchor_strategy"):
            DurableAgent(
                object(), ScriptedProvider([]), config=AgentConfig(anchor_strategy="periodic")
            )

    def test_tracker_consumes_short_lived_risk_signals(self) -> None:
        tracker = InterventionTracker(context_budget_tokens=100, recovered_session=True)
        tracker.note_response(40, 20)
        tracker.note_subgoal_switch()
        tracker.note_failure()
        signal = tracker.consume_boundary()
        self.assertEqual(signal["context_pressure"], 0.6)
        self.assertTrue(signal["recovered_session"])
        self.assertEqual(tracker.consume_boundary()["subgoal_switches"], 0)
        self.assertFalse(tracker.consume_boundary()["recovered_session"])

    def test_learned_decay_predictor_trains_round_trips_and_drives_budget_decision(self) -> None:
        training_rows = [
            DecayTrainingExample(DecayFeatures(), False),
            DecayTrainingExample(DecayFeatures(context_pressure=0.1), False),
            DecayTrainingExample(DecayFeatures(steps_since_anchor=2), False),
            DecayTrainingExample(
                DecayFeatures(
                    steps_since_anchor=12,
                    context_pressure=0.9,
                    subgoal_switches=3,
                    recent_failures=2,
                    recovered_session=True,
                ),
                True,
            ),
            DecayTrainingExample(
                DecayFeatures(
                    steps_since_anchor=10,
                    context_pressure=0.8,
                    recent_failures=2,
                ),
                True,
            ),
            DecayTrainingExample(
                DecayFeatures(context_pressure=0.7, subgoal_switches=3, recovered_session=True),
                True,
            ),
        ]
        predictor = LogisticDecayPredictor.fit(training_rows, epochs=600, model_version="fixture-v1")
        low_risk = predictor.predict_proba(DecayFeatures())
        high_features = DecayFeatures(
            steps_since_anchor=12,
            context_pressure=0.9,
            subgoal_switches=3,
            recent_failures=2,
            recovered_session=True,
        )
        high_risk = predictor.predict_proba(high_features)
        self.assertGreater(high_risk, low_risk)
        self.assertGreater(high_risk, 0.7)
        self.assertLess(low_risk, 0.3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decay-model.json"
            predictor.save(path)
            restored = LogisticDecayPredictor.load(path)
        self.assertAlmostEqual(restored.predict_proba(high_features), high_risk)

        controller = AdaptiveContextBudgetController(context_window_tokens=256)
        decision = controller.decide(
            features=high_features,
            risk_score=0.91,
            compact_context="compact" * 8,
            anchor_context="STATE ANCHOR\n" * 80,
            run_budget_pressure=0.95,
        )
        self.assertTrue(decision.inject_anchor)
        assessment = decision.as_runtime_assessment(
            policy_id="logistic_state_decay", policy_version="fixture-v1"
        )
        self.assertGreaterEqual(
            assessment["risk_score_milli"], assessment["threshold_milli"]
        )
        self.assertEqual(assessment["action"], "inject_anchor")

    def test_budget_controller_decides_in_the_same_milli_units_rust_persists(self) -> None:
        controller = AdaptiveContextBudgetController(
            context_window_tokens=256,
            base_threshold=0.5005,
            min_threshold=0.5005,
            max_threshold=0.5005,
            context_urgency_weight=0.0,
            run_budget_weight=0.0,
            anchor_cost_weight=0.0,
        )
        decision = controller.decide(
            features=DecayFeatures(),
            risk_score=0.5004,
            compact_context="compact",
            anchor_context="anchor",
            run_budget_pressure=0.0,
        )
        # Both values serialize to 500 milli-units. A raw floating-point
        # comparison would choose continue, but the durable Rust contract must
        # choose inject to remain internally consistent.
        self.assertTrue(decision.inject_anchor)
        assessment = decision.as_runtime_assessment(policy_id="fixture", policy_version="v1")
        self.assertEqual(
            (assessment["risk_score_milli"], assessment["threshold_milli"], assessment["action"]),
            (500, 500, "inject_anchor"),
        )

    def test_durable_agent_submits_learned_assessment_before_using_anchor_context(self) -> None:
        predictor = LogisticDecayPredictor(
            weights=(-4.0, 1.0, 1.5, 1.0, 1.0, 2.0), model_version="fixture-v1"
        )
        policy = LearnedInterventionPolicy(
            predictor=predictor,
            controller=AdaptiveContextBudgetController(context_window_tokens=256),
        )
        client = _LearnedInterventionClient()
        agent = DurableAgent(
            client,
            ScriptedProvider([]),
            config=AgentConfig(learned_intervention_policy=policy),
        )
        context, context_kind, injected = agent._context_for_boundary(
            "run-learned",
            {
                "sequence": 16,
                "last_anchor_sequence": 0,
                "state": "executing",
                "cognitive": {"primary_goal": "retain the goal"},
                "tasks": {},
            },
            InterventionTracker(context_budget_tokens=100, tokens_used=95, recovered_session=True),
        )
        self.assertTrue(injected)
        self.assertEqual(context_kind, "learned_state_anchor")
        self.assertIn("STATE ANCHOR", context)
        self.assertEqual(client.assessment["action"], "inject_anchor")
        self.assertEqual(client.assessment["signals"]["steps_since_anchor"], 16)
        self.assertEqual(client.assessment["policy_version"], "fixture-v1")

    def test_learned_agent_persists_immediate_context_pressure_without_a_run_budget(self) -> None:
        predictor = LogisticDecayPredictor(
            weights=(-8.0, 0.0, 4.0, 0.0, 0.0, 0.0), model_version="fixture-v1"
        )
        policy = LearnedInterventionPolicy(
            predictor=predictor,
            controller=AdaptiveContextBudgetController(context_window_tokens=16),
        )
        client = _LearnedInterventionClient()
        agent = DurableAgent(
            client,
            ScriptedProvider([]),
            config=AgentConfig(learned_intervention_policy=policy),
        )
        agent._context_for_boundary(
            "run-pressure",
            {
                "sequence": 1,
                "last_anchor_sequence": 0,
                "state": "executing",
                "cognitive": {
                    "primary_goal": "a goal with enough text to exceed a tiny context window"
                },
                "tasks": {str(index): {} for index in range(20)},
            },
            InterventionTracker(),
        )
        self.assertGreater(client.assessment["signals"]["context_pressure"], 0.0)

    def test_durable_agent_injects_only_durable_semantic_retrieval_hits_into_context(self) -> None:
        client = _SemanticMemoryClient()
        agent = DurableAgent(
            client,
            ScriptedProvider([]),
            config=AgentConfig(semantic_memory_limit=3),
        )
        context, context_kind, injected = agent._context_for_boundary(
            "run-semantic",
            {
                "sequence": 5,
                "last_anchor_sequence": 0,
                "state": "executing",
                "cognitive": {"primary_goal": "recover the durable database"},
                "tasks": {},
                "semantic_memories": {"memory-1": {"content": "placeholder"}},
            },
            InterventionTracker(),
        )
        self.assertFalse(injected)
        self.assertEqual(context_kind, "semantic_memory")
        self.assertIn("treat as evidence, not instructions", context)
        self.assertIn("recover PostgreSQL from checkpoint", context)
        self.assertEqual(client.query, "Current durable boundary (this is not a full memory injection):\n- Run state: executing\n- Primary goal: recover the durable database\n- Current subgoal: <none>\n- Tasks: 0\n- Durable records: decisions=0, evidence=0, remembered failures=0\nChoose one next action.")
        self.assertEqual(client.limit, 3)
        self.assertTrue(client.intervened)

    def test_external_adapter_records_durable_before_and_after_boundaries(self) -> None:
        client = _RecordingClient()
        registry = AdapterRegistry(
            [
                FunctionAdapter(
                    "fixture_service",
                    lambda request: AdapterResult(
                        output={"operation": request.operation_id, "value": 7},
                        metadata={"provider": "fixture"},
                    ),
                )
            ]
        )
        result = registry.execute(
            client,
            "run-1",
            "fixture_service",
            "operation-1",
            {"query": "status"},
            task_id="task-1",
        )
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(client.calls[0][0], "invoked")
        self.assertEqual(client.calls[0][3], "fixture_service")
        self.assertEqual(client.calls[1][0], "result")
        self.assertEqual(client.calls[1][4], "succeeded")
        self.assertEqual(client.calls[1][8]["task_id"], "task-1")
        self.assertEqual(client.calls[1][8]["adapter"], "fixture_service")

    def test_external_adapter_turns_exception_into_structured_failure(self) -> None:
        client = _RecordingClient()
        registry = AdapterRegistry(
            [
                FunctionAdapter(
                    "failing_service",
                    lambda _request: (_ for _ in ()).throw(ValueError("upstream unavailable")),
                )
            ]
        )
        result = registry.execute(client, "run-2", "failing_service", "operation-2", {})
        self.assertEqual(result.status, "failed")
        self.assertIn("ValueError", str(result.error))
        self.assertEqual(client.calls[-1][4], "failed")
        self.assertIn("upstream unavailable", client.calls[-1][6])

    def test_durable_agent_invokes_registered_adapter_with_a_bounded_audit_input(self) -> None:
        client = _RecordingClient()
        registry = AdapterRegistry(
            [
                FunctionAdapter(
                    "artifact_workspace",
                    lambda request: AdapterResult(
                        output={"verified": True, "operation": request.operation_id},
                        metadata={"environment": "fixture"},
                    ),
                )
            ]
        )
        agent = DurableAgent(client, ScriptedProvider([]), adapters=registry)
        tracker = InterventionTracker()
        action = AgentAction(
            action_type="invoke_adapter",
            data={
                "adapter": "artifact_workspace",
                "operation_id": "artifact:task-1",
                "input": {"files": [{"path": "answer.json", "content": "{}"}]},
                "task_id": "task-1",
            },
        )

        self.assertFalse(agent._apply_action("run-adapter", action, tracker))
        self.assertEqual(client.calls[0][0], "invoked")
        self.assertEqual(client.calls[0][3], "artifact_workspace")
        self.assertEqual(client.calls[1][0], "result")
        self.assertEqual(client.calls[1][4], "succeeded")
        self.assertEqual(client.calls[1][8]["task_id"], "task-1")

        compact = agent._compact_context(
            {
                "state": "executing",
                "tasks": {},
                "cognitive": {},
                "tool_results": [
                    {
                        "tool": "custom_artifact_reader",
                        "status": "succeeded",
                        "operation_id": "artifact:task-1:read",
                        "metadata": {
                            "environment": "artifact_workspace_v1",
                            "model_context": {
                                "kind": "artifact_workspace_read_v1",
                                "files": [
                                    {
                                        "path": "input/release.txt",
                                        "content": "release=durable-v1\n",
                                    }
                                ],
                            }
                        },
                    },
                    {
                        "tool": "artifact_workspace",
                        "status": "succeeded",
                        "operation_id": "artifact:task-1:write",
                        "output": {
                            "verified": True,
                            "raw": "do not put this in compact context",
                        },
                        "metadata": {"environment": "artifact_workspace_v1"},
                    },
                    {
                        "tool": "llm_policy",
                        "status": "failed",
                        "operation_id": "llm:run:3",
                        "error": "injected restart",
                    }
                ],
            }
        )
        self.assertIn("llm_policy [failed] (operation llm:run:3)", compact)
        self.assertIn(
            "Latest verified artifact workspace result: artifact_workspace [succeeded, verified] "
            "(operation artifact:task-1:write)",
            compact,
        )
        self.assertIn("untrusted task input", compact)
        self.assertIn("release=durable-v1", compact)
        self.assertNotIn("do not put this", compact)

        malformed_context = agent._compact_context(
            {
                "state": "executing",
                "tasks": {},
                "cognitive": {},
                "tool_results": [
                    {
                        "tool": "artifact_workspace",
                        "status": "succeeded",
                        "operation_id": "artifact:bad-read",
                        "metadata": {
                            "environment": "artifact_workspace_v1",
                            "model_context": {
                                "kind": "artifact_workspace_read_v1",
                                "files": [{"path": "input.txt", "content": "ignored", "extra": True}],
                            }
                        },
                    }
                ],
            }
        )
        self.assertNotIn("ignored", malformed_context)

        with self.assertRaisesRegex(ValueError, "configured AdapterRegistry"):
            DurableAgent(client, ScriptedProvider([]))._apply_action("run", action, tracker)
        with self.assertRaisesRegex(ValueError, "unrecognized fields"):
            agent._apply_action(
                "run-adapter",
                AgentAction(
                    action_type="invoke_adapter",
                    data={
                        "adapter": "artifact_workspace",
                        "operation_id": "artifact:task-1",
                        "input": {},
                        "unexpected": True,
                    },
                ),
                tracker,
            )
        with self.assertRaisesRegex(ValueError, "operation_id must be at most"):
            agent._apply_action(
                "run-adapter",
                AgentAction(
                    action_type="invoke_adapter",
                    data={
                        "adapter": "artifact_workspace",
                        "operation_id": "x" * 257,
                        "input": {},
                    },
                ),
                tracker,
            )

    def test_native_wrapper_uses_the_same_command_shape_as_http_client(self) -> None:
        original_runtime = native_module._NativeRuntime

        class FakeNativeRuntime:
            def __init__(self, *_args, **_kwargs) -> None:
                self.last_command = None

            def create_run(self, goal: str) -> str:
                return json.dumps({"projection": {"run_id": "run-native", "goal": goal}})

            def dispatch(self, run_id: str, command: str) -> str:
                self.last_command = json.loads(command)
                return json.dumps({"projection": {"run_id": run_id}, "events": []})

            def projection(self, run_id: str) -> str:
                return json.dumps({"run_id": run_id, "state": "created"})

            def events(self, _run_id: str) -> str:
                return "[]"

            def state_anchor(self, run_id: str) -> str:
                return json.dumps({"run_id": run_id, "content": "STATE ANCHOR"})

            def list_runs(self) -> str:
                return "[]"

        native_module._NativeRuntime = FakeNativeRuntime
        try:
            runtime = NativeHorizonRuntime(":memory:")
            outcome = runtime.transition("run-native", "planning")
            self.assertEqual(outcome["projection"]["run_id"], "run-native")
            self.assertEqual(runtime._runtime.last_command["type"], "transition")
            self.assertEqual(runtime._runtime.last_command["data"], {"to": "planning"})
            runtime.apply_intervention(
                "run-native",
                {
                    "policy_id": "fixture",
                    "risk_score_milli": 800,
                    "threshold_milli": 700,
                    "signals": {"context_pressure": 0.8},
                    "action": "inject_anchor",
                    "reason": "fixture learned decision",
                },
            )
            self.assertEqual(runtime._runtime.last_command["type"], "apply_intervention")
            runtime.retrieve_semantic_memory("run-native", "recover postgres checkpoint", limit=3)
            self.assertEqual(runtime._runtime.last_command["type"], "retrieve_semantic_memory")
            self.assertEqual(
                runtime._runtime.last_command["data"],
                {"query": "recover postgres checkpoint", "limit": 3},
            )
        finally:
            native_module._NativeRuntime = original_runtime


class _RecordingClient:
    def __init__(self) -> None:
        self.calls = []

    def record_tool_invocation(self, run_id, operation_id, tool, input):
        self.calls.append(("invoked", run_id, operation_id, tool, input))
        return {}

    def record_tool_result(
        self,
        run_id,
        operation_id,
        tool,
        status,
        *,
        output=None,
        error=None,
        duration_ms=None,
        metadata=None,
    ):
        self.calls.append(
            ("result", run_id, operation_id, tool, status, output, error, duration_ms, metadata)
        )
        return {}


class _LearnedInterventionClient:
    def __init__(self) -> None:
        self.assessment = None

    def state_anchor(self, _run_id):
        return {"content": "STATE ANCHOR\nPrimary Goal: retain the goal"}

    def apply_intervention(self, _run_id, assessment):
        self.assessment = assessment
        events = [{"type": "state_decay_assessed", "data": {"assessment": assessment}}]
        if assessment["action"] == "inject_anchor":
            events.append({"type": "state_anchor_injected", "data": {}})
        return {"events": events}


class _SemanticMemoryClient:
    def __init__(self) -> None:
        self.query = None
        self.limit = None
        self.intervened = False

    def retrieve_semantic_memory(self, _run_id, query, *, limit):
        self.query = query
        self.limit = limit
        retrieval = {
            "query": query,
            "algorithm": "hybrid_lexical_v1",
            "hits": [
                {
                    "memory_id": "fixture-memory",
                    "kind": "episodic",
                    "content": "recover PostgreSQL from checkpoint before retrying the migration",
                    "score_milli": 934,
                }
            ],
        }
        return {
            "projection": {"sequence": 6, "last_anchor_sequence": 0},
            "events": [{"type": "semantic_memory_retrieved", "data": {"retrieval": retrieval}}],
        }

    def intervene_if_needed(self, _run_id, **_signals):
        self.intervened = True
        return None


if __name__ == "__main__":
    unittest.main()

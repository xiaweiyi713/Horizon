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


if __name__ == "__main__":
    unittest.main()

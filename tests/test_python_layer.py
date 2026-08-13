"""Unit tests for dependency-free Python policy helpers."""

from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from horizon_agent.adapters import AdapterRegistry, AdapterResult, FunctionAdapter  # noqa: E402
from horizon_agent.memory import InterventionTracker  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()

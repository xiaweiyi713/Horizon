"""Unit tests for dependency-free Python policy helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from horizon_agent.memory import InterventionTracker  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()

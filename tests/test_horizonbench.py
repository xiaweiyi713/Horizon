"""Regression tests for HorizonBench metric definitions and fixture integrity."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.run import default_strategies, load_scenarios, run_suite  # noqa: E402
from benchmarks.horizonbench.score import EpisodeResult, score_results  # noqa: E402


class HorizonBenchTests(unittest.TestCase):
    def test_fixture_has_six_balanced_categories(self) -> None:
        scenarios = load_scenarios()
        categories = {}
        for scenario in scenarios:
            categories[scenario.category] = categories.get(scenario.category, 0) + 1
        self.assertEqual(len(scenarios), 30)
        self.assertEqual(set(categories.values()), {5})
        self.assertEqual(len(categories), 6)

    def test_metric_repeated_failure_rate(self) -> None:
        metrics = score_results(
            [
                EpisodeResult(
                    task_id="one",
                    category="failure_avoidance",
                    success=False,
                    goal_retained=True,
                    constraint_violations=0,
                    failure_actions=("bad approach", "new approach"),
                    known_failed_approaches=("bad approach",),
                    tokens=20,
                ),
                EpisodeResult(
                    task_id="two",
                    category="goal_retention",
                    success=True,
                    goal_retained=True,
                    constraint_violations=0,
                    tokens=10,
                ),
            ]
        )
        self.assertEqual(metrics.repeated_failure_rate, 0.5)
        self.assertEqual(metrics.tokens_per_successful_task, 30.0)

    def test_reference_suite_includes_requested_ablations(self) -> None:
        report = run_suite(load_scenarios(), default_strategies(include_ablations=True))
        names = set(report["strategies"])
        self.assertIn("Horizon_full", names)
        self.assertIn("Horizon_minus_state_anchor", names)
        self.assertIn("Horizon_minus_failure_memory", names)
        self.assertIn("Horizon_minus_checkpoint_recovery", names)
        self.assertIn("Horizon_minus_structured_cognitive_state", names)

    def test_research_controls_exercise_learned_adaptive_and_fixed_threshold_paths(self) -> None:
        report = run_suite(load_scenarios(), default_strategies(include_ablations=False))
        adaptive = report["strategies"]["Horizon_learned_adaptive"]
        fixed = report["strategies"]["Horizon_learned_fixed_threshold"]
        self.assertGreater(adaptive["anchors_injected"], 0)
        self.assertGreater(fixed["anchors_injected"], 0)
        # The fixture gives the adaptive controller a high-context boundary
        # that the fixed-threshold control deliberately misses. This validates
        # that the benchmark invokes distinct controller behavior, not aliases.
        self.assertNotEqual(adaptive["anchors_injected"], fixed["anchors_injected"])

    def test_semantic_memory_on_off_control_is_not_an_alias(self) -> None:
        report = run_suite(load_scenarios(), default_strategies(include_ablations=False))
        retrieved = report["strategies"]["Horizon_semantic_memory"]
        retrieval_off = report["strategies"]["Horizon_semantic_memory_off"]
        self.assertGreater(retrieved["task_success_rate"], retrieval_off["task_success_rate"])
        self.assertGreater(retrieved["tokens_per_successful_task"], 0)


if __name__ == "__main__":
    unittest.main()

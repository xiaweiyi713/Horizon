"""Deterministic HorizonBench smoke harness and ablation scaffold.

This file is deliberately not a substitute for a model-backed paper result. It
serves three practical purposes: regression tests for metric plumbing, a clear
adapter contract, and a reproducible example of how proactive state intervention
is evaluated against baselines. Pass externally collected JSONL to ``score.py``
for real model runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

try:  # Supports both `python file.py` and `python -m benchmarks.horizonbench.run`.
    from .score import EpisodeResult, score_results, write_jsonl
except ImportError:  # pragma: no cover - direct-script compatibility
    from score import EpisodeResult, score_results, write_jsonl


ROOT = Path(__file__).resolve().parent
DEFAULT_TASKS = ROOT / "tasks.json"


@dataclass(frozen=True)
class Scenario:
    task_id: str
    category: str
    goal: str
    constraint: str
    failed_approach: str
    needs_recovery: bool
    context_steps: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Scenario":
        return cls(
            task_id=str(value["id"]),
            category=str(value["category"]),
            goal=str(value["goal"]),
            constraint=str(value["constraint"]),
            failed_approach=str(value["failed_approach"]),
            needs_recovery=bool(value.get("needs_recovery", False)),
            context_steps=int(value.get("context_steps", 16)),
        )


class Strategy:
    name = "strategy"

    def run(self, scenario: Scenario) -> EpisodeResult:
        raise NotImplementedError


class RawContextStrategy(Strategy):
    name = "B0_raw_context"

    def run(self, scenario: Scenario) -> EpisodeResult:
        # A fixed history window loses early instructions on deliberately long
        # trajectories. It is a transparent approximation of raw truncation.
        retained = scenario.context_steps <= 5
        failure_visible = retained
        return _result(
            scenario,
            goal_retained=retained,
            constraint_retained=retained,
            failure_visible=failure_visible,
            recovery_ok=not scenario.needs_recovery or retained,
            state_consistent=retained,
            tokens=1_800,
        )


class PeriodicSummaryStrategy(Strategy):
    name = "B1_periodic_summary"

    def run(self, scenario: Scenario) -> EpisodeResult:
        # Periodic summary preserves declared goals and constraints but commonly
        # drops operational failure reasons and exact environment state.
        recovery_ok = not scenario.needs_recovery or scenario.category != "fault_recovery"
        return _result(
            scenario,
            goal_retained=True,
            constraint_retained=True,
            failure_visible=False,
            recovery_ok=recovery_ok,
            state_consistent=recovery_ok,
            tokens=2_100,
        )


class VectorRetrievalStrategy(Strategy):
    name = "B2_vector_retrieval"

    def run(self, scenario: Scenario) -> EpisodeResult:
        # Passive semantic retrieval helps goal/constraint lookup but is weaker
        # when an interruption changes the relevant action without a query.
        failure_visible = scenario.category != "failure_avoidance"
        recovery_ok = not scenario.needs_recovery or scenario.category != "cross_session_recovery"
        return _result(
            scenario,
            goal_retained=True,
            constraint_retained=True,
            failure_visible=failure_visible,
            recovery_ok=recovery_ok,
            state_consistent=recovery_ok,
            tokens=2_500,
        )


class StructuredPassiveStrategy(Strategy):
    name = "B3_structured_passive"

    def run(self, scenario: Scenario) -> EpisodeResult:
        # State is stored correctly, but without intervention some high-pressure
        # decision boundaries still fail to make the stored state behavioral.
        pressure_loss = scenario.category in {"long_context_degradation", "fault_recovery"}
        return _result(
            scenario,
            goal_retained=True,
            constraint_retained=not pressure_loss,
            failure_visible=not pressure_loss,
            recovery_ok=not scenario.needs_recovery or not pressure_loss,
            state_consistent=not pressure_loss,
            tokens=2_250,
        )


class HorizonStrategy(Strategy):
    name = "Horizon_full"

    def run(self, scenario: Scenario) -> EpisodeResult:
        # The deterministic reference behavior represents a correct state-anchor
        # policy: inject once when the scenario's risk signal is high, keeping
        # the structured execution state actionable without always-on tokens.
        # Mirrors the intended conditional protocol: failure/recovery boundaries
        # and genuinely high context pressure need anchoring, while routine
        # low-risk goal/constraint checks do not pay an always-on token cost.
        high_risk = (
            scenario.category == "failure_avoidance"
            or scenario.needs_recovery
            or scenario.context_steps >= 18
        )
        result = _result(
            scenario,
            goal_retained=True,
            constraint_retained=True,
            failure_visible=True,
            recovery_ok=True,
            state_consistent=True,
            tokens=1_950 + (180 if high_risk else 0),
        )
        return EpisodeResult(**{**result.__dict__, "anchors_injected": int(high_risk)})


class AlwaysOnAnchorStrategy(HorizonStrategy):
    """Control for H3: inject a full anchor on every boundary."""

    name = "Horizon_always_on_anchor"

    def run(self, scenario: Scenario) -> EpisodeResult:
        result = _result(
            scenario,
            goal_retained=True,
            constraint_retained=True,
            failure_visible=True,
            recovery_ok=True,
            state_consistent=True,
            tokens=2_130,
        )
        return EpisodeResult(**{**result.__dict__, "anchors_injected": 1})


class AblationStrategy(HorizonStrategy):
    def __init__(self, component: str) -> None:
        self.component = component
        self.name = "Horizon_minus_" + component

    def run(self, scenario: Scenario) -> EpisodeResult:
        if self.component == "state_anchor":
            return StructuredPassiveStrategy().run(scenario)
        if self.component == "failure_memory":
            result = super().run(scenario)
            if scenario.category == "failure_avoidance":
                return EpisodeResult(**{**result.__dict__, "success": False, "failure_actions": (scenario.failed_approach,)})
            return result
        if self.component == "checkpoint_recovery":
            result = super().run(scenario)
            if scenario.needs_recovery:
                return EpisodeResult(
                    **{
                        **result.__dict__,
                        "success": False,
                        "recovery_succeeded": False,
                        "recovery_distance": None,
                        "state_consistent": False,
                    }
                )
            return result
        if self.component == "structured_cognitive_state":
            return PeriodicSummaryStrategy().run(scenario)
        raise ValueError("unknown ablation component: {}".format(self.component))


def _result(
    scenario: Scenario,
    *,
    goal_retained: bool,
    constraint_retained: bool,
    failure_visible: bool,
    recovery_ok: bool,
    state_consistent: bool,
    tokens: int,
) -> EpisodeResult:
    repeated_failure = scenario.category == "failure_avoidance" and not failure_visible
    constraint_violation = int(not constraint_retained)
    success = goal_retained and not constraint_violation and not repeated_failure and recovery_ok
    return EpisodeResult(
        task_id=scenario.task_id,
        category=scenario.category,
        success=success,
        goal_retained=goal_retained,
        constraint_violations=constraint_violation,
        failure_actions=(scenario.failed_approach,) if repeated_failure else (),
        known_failed_approaches=(scenario.failed_approach,),
        recovery_attempted=scenario.needs_recovery,
        recovery_succeeded=(recovery_ok if scenario.needs_recovery else False),
        recovery_distance=(1 if scenario.needs_recovery and recovery_ok else None),
        state_consistent=state_consistent,
        tokens=tokens,
    )


def load_scenarios(path: Path = DEFAULT_TASKS) -> list[Scenario]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("HorizonBench task fixture must be a JSON list")
    scenarios = [Scenario.from_mapping(item) for item in raw]
    if len(scenarios) != 30:
        raise ValueError("HorizonBench v0.1 expects exactly 30 tasks, found {}".format(len(scenarios)))
    identifiers = [item.task_id for item in scenarios]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("HorizonBench task ids must be unique")
    return scenarios


def default_strategies(include_ablations: bool = True) -> list[Strategy]:
    strategies: list[Strategy] = [
        RawContextStrategy(),
        PeriodicSummaryStrategy(),
        VectorRetrievalStrategy(),
        StructuredPassiveStrategy(),
        AlwaysOnAnchorStrategy(),
        HorizonStrategy(),
    ]
    if include_ablations:
        strategies.extend(
            AblationStrategy(component)
            for component in ("state_anchor", "failure_memory", "checkpoint_recovery", "structured_cognitive_state")
        )
    return strategies


def run_suite(
    scenarios: Sequence[Scenario],
    strategies: Iterable[Strategy],
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "protocol": "synthetic_deterministic_smoke",
        "warning": (
            "This is a deterministic CI harness, not a model-backed research result. "
            "Use score.py on adapter-produced JSONL for empirical evaluation."
        ),
        "task_count": len(scenarios),
        "strategies": {},
    }
    for strategy in strategies:
        rows = [strategy.run(scenario) for scenario in scenarios]
        if output_dir is not None:
            write_jsonl(output_dir / (strategy.name + ".jsonl"), rows)
        report["strategies"][strategy.name] = score_results(rows).as_dict()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HorizonBench deterministic smoke/ablation suite")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of a Markdown table")
    args = parser.parse_args()
    report = run_suite(
        load_scenarios(args.tasks),
        default_strategies(include_ablations=not args.no_ablations),
        args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Synthetic HorizonBench smoke suite (not an empirical model result)\n")
        print("| Strategy | Success | Goal retention | Constraint violations/task | Repeated failure | Recovery | Tokens/success |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name, metrics in report["strategies"].items():
            recovery = metrics["recovery_success_rate"]
            print(
                "| {} | {:.1%} | {:.1%} | {:.2f} | {:.1%} | {} | {} |".format(
                    name,
                    metrics["task_success_rate"],
                    metrics["goal_retention_rate"],
                    metrics["constraint_violation_rate"],
                    metrics["repeated_failure_rate"],
                    "—" if recovery is None else "{:.1%}".format(recovery),
                    "—"
                    if metrics["tokens_per_successful_task"] is None
                    else "{:.0f}".format(metrics["tokens_per_successful_task"]),
                )
            )


if __name__ == "__main__":
    main()

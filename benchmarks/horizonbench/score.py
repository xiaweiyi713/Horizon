"""Canonical HorizonBench metric definitions and JSONL scoring.

The scorer does not assume a particular model, provider, prompt, or runtime.
An adapter emits one JSON object per task, then this module derives the metrics
described in Horizon's research plan from those observed outcomes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


_MAX_DIAGNOSTIC_CHARS = 512


def _normalize_diagnostic(value: Any) -> Optional[str]:
    """Validate a small operational label without allowing unbounded payloads."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("benchmark result diagnostic must be a string or null")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError("benchmark result diagnostic must not be blank")
    if len(normalized) > _MAX_DIAGNOSTIC_CHARS:
        raise ValueError(
            "benchmark result diagnostic must be at most {} characters".format(_MAX_DIAGNOSTIC_CHARS)
        )
    return normalized


@dataclass(frozen=True)
class EpisodeResult:
    """Observable outcome for one benchmark task.

    ``failure_actions`` contains all failed approaches taken by the adapter;
    ``known_failed_approaches`` contains approaches revealed as failed earlier in
    the same trajectory. Their intersection measures repeated failures.
    """

    task_id: str
    category: str
    success: bool
    goal_retained: bool
    constraint_violations: int
    failure_actions: tuple[str, ...] = ()
    known_failed_approaches: tuple[str, ...] = ()
    recovery_attempted: bool = False
    recovery_succeeded: bool = False
    recovery_distance: Optional[int] = None
    state_consistent: bool = False
    tokens: int = 0
    anchors_injected: int = 0
    # Optional bounded operational context. It is deliberately not a score and
    # must not contain an unredacted provider transcript or credential.
    diagnostic: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostic", _normalize_diagnostic(self.diagnostic))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EpisodeResult":
        required = ("task_id", "category", "success", "goal_retained", "constraint_violations")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError("benchmark result is missing required fields: {}".format(", ".join(missing)))
        return cls(
            task_id=str(value["task_id"]),
            category=str(value["category"]),
            success=bool(value["success"]),
            goal_retained=bool(value["goal_retained"]),
            constraint_violations=int(value["constraint_violations"]),
            failure_actions=tuple(map(str, value.get("failure_actions", ()))),
            known_failed_approaches=tuple(map(str, value.get("known_failed_approaches", ()))),
            recovery_attempted=bool(value.get("recovery_attempted", False)),
            recovery_succeeded=bool(value.get("recovery_succeeded", False)),
            recovery_distance=(
                int(value["recovery_distance"])
                if value.get("recovery_distance") is not None
                else None
            ),
            state_consistent=bool(value.get("state_consistent", False)),
            tokens=max(0, int(value.get("tokens", 0))),
            anchors_injected=max(0, int(value.get("anchors_injected", 0))),
            diagnostic=value.get("diagnostic"),
        )

    def repeated_failure_count(self) -> int:
        known = {_normalize(item) for item in self.known_failed_approaches}
        return sum(_normalize(action) in known for action in self.failure_actions)


@dataclass(frozen=True)
class BenchmarkMetrics:
    task_count: int
    task_success_rate: float
    goal_retention_rate: float
    constraint_violation_rate: float
    repeated_failure_rate: float
    recovery_success_rate: Optional[float]
    recovery_distance: Optional[float]
    state_consistency_rate: float
    token_consumption: int
    tokens_per_successful_task: float | None
    anchors_injected: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_results(results: Iterable[EpisodeResult]) -> BenchmarkMetrics:
    """Aggregate results using the metric definitions published by Horizon."""

    rows = list(results)
    if not rows:
        raise ValueError("cannot score an empty benchmark result set")
    task_count = len(rows)
    successes = sum(row.success for row in rows)
    constraint_violations = sum(row.constraint_violations for row in rows)
    all_failed_actions = sum(len(row.failure_actions) for row in rows)
    repeated_failures = sum(row.repeated_failure_count() for row in rows)
    recoveries = [row for row in rows if row.recovery_attempted]
    recovery_distances = [row.recovery_distance for row in recoveries if row.recovery_distance is not None]
    tokens = sum(row.tokens for row in rows)
    return BenchmarkMetrics(
        task_count=task_count,
        task_success_rate=successes / task_count,
        goal_retention_rate=sum(row.goal_retained for row in rows) / task_count,
        constraint_violation_rate=constraint_violations / task_count,
        repeated_failure_rate=(repeated_failures / all_failed_actions if all_failed_actions else 0.0),
        recovery_success_rate=(
            sum(row.recovery_succeeded for row in recoveries) / len(recoveries)
            if recoveries
            else None
        ),
        recovery_distance=(sum(recovery_distances) / len(recovery_distances) if recovery_distances else None),
        state_consistency_rate=sum(row.state_consistent for row in rows) / task_count,
        token_consumption=tokens,
        tokens_per_successful_task=(tokens / successes if successes else None),
        anchors_injected=sum(row.anchors_injected for row in rows),
    )


def read_jsonl(path: Path) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("invalid JSON on line {} of {}: {}".format(line_number, path, error)) from error
        if not isinstance(value, Mapping):
            raise ValueError("line {} of {} is not a JSON object".format(line_number, path))
        results.append(EpisodeResult.from_mapping(value))
    return results


def write_jsonl(path: Path, results: Sequence[EpisodeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in results:
            output.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())

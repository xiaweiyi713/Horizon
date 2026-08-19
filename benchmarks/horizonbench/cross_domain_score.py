"""Score manifest-bound HorizonBench runs by long-horizon workflow and domain.

The generic matrix scorer answers whether every frozen task was completed. This
module adds the cross-domain view required for long-horizon studies: it checks
the same immutable task identity, then reports task metrics alongside workflow
completion, prerequisite integrity, recovery-boundary success, and
evidence-required task success. It does not invoke a provider.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Optional, Sequence

try:  # Supports both `python file.py` and module execution.
    from .protocol import (
        ManifestValidationError,
        RunManifest,
        TaskSetIdentity,
        manifest_summary,
        read_manifest_jsonl,
        validate_comparable_matrix,
        validate_result_task_ids,
    )
    from .score import EpisodeResult, read_jsonl, score_results
    from .score_matrix import score_matrix
    from .workflows import (
        DEFAULT_CROSS_DOMAIN_TASKS,
        CrossDomainWorkflowSuite,
        WorkflowDefinition,
        WorkflowTask,
        WorkflowValidationError,
        load_cross_domain_workflows,
    )
except ImportError:  # pragma: no cover - direct-script compatibility
    from protocol import (  # type: ignore[no-redef]
        ManifestValidationError,
        RunManifest,
        TaskSetIdentity,
        manifest_summary,
        read_manifest_jsonl,
        validate_comparable_matrix,
        validate_result_task_ids,
    )
    from score import EpisodeResult, read_jsonl, score_results
    from score_matrix import score_matrix
    from workflows import (  # type: ignore[no-redef]
        DEFAULT_CROSS_DOMAIN_TASKS,
        CrossDomainWorkflowSuite,
        WorkflowDefinition,
        WorkflowTask,
        WorkflowValidationError,
        load_cross_domain_workflows,
    )


def _numeric_summary(values: Iterable[Optional[float]]) -> dict[str, Any]:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numbers:
        return {"observations": 0, "mean": None, "sample_stddev": None}
    return {
        "observations": len(numbers),
        "mean": mean(numbers),
        "sample_stddev": stdev(numbers) if len(numbers) > 1 else 0.0,
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    keys = sorted({key for row in rows for key in row})
    summary: dict[str, dict[str, Any]] = {}
    for key in keys:
        values: list[Optional[float]] = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, bool) or value is None:
                values.append(None)
            elif isinstance(value, (int, float)):
                values.append(float(value))
            else:
                values.append(None)
        summary[key] = _numeric_summary(values)
    return summary


def _rate(rows: Sequence[bool]) -> Optional[float]:
    if not rows:
        return None
    return sum(rows) / len(rows)


def _workflow_entry(
    workflow: WorkflowDefinition, result_by_id: Mapping[str, EpisodeResult]
) -> dict[str, Any]:
    task_rows = [result_by_id[task.task_id] for task in workflow.tasks]
    prerequisite_integrity = all(
        not result_by_id[task.task_id].success
        or all(result_by_id[dependency].success for dependency in task.depends_on)
        for task in workflow.tasks
    )
    recovery_rows = [result_by_id[task.task_id].success for task in workflow.tasks if task.recovery_boundary]
    evidence_rows = [result_by_id[task.task_id].success for task in workflow.tasks if task.evidence_required]
    all_tasks_success = all(row.success for row in task_rows)
    return {
        "workflow_id": workflow.workflow_id,
        "workflow_title": workflow.workflow_title,
        "domain": workflow.domain,
        "task_ids": [task.task_id for task in workflow.tasks],
        "task_count": len(task_rows),
        "expected_boundaries": workflow.expected_boundaries,
        "completed_steps": sum(row.success for row in task_rows),
        "workflow_success": all_tasks_success and prerequisite_integrity,
        "prerequisite_integrity": prerequisite_integrity,
        "recovery_boundary_count": len(recovery_rows),
        "recovery_boundary_task_success_rate": _rate(recovery_rows),
        "evidence_task_count": len(evidence_rows),
        "evidence_task_success_rate": _rate(evidence_rows),
        "metrics": score_results(task_rows).as_dict(),
    }


def _domain_entry(
    workflow_definitions: Sequence[WorkflowDefinition],
    workflow_entries: Sequence[Mapping[str, Any]],
    result_by_id: Mapping[str, EpisodeResult],
) -> dict[str, Any]:
    if not workflow_definitions or not workflow_entries:
        raise ManifestValidationError("a domain must contain at least one workflow")
    if len(workflow_definitions) != len(workflow_entries):
        raise ManifestValidationError("workflow definitions and outcomes are out of sync")
    domain = workflow_definitions[0].domain
    task_rows = [
        result_by_id[task.task_id]
        for workflow in workflow_definitions
        for task in workflow.tasks
    ]
    recovery_rows = [
        result_by_id[task.task_id].success
        for workflow in workflow_definitions
        for task in workflow.tasks
        if task.recovery_boundary
    ]
    evidence_rows = [
        result_by_id[task.task_id].success
        for workflow in workflow_definitions
        for task in workflow.tasks
        if task.evidence_required
    ]
    return {
        "domain": domain,
        "workflow_count": len(workflow_definitions),
        "task_count": len(task_rows),
        "metrics": score_results(task_rows).as_dict(),
        "workflow_success_rate": _rate(
            [bool(workflow["workflow_success"]) for workflow in workflow_entries]
        ),
        "prerequisite_integrity_rate": _rate(
            [bool(workflow["prerequisite_integrity"]) for workflow in workflow_entries]
        ),
        "recovery_boundary_task_success_rate": _rate(recovery_rows),
        "evidence_task_success_rate": _rate(evidence_rows),
        "mean_expected_boundaries": mean(
            [float(workflow.expected_boundaries) for workflow in workflow_definitions]
        ),
    }


def _domain_entries(
    suite: CrossDomainWorkflowSuite, result_by_id: Mapping[str, EpisodeResult]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workflow_entries: list[dict[str, Any]] = []
    definitions_by_domain: dict[str, list[WorkflowDefinition]] = defaultdict(list)
    entries_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for workflow in sorted(suite.workflows, key=lambda item: (item.domain, item.workflow_id)):
        entry = _workflow_entry(workflow, result_by_id)
        workflow_entries.append(entry)
        definitions_by_domain[workflow.domain].append(workflow)
        entries_by_domain[workflow.domain].append(entry)
    domain_entries = [
        _domain_entry(definitions_by_domain[domain], entries_by_domain[domain], result_by_id)
        for domain in sorted(definitions_by_domain)
    ]
    return domain_entries, workflow_entries


def _summary_entries(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        manifest = run["manifest"]
        grouped[(str(manifest["model_id"]), str(manifest["condition_id"]))].append(run)
    summaries: list[dict[str, Any]] = []
    for (model_id, condition_id), group in sorted(grouped.items()):
        domain_names = sorted(
            {str(domain["domain"]) for run in group for domain in run["domains"]}
        )
        domains: list[dict[str, Any]] = []
        for domain_name in domain_names:
            domain_runs = [
                next(domain for domain in run["domains"] if domain["domain"] == domain_name)
                for run in group
            ]
            domains.append(
                {
                    "domain": domain_name,
                    "run_count": len(domain_runs),
                    "metrics": _metric_summary([domain["metrics"] for domain in domain_runs]),
                    "workflow_success_rate": _numeric_summary(
                        [domain["workflow_success_rate"] for domain in domain_runs]
                    ),
                    "prerequisite_integrity_rate": _numeric_summary(
                        [domain["prerequisite_integrity_rate"] for domain in domain_runs]
                    ),
                    "recovery_boundary_task_success_rate": _numeric_summary(
                        [domain["recovery_boundary_task_success_rate"] for domain in domain_runs]
                    ),
                    "evidence_task_success_rate": _numeric_summary(
                        [domain["evidence_task_success_rate"] for domain in domain_runs]
                    ),
                    "mean_expected_boundaries": _numeric_summary(
                        [domain["mean_expected_boundaries"] for domain in domain_runs]
                    ),
                }
            )
        summaries.append(
            {
                "model_id": model_id,
                "condition_id": condition_id,
                "run_count": len(group),
                "overall_metrics": _metric_summary([run["overall_metrics"] for run in group]),
                "domains": domains,
            }
        )
    return summaries


def _validate_suite_matches_manifests(
    suite: CrossDomainWorkflowSuite, manifests: Sequence[RunManifest]
) -> TaskSetIdentity:
    identity = TaskSetIdentity.from_task_set(suite.task_set)
    for manifest in manifests:
        if manifest.task_set != identity:
            raise ManifestValidationError(
                "cross-domain workflow suite does not match frozen manifest task set"
            )
    return identity


def score_cross_domain(
    matrix_path: Path,
    results_dir: Path,
    *,
    workflow_tasks_path: Path = DEFAULT_CROSS_DOMAIN_TASKS,
    source_name: str = "horizon-cross-domain-fixture",
    source_revision: str = "v1",
    split: str = "heldout",
    result_suffix: str = ".jsonl",
) -> dict[str, Any]:
    """Score observed model runs against a frozen cross-domain workflow suite."""

    suite = load_cross_domain_workflows(
        workflow_tasks_path,
        source_name=source_name,
        source_revision=source_revision,
        split=split,
    )
    manifests = read_manifest_jsonl(matrix_path)
    validate_comparable_matrix(manifests)
    task_identity = _validate_suite_matches_manifests(suite, manifests)
    base_report = score_matrix(matrix_path, results_dir, result_suffix=result_suffix)
    metrics_by_run_id = {str(run["manifest"]["run_id"]): run["metrics"] for run in base_report["runs"]}
    runs: list[dict[str, Any]] = []
    for manifest in manifests:
        result_path = results_dir / (manifest.run_id + result_suffix)
        results = read_jsonl(result_path)
        validate_result_task_ids(results, manifest)
        result_by_id = {result.task_id: result for result in results}
        domains, workflows = _domain_entries(suite, result_by_id)
        runs.append(
            {
                "manifest": manifest_summary(manifest),
                "result_path": str(result_path),
                "overall_metrics": metrics_by_run_id[manifest.run_id],
                "domains": domains,
                "workflows": workflows,
            }
        )
    return {
        "protocol": "horizonbench_cross_domain_score_v1",
        "suite": suite.identity(),
        "matrix": {
            "matrix_path": str(matrix_path),
            "run_count": len(runs),
            "task_set": task_identity.as_dict(),
            "prompt_sha256": base_report["prompt_sha256"],
            "runtime_revision": base_report["runtime_revision"],
        },
        "runs": runs,
        "summaries": _summary_entries(runs),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score manifest-bound HorizonBench results by cross-domain workflow"
    )
    parser.add_argument("--matrix", type=Path, required=True, help="JSONL matrix from plan_matrix.py")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--workflow-tasks", type=Path, default=DEFAULT_CROSS_DOMAIN_TASKS)
    parser.add_argument("--source-name", default="horizon-cross-domain-fixture")
    parser.add_argument("--source-revision", default="v1")
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--result-suffix", default=".jsonl")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = score_cross_domain(
            args.matrix,
            args.results_dir,
            workflow_tasks_path=args.workflow_tasks,
            source_name=args.source_name,
            source_revision=args.source_revision,
            split=args.split,
            result_suffix=args.result_suffix,
        )
    except (ManifestValidationError, WorkflowValidationError, ValueError) as error:
        parser.error(str(error))
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

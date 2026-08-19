"""Render a concise, auditable Markdown report from cross-domain score JSON.

The renderer intentionally does not infer causal conclusions. It turns an
already validated ``cross_domain_score.py`` output into a portable technical
report that retains the frozen provenance and makes missing empirical context
visible to reviewers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


class ReportValidationError(ValueError):
    """Raised when a supplied score report cannot support an audit trail."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportValidationError("{} must be an object".format(field))
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReportValidationError("{} must be an array".format(field))
    return value


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip() or "—"


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: Mapping[str, Any], metric: str) -> Optional[float]:
    summary = values.get(metric)
    if not isinstance(summary, Mapping):
        return None
    return _number(summary.get("mean"))


def _rate(value: Optional[float]) -> str:
    return "—" if value is None else "{:.1%}".format(value)


def _decimal(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value.is_integer():
        return str(int(value))
    return "{:.2f}".format(value)


def _boolean_rate(value: Any) -> str:
    if value is True:
        return _rate(1.0)
    if value is False:
        return _rate(0.0)
    return "—"


def load_cross_domain_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReportValidationError("score report does not exist: {}".format(path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReportValidationError("invalid JSON in score report {}: {}".format(path, error)) from error
    report = dict(_mapping(value, "score report"))
    if report.get("protocol") != "horizonbench_cross_domain_score_v1":
        raise ReportValidationError("score report has unsupported protocol")
    _mapping(report.get("suite"), "suite")
    matrix = _mapping(report.get("matrix"), "matrix")
    _mapping(matrix.get("task_set"), "matrix.task_set")
    _list(report.get("runs"), "runs")
    _list(report.get("summaries"), "summaries")
    return report


def render_technical_report(
    report: Mapping[str, Any],
    *,
    title: str = "Horizon Cross-Domain Long-Horizon Evaluation Report",
) -> str:
    """Render a deterministic Markdown report from a validated score mapping."""

    if not isinstance(title, str) or not title.strip() or "\n" in title:
        raise ReportValidationError("title must be a non-empty string")
    if report.get("protocol") != "horizonbench_cross_domain_score_v1":
        raise ReportValidationError("report has unsupported protocol")
    suite = _mapping(report.get("suite"), "suite")
    matrix = _mapping(report.get("matrix"), "matrix")
    task_set = _mapping(matrix.get("task_set"), "matrix.task_set")
    runs = _list(report.get("runs"), "runs")
    summaries = _list(report.get("summaries"), "summaries")

    lines = [
        "# {}".format(title.strip()),
        "",
        "> This report renders supplied, manifest-bound observed outcomes. It does not by itself "
        "establish a causal, cross-model, or generalization claim.",
        "",
        "## Audit scope",
        "",
        "| Field | Frozen value |",
        "|---|---|",
        "| Source | {} |".format(_cell(task_set.get("source_name"))),
        "| Source revision | {} |".format(_cell(task_set.get("source_revision"))),
        "| Split | {} |".format(_cell(task_set.get("split"))),
        "| Source SHA-256 | {} |".format(_cell(task_set.get("source_sha256"))),
        "| Selected task set SHA-256 | {} |".format(_cell(task_set.get("task_set_sha256"))),
        "| Task count | {} |".format(_cell(len(task_set.get("task_ids", ())))),
        "| Prompt SHA-256 | {} |".format(_cell(matrix.get("prompt_sha256"))),
        "| Runtime revision | {} |".format(_cell(matrix.get("runtime_revision"))),
        "| Observed runs | {} |".format(_cell(matrix.get("run_count"))),
        "| Workflow suite identity | {} |".format(_cell(suite.get("task_set_sha256", "recorded"))),
        "",
        "## Aggregate outcomes",
        "",
        "| Model | Condition | Runs | Task success | Goal retention | Constraint violations/task | Tokens/success |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for raw_summary in summaries:
        summary = _mapping(raw_summary, "summary")
        metrics = _mapping(summary.get("overall_metrics"), "summary.overall_metrics")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                _cell(summary.get("model_id")),
                _cell(summary.get("condition_id")),
                _cell(summary.get("run_count")),
                _rate(_mean(metrics, "task_success_rate")),
                _rate(_mean(metrics, "goal_retention_rate")),
                _decimal(_mean(metrics, "constraint_violation_rate")),
                _decimal(_mean(metrics, "tokens_per_successful_task")),
            )
        )

    lines.extend(
        [
            "",
            "## Cross-domain workflow outcomes",
            "",
            "| Model | Condition | Domain | Runs | Task success | Workflow success | Prerequisite integrity | Evidence-task success | Recovery-boundary success |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for raw_summary in summaries:
        summary = _mapping(raw_summary, "summary")
        for raw_domain in _list(summary.get("domains"), "summary.domains"):
            domain = _mapping(raw_domain, "domain summary")
            metrics = _mapping(domain.get("metrics"), "domain.metrics")
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    _cell(summary.get("model_id")),
                    _cell(summary.get("condition_id")),
                    _cell(domain.get("domain")),
                    _cell(domain.get("run_count")),
                    _rate(_mean(metrics, "task_success_rate")),
                    _rate(_mean(domain, "workflow_success_rate")),
                    _rate(_mean(domain, "prerequisite_integrity_rate")),
                    _rate(_mean(domain, "evidence_task_success_rate")),
                    _rate(_mean(domain, "recovery_boundary_task_success_rate")),
                )
            )

    lines.extend(
        [
            "",
            "## Per-run workflow ledger",
            "",
            "| Run | Model | Condition | Seed | Domain | Workflow | Completed steps | Success | Prerequisites |",
            "|---|---|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for raw_run in runs:
        run = _mapping(raw_run, "run")
        manifest = _mapping(run.get("manifest"), "run.manifest")
        for raw_workflow in _list(run.get("workflows"), "run.workflows"):
            workflow = _mapping(raw_workflow, "workflow")
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {}/{} | {} | {} |".format(
                    _cell(manifest.get("run_id")),
                    _cell(manifest.get("model_id")),
                    _cell(manifest.get("condition_id")),
                    _cell(manifest.get("seed")),
                    _cell(workflow.get("domain")),
                    _cell(workflow.get("workflow_id")),
                    _cell(workflow.get("completed_steps")),
                    _cell(workflow.get("task_count")),
                    _boolean_rate(workflow.get("workflow_success")),
                    _boolean_rate(workflow.get("prerequisite_integrity")),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Use only records generated against the frozen manifest, task split, prompt, and runtime revision above.",
            "- The report summarizes observed episodes; it does not replace an external benchmark license, task provenance, or statistical analysis plan.",
            "- Compare conditions on the same model/seed set, and retain raw JSONL outcomes and durable runtime events for audit.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Render a Markdown technical report from cross-domain score JSON")
    parser.add_argument("input", type=Path, help="JSON output from cross_domain_score.py")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report destination")
    parser.add_argument("--title", default="Horizon Cross-Domain Long-Horizon Evaluation Report")
    args = parser.parse_args(argv)
    try:
        report = load_cross_domain_report(args.input)
        rendered = render_technical_report(report, title=args.title)
    except ReportValidationError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print("wrote {}".format(args.output))


if __name__ == "__main__":
    main()

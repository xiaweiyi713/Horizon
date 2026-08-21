"""Render and audit a local Ollama durable-trace mechanism-smoke artifact.

This renderer is deliberately scoped to :mod:`ollama_durable_trace`.  It
re-opens the immutable matrix, preflight report, execution receipts, raw
episode outcomes, and score report before emitting Markdown.  Therefore an
otherwise plausible-looking score JSON cannot be presented without its
matching provenance and completed receipt-bound outcomes.

The output is an audit artifact for a narrow runtime-mechanism smoke.  It is
not a public benchmark report and never draws a causal or model-quality
conclusion from its observations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.execute_matrix import (  # noqa: E402
    EXECUTION_PROTOCOL,
    RECEIPT_SCHEMA_VERSION,
)
from benchmarks.horizonbench.preflight_matrix import PREFLIGHT_PROTOCOL  # noqa: E402
from benchmarks.horizonbench.protocol import (  # noqa: E402
    ManifestValidationError,
    RunManifest,
    TaskSetIdentity,
    read_manifest_jsonl,
)
from benchmarks.horizonbench.score import EpisodeResult, read_jsonl  # noqa: E402
from benchmarks.horizonbench.score_matrix import score_matrix  # noqa: E402


EXPERIMENT_KIND = "real_model_durable_trace_mechanism_smoke"
CLAIM_BOUNDARY = "not_a_public_benchmark_or_model_improvement_claim"
SCORE_PROTOCOL = "horizonbench_cross_model_score_v1"
DEFAULT_TITLE = "Horizon 本地 Ollama 持久化轨迹机制 Smoke 报告"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_DIAGNOSTICS = frozenset(
    {
        "configured budget exhausted",
        "agent reached max_steps",
        "LLM policy call failed",
        "policy action rejected",
        "agent stopped with an unclassified error",
    }
)


class SmokeReportValidationError(ValueError):
    """Raised when artifacts cannot support an auditable smoke report."""


@dataclass(frozen=True)
class SmokeArtifacts:
    """Validated, mutually consistent artifacts for one smoke matrix."""

    artifact_dir: Path
    experiment: Mapping[str, Any]
    manifests: tuple[RunManifest, ...]
    score: Mapping[str, Any]
    preflight: Mapping[str, Any]
    execution: Mapping[str, Any]
    outcomes_by_run: Mapping[str, tuple[EpisodeResult, ...]]


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SmokeReportValidationError("{} is not a regular file: {}".format(label, path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SmokeReportValidationError("invalid JSON in {} {}: {}".format(label, path, error)) from error
    if not isinstance(value, Mapping):
        raise SmokeReportValidationError("{} must contain a JSON object".format(label))
    return dict(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError("invalid JSON constant {}".format(value))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SmokeReportValidationError("{} must be an object".format(label))
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SmokeReportValidationError("{} must be an array".format(label))
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SmokeReportValidationError("{} must be a non-empty string".format(label))
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SmokeReportValidationError("{} must be a positive integer".format(label))
    return value


def _sha256(value: Any, label: str) -> str:
    value = _string(value, label)
    if not _SHA256_RE.fullmatch(value):
        raise SmokeReportValidationError("{} must be a lowercase SHA-256 digest".format(label))
    return value


def _artifact_file(artifact_dir: Path, name: str) -> Path:
    path = artifact_dir / name
    if not path.is_file() or path.is_symlink():
        raise SmokeReportValidationError("required artifact is not a regular file: {}".format(path))
    return path


def _task_set(value: Any, label: str) -> TaskSetIdentity:
    try:
        return TaskSetIdentity.from_mapping(_mapping(value, label))
    except ManifestValidationError as error:
        raise SmokeReportValidationError("{} is invalid: {}".format(label, error)) from error


def _same_task_set(left: TaskSetIdentity, right: TaskSetIdentity, label: str) -> None:
    if left != right:
        raise SmokeReportValidationError("{} does not match the frozen matrix task set".format(label))


def _reported_filename(value: Any, expected: str, label: str) -> None:
    reported = _string(value, label)
    if Path(reported).name != expected:
        raise SmokeReportValidationError("{} does not name expected artifact {!r}".format(label, expected))


def _model_records(experiment: Mapping[str, Any]) -> dict[str, str]:
    if experiment.get("experiment_kind") != EXPERIMENT_KIND:
        raise SmokeReportValidationError("experiment has an unsupported experiment_kind")
    if experiment.get("claim_boundary") != CLAIM_BOUNDARY:
        raise SmokeReportValidationError("experiment has an unsupported claim_boundary")
    _string(experiment.get("output_dir"), "experiment.output_dir")
    for field, expected in (
        ("preflight", "preflight.json"),
        ("execution", "execution.json"),
        ("score", "score.json"),
    ):
        if experiment.get(field) != expected:
            raise SmokeReportValidationError("experiment.{} must be {!r}".format(field, expected))
    report_name = experiment.get("report")
    if report_name is not None and report_name != "report.md":
        raise SmokeReportValidationError("experiment.report must be 'report.md' when supplied")

    rows = _list(experiment.get("models"), "experiment.models")
    if _positive_int(experiment.get("model_count"), "experiment.model_count") != len(rows):
        raise SmokeReportValidationError("experiment.model_count does not match experiment.models")
    models: dict[str, str] = {}
    for index, raw in enumerate(rows, 1):
        row = _mapping(raw, "experiment.models[{}]".format(index))
        model = _string(row.get("model"), "experiment.models[{}].model".format(index))
        digest = _sha256(row.get("digest"), "experiment.models[{}].digest".format(index))
        if model in models:
            raise SmokeReportValidationError("experiment.models contains duplicate model {!r}".format(model))
        if digest in models.values():
            raise SmokeReportValidationError("experiment.models contains duplicate model digest")
        models[model] = digest
    if not models:
        raise SmokeReportValidationError("experiment.models must not be empty")

    conditions = _list(experiment.get("conditions"), "experiment.conditions")
    if not conditions or not all(isinstance(item, str) and item.strip() for item in conditions):
        raise SmokeReportValidationError("experiment.conditions must contain non-empty strings")
    if len(set(conditions)) != len(conditions):
        raise SmokeReportValidationError("experiment.conditions contains duplicates")
    _positive_int(experiment.get("matrix_runs"), "experiment.matrix_runs")
    _positive_int(experiment.get("task_count"), "experiment.task_count")
    return models


def _validate_matrix(
    manifests: Sequence[RunManifest],
    experiment: Mapping[str, Any],
    models: Mapping[str, str],
) -> None:
    if len(manifests) != experiment["matrix_runs"]:
        raise SmokeReportValidationError("experiment.matrix_runs does not match matrix rows")
    if not manifests:
        raise SmokeReportValidationError("matrix must contain at least one row")
    task_set = manifests[0].task_set
    if len(task_set.task_ids) != experiment["task_count"]:
        raise SmokeReportValidationError("experiment.task_count does not match the frozen task set")
    expected_conditions = tuple(experiment["conditions"])
    observed_conditions: set[str] = set()
    profiles: dict[str, tuple[str, str]] = {}
    seeds_by_pair: dict[tuple[str, str], set[Optional[int]]] = defaultdict(set)

    for manifest in manifests:
        _same_task_set(manifest.task_set, task_set, "matrix manifest task_set")
        if manifest.metadata.get("experiment_kind") != EXPERIMENT_KIND:
            raise SmokeReportValidationError("matrix manifest has an unexpected experiment_kind")
        if manifest.metadata.get("claim_boundary") != CLAIM_BOUNDARY:
            raise SmokeReportValidationError("matrix manifest has an unexpected claim_boundary")
        expected_digest = models.get(manifest.model.model)
        if expected_digest is None:
            raise SmokeReportValidationError("matrix references a model absent from experiment.models")
        if manifest.model.model_revision != "ollama-digest:" + expected_digest:
            raise SmokeReportValidationError("matrix model revision does not match experiment digest")
        profile = (manifest.model.model_id, manifest.model.model_revision)
        prior_profile = profiles.setdefault(manifest.model.model, profile)
        if profile != prior_profile:
            raise SmokeReportValidationError("matrix changes the frozen profile for one model")
        if manifest.condition.condition_id not in expected_conditions:
            raise SmokeReportValidationError("matrix references a condition absent from experiment.conditions")
        observed_conditions.add(manifest.condition.condition_id)
        pair = (manifest.model.model, manifest.condition.condition_id)
        if manifest.seed in seeds_by_pair[pair]:
            raise SmokeReportValidationError("matrix has duplicate model/condition/seed rows")
        seeds_by_pair[pair].add(manifest.seed)

    if set(profiles) != set(models):
        raise SmokeReportValidationError("matrix does not cover every experiment model")
    if observed_conditions != set(expected_conditions):
        raise SmokeReportValidationError("matrix does not cover every experiment condition")
    expected_pairs = {(model, condition) for model in models for condition in expected_conditions}
    if set(seeds_by_pair) != expected_pairs:
        raise SmokeReportValidationError("matrix does not cover every model/condition pair")
    seed_sets = {tuple(sorted(values, key=lambda value: (-1 if value is None else value))) for values in seeds_by_pair.values()}
    if len(seed_sets) != 1:
        raise SmokeReportValidationError("matrix does not use the same seed set for every model/condition pair")
    seed_count = len(next(iter(seed_sets)))
    if len(manifests) != len(models) * len(expected_conditions) * seed_count:
        raise SmokeReportValidationError("matrix row count does not match its full model/condition/seed shape")


def _validate_preflight(
    report: Mapping[str, Any],
    manifests: Sequence[RunManifest],
) -> None:
    if report.get("protocol") != PREFLIGHT_PROTOCOL:
        raise SmokeReportValidationError("preflight has an unsupported protocol")
    _reported_filename(report.get("matrix_path"), "matrix.jsonl", "preflight.matrix_path")
    _string(report.get("executor"), "preflight.executor")
    task_set = _task_set(report.get("task_set"), "preflight.task_set")
    _same_task_set(task_set, manifests[0].task_set, "preflight.task_set")
    runs = _list(report.get("runs"), "preflight.runs")
    if _positive_int(report.get("run_count"), "preflight.run_count") != len(manifests) or len(runs) != len(manifests):
        raise SmokeReportValidationError("preflight run count does not match matrix")
    expected_total = len(manifests) * len(task_set.task_ids)
    if _positive_int(report.get("task_count"), "preflight.task_count") != expected_total:
        raise SmokeReportValidationError("preflight task_count does not match matrix task attempts")
    by_id = {manifest.run_id: manifest for manifest in manifests}
    seen: set[str] = set()
    for index, raw in enumerate(runs, 1):
        run = _mapping(raw, "preflight.runs[{}]".format(index))
        run_id = _string(run.get("run_id"), "preflight.runs[{}].run_id".format(index))
        manifest = by_id.get(run_id)
        if manifest is None or run_id in seen:
            raise SmokeReportValidationError("preflight runs do not uniquely match matrix rows")
        seen.add(run_id)
        if run.get("manifest_sha256") != manifest.manifest_sha256:
            raise SmokeReportValidationError("preflight manifest hash does not match matrix")
        if run.get("task_count") != len(task_set.task_ids):
            raise SmokeReportValidationError("preflight run task_count does not match frozen task set")
        checks = _list(run.get("checks"), "preflight.runs[{}].checks".format(index))
        check_ids = []
        for check in checks:
            check_map = _mapping(check, "preflight check")
            check_ids.append(_string(check_map.get("task_id"), "preflight check.task_id"))
            _mapping(check_map.get("preflight"), "preflight check.preflight")
        if len(check_ids) != len(task_set.task_ids) or set(check_ids) != set(task_set.task_ids):
            raise SmokeReportValidationError("preflight checks do not cover the frozen task set")
    if seen != set(by_id):
        raise SmokeReportValidationError("preflight is missing matrix rows")


def _validate_receipt(
    path: Path,
    *,
    manifest: RunManifest,
    executor: str,
) -> None:
    receipt = _read_json_object(path, "receipt")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise SmokeReportValidationError("receipt has an unsupported schema_version")
    if receipt.get("protocol") != EXECUTION_PROTOCOL:
        raise SmokeReportValidationError("receipt has an unsupported protocol")
    if receipt.get("run_id") != manifest.run_id:
        raise SmokeReportValidationError("receipt run_id does not match matrix")
    if receipt.get("manifest_sha256") != manifest.manifest_sha256:
        raise SmokeReportValidationError("receipt manifest hash does not match matrix")
    if receipt.get("executor") != executor:
        raise SmokeReportValidationError("receipt executor does not match execution report")
    if receipt.get("status") != "completed" or receipt.get("error") is not None:
        raise SmokeReportValidationError("receipt does not record a completed run")
    receipt_task_set = _task_set(receipt.get("task_set"), "receipt.task_set")
    _same_task_set(receipt_task_set, manifest.task_set, "receipt.task_set")
    completed = _list(receipt.get("completed_task_ids"), "receipt.completed_task_ids")
    if not all(isinstance(task_id, str) and task_id for task_id in completed):
        raise SmokeReportValidationError("receipt completed task IDs must be non-empty strings")
    if set(completed) != set(manifest.task_set.task_ids) or len(completed) != len(manifest.task_set.task_ids):
        raise SmokeReportValidationError("receipt completed task IDs do not match matrix task set")
    attempts = _list(receipt.get("attempts"), "receipt.attempts")
    succeeded: set[str] = set()
    for attempt in attempts:
        attempt_map = _mapping(attempt, "receipt attempt")
        task_id = _string(attempt_map.get("task_id"), "receipt attempt.task_id")
        if task_id not in manifest.task_set.task_ids:
            raise SmokeReportValidationError("receipt attempt references a task outside the matrix")
        if attempt_map.get("status") == "succeeded":
            succeeded.add(task_id)
    if succeeded != set(manifest.task_set.task_ids):
        raise SmokeReportValidationError("receipt lacks a successful attempt for a frozen task")


def _validate_execution(
    report: Mapping[str, Any],
    manifests: Sequence[RunManifest],
    results_dir: Path,
) -> None:
    if report.get("protocol") != EXECUTION_PROTOCOL:
        raise SmokeReportValidationError("execution has an unsupported protocol")
    _reported_filename(report.get("matrix_path"), "matrix.jsonl", "execution.matrix_path")
    executor = _string(report.get("executor"), "execution.executor")
    task_set = _task_set(report.get("task_set"), "execution.task_set")
    _same_task_set(task_set, manifests[0].task_set, "execution.task_set")
    runs = _list(report.get("runs"), "execution.runs")
    if _positive_int(report.get("run_count"), "execution.run_count") != len(manifests) or len(runs) != len(manifests):
        raise SmokeReportValidationError("execution run count does not match matrix")
    by_id = {manifest.run_id: manifest for manifest in manifests}
    seen: set[str] = set()
    for index, raw in enumerate(runs, 1):
        run = _mapping(raw, "execution.runs[{}]".format(index))
        run_id = _string(run.get("run_id"), "execution.runs[{}].run_id".format(index))
        manifest = by_id.get(run_id)
        if manifest is None or run_id in seen:
            raise SmokeReportValidationError("execution runs do not uniquely match matrix rows")
        seen.add(run_id)
        if run.get("status") != "completed":
            raise SmokeReportValidationError("execution includes a non-completed run")
        if run.get("task_count") != len(task_set.task_ids):
            raise SmokeReportValidationError("execution run task_count does not match frozen task set")
        executed = _list(run.get("executed_task_ids"), "execution.executed_task_ids")
        if not all(isinstance(task_id, str) and task_id for task_id in executed):
            raise SmokeReportValidationError("execution task IDs must be non-empty strings")
        if set(executed) != set(task_set.task_ids) or len(executed) != len(task_set.task_ids):
            raise SmokeReportValidationError("execution task IDs do not match frozen task set")
        _reported_filename(run.get("result_path"), run_id + ".jsonl", "execution.result_path")
        _reported_filename(
            run.get("receipt_path"), run_id + ".receipt.json", "execution.receipt_path"
        )
        _validate_receipt(results_dir / (run_id + ".receipt.json"), manifest=manifest, executor=executor)
    if seen != set(by_id):
        raise SmokeReportValidationError("execution is missing matrix rows")


def _validate_score(
    score: Mapping[str, Any],
    matrix_path: Path,
    results_dir: Path,
) -> Mapping[str, Any]:
    if score.get("protocol") != SCORE_PROTOCOL:
        raise SmokeReportValidationError("score has an unsupported protocol")
    _reported_filename(score.get("matrix_path"), "matrix.jsonl", "score.matrix_path")
    try:
        recomputed = score_matrix(matrix_path, results_dir)
    except (ManifestValidationError, OSError, ValueError) as error:
        raise SmokeReportValidationError("could not recompute score from raw outcomes: {}".format(error)) from error
    if set(score) != set(recomputed):
        raise SmokeReportValidationError("score fields do not match the supported score protocol")
    comparable_score = _portable_score(score)
    comparable_recomputed = _portable_score(recomputed)
    for field, expected in comparable_recomputed.items():
        if comparable_score.get(field) != expected:
            raise SmokeReportValidationError("score.{} does not match recomputed raw outcomes".format(field))
    return recomputed


def _portable_score(score: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only output-directory paths before score equality comparison.

    A complete artifact directory can be copied to another machine.  The score
    schema records result paths, so compare their manifest-bound filenames
    rather than their old absolute directory names.  All scored facts and
    provenance remain exact.
    """

    normalized = dict(score)
    normalized.pop("matrix_path", None)
    runs = _list(normalized.get("runs"), "score.runs")
    portable_runs: list[dict[str, Any]] = []
    for index, raw in enumerate(runs, 1):
        run = dict(_mapping(raw, "score.runs[{}]".format(index)))
        manifest = _mapping(run.get("manifest"), "score.runs[{}].manifest".format(index))
        run_id = _string(manifest.get("run_id"), "score.runs[{}].manifest.run_id".format(index))
        _reported_filename(run.get("result_path"), run_id + ".jsonl", "score.runs[{}].result_path".format(index))
        run["result_path"] = run_id + ".jsonl"
        portable_runs.append(run)
    normalized["runs"] = portable_runs
    return normalized


def load_ollama_trace_artifacts(artifact_dir: Path) -> SmokeArtifacts:
    """Load a complete artifact directory and verify all auditable bindings."""

    if not artifact_dir.is_dir() or artifact_dir.is_symlink():
        raise SmokeReportValidationError("artifact_dir is not a regular directory: {}".format(artifact_dir))
    artifact_dir = artifact_dir.resolve()
    experiment = _read_json_object(_artifact_file(artifact_dir, "experiment.json"), "experiment")
    models = _model_records(experiment)
    matrix_path = _artifact_file(artifact_dir, "matrix.jsonl")
    try:
        manifests = tuple(read_manifest_jsonl(matrix_path))
    except (ManifestValidationError, OSError) as error:
        raise SmokeReportValidationError("invalid matrix: {}".format(error)) from error
    _validate_matrix(manifests, experiment, models)

    results_dir = artifact_dir / "results"
    if not results_dir.is_dir() or results_dir.is_symlink():
        raise SmokeReportValidationError("results is not a regular directory: {}".format(results_dir))
    score = _read_json_object(_artifact_file(artifact_dir, "score.json"), "score")
    recomputed_score = _validate_score(score, matrix_path, results_dir)
    score_task_set = _task_set(score.get("task_set"), "score.task_set")
    _same_task_set(score_task_set, manifests[0].task_set, "score.task_set")
    if score.get("runtime_revision") != manifests[0].runtime_revision:
        raise SmokeReportValidationError("score.runtime_revision does not match matrix")
    if score.get("prompt_sha256") != manifests[0].prompt.sha256:
        raise SmokeReportValidationError("score.prompt_sha256 does not match matrix")

    preflight = _read_json_object(_artifact_file(artifact_dir, "preflight.json"), "preflight")
    _validate_preflight(preflight, manifests)
    execution = _read_json_object(_artifact_file(artifact_dir, "execution.json"), "execution")
    _validate_execution(execution, manifests, results_dir)

    outcomes_by_run: dict[str, tuple[EpisodeResult, ...]] = {}
    for manifest in manifests:
        try:
            outcomes = tuple(read_jsonl(results_dir / (manifest.run_id + ".jsonl")))
        except (OSError, ValueError) as error:
            raise SmokeReportValidationError("could not read run outcomes: {}".format(error)) from error
        outcomes_by_run[manifest.run_id] = outcomes

    return SmokeArtifacts(
        artifact_dir=artifact_dir,
        experiment=experiment,
        manifests=manifests,
        score=recomputed_score,
        preflight=preflight,
        execution=execution,
        outcomes_by_run=outcomes_by_run,
    )


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip() or "—"


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric(summary: Mapping[str, Any], name: str) -> Optional[float]:
    metrics = _mapping(summary.get("metrics"), "score summary.metrics")
    value = _mapping(metrics.get(name), "score summary.metrics.{}".format(name))
    return _number(value.get("mean"))


def _rate(value: Optional[float]) -> str:
    return "—" if value is None else "{:.1%}".format(value)


def _decimal(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value.is_integer():
        return str(int(value))
    return "{:.2f}".format(value)


def _seed_label(seed: Optional[int]) -> str:
    return "无固定 seed" if seed is None else str(seed)


def _display_diagnostic(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value in _SAFE_DIAGNOSTICS:
        return value
    return "未识别诊断（已抑制原始内容）"


def render_ollama_trace_report(
    artifacts: SmokeArtifacts,
    *,
    title: str = DEFAULT_TITLE,
) -> str:
    """Render deterministic Chinese Markdown from validated smoke artifacts."""

    if not isinstance(artifacts, SmokeArtifacts):
        raise SmokeReportValidationError("artifacts must be SmokeArtifacts")
    if not isinstance(title, str) or not title.strip() or "\n" in title:
        raise SmokeReportValidationError("title must be a non-empty single-line string")

    task_set = artifacts.manifests[0].task_set
    experiment = artifacts.experiment
    model_order = [row["model"] for row in experiment["models"]]
    digest_by_model = {row["model"]: row["digest"] for row in experiment["models"]}
    condition_order = list(experiment["conditions"])
    profile_by_model = {
        manifest.model.model: manifest.model.model_id for manifest in artifacts.manifests
    }
    seeds_by_pair: dict[tuple[str, str], list[Optional[int]]] = defaultdict(list)
    for manifest in artifacts.manifests:
        seeds_by_pair[(manifest.model.model, manifest.condition.condition_id)].append(manifest.seed)
    for seeds in seeds_by_pair.values():
        seeds.sort(key=lambda value: (-1 if value is None else value))
    summaries = {
        (summary["model_id"], summary["condition_id"]): summary
        for summary in artifacts.score["summaries"]
    }

    diagnostic_counts: Counter[tuple[str, str, str]] = Counter()
    for manifest in artifacts.manifests:
        for outcome in artifacts.outcomes_by_run[manifest.run_id]:
            diagnostic = _display_diagnostic(outcome.diagnostic)
            if diagnostic is not None:
                diagnostic_counts[(manifest.model.model, manifest.condition.condition_id, diagnostic)] += 1

    seed_set = seeds_by_pair[(model_order[0], condition_order[0])]
    episode_count = sum(len(rows) for rows in artifacts.outcomes_by_run.values())
    lines = [
        "# {}".format(title.strip()),
        "",
        "> 研究边界：这是对本地 Ollama、Python policy 与 Rust runtime 端到端路径的窄机制 smoke。它不是公共 benchmark，不能据此声称 Horizon 提升某个模型，或得到跨模型的因果／泛化结论。",
        "",
        "## 审计身份",
        "",
        "| 字段 | 冻结值 |",
        "|---|---|",
        "| 实验类型 | {} |".format(_cell(experiment["experiment_kind"])),
        "| 任务源 | {} |".format(_cell(task_set.source_name)),
        "| 任务源版本 | {} |".format(_cell(task_set.source_revision)),
        "| 切分 | {} |".format(_cell(task_set.split)),
        "| 源文件 SHA-256 | {} |".format(_cell(task_set.source_sha256)),
        "| 选中任务集 SHA-256 | {} |".format(_cell(task_set.task_set_sha256)),
        "| Prompt SHA-256 | {} |".format(_cell(artifacts.score["prompt_sha256"])),
        "| Runtime revision | {} |".format(_cell(artifacts.score["runtime_revision"])),
        "| 模型数 × 条件数 × 每组合 seeds | {} × {} × {} |".format(
            len(model_order), len(condition_order), len(seed_set)
        ),
        "| 冻结 seeds | {} |".format(_cell(", ".join(_seed_label(seed) for seed in seed_set))),
        "| 每次运行任务数 | {} |".format(len(task_set.task_ids)),
        "",
        "## 输入完整性",
        "",
        "| 已核验工件 | 状态 |",
        "|---|---|",
        "| `matrix.jsonl` | {} 个自校验 manifest，覆盖每个模型 × 条件 × seed 组合 |".format(
            len(artifacts.manifests)
        ),
        "| `preflight.json` | {} 个运行、{} 个静态任务检查；未将 provider 或 runtime 调用计入 preflight |".format(
            artifacts.preflight["run_count"], artifacts.preflight["task_count"]
        ),
        "| `execution.json` 与 receipt | {} 个运行均为 completed，且每条 receipt 绑定对应 manifest 与完整任务集 |".format(
            artifacts.execution["run_count"]
        ),
        "| 原始 EpisodeResult JSONL | {} 条客观观察结果（{} 运行 × {} 任务） |".format(
            episode_count, len(artifacts.manifests), len(task_set.task_ids)
        ),
        "| `score.json` | 已从上述原始结果重新计算，并逐字段匹配 |",
        "",
        "## 冻结模型身份",
        "",
        "| Ollama 模型标签 | 不可变 digest | Model profile ID |",
        "|---|---|---|",
    ]
    for model in model_order:
        lines.append(
            "| {} | {} | {} |".format(
                _cell(model), _cell(digest_by_model[model]), _cell(profile_by_model[model])
            )
        )

    lines.extend(
        [
            "",
            "## 按条件观察到的结果",
            "",
            "| 模型 | 条件 | Seeds | 运行数 | 任务通过率 | 状态一致率 | 锚点/运行 | Tokens/运行 | Tokens/成功任务 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in model_order:
        model_id = profile_by_model[model]
        for condition in condition_order:
            summary = _mapping(summaries[(model_id, condition)], "score summary")
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    _cell(model),
                    _cell(condition),
                    _cell(", ".join(_seed_label(seed) for seed in seeds_by_pair[(model, condition)])),
                    _cell(summary["run_count"]),
                    _rate(_metric(summary, "task_success_rate")),
                    _rate(_metric(summary, "state_consistency_rate")),
                    _decimal(_metric(summary, "anchors_injected")),
                    _decimal(_metric(summary, "token_consumption")),
                    _decimal(_metric(summary, "tokens_per_successful_task")),
                )
            )

    lines.extend(
        [
            "",
            "## 有界运行诊断（非评分指标）",
            "",
            "下表只呈现 allowlist 中的运行类别；未识别值会被抑制，避免把模型文本、provider 响应或凭据带入报告。",
            "",
            "| 模型 | 条件 | 诊断类别 | Episode 数 |",
            "|---|---|---|---:|",
        ]
    )
    if diagnostic_counts:
        for (model, condition, diagnostic), count in sorted(diagnostic_counts.items()):
            lines.append(
                "| {} | {} | {} | {} |".format(
                    _cell(model), _cell(condition), _cell(diagnostic), count
                )
            )
    else:
        lines.append("| — | — | 未记录诊断 | 0 |")

    lines.extend(
        [
            "",
            "## 解读限制",
            "",
            "- 任务是模型可读的 durable-trace 机制任务；它验证的是机制链路和客观 trace judge，不是对开放域能力的测量。",
            "- 本次每个模型 × 条件组合只有 {} 个 seed；表格是冻结观察值，不构成方差估计或统计显著性结论。".format(
                len(seed_set)
            ),
            "- 公开 artifact／domain 任务以及故障注入，仍需要各自的任务环境、实际故障调度和环境专属 judge 后才能报告经验性结果。",
            "- 保留 `matrix.jsonl`、`preflight.json`、`execution.json`、receipt、原始 JSONL、`score.json` 与 durable runtime 事件，供独立复核；本报告不复述 prompt、模型响应、provider 正文或凭据。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit a complete local Ollama durable-trace smoke artifact and render Chinese Markdown"
    )
    parser.add_argument("artifact_dir", type=Path, help="directory written by ollama_durable_trace.py")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report destination")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="single-line Markdown report title")
    args = parser.parse_args(argv)
    try:
        artifacts = load_ollama_trace_artifacts(args.artifact_dir)
        rendered = render_ollama_trace_report(artifacts, title=args.title)
    except SmokeReportValidationError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print("wrote {}".format(args.output))


if __name__ == "__main__":
    main()

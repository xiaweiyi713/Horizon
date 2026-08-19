"""Execute frozen HorizonBench matrix rows through an objective task adapter.

This module deliberately does not turn an LLM response into a benchmark
outcome.  The selected executor owns the task environment and must return an
``EpisodeResult`` observed by that environment.  The runner freezes the
manifest/task binding, persists each completed outcome atomically, and writes
a sidecar receipt so an interrupted run can resume without silently mixing
models, task sets, or executors.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

try:  # Supports both `python file.py` and module execution.
    from .adapters import BenchmarkTask, BenchmarkTaskSet, JsonlTaskAdapter
    from .execution import (
        EpisodeExecutor,
        ExecutionContext,
        ExecutionValidationError,
        load_executor,
        normalize_episode_result,
    )
    from .protocol import ManifestValidationError, RunManifest, TaskSetIdentity, read_manifest_jsonl
    from .score import EpisodeResult, read_jsonl
except ImportError:  # pragma: no cover - direct-script compatibility
    from adapters import BenchmarkTask, BenchmarkTaskSet, JsonlTaskAdapter
    from execution import (  # type: ignore[no-redef]
        EpisodeExecutor,
        ExecutionContext,
        ExecutionValidationError,
        load_executor,
        normalize_episode_result,
    )
    from protocol import ManifestValidationError, RunManifest, TaskSetIdentity, read_manifest_jsonl
    from score import EpisodeResult, read_jsonl


EXECUTION_PROTOCOL = "horizonbench_matrix_execution_v1"
RECEIPT_SCHEMA_VERSION = 1
RESULT_SUFFIX = ".jsonl"
RECEIPT_SUFFIX = ".receipt.json"
_RECEIPT_STATUSES = frozenset(("running", "failed", "completed"))
_ATTEMPT_STATUSES = frozenset(("succeeded", "failed"))


class MatrixExecutionError(RuntimeError):
    """Raised after the runner has recorded an executor failure in its receipt."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_plain_run_id(run_id: str) -> str:
    """Reject path-like run IDs before making a result/receipt filename."""

    if not isinstance(run_id, str) or not run_id:
        raise ExecutionValidationError("manifest run_id must be a non-empty string")
    candidate = Path(run_id)
    if candidate.name != run_id or run_id in {".", ".."} or "\\" in run_id:
        raise ExecutionValidationError(
            "manifest run_id must be a plain filename when executing a matrix: {!r}".format(run_id)
        )
    return run_id


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one local artifact without leaving a partially written JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_file():
        raise ExecutionValidationError("execution artifact is not a regular file: {}".format(path))
    if path.is_symlink():
        raise ExecutionValidationError("refusing to write execution artifact through symlink: {}".format(path))
    temporary = path.with_name(".{}.{}.tmp".format(path.name, uuid4().hex))
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_results(path: Path, results: Sequence[EpisodeResult]) -> None:
    encoded = "".join(
        json.dumps(asdict(result), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for result in results
    )
    _atomic_write_text(path, encoded)


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ExecutionValidationError("{} is not a regular file: {}".format(label, path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ExecutionValidationError("invalid JSON in {} {}: {}".format(label, path, error)) from error
    if not isinstance(value, Mapping):
        raise ExecutionValidationError("{} must contain a JSON object: {}".format(label, path))
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError("invalid JSON constant {}".format(value))


def _result_paths(results_dir: Path, manifest: RunManifest) -> tuple[Path, Path]:
    run_id = _require_plain_run_id(manifest.run_id)
    return results_dir / (run_id + RESULT_SUFFIX), results_dir / (run_id + RECEIPT_SUFFIX)


def _task_index(task_set: BenchmarkTaskSet) -> dict[str, BenchmarkTask]:
    return {task.task_id: task for task in task_set.tasks}


def validate_manifest_task_set(manifest: RunManifest, task_set: BenchmarkTaskSet) -> None:
    identity = TaskSetIdentity.from_task_set(task_set)
    if manifest.task_set != identity:
        raise ExecutionValidationError(
            "task source selected for run {!r} does not match its frozen manifest task set".format(
                manifest.run_id
            )
        )


def _read_partial_results(
    path: Path,
    manifest: RunManifest,
    task_by_id: Mapping[str, BenchmarkTask],
) -> list[EpisodeResult]:
    if not path.exists():
        return []
    if not path.is_file() or path.is_symlink():
        raise ExecutionValidationError("result artifact is not a regular file: {}".format(path))
    rows = read_jsonl(path)
    normalized: list[EpisodeResult] = []
    seen: set[str] = set()
    for row in rows:
        task = task_by_id.get(row.task_id)
        if task is None:
            raise ExecutionValidationError(
                "result artifact {} contains task {!r} outside frozen task set".format(path, row.task_id)
            )
        if row.task_id in seen:
            raise ExecutionValidationError(
                "result artifact {} contains duplicate task id {!r}".format(path, row.task_id)
            )
        seen.add(row.task_id)
        normalized.append(normalize_episode_result(row, ExecutionContext(manifest, task)))
    return normalized


def _new_receipt(manifest: RunManifest, task_set: BenchmarkTaskSet, executor_spec: str) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol": EXECUTION_PROTOCOL,
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.manifest_sha256,
        "task_set": TaskSetIdentity.from_task_set(task_set).as_dict(),
        "executor": executor_spec,
        "status": "running",
        "started_at": now,
        "updated_at": now,
        "completed_task_ids": [],
        "attempts": [],
        "error": None,
    }


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExecutionValidationError("receipt {} must be a non-empty string".format(label))
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExecutionValidationError("receipt {} must be a positive integer".format(label))
    return value


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    manifest: RunManifest,
    task_set: BenchmarkTaskSet,
    executor_spec: str,
    results: Sequence[EpisodeResult],
) -> dict[str, Any]:
    """Validate a sidecar receipt and reconcile a crash between two atomic writes.

    A completed result may exist before its receipt update reaches disk.  That
    is safe to reconcile because the manifest-bound outcome JSONL is the
    source of task completion.  The reverse direction (receipt says complete
    but an outcome is absent) is rejected as possible data loss.
    """

    allowed = {
        "schema_version",
        "protocol",
        "run_id",
        "manifest_sha256",
        "task_set",
        "executor",
        "status",
        "started_at",
        "updated_at",
        "completed_task_ids",
        "attempts",
        "error",
    }
    unknown = sorted(set(receipt) - allowed)
    if unknown:
        raise ExecutionValidationError("receipt has unrecognized fields: {}".format(", ".join(unknown)))
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ExecutionValidationError("receipt has an unsupported schema_version")
    if receipt.get("protocol") != EXECUTION_PROTOCOL:
        raise ExecutionValidationError("receipt has an unsupported protocol")
    if receipt.get("run_id") != manifest.run_id:
        raise ExecutionValidationError("receipt run_id does not match selected manifest")
    if receipt.get("manifest_sha256") != manifest.manifest_sha256:
        raise ExecutionValidationError("receipt manifest_sha256 does not match selected manifest")
    raw_task_set = receipt.get("task_set")
    if not isinstance(raw_task_set, Mapping):
        raise ExecutionValidationError("receipt task_set must be an object")
    try:
        receipt_task_set = TaskSetIdentity.from_mapping(raw_task_set)
    except ManifestValidationError as error:
        raise ExecutionValidationError("receipt task_set is invalid: {}".format(error)) from error
    if receipt_task_set != TaskSetIdentity.from_task_set(task_set):
        raise ExecutionValidationError("receipt task_set does not match selected task source")
    if receipt.get("executor") != executor_spec:
        raise ExecutionValidationError(
            "receipt executor does not match requested executor; use a new result directory or --overwrite"
        )
    status = receipt.get("status")
    if status not in _RECEIPT_STATUSES:
        raise ExecutionValidationError("receipt status is invalid")
    _require_string(receipt.get("started_at"), "started_at")
    _require_string(receipt.get("updated_at"), "updated_at")
    completed = receipt.get("completed_task_ids")
    if not isinstance(completed, list) or not all(isinstance(item, str) and item for item in completed):
        raise ExecutionValidationError("receipt completed_task_ids must be an array of non-empty strings")
    if len(set(completed)) != len(completed):
        raise ExecutionValidationError("receipt completed_task_ids contains duplicates")
    result_ids = {row.task_id for row in results}
    expected_ids = set(manifest.task_set.task_ids)
    completed_ids = set(completed)
    if not completed_ids <= expected_ids:
        raise ExecutionValidationError("receipt completed_task_ids contains tasks outside frozen task set")
    if not completed_ids <= result_ids:
        raise ExecutionValidationError("receipt claims a completed task with no persisted outcome")
    raw_attempts = receipt.get("attempts")
    if not isinstance(raw_attempts, list):
        raise ExecutionValidationError("receipt attempts must be an array")
    attempts: list[dict[str, Any]] = []
    next_attempt_by_task: dict[str, int] = {}
    for index, raw in enumerate(raw_attempts, 1):
        if not isinstance(raw, Mapping):
            raise ExecutionValidationError("receipt attempt {} must be an object".format(index))
        if set(raw) != {"task_id", "attempt", "started_at", "finished_at", "duration_ms", "status", "error"}:
            raise ExecutionValidationError("receipt attempt {} has an invalid field set".format(index))
        task_id = _require_string(raw.get("task_id"), "attempt.task_id")
        if task_id not in expected_ids:
            raise ExecutionValidationError("receipt attempt references task outside frozen task set")
        attempt = _require_positive_int(raw.get("attempt"), "attempt.attempt")
        expected_attempt = next_attempt_by_task.get(task_id, 1)
        if attempt != expected_attempt:
            raise ExecutionValidationError(
                "receipt attempt numbers must be contiguous for task {!r}".format(task_id)
            )
        next_attempt_by_task[task_id] = attempt + 1
        duration_ms = raw.get("duration_ms")
        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms < 0:
            raise ExecutionValidationError("receipt attempt.duration_ms must be a non-negative integer")
        attempt_status = raw.get("status")
        if attempt_status not in _ATTEMPT_STATUSES:
            raise ExecutionValidationError("receipt attempt.status is invalid")
        _require_string(raw.get("started_at"), "attempt.started_at")
        _require_string(raw.get("finished_at"), "attempt.finished_at")
        error = raw.get("error")
        if error is not None and not isinstance(error, str):
            raise ExecutionValidationError("receipt attempt.error must be a string or null")
        if (attempt_status == "succeeded") != (error is None):
            raise ExecutionValidationError("receipt attempt error does not match its status")
        if attempt_status == "succeeded" and task_id not in result_ids:
            raise ExecutionValidationError("receipt records a successful attempt with no persisted outcome")
        attempts.append(
            {
                "task_id": task_id,
                "attempt": attempt,
                "started_at": raw["started_at"],
                "finished_at": raw["finished_at"],
                "duration_ms": duration_ms,
                "status": attempt_status,
                "error": error,
            }
        )
    error = receipt.get("error")
    if error is not None and not isinstance(error, str):
        raise ExecutionValidationError("receipt error must be a string or null")
    if status == "completed" and error is not None:
        raise ExecutionValidationError("completed receipt cannot carry an error")
    if status == "completed" and result_ids != expected_ids:
        raise ExecutionValidationError("completed receipt does not have every frozen task outcome")

    # Detach untrusted JSON before mutation, then repair only the safe,
    # documented result-first crash window described above.
    normalized = dict(receipt)
    normalized["attempts"] = attempts
    normalized["completed_task_ids"] = [
        task_id for task_id in manifest.task_set.task_ids if task_id in result_ids
    ]
    return normalized


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(dict(receipt), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )


def _attempt_number(receipt: Mapping[str, Any], task_id: str) -> int:
    attempts = receipt["attempts"]
    return 1 + sum(1 for entry in attempts if entry["task_id"] == task_id)


def _record_attempt(
    receipt: dict[str, Any],
    *,
    task_id: str,
    attempt: int,
    started_at: str,
    duration_ms: int,
    status: str,
    error: Optional[str],
) -> None:
    receipt["attempts"].append(
        {
            "task_id": task_id,
            "attempt": attempt,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "duration_ms": duration_ms,
            "status": status,
            "error": error,
        }
    )
    receipt["updated_at"] = _utc_now()


def _reset_for_resume(receipt: dict[str, Any]) -> None:
    receipt["status"] = "running"
    receipt["error"] = None
    receipt["updated_at"] = _utc_now()


def _remove_output(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_file() or path.is_symlink():
        raise ExecutionValidationError("refusing to replace non-regular execution artifact: {}".format(path))
    path.unlink()


def _summary(
    manifest: RunManifest,
    result_path: Path,
    receipt_path: Path,
    *,
    executed_task_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "run_id": manifest.run_id,
        "status": "completed",
        "result_path": str(result_path),
        "receipt_path": str(receipt_path),
        "task_count": len(manifest.task_set.task_ids),
        "executed_task_ids": list(executed_task_ids),
    }


def execute_manifest(
    manifest: RunManifest,
    *,
    task_set: BenchmarkTaskSet,
    executor: EpisodeExecutor,
    executor_spec: str,
    results_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute or resume one frozen manifest against its exact task source."""

    if not callable(executor):
        raise ExecutionValidationError("executor must be callable")
    if not isinstance(executor_spec, str) or not executor_spec:
        raise ExecutionValidationError("executor_spec must be a non-empty string")
    validate_manifest_task_set(manifest, task_set)
    result_path, receipt_path = _result_paths(results_dir, manifest)
    if overwrite:
        _remove_output(result_path)
        _remove_output(receipt_path)

    task_by_id = _task_index(task_set)
    results = _read_partial_results(result_path, manifest, task_by_id)
    if result_path.exists() and not receipt_path.exists():
        raise ExecutionValidationError(
            "result artifact exists without a receipt; refuse to adopt unprovenanced outcomes: {}".format(
                result_path
            )
        )
    if receipt_path.exists():
        receipt = _validate_receipt(
            _read_json_object(receipt_path, "execution receipt"),
            manifest=manifest,
            task_set=task_set,
            executor_spec=executor_spec,
            results=results,
        )
    else:
        receipt = _new_receipt(manifest, task_set, executor_spec)
        _write_receipt(receipt_path, receipt)

    completed_ids = {result.task_id for result in results}
    expected_ids = set(manifest.task_set.task_ids)
    if completed_ids == expected_ids:
        receipt["status"] = "completed"
        receipt["error"] = None
        receipt["updated_at"] = _utc_now()
        _write_receipt(receipt_path, receipt)
        return _summary(manifest, result_path, receipt_path, executed_task_ids=[])

    _reset_for_resume(receipt)
    _write_receipt(receipt_path, receipt)
    executed_task_ids: list[str] = []
    for task in task_set.tasks:
        if task.task_id in completed_ids:
            continue
        attempt = _attempt_number(receipt, task.task_id)
        started_at = _utc_now()
        started_monotonic = time.monotonic()
        try:
            context = ExecutionContext(manifest, task, attempt=attempt)
            outcome = normalize_episode_result(executor(context), context)
        except Exception as error:
            duration_ms = max(0, int((time.monotonic() - started_monotonic) * 1000))
            message = "{}: {}".format(type(error).__name__, error)
            _record_attempt(
                receipt,
                task_id=task.task_id,
                attempt=attempt,
                started_at=started_at,
                duration_ms=duration_ms,
                status="failed",
                error=message,
            )
            receipt["status"] = "failed"
            receipt["error"] = message
            _write_receipt(receipt_path, receipt)
            raise MatrixExecutionError(
                "run {!r} failed on task {!r} attempt {}: {}".format(
                    manifest.run_id, task.task_id, attempt, message
                )
            ) from error
        duration_ms = max(0, int((time.monotonic() - started_monotonic) * 1000))
        results.append(outcome)
        completed_ids.add(task.task_id)
        _write_results(result_path, results)
        _record_attempt(
            receipt,
            task_id=task.task_id,
            attempt=attempt,
            started_at=started_at,
            duration_ms=duration_ms,
            status="succeeded",
            error=None,
        )
        receipt["completed_task_ids"] = [
            task_id for task_id in manifest.task_set.task_ids if task_id in completed_ids
        ]
        _write_receipt(receipt_path, receipt)
        executed_task_ids.append(task.task_id)

    if completed_ids != expected_ids:  # Defensive: protects future task iterator changes.
        raise ExecutionValidationError("executor finished without every frozen task outcome")
    receipt["status"] = "completed"
    receipt["error"] = None
    receipt["updated_at"] = _utc_now()
    _write_receipt(receipt_path, receipt)
    return _summary(manifest, result_path, receipt_path, executed_task_ids=executed_task_ids)


def select_manifests(manifests: Sequence[RunManifest], run_ids: Optional[Sequence[str]]) -> list[RunManifest]:
    if not run_ids:
        return list(manifests)
    requested = list(run_ids)
    if len(set(requested)) != len(requested):
        raise ExecutionValidationError("--run-id values must be unique")
    by_id = {manifest.run_id: manifest for manifest in manifests}
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ExecutionValidationError("run_id was not found in matrix: {}".format(", ".join(missing)))
    return [by_id[run_id] for run_id in requested]


def execute_matrix(
    matrix_path: Path,
    *,
    tasks_path: Path,
    source_name: str,
    source_revision: str,
    split: str,
    executor: EpisodeExecutor,
    executor_spec: str,
    results_dir: Path,
    run_ids: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute selected matrix rows, resuming only receipt-bound local outputs."""

    manifests = select_manifests(read_manifest_jsonl(matrix_path), run_ids)
    task_set = JsonlTaskAdapter(source_name, source_revision).load(tasks_path, split=split)
    for manifest in manifests:
        validate_manifest_task_set(manifest, task_set)
    runs = [
        execute_manifest(
            manifest,
            task_set=task_set,
            executor=executor,
            executor_spec=executor_spec,
            results_dir=results_dir,
            overwrite=overwrite,
        )
        for manifest in manifests
    ]
    return {
        "protocol": EXECUTION_PROTOCOL,
        "matrix_path": str(matrix_path),
        "results_dir": str(results_dir),
        "executor": executor_spec,
        "task_set": TaskSetIdentity.from_task_set(task_set).as_dict(),
        "run_count": len(runs),
        "runs": runs,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen HorizonBench matrix rows through an objective executor plugin"
    )
    parser.add_argument("--matrix", type=Path, required=True, help="JSONL matrix from plan_matrix.py")
    parser.add_argument("--tasks", type=Path, required=True, help="the exact local task JSONL used to plan")
    parser.add_argument("--source-name", required=True, help="task-source identity used when planning")
    parser.add_argument("--source-revision", required=True, help="immutable task-source revision used when planning")
    parser.add_argument("--split", default="heldout", help="task split to execute (default: heldout)")
    parser.add_argument(
        "--executor",
        required=True,
        help="import path module:callable; it must return environment-observed EpisodeResults",
    )
    parser.add_argument("--results-dir", type=Path, required=True, help="destination for result JSONL and receipts")
    parser.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="run only this manifest ID (repeatable; default executes every matrix row)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing result/receipt pairs instead of resuming them",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = make_parser().parse_args(argv)
    try:
        report = execute_matrix(
            args.matrix,
            tasks_path=args.tasks,
            source_name=args.source_name,
            source_revision=args.source_revision,
            split=args.split,
            executor=load_executor(args.executor),
            executor_spec=args.executor,
            results_dir=args.results_dir,
            run_ids=args.run_id,
            overwrite=args.overwrite,
        )
    except (ExecutionValidationError, ManifestValidationError, MatrixExecutionError, ValueError) as error:
        raise SystemExit("error: {}".format(error)) from error
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

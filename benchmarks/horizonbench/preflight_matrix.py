"""Statically validate frozen HorizonBench matrix rows before execution.

Unlike :mod:`execute_matrix`, this command writes no results or receipts and
never invokes an episode executor.  It calls only an executor's explicit
``preflight(context)`` hook, allowing model/runtime configuration errors to be
caught before an experiment crosses an external boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

try:  # Supports both `python file.py` and module execution.
    from .adapters import JsonlTaskAdapter
    from .execution import (
        EpisodeExecutor,
        ExecutionContext,
        ExecutionValidationError,
        load_executor,
        preflight_executor,
    )
    from .execute_matrix import select_manifests, validate_manifest_task_set
    from .protocol import ManifestValidationError, TaskSetIdentity, read_manifest_jsonl
except ImportError:  # pragma: no cover - direct-script compatibility
    from adapters import JsonlTaskAdapter
    from execution import (  # type: ignore[no-redef]
        EpisodeExecutor,
        ExecutionContext,
        ExecutionValidationError,
        load_executor,
        preflight_executor,
    )
    from execute_matrix import select_manifests, validate_manifest_task_set
    from protocol import ManifestValidationError, TaskSetIdentity, read_manifest_jsonl


PREFLIGHT_PROTOCOL = "horizonbench_matrix_preflight_v1"


def preflight_matrix(
    matrix_path: Path,
    *,
    tasks_path: Path,
    source_name: str,
    source_revision: str,
    split: str,
    executor: EpisodeExecutor,
    executor_spec: str,
    run_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Check selected rows against immutable task input without running them.

    The returned report is intentionally stdout-friendly rather than persisted:
    a preflight must not create resumable artifacts that could be mistaken for
    observed benchmark outcomes.
    """

    if not callable(executor):
        raise ExecutionValidationError("executor must be callable")
    if not isinstance(executor_spec, str) or not executor_spec:
        raise ExecutionValidationError("executor_spec must be a non-empty string")

    manifests = select_manifests(read_manifest_jsonl(matrix_path), run_ids)
    task_set = JsonlTaskAdapter(source_name, source_revision).load(tasks_path, split=split)
    for manifest in manifests:
        validate_manifest_task_set(manifest, task_set)

    runs: list[dict[str, Any]] = []
    for manifest in manifests:
        checks: list[dict[str, Any]] = []
        for task in task_set.tasks:
            context = ExecutionContext(manifest, task)
            checks.append(
                {
                    "task_id": task.task_id,
                    "preflight": preflight_executor(executor, context),
                }
            )
        runs.append(
            {
                "run_id": manifest.run_id,
                "manifest_sha256": manifest.manifest_sha256,
                "task_count": len(checks),
                "checks": checks,
            }
        )

    return {
        "protocol": PREFLIGHT_PROTOCOL,
        "matrix_path": str(matrix_path),
        "executor": executor_spec,
        "task_set": TaskSetIdentity.from_task_set(task_set).as_dict(),
        "run_count": len(runs),
        "task_count": sum(run["task_count"] for run in runs),
        "runs": runs,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate frozen HorizonBench rows without executing a runtime, model, or task environment"
        )
    )
    parser.add_argument("--matrix", type=Path, required=True, help="JSONL matrix from plan_matrix.py")
    parser.add_argument("--tasks", type=Path, required=True, help="the exact local task JSONL used to plan")
    parser.add_argument("--source-name", required=True, help="task-source identity used when planning")
    parser.add_argument(
        "--source-revision",
        required=True,
        help="immutable task-source revision used when planning",
    )
    parser.add_argument("--split", default="heldout", help="task split to validate (default: heldout)")
    parser.add_argument(
        "--executor",
        required=True,
        help="import path module:callable; callable must expose static preflight(context)",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="validate only this manifest ID (repeatable; default checks every matrix row)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = make_parser().parse_args(argv)
    try:
        report = preflight_matrix(
            args.matrix,
            tasks_path=args.tasks,
            source_name=args.source_name,
            source_revision=args.source_revision,
            split=args.split,
            executor=load_executor(args.executor),
            executor_spec=args.executor,
            run_ids=args.run_id,
        )
    except (ExecutionValidationError, ManifestValidationError, ValueError) as error:
        raise SystemExit("error: {}".format(error)) from error
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

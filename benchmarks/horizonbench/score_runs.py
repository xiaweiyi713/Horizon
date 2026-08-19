"""Score adapter-produced HorizonBench JSONL traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

try:  # Supports both `python file.py` and module execution.
    from .score import read_jsonl, score_results
    from .protocol import (
        ManifestValidationError,
        RunManifest,
        load_manifest,
        manifest_summary,
        read_manifest_jsonl,
        validate_result_task_ids,
    )
except ImportError:  # pragma: no cover - direct-script compatibility
    from score import read_jsonl, score_results
    from protocol import (  # type: ignore[no-redef]
        ManifestValidationError,
        RunManifest,
        load_manifest,
        manifest_summary,
        read_manifest_jsonl,
        validate_result_task_ids,
    )


def _manifest_from_matrix(path: Path, run_id: str) -> RunManifest:
    matches = [manifest for manifest in read_manifest_jsonl(path) if manifest.run_id == run_id]
    if not matches:
        raise ManifestValidationError("run_id {!r} was not found in {}".format(run_id, path))
    if len(matches) != 1:  # read_manifest_jsonl also rejects duplicates; keep this boundary explicit.
        raise ManifestValidationError("run_id {!r} occurs more than once in {}".format(run_id, path))
    return matches[0]


def score_run(
    input_path: Path,
    *,
    manifest_path: Optional[Path] = None,
    manifest_matrix_path: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """Score a result trace and, when supplied, bind it to a frozen manifest."""

    if manifest_path is not None and manifest_matrix_path is not None:
        raise ManifestValidationError("supply either manifest_path or manifest_matrix_path, not both")
    if manifest_matrix_path is not None and not run_id:
        raise ManifestValidationError("run_id is required with manifest_matrix_path")
    if manifest_path is not None and run_id is not None:
        raise ManifestValidationError("run_id is only valid with manifest_matrix_path")

    results = read_jsonl(input_path)
    metrics = score_results(results).as_dict()
    manifest: Optional[RunManifest] = None
    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
    elif manifest_matrix_path is not None:
        manifest = _manifest_from_matrix(manifest_matrix_path, run_id or "")
    if manifest is None:
        return metrics
    validate_result_task_ids(results, manifest)
    return {"manifest": manifest_summary(manifest), "metrics": metrics}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Score JSONL episode outcomes against HorizonBench metrics")
    parser.add_argument("input", type=Path, help="one JSON object per completed benchmark task")
    parser.add_argument("--output", type=Path, default=None)
    manifests = parser.add_mutually_exclusive_group()
    manifests.add_argument("--manifest", type=Path, default=None, help="single frozen run manifest JSON")
    manifests.add_argument(
        "--manifest-matrix", type=Path, default=None, help="JSONL matrix produced by plan_matrix.py"
    )
    parser.add_argument("--run-id", default=None, help="required when selecting a row from --manifest-matrix")
    args = parser.parse_args(argv)
    try:
        report = score_run(
            args.input,
            manifest_path=args.manifest,
            manifest_matrix_path=args.manifest_matrix,
            run_id=args.run_id,
        )
    except (ManifestValidationError, ValueError) as error:
        parser.error(str(error))
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

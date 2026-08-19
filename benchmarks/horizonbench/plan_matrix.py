"""Plan reproducible model × condition HorizonBench runs without invoking a provider.

The generated JSONL is intentionally a run plan, not an LLM result.  Execute
each row through a provider-specific adapter, emit one observed EpisodeResult
per frozen task, then pass the result file and its single manifest to
``score_runs.py --manifest``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

try:  # Supports both `python file.py` and module execution.
    from .adapters import JsonlTaskAdapter
    from .protocol import (
        EvaluationCondition,
        ManifestValidationError,
        ModelProfile,
        PromptArtifact,
        TaskSetIdentity,
        build_cross_model_matrix,
        write_manifest_jsonl,
    )
except ImportError:  # pragma: no cover - direct-script compatibility
    from adapters import JsonlTaskAdapter
    from protocol import (  # type: ignore[no-redef]
        EvaluationCondition,
        ManifestValidationError,
        ModelProfile,
        PromptArtifact,
        TaskSetIdentity,
        build_cross_model_matrix,
        write_manifest_jsonl,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError("invalid JSON constant {}".format(value))


def _load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise ManifestValidationError("{} file does not exist: {}".format(label, path))
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ManifestValidationError("invalid JSON in {} {}: {}".format(label, path, error)) from error


def _load_array(path: Path, key: str) -> list[Mapping[str, Any]]:
    raw = _load_json(path, key)
    if isinstance(raw, Mapping):
        raw = raw.get(key)
    if not isinstance(raw, list) or not raw:
        raise ManifestValidationError("{} must contain a non-empty JSON array".format(path))
    if not all(isinstance(item, Mapping) for item in raw):
        raise ManifestValidationError("{} entries must all be JSON objects".format(path))
    return list(raw)


def _load_optional_object(path: Optional[Path], label: str, default: Any) -> Any:
    if path is None:
        return default
    raw = _load_json(path, label)
    if not isinstance(raw, type(default)):
        expected = "object" if isinstance(default, dict) else "array"
        raise ManifestValidationError("{} must contain a JSON {}".format(path, expected))
    return raw


def _parse_seeds(value: str) -> list[Optional[int]]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ManifestValidationError("--seeds must be comma-separated integers or 'none'")
    seeds: list[Optional[int]] = []
    for part in parts:
        if part.lower() == "none":
            seeds.append(None)
            continue
        try:
            seeds.append(int(part))
        except ValueError as error:
            raise ManifestValidationError(
                "--seeds must contain integers or 'none', got {!r}".format(part)
            ) from error
    return seeds


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a frozen, provider-neutral HorizonBench model evaluation matrix"
    )
    parser.add_argument("--tasks", type=Path, required=True, help="public benchmark task JSONL")
    parser.add_argument("--source-name", required=True, help="stable public benchmark/source identifier")
    parser.add_argument("--source-revision", required=True, help="dataset release, commit, or immutable revision")
    parser.add_argument("--split", default="heldout", help="task split to evaluate (default: heldout)")
    parser.add_argument("--models", type=Path, required=True, help="JSON array or {models: [...]} profile file")
    parser.add_argument(
        "--conditions", type=Path, required=True, help="JSON array or {conditions: [...]} control file"
    )
    parser.add_argument("--prompt", type=Path, required=True, help="system prompt text file to embed and hash")
    parser.add_argument("--prompt-revision", required=True, help="declared prompt revision")
    parser.add_argument("--runtime-revision", required=True, help="Horizon/runtime source revision")
    parser.add_argument(
        "--seeds",
        default="0",
        help="comma-separated seeds; use 'none' only if a provider has no seed control (default: 0)",
    )
    parser.add_argument("--checkpoint-cadence", type=int, default=1)
    parser.add_argument("--fault-schedule", type=Path, default=None, help="optional JSON array")
    parser.add_argument("--metadata", type=Path, default=None, help="optional JSON object")
    parser.add_argument("--output", type=Path, required=True, help="destination JSONL manifest matrix")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    return parser


def plan_matrix(args: argparse.Namespace) -> list[Any]:
    """Create matrix rows from parsed CLI arguments; exposed for testability."""

    if args.output.exists() and not args.overwrite:
        raise ManifestValidationError("refusing to overwrite existing manifest file: {}".format(args.output))
    models = [ModelProfile.from_mapping(item) for item in _load_array(args.models, "models")]
    conditions = [
        EvaluationCondition.from_mapping(item) for item in _load_array(args.conditions, "conditions")
    ]
    if not args.prompt.is_file():
        raise ManifestValidationError("prompt file does not exist: {}".format(args.prompt))
    prompt = PromptArtifact(args.prompt_revision, args.prompt.read_text(encoding="utf-8"))
    adapter = JsonlTaskAdapter(args.source_name, args.source_revision)
    selected_tasks = adapter.load(args.tasks, split=args.split)
    task_set = TaskSetIdentity.from_task_set(selected_tasks)
    manifests = build_cross_model_matrix(
        models=models,
        conditions=conditions,
        task_set=task_set,
        prompt=prompt,
        seeds=_parse_seeds(args.seeds),
        runtime_revision=args.runtime_revision,
        checkpoint_cadence=args.checkpoint_cadence,
        fault_schedule=_load_optional_object(args.fault_schedule, "fault schedule", []),
        metadata=_load_optional_object(args.metadata, "metadata", {}),
    )
    write_manifest_jsonl(args.output, manifests)
    return manifests


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = make_parser().parse_args(argv)
    try:
        manifests = plan_matrix(args)
    except (ManifestValidationError, ValueError) as error:
        raise SystemExit("error: {}".format(error)) from error
    first = manifests[0]
    print(
        json.dumps(
            {
                "protocol": "horizonbench_cross_model_manifest_v1",
                "manifest_count": len(manifests),
                "output": str(args.output),
                "task_set_sha256": first.task_set.task_set_sha256,
                "task_count": len(first.task_set.task_ids),
                "split": first.task_set.split,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

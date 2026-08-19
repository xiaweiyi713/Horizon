"""Strict public/held-out JSONL adapter for HorizonBench task sources.

The adapter is dependency-free and never downloads data or contacts the
network. It only normalizes local UTF-8 JSONL records into immutable task
objects suitable for a run manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Union


_NORMALIZED_FIELDS = frozenset({"id", "category", "goal", "constraint", "split"})
_REQUIRED_FIELDS = ("id", "category", "goal", "constraint", "split")
PathLike = Union[str, Path]


@dataclass(frozen=True)
class BenchmarkTask:
    """Normalized benchmark task with JSON-safe metadata."""

    task_id: str
    category: str
    goal: str
    constraint: str
    split: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "goal": self.goal,
            "constraint": self.constraint,
            "split": self.split,
            "metadata": _json_ready(self.metadata),
        }


@dataclass(frozen=True)
class BenchmarkTaskSet:
    """Loaded selection of tasks plus source identity for a run manifest."""

    source_name: str
    source_revision: str
    split: str
    source_sha256: str
    task_set_sha256: str
    tasks: tuple[BenchmarkTask, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    def identity(self) -> dict[str, Any]:
        """Compact JSON-safe identity suitable for a run manifest."""

        return {
            "source_name": self.source_name,
            "source_revision": self.source_revision,
            "split": self.split,
            "source_sha256": self.source_sha256,
            "task_set_sha256": self.task_set_sha256,
            "task_ids": list(self.task_ids),
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.identity()
        payload["tasks"] = [task.as_dict() for task in self.tasks]
        return payload


class JsonlTaskAdapter:
    """Load a local JSONL task source and select one split."""

    def __init__(self, source_name: str, source_revision: str) -> None:
        self._source_name = _require_nonempty_string(source_name, "source_name")
        self._source_revision = _require_nonempty_string(source_revision, "source_revision")

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def source_revision(self) -> str:
        return self._source_revision

    def load(self, path: PathLike, *, split: str = "heldout") -> BenchmarkTaskSet:
        split = _require_nonempty_string(split, "split")
        source_path = Path(path)
        if not source_path.exists() or not source_path.is_file():
            raise ValueError("benchmark task source is missing or is not a file: {}".format(source_path))

        raw = source_path.read_bytes()
        source_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("source {} is not valid UTF-8: {}".format(source_path, error)) from error

        selected: list[BenchmarkTask] = []
        seen_ids: dict[str, int] = {}
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                raise ValueError("blank JSON line {} of {}".format(line_number, source_path))
            task = _parse_source_record(line, source_path, line_number)
            first_line = seen_ids.get(task.task_id)
            if first_line is not None:
                raise ValueError(
                    "duplicate task id {!r} on line {} of {} (first seen on line {})".format(
                        task.task_id,
                        line_number,
                        source_path,
                        first_line,
                    )
                )
            seen_ids[task.task_id] = line_number
            if task.split == split:
                selected.append(task)

        if not selected:
            raise ValueError("{} contains no tasks for split {!r}".format(source_path, split))

        return BenchmarkTaskSet(
            source_name=self._source_name,
            source_revision=self._source_revision,
            split=split,
            source_sha256=source_sha256,
            task_set_sha256=_task_set_sha256(selected),
            tasks=tuple(selected),
        )

    def load_heldout(self, path: PathLike) -> BenchmarkTaskSet:
        return self.load(path, split="heldout")


def _parse_source_record(line: str, source_path: Path, line_number: int) -> BenchmarkTask:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError("invalid JSON on line {} of {}: {}".format(line_number, source_path, error)) from error
    if not isinstance(value, dict):
        raise ValueError("line {} of {} is not a JSON object".format(line_number, source_path))

    fields: dict[str, str] = {}
    for field in _REQUIRED_FIELDS:
        if field not in value:
            raise ValueError("line {} of {} is missing required field {}".format(line_number, source_path, field))
        fields[field] = _require_record_string(value[field], field, source_path, line_number)

    metadata = _extract_metadata(value, source_path, line_number)
    return BenchmarkTask(
        task_id=fields["id"],
        category=fields["category"],
        goal=fields["goal"],
        constraint=fields["constraint"],
        split=fields["split"],
        metadata=metadata,
    )


def _extract_metadata(value: Mapping[str, Any], source_path: Path, line_number: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if "metadata" in value:
        raw_metadata = value["metadata"]
        if not isinstance(raw_metadata, dict):
            raise ValueError("line {} of {} has non-object metadata".format(line_number, source_path))
        _ensure_json_safe(raw_metadata, "line {} of {} field metadata".format(line_number, source_path))
        metadata.update(raw_metadata)

    for key, item in value.items():
        if key in _NORMALIZED_FIELDS or key == "metadata":
            continue
        where = "line {} of {} field {}".format(line_number, source_path, key)
        _ensure_json_safe(item, where)
        if key in metadata:
            raise ValueError(
                "line {} of {} has conflicting metadata field {}".format(line_number, source_path, key)
            )
        metadata[key] = item
    return metadata


def _task_set_sha256(tasks: Sequence[BenchmarkTask]) -> str:
    canonical_records = sorted((task.as_dict() for task in tasks), key=lambda item: item["task_id"])
    encoded = _canonical_json(canonical_records).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a nonempty string".format(field))
    return value


def _require_record_string(value: Any, field: str, source_path: Path, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "line {} of {} has invalid required field {}".format(line_number, source_path, field)
        )
    return value


def _ensure_json_safe(value: Any, where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("{} has a non-string metadata key".format(where))
            _ensure_json_safe(item, where)
        return
    if isinstance(value, list):
        for item in value:
            _ensure_json_safe(item, where)
        return
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("{} contains an unsupported non-finite number".format(where))
        return
    raise ValueError("{} contains an unsupported metadata JSON value".format(where))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["BenchmarkTask", "BenchmarkTaskSet", "JsonlTaskAdapter"]

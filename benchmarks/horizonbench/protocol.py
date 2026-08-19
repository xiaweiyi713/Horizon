"""Reproducible cross-model evaluation manifests for HorizonBench.

The benchmark scorer intentionally understands only observed episode outcomes.
This module supplies the adjacent experiment contract: freeze the selected task
set, model and decoding configuration, prompt, policy condition, and execution
controls before a model run begins.  It is dependency-free so public benchmark
mirrors can be evaluated without adding a provider SDK to Horizon.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence


MANIFEST_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class ManifestValidationError(ValueError):
    """Raised when a manifest cannot safely identify a reproducible run."""


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError("{} must be a non-empty string".format(field))
    return value


def _require_sha256(value: Any, field: str) -> str:
    value = _require_string(value, field)
    if not _SHA256_RE.fullmatch(value):
        raise ManifestValidationError("{} must be a lowercase SHA-256 hex digest".format(field))
    return value


def _normalize_json(value: Any, field: str = "value") -> Any:
    """Return a detached JSON-safe value, rejecting nonportable Python data."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestValidationError("{} must not contain NaN or infinity".format(field))
        return value
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, "{}[{}]".format(field, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ManifestValidationError("{} has a non-string key".format(field))
            normalized[key] = _normalize_json(item, "{}.{}".format(field, key))
        return normalized
    raise ManifestValidationError("{} must contain only JSON values".format(field))


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError("{} must be an object".format(field))
    normalized = _normalize_json(value, field)
    if not isinstance(normalized, dict):  # Keeps type checkers honest.
        raise ManifestValidationError("{} must be an object".format(field))
    return normalized


def _reject_unknown_fields(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ManifestValidationError(
            "{} contains unrecognized fields: {}".format(field, ", ".join(unknown))
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _freeze_json(value: Any) -> Any:
    """Recursively freeze validated JSON so a manifest cannot drift in memory."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    return slug or "run"


@dataclass(frozen=True)
class TaskSetIdentity:
    """Immutable identity of a selected benchmark split, not its task text."""

    source_name: str
    source_revision: str
    split: str
    source_sha256: str
    task_set_sha256: str
    task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_name", _require_string(self.source_name, "source_name"))
        object.__setattr__(self, "source_revision", _require_string(self.source_revision, "source_revision"))
        object.__setattr__(self, "split", _require_string(self.split, "split"))
        object.__setattr__(self, "source_sha256", _require_sha256(self.source_sha256, "source_sha256"))
        object.__setattr__(self, "task_set_sha256", _require_sha256(self.task_set_sha256, "task_set_sha256"))
        if isinstance(self.task_ids, (str, bytes)):
            raise ManifestValidationError("task_ids must be an array of strings")
        try:
            task_ids = tuple(_require_string(item, "task_ids") for item in self.task_ids)
        except TypeError as error:
            raise ManifestValidationError("task_ids must be an array of strings") from error
        if not task_ids:
            raise ManifestValidationError("task_ids must not be empty")
        if len(set(task_ids)) != len(task_ids):
            raise ManifestValidationError("task_ids must be unique")
        object.__setattr__(self, "task_ids", tuple(sorted(task_ids)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskSetIdentity":
        _reject_unknown_fields(
            value,
            {"source_name", "source_revision", "split", "source_sha256", "task_set_sha256", "task_ids"},
            "task_set",
        )
        task_ids = value.get("task_ids")
        if not isinstance(task_ids, (list, tuple)):
            raise ManifestValidationError("task_ids must be a JSON array")
        return cls(
            source_name=value.get("source_name"),
            source_revision=value.get("source_revision"),
            split=value.get("split"),
            source_sha256=value.get("source_sha256"),
            task_set_sha256=value.get("task_set_sha256"),
            task_ids=tuple(task_ids),
        )

    @classmethod
    def from_task_set(cls, task_set: Any) -> "TaskSetIdentity":
        """Build an identity from a compatible held-out task adapter result."""

        return cls(
            source_name=getattr(task_set, "source_name"),
            source_revision=getattr(task_set, "source_revision"),
            split=getattr(task_set, "split"),
            source_sha256=getattr(task_set, "source_sha256"),
            task_set_sha256=getattr(task_set, "task_set_sha256"),
            task_ids=tuple(getattr(task_set, "task_ids")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_revision": self.source_revision,
            "split": self.split,
            "source_sha256": self.source_sha256,
            "task_set_sha256": self.task_set_sha256,
            "task_ids": list(self.task_ids),
        }


@dataclass(frozen=True)
class ModelProfile:
    """A frozen provider/model/decoding identity for a matrix row."""

    model_id: str
    provider: str
    model: str
    model_revision: str
    decoding: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _require_string(self.model_id, "model_id"))
        object.__setattr__(self, "provider", _require_string(self.provider, "provider"))
        object.__setattr__(self, "model", _require_string(self.model, "model"))
        object.__setattr__(self, "model_revision", _require_string(self.model_revision, "model_revision"))
        object.__setattr__(self, "decoding", _freeze_json(_require_mapping(self.decoding, "decoding")))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelProfile":
        _reject_unknown_fields(
            value,
            {"model_id", "provider", "model", "model_revision", "decoding"},
            "model",
        )
        return cls(
            model_id=value.get("model_id"),
            provider=value.get("provider"),
            model=value.get("model"),
            model_revision=value.get("model_revision"),
            decoding=value.get("decoding"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "model": self.model,
            "model_revision": self.model_revision,
            "decoding": _normalize_json(self.decoding, "decoding"),
        }


@dataclass(frozen=True)
class EvaluationCondition:
    """One policy/control condition that is held fixed across model rows."""

    condition_id: str
    policy_revision: str
    configuration: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", _require_string(self.condition_id, "condition_id"))
        object.__setattr__(self, "policy_revision", _require_string(self.policy_revision, "policy_revision"))
        object.__setattr__(
            self, "configuration", _freeze_json(_require_mapping(self.configuration, "configuration"))
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationCondition":
        _reject_unknown_fields(
            value,
            {"condition_id", "policy_revision", "configuration"},
            "condition",
        )
        return cls(
            condition_id=value.get("condition_id"),
            policy_revision=value.get("policy_revision"),
            configuration=value.get("configuration"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "policy_revision": self.policy_revision,
            "configuration": _normalize_json(self.configuration, "configuration"),
        }


@dataclass(frozen=True)
class PromptArtifact:
    """Prompt text plus its declared revision and content hash."""

    prompt_revision: str
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_revision", _require_string(self.prompt_revision, "prompt_revision"))
        if not isinstance(self.text, str) or not self.text:
            raise ManifestValidationError("prompt text must be a non-empty string")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PromptArtifact":
        _reject_unknown_fields(value, {"prompt_revision", "text", "sha256"}, "prompt")
        prompt = cls(prompt_revision=value.get("prompt_revision"), text=value.get("text"))
        supplied_hash = value.get("sha256")
        if supplied_hash is not None and _require_sha256(supplied_hash, "prompt.sha256") != prompt.sha256:
            raise ManifestValidationError("prompt.sha256 does not match prompt text")
        return prompt

    def as_dict(self) -> dict[str, str]:
        return {
            "prompt_revision": self.prompt_revision,
            "text": self.text,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class RunManifest:
    """Complete, self-validating protocol record for one empirical run."""

    run_id: str
    model: ModelProfile
    condition: EvaluationCondition
    task_set: TaskSetIdentity
    prompt: PromptArtifact
    seed: Optional[int]
    runtime_revision: str
    checkpoint_cadence: int
    fault_schedule: Sequence[Any]
    metadata: Mapping[str, Any]
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_string(self.run_id, "run_id"))
        if not isinstance(self.model, ModelProfile):
            raise ManifestValidationError("model must be a ModelProfile")
        if not isinstance(self.condition, EvaluationCondition):
            raise ManifestValidationError("condition must be an EvaluationCondition")
        if not isinstance(self.task_set, TaskSetIdentity):
            raise ManifestValidationError("task_set must be a TaskSetIdentity")
        if not isinstance(self.prompt, PromptArtifact):
            raise ManifestValidationError("prompt must be a PromptArtifact")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ManifestValidationError("seed must be an integer or null")
        object.__setattr__(self, "runtime_revision", _require_string(self.runtime_revision, "runtime_revision"))
        if isinstance(self.checkpoint_cadence, bool) or not isinstance(self.checkpoint_cadence, int):
            raise ManifestValidationError("checkpoint_cadence must be a positive integer")
        if self.checkpoint_cadence < 1:
            raise ManifestValidationError("checkpoint_cadence must be a positive integer")
        normalized_schedule = _normalize_json(self.fault_schedule, "fault_schedule")
        if not isinstance(normalized_schedule, list):
            raise ManifestValidationError("fault_schedule must be a JSON list")
        object.__setattr__(self, "fault_schedule", _freeze_json(normalized_schedule))
        object.__setattr__(self, "metadata", _freeze_json(_require_mapping(self.metadata, "metadata")))
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestValidationError(
                "unsupported manifest schema version {}".format(self.schema_version)
            )

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "model": self.model.as_dict(),
            "condition": self.condition.as_dict(),
            "task_set": self.task_set.as_dict(),
            "prompt": self.prompt.as_dict(),
            "seed": self.seed,
            "runtime_revision": self.runtime_revision,
            "checkpoint_cadence": self.checkpoint_cadence,
            "fault_schedule": _normalize_json(self.fault_schedule, "fault_schedule"),
            "metadata": _normalize_json(self.metadata, "metadata"),
        }

    def run_identity_dict(self) -> dict[str, Any]:
        """Stable run inputs excluding the filename-oriented ``run_id`` itself."""

        value = self.unsigned_dict()
        del value["run_id"]
        return value

    @property
    def run_identity_sha256(self) -> str:
        """Digest used in generated run IDs to prevent stale-file collisions."""

        return _sha256_json(self.run_identity_dict())

    @property
    def manifest_sha256(self) -> str:
        return _sha256_json(self.unsigned_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunManifest":
        if not isinstance(value, Mapping):
            raise ManifestValidationError("manifest must be a JSON object")
        _reject_unknown_fields(
            value,
            {
                "schema_version",
                "run_id",
                "model",
                "condition",
                "task_set",
                "prompt",
                "seed",
                "runtime_revision",
                "checkpoint_cadence",
                "fault_schedule",
                "metadata",
                "manifest_sha256",
            },
            "manifest",
        )
        manifest = cls(
            schema_version=value.get("schema_version"),
            run_id=value.get("run_id"),
            model=ModelProfile.from_mapping(_require_mapping(value.get("model"), "model")),
            condition=EvaluationCondition.from_mapping(
                _require_mapping(value.get("condition"), "condition")
            ),
            task_set=TaskSetIdentity.from_mapping(_require_mapping(value.get("task_set"), "task_set")),
            prompt=PromptArtifact.from_mapping(_require_mapping(value.get("prompt"), "prompt")),
            seed=value.get("seed"),
            runtime_revision=value.get("runtime_revision"),
            checkpoint_cadence=value.get("checkpoint_cadence"),
            fault_schedule=value.get("fault_schedule", []),
            metadata=_require_mapping(value.get("metadata", {}), "metadata"),
        )
        supplied_hash = value.get("manifest_sha256")
        if supplied_hash is not None and _require_sha256(supplied_hash, "manifest_sha256") != manifest.manifest_sha256:
            raise ManifestValidationError("manifest_sha256 does not match manifest content")
        return manifest

    def as_dict(self) -> dict[str, Any]:
        value = self.unsigned_dict()
        value["manifest_sha256"] = self.manifest_sha256
        return value


def default_run_id(
    model_id: str,
    condition_id: str,
    seed: Optional[int],
    task_set_sha256: str,
    run_identity_sha256: str,
) -> str:
    """Create a readable, collision-resistant matrix identifier."""

    seed_part = "unseeded" if seed is None else "seed-{}".format(seed)
    return "{}__{}__{}__{}__{}".format(
        _slug(_require_string(model_id, "model_id")),
        _slug(_require_string(condition_id, "condition_id")),
        seed_part,
        _require_sha256(task_set_sha256, "task_set_sha256")[:12],
        _require_sha256(run_identity_sha256, "run_identity_sha256")[:12],
    )


def build_cross_model_matrix(
    *,
    models: Iterable[ModelProfile],
    conditions: Iterable[EvaluationCondition],
    task_set: TaskSetIdentity,
    prompt: PromptArtifact,
    seeds: Sequence[Optional[int]],
    runtime_revision: str,
    checkpoint_cadence: int = 1,
    fault_schedule: Sequence[Any] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> list[RunManifest]:
    """Build one manifest for every model × condition × seed combination."""

    frozen_models = tuple(models)
    frozen_conditions = tuple(conditions)
    frozen_seeds = tuple(seeds)
    if not frozen_models:
        raise ManifestValidationError("models must not be empty")
    if not frozen_conditions:
        raise ManifestValidationError("conditions must not be empty")
    if not frozen_seeds:
        raise ManifestValidationError("seeds must not be empty")
    if len({item.model_id for item in frozen_models}) != len(frozen_models):
        raise ManifestValidationError("model_id values must be unique")
    if len({item.condition_id for item in frozen_conditions}) != len(frozen_conditions):
        raise ManifestValidationError("condition_id values must be unique")
    if len(set(frozen_seeds)) != len(frozen_seeds):
        raise ManifestValidationError("seeds must be unique")

    common_metadata = _require_mapping(metadata or {}, "metadata")
    manifests: list[RunManifest] = []
    for model in sorted(frozen_models, key=lambda item: item.model_id):
        for condition in sorted(frozen_conditions, key=lambda item: item.condition_id):
            for seed in sorted(frozen_seeds, key=lambda item: (-1 if item is None else item)):
                template = RunManifest(
                    run_id="pending",
                    model=model,
                    condition=condition,
                    task_set=task_set,
                    prompt=prompt,
                    seed=seed,
                    runtime_revision=runtime_revision,
                    checkpoint_cadence=checkpoint_cadence,
                    fault_schedule=fault_schedule,
                    metadata=common_metadata,
                )
                manifests.append(
                    RunManifest(
                        run_id=default_run_id(
                            model.model_id,
                            condition.condition_id,
                            seed,
                            task_set.task_set_sha256,
                            template.run_identity_sha256,
                        ),
                        model=model,
                        condition=condition,
                        task_set=task_set,
                        prompt=prompt,
                        seed=seed,
                        runtime_revision=runtime_revision,
                        checkpoint_cadence=checkpoint_cadence,
                        fault_schedule=fault_schedule,
                        metadata=common_metadata,
                    )
                )
    if len({item.run_id for item in manifests}) != len(manifests):
        raise ManifestValidationError("generated run ids collide; choose distinct model/condition ids")
    return manifests


def write_manifest_jsonl(path: Path, manifests: Sequence[RunManifest]) -> None:
    """Write a matrix in canonical, newline-delimited JSON for execution."""

    if not manifests:
        raise ManifestValidationError("cannot write an empty manifest matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for manifest in manifests:
            output.write(_canonical_json(manifest.as_dict()) + "\n")


def read_manifest_jsonl(path: Path) -> list[RunManifest]:
    if not path.is_file():
        raise ManifestValidationError("manifest file does not exist: {}".format(path))
    manifests: list[RunManifest] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ManifestValidationError("blank manifest line {} in {}".format(line_number, path))
        try:
            raw = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ManifestValidationError(
                "invalid JSON on manifest line {} in {}: {}".format(line_number, path, error)
            ) from error
        manifests.append(RunManifest.from_mapping(_require_mapping(raw, "manifest")))
    if not manifests:
        raise ManifestValidationError("manifest file is empty: {}".format(path))
    run_ids = [item.run_id for item in manifests]
    if len(set(run_ids)) != len(run_ids):
        raise ManifestValidationError("manifest file contains duplicate run_id values")
    return manifests


def load_manifest(path: Path) -> RunManifest:
    """Load one self-validating JSON manifest (not a JSONL matrix)."""

    if not path.is_file():
        raise ManifestValidationError("manifest file does not exist: {}".format(path))
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ManifestValidationError("invalid JSON in manifest {}: {}".format(path, error)) from error
    return RunManifest.from_mapping(_require_mapping(raw, "manifest"))


def validate_result_task_ids(results: Iterable[Any], manifest: RunManifest) -> None:
    """Require exactly one observed outcome for every frozen manifest task."""

    observed: list[str] = []
    seen: set[str] = set()
    duplicate_ids: set[str] = set()
    for row in results:
        task_id = getattr(row, "task_id", None)
        if not isinstance(task_id, str) or not task_id:
            raise ManifestValidationError("each result row must expose a non-empty task_id")
        if task_id in seen:
            duplicate_ids.add(task_id)
        seen.add(task_id)
        observed.append(task_id)
    if duplicate_ids:
        raise ManifestValidationError(
            "result rows contain duplicate task ids: {}".format(", ".join(sorted(duplicate_ids)))
        )
    expected = set(manifest.task_set.task_ids)
    actual = set(observed)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing {}".format(", ".join(missing)))
        if unexpected:
            details.append("unexpected {}".format(", ".join(unexpected)))
        raise ManifestValidationError("result task ids do not match manifest: {}".format("; ".join(details)))


def manifest_summary(manifest: RunManifest) -> dict[str, Any]:
    """Small provenance object embedded beside a scored aggregate report."""

    return {
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.manifest_sha256,
        "model_id": manifest.model.model_id,
        "provider": manifest.model.provider,
        "model": manifest.model.model,
        "condition_id": manifest.condition.condition_id,
        "seed": manifest.seed,
        "task_set_sha256": manifest.task_set.task_set_sha256,
        "task_count": len(manifest.task_set.task_ids),
    }


def validate_comparable_matrix(manifests: Sequence[RunManifest]) -> None:
    """Reject a matrix that would silently compare different experiments.

    Conditions are deliberately allowed to differ.  The task split, prompt,
    runtime revision, checkpoint cadence, and injected fault schedule must not:
    changing any of those creates a distinct experiment rather than another row
    in a cross-model comparison.
    """

    if not manifests:
        raise ManifestValidationError("manifest matrix must not be empty")
    reference = manifests[0]
    reference_task_set = _canonical_json(reference.task_set.as_dict())
    reference_prompt = reference.prompt.sha256
    reference_runtime = reference.runtime_revision
    reference_cadence = reference.checkpoint_cadence
    reference_fault_schedule = _canonical_json(reference.fault_schedule)
    for manifest in manifests[1:]:
        if _canonical_json(manifest.task_set.as_dict()) != reference_task_set:
            raise ManifestValidationError("matrix manifests do not share the same frozen task set")
        if manifest.prompt.sha256 != reference_prompt:
            raise ManifestValidationError("matrix manifests do not share the same prompt content")
        if manifest.runtime_revision != reference_runtime:
            raise ManifestValidationError("matrix manifests do not share the same runtime revision")
        if manifest.checkpoint_cadence != reference_cadence:
            raise ManifestValidationError("matrix manifests do not share the same checkpoint cadence")
        if _canonical_json(manifest.fault_schedule) != reference_fault_schedule:
            raise ManifestValidationError("matrix manifests do not share the same fault schedule")


def _reject_json_constant(value: str) -> None:
    raise ValueError("invalid JSON constant {}".format(value))

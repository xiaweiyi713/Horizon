"""A bounded, deterministic artifact workspace for objective agent tasks.

The environment deliberately does not execute a shell command or accept an
arbitrary filesystem path from a model.  A task owns an immutable template,
an allowlist of writable relative paths, and one exact text/JSON verifier.
The policy can submit candidate file contents only through
``ArtifactWorkspaceAdapter``; :class:`~horizon_agent.AdapterRegistry` records
the invocation and terminal verification outcome around that side effect.

This is a small task-environment primitive, not a general sandbox.  It gives
HorizonBench artifact tasks an objective, reproducible success boundary before
adding a broader container/process environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

from .adapters import AdapterRequest, AdapterResult


ARTIFACT_WORKSPACE_METADATA_KEY = "horizon_artifact_workspace_v1"
ARTIFACT_WORKSPACE_SCHEMA_VERSION = 1
_WORKSPACE_MARKER = ".horizon-artifact-workspace.json"
_WORKSPACE_OPERATIONS = ".horizon-artifact-operations.json"
_OPERATION_RECEIPT_SCHEMA_VERSION = 1
_MAX_TEMPLATE_FILES = 16
_MAX_WRITABLE_FILES = 8
_MAX_FILE_BYTES = 16 * 1024
_MAX_TEMPLATE_BYTES = 128 * 1024
_MAX_OPERATION_RECEIPTS = 64
_MAX_OPERATION_ID_BYTES = 256
_ADAPTER_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OBSERVATION_REASONS = frozenset(
    {
        "artifact matches expected text",
        "artifact matches expected JSON",
        "required artifact is absent",
        "required artifact is unreadable",
        "artifact exceeds byte limit",
        "artifact content does not match expected text",
        "artifact JSON is invalid",
        "artifact JSON does not match expected value",
        "workspace candidate rejected",
    }
)
_VERIFIED_OBSERVATION_REASONS = frozenset(
    {"artifact matches expected text", "artifact matches expected JSON"}
)


class ArtifactWorkspaceError(ValueError):
    """Raised when a workspace contract or candidate is unsafe or invalid."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactWorkspaceError("{} must be an object".format(label))
    return value


def _list(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ArtifactWorkspaceError("{} must be an array".format(label))
    return list(value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactWorkspaceError("{} must be a non-empty string".format(label))
    return value


def _json_value(value: Any, label: str) -> Any:
    """Detach a finite JSON value without accepting NaN or custom objects."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactWorkspaceError("{} must not contain NaN or infinity".format(label))
        return value
    if isinstance(value, Mapping):
        detached: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactWorkspaceError("{} has a non-string object key".format(label))
            detached[key] = _json_value(item, "{}.{}".format(label, key))
        return detached
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, "{}[]".format(label)) for item in value]
    raise ArtifactWorkspaceError("{} must contain only JSON values".format(label))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _bounded_utf8(value: str, label: str, *, limit: int = _MAX_FILE_BYTES) -> str:
    if not isinstance(value, str):
        raise ArtifactWorkspaceError("{} must be a string".format(label))
    if len(value.encode("utf-8")) > limit:
        raise ArtifactWorkspaceError("{} must be at most {} UTF-8 bytes".format(label, limit))
    return value


def _operation_id(value: Any, label: str = "operation_id") -> str:
    """Validate a durable adapter identity without treating it as a path."""

    value = _string(value, label)
    if "\x00" in value or len(value.encode("utf-8")) > _MAX_OPERATION_ID_BYTES:
        raise ArtifactWorkspaceError(
            "{} must be at most {} UTF-8 bytes and cannot contain NUL".format(
                label, _MAX_OPERATION_ID_BYTES
            )
        )
    return value


def _sha256(value: Any, label: str, *, allow_none: bool = False) -> Optional[str]:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ArtifactWorkspaceError("{} must be a lowercase SHA-256 digest".format(label))
    return value


def _relative_path(value: Any, label: str) -> str:
    value = _string(value, label)
    if "\\" in value or "\x00" in value:
        raise ArtifactWorkspaceError("{} must be a normalized relative POSIX path".format(label))
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactWorkspaceError("{} must be a normalized relative POSIX path".format(label))
    if any(part.startswith(".") for part in path.parts):
        raise ArtifactWorkspaceError("{} cannot contain hidden path segments".format(label))
    if path.as_posix() != value:
        raise ArtifactWorkspaceError("{} must be a normalized relative POSIX path".format(label))
    return value


def _strict_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ArtifactWorkspaceError("{} has unrecognized fields: {}".format(label, ", ".join(unknown)))


@dataclass(frozen=True)
class WorkspaceTemplateFile:
    """One immutable task-owned file placed in a workspace before execution."""

    path: str
    content: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, "template file path"))
        object.__setattr__(self, "content", _bounded_utf8(self.content, "template file content"))

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "content": self.content}


@dataclass(frozen=True)
class ArtifactVerifier:
    """One deterministic check over an allowlisted artifact file."""

    kind: str
    path: str
    expected: Any

    def __post_init__(self) -> None:
        if self.kind not in {"text_exact", "json_exact"}:
            raise ArtifactWorkspaceError("verifier.kind must be text_exact or json_exact")
        object.__setattr__(self, "path", _relative_path(self.path, "verifier.path"))
        expected = _json_value(self.expected, "verifier.expected")
        if self.kind == "text_exact":
            expected = _bounded_utf8(expected, "verifier.expected")
        elif len(_canonical_json(expected).encode("utf-8")) > _MAX_FILE_BYTES:
            raise ArtifactWorkspaceError(
                "verifier.expected must be at most {} UTF-8 bytes when rendered as JSON".format(
                    _MAX_FILE_BYTES
                )
            )
        object.__setattr__(self, "expected", expected)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": self.path, "expected": self.expected}


@dataclass(frozen=True)
class ArtifactWorkspaceSpec:
    """Immutable task-owned contract for one artifact workspace."""

    adapter_name: str
    template_files: tuple[WorkspaceTemplateFile, ...]
    writable_paths: tuple[str, ...]
    verifier: ArtifactVerifier

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_name, str) or not _ADAPTER_NAME_RE.fullmatch(self.adapter_name):
            raise ArtifactWorkspaceError(
                "adapter_name must match lowercase [a-z][a-z0-9_-]{0,63}"
            )
        if not self.template_files or len(self.template_files) > _MAX_TEMPLATE_FILES:
            raise ArtifactWorkspaceError(
                "template_files must contain 1 to {} files".format(_MAX_TEMPLATE_FILES)
            )
        if not self.writable_paths or len(self.writable_paths) > _MAX_WRITABLE_FILES:
            raise ArtifactWorkspaceError(
                "writable_paths must contain 1 to {} paths".format(_MAX_WRITABLE_FILES)
            )
        templates = tuple(self.template_files)
        if not all(isinstance(item, WorkspaceTemplateFile) for item in templates):
            raise ArtifactWorkspaceError("template_files must contain WorkspaceTemplateFile objects")
        writable = tuple(_relative_path(path, "writable path") for path in self.writable_paths)
        template_paths = tuple(item.path for item in templates)
        if len(set(template_paths)) != len(template_paths):
            raise ArtifactWorkspaceError("template_files contains duplicate paths")
        if len(set(writable)) != len(writable):
            raise ArtifactWorkspaceError("writable_paths contains duplicate paths")
        if set(template_paths) & set(writable):
            raise ArtifactWorkspaceError("template_files and writable_paths must not overlap")
        if sum(len(item.content.encode("utf-8")) for item in templates) > _MAX_TEMPLATE_BYTES:
            raise ArtifactWorkspaceError(
                "template_files must total at most {} UTF-8 bytes".format(_MAX_TEMPLATE_BYTES)
            )
        if not isinstance(self.verifier, ArtifactVerifier):
            raise ArtifactWorkspaceError("verifier must be an ArtifactVerifier")
        if self.verifier.path not in writable:
            raise ArtifactWorkspaceError("verifier.path must be included in writable_paths")
        object.__setattr__(self, "template_files", templates)
        object.__setattr__(self, "writable_paths", writable)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "ArtifactWorkspaceSpec":
        raw = _mapping(metadata, "task metadata").get(ARTIFACT_WORKSPACE_METADATA_KEY)
        value = _mapping(raw, ARTIFACT_WORKSPACE_METADATA_KEY)
        _strict_fields(
            value,
            {"schema_version", "adapter_name", "template_files", "writable_paths", "verifier"},
            ARTIFACT_WORKSPACE_METADATA_KEY,
        )
        if value.get("schema_version") != ARTIFACT_WORKSPACE_SCHEMA_VERSION:
            raise ArtifactWorkspaceError(
                "{}.schema_version must be {}".format(
                    ARTIFACT_WORKSPACE_METADATA_KEY, ARTIFACT_WORKSPACE_SCHEMA_VERSION
                )
            )
        raw_templates = _list(value.get("template_files"), "{}.template_files".format(ARTIFACT_WORKSPACE_METADATA_KEY))
        templates: list[WorkspaceTemplateFile] = []
        for index, raw_template in enumerate(raw_templates, 1):
            template = _mapping(raw_template, "template_files[{}]".format(index))
            _strict_fields(template, {"path", "content"}, "template_files[{}]".format(index))
            templates.append(
                WorkspaceTemplateFile(
                    path=template.get("path"), content=template.get("content")
                )
            )
        raw_writable = _list(value.get("writable_paths"), "{}.writable_paths".format(ARTIFACT_WORKSPACE_METADATA_KEY))
        verifier_raw = _mapping(value.get("verifier"), "{}.verifier".format(ARTIFACT_WORKSPACE_METADATA_KEY))
        _strict_fields(verifier_raw, {"kind", "path", "expected"}, "{}.verifier".format(ARTIFACT_WORKSPACE_METADATA_KEY))
        return cls(
            adapter_name=value.get("adapter_name"),
            template_files=tuple(templates),
            writable_paths=tuple(raw_writable),
            verifier=ArtifactVerifier(
                kind=verifier_raw.get("kind"),
                path=verifier_raw.get("path"),
                expected=verifier_raw.get("expected"),
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_WORKSPACE_SCHEMA_VERSION,
            "adapter_name": self.adapter_name,
            "template_files": [item.as_dict() for item in self.template_files],
            "writable_paths": list(self.writable_paths),
            "verifier": self.verifier.as_dict(),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.as_dict()).encode("utf-8")).hexdigest()

    def preflight_summary(self) -> dict[str, Any]:
        """Describe the environment without copying template/expected content."""

        return {
            "environment": "artifact_workspace_v1",
            "spec_sha256": self.sha256,
            "adapter_name": self.adapter_name,
            "template_file_count": len(self.template_files),
            "writable_paths": list(self.writable_paths),
            "verifier_kind": self.verifier.kind,
            "verifier_path": self.verifier.path,
        }


@dataclass(frozen=True)
class ArtifactObservation:
    """Safe, objective outcome of one workspace verification."""

    verified: bool
    reason: str
    artifact_sha256: Optional[str]
    written_paths: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "reason": self.reason,
            "artifact_sha256": self.artifact_sha256,
            "written_paths": list(self.written_paths),
        }


class ArtifactWorkspace:
    """Materialize and verify one immutable workspace under a controlled root."""

    def __init__(self, root: Path, spec: ArtifactWorkspaceSpec) -> None:
        if not isinstance(root, Path):
            raise ArtifactWorkspaceError("workspace root must be a pathlib.Path")
        if not isinstance(spec, ArtifactWorkspaceSpec):
            raise ArtifactWorkspaceError("workspace spec must be an ArtifactWorkspaceSpec")
        self.root = root
        self.spec = spec
        self._prepare()

    @classmethod
    def for_attempt(
        cls,
        artifact_root: Path,
        *,
        run_id: str,
        task_id: str,
        spec: ArtifactWorkspaceSpec,
    ) -> "ArtifactWorkspace":
        """Create a stable, non-user-controlled workspace name for one attempt."""

        if not isinstance(artifact_root, Path):
            raise ArtifactWorkspaceError("artifact_root must be a pathlib.Path")
        run_id = _string(run_id, "run_id")
        task_id = _string(task_id, "task_id")
        digest = hashlib.sha256((run_id + "\0" + task_id).encode("utf-8")).hexdigest()
        return cls(artifact_root / ("workspace-" + digest), spec)

    def _prepare(self) -> None:
        if self.root.exists():
            if not self.root.is_dir() or self.root.is_symlink():
                raise ArtifactWorkspaceError("workspace root is not a regular directory")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
        self._assert_no_symlinks()
        marker = self.root / _WORKSPACE_MARKER
        if marker.exists():
            self._validate_marker(marker)
            self._validate_existing_layout()
            return
        if any(self.root.iterdir()):
            raise ArtifactWorkspaceError("new workspace root must be empty")
        self._write_marker(marker)
        for template in self.spec.template_files:
            self._write_text(template.path, template.content)

    def _write_marker(self, marker: Path) -> None:
        self._atomic_write(
            marker,
            _canonical_json(
                {
                    "schema_version": ARTIFACT_WORKSPACE_SCHEMA_VERSION,
                    "spec_sha256": self.spec.sha256,
                }
            )
            + "\n",
        )

    def _validate_marker(self, marker: Path) -> None:
        if not marker.is_file() or marker.is_symlink():
            raise ArtifactWorkspaceError("workspace marker is not a regular file")
        try:
            value = json.loads(marker.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ArtifactWorkspaceError("workspace marker is invalid") from error
        if not isinstance(value, Mapping):
            raise ArtifactWorkspaceError("workspace marker is invalid")
        if value.get("schema_version") != ARTIFACT_WORKSPACE_SCHEMA_VERSION:
            raise ArtifactWorkspaceError("workspace marker has an unsupported schema version")
        if value.get("spec_sha256") != self.spec.sha256:
            raise ArtifactWorkspaceError("workspace belongs to a different immutable task environment")

    def _validate_existing_layout(self) -> None:
        template_by_path = {item.path: item.content for item in self.spec.template_files}
        allowed_files = set(template_by_path) | set(self.spec.writable_paths) | {
            _WORKSPACE_MARKER,
            _WORKSPACE_OPERATIONS,
        }
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ArtifactWorkspaceError("workspace cannot contain symlinks")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ArtifactWorkspaceError("workspace contains an unsupported filesystem entry")
            relative = path.relative_to(self.root).as_posix()
            if relative not in allowed_files:
                raise ArtifactWorkspaceError("workspace contains an unexpected file")
        for relative, expected in template_by_path.items():
            path = self._path(relative)
            if not path.is_file() or path.is_symlink():
                raise ArtifactWorkspaceError("workspace template file is missing")
            try:
                actual = path.read_text(encoding="utf-8")
            except OSError as error:
                raise ArtifactWorkspaceError("workspace template file is unreadable") from error
            if actual != expected:
                raise ArtifactWorkspaceError("workspace template file differs from immutable task input")
        self._load_operation_receipts()

    @property
    def _operations_path(self) -> Path:
        return self.root / _WORKSPACE_OPERATIONS

    def _load_operation_receipts(self) -> dict[str, tuple[Optional[str], ArtifactObservation]]:
        """Load safe operation receipts persisted by this task environment.

        The receipt contains a digest of the submitted input and the bounded
        verification observation, never the candidate contents.  This lets a
        new Python process replay an adapter call with the same stable
        operation ID without writing the workspace again.
        """

        path = self._operations_path
        if not path.exists():
            return {}
        if not path.is_file() or path.is_symlink():
            raise ArtifactWorkspaceError("workspace operation receipts are not a regular file")
        try:
            value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ArtifactWorkspaceError("workspace operation receipts are invalid") from error
        raw = _mapping(value, "workspace operation receipts")
        _strict_fields(
            raw,
            {"schema_version", "spec_sha256", "operations"},
            "workspace operation receipts",
        )
        if raw.get("schema_version") != _OPERATION_RECEIPT_SCHEMA_VERSION:
            raise ArtifactWorkspaceError("workspace operation receipts have an unsupported schema version")
        if raw.get("spec_sha256") != self.spec.sha256:
            raise ArtifactWorkspaceError("workspace operation receipts belong to a different task environment")
        operations = _mapping(raw.get("operations"), "workspace operation receipts.operations")
        if len(operations) > _MAX_OPERATION_RECEIPTS:
            raise ArtifactWorkspaceError("workspace operation receipts exceed the configured limit")
        receipts: dict[str, tuple[Optional[str], ArtifactObservation]] = {}
        for operation_id, raw_receipt in operations.items():
            if not isinstance(operation_id, str) or _operation_id(operation_id) != operation_id:
                raise ArtifactWorkspaceError("workspace operation receipt has an invalid operation ID")
            receipt = _mapping(raw_receipt, "workspace operation receipt")
            _strict_fields(
                receipt,
                {"input_sha256", "observation"},
                "workspace operation receipt",
            )
            input_sha256 = _sha256(
                receipt.get("input_sha256"),
                "workspace operation receipt.input_sha256",
                allow_none=True,
            )
            observation = self._observation_from_receipt(receipt.get("observation"))
            receipts[operation_id] = (input_sha256, observation)
        return receipts

    def _observation_from_receipt(self, value: Any) -> ArtifactObservation:
        raw = _mapping(value, "workspace operation receipt.observation")
        _strict_fields(
            raw,
            {"verified", "reason", "artifact_sha256", "written_paths"},
            "workspace operation receipt.observation",
        )
        verified = raw.get("verified")
        reason = raw.get("reason")
        if (
            not isinstance(verified, bool)
            or not isinstance(reason, str)
            or reason not in _OBSERVATION_REASONS
        ):
            raise ArtifactWorkspaceError("workspace operation receipt has an invalid observation")
        if verified != (reason in _VERIFIED_OBSERVATION_REASONS):
            raise ArtifactWorkspaceError("workspace operation receipt has an inconsistent observation")
        digest = _sha256(
            raw.get("artifact_sha256"),
            "workspace operation receipt.observation.artifact_sha256",
            allow_none=True,
        )
        raw_paths = _list(
            raw.get("written_paths"), "workspace operation receipt.observation.written_paths"
        )
        if len(raw_paths) > _MAX_WRITABLE_FILES:
            raise ArtifactWorkspaceError("workspace operation receipt has too many written paths")
        written_paths = tuple(
            _relative_path(path, "workspace operation receipt written path") for path in raw_paths
        )
        if len(set(written_paths)) != len(written_paths) or any(
            path not in self.spec.writable_paths for path in written_paths
        ):
            raise ArtifactWorkspaceError("workspace operation receipt has invalid written paths")
        return ArtifactObservation(verified, reason, digest, written_paths)

    def operation_observation(self, operation_id: str) -> Optional[ArtifactObservation]:
        """Return a previously completed operation outcome, if any."""

        receipt = self._load_operation_receipts().get(_operation_id(operation_id))
        return None if receipt is None else receipt[1]

    def record_operation(
        self,
        operation_id: str,
        input_sha256: Optional[str],
        observation: ArtifactObservation,
    ) -> ArtifactObservation:
        """Persist one terminal adapter operation without retaining its input."""

        operation_id = _operation_id(operation_id)
        input_sha256 = _sha256(input_sha256, "operation input SHA-256", allow_none=True)
        if not isinstance(observation, ArtifactObservation):
            raise ArtifactWorkspaceError("operation observation must be an ArtifactObservation")
        receipts = self._load_operation_receipts()
        existing = receipts.get(operation_id)
        if existing is not None:
            return existing[1]
        if len(receipts) >= _MAX_OPERATION_RECEIPTS:
            raise ArtifactWorkspaceError("workspace cannot record more adapter operations")
        receipts[operation_id] = (input_sha256, observation)
        self._atomic_write(
            self._operations_path,
            _canonical_json(
                {
                    "schema_version": _OPERATION_RECEIPT_SCHEMA_VERSION,
                    "spec_sha256": self.spec.sha256,
                    "operations": {
                        key: {
                            "input_sha256": digest,
                            "observation": item.as_dict(),
                        }
                        for key, (digest, item) in receipts.items()
                    },
                }
            )
            + "\n",
        )
        return observation

    def _assert_no_symlinks(self) -> None:
        if self.root.is_symlink():
            raise ArtifactWorkspaceError("workspace root cannot be a symlink")
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ArtifactWorkspaceError("workspace cannot contain symlinks")

    def _path(self, relative: str) -> Path:
        path = self.root.joinpath(*PurePosixPath(relative).parts)
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError as error:
            raise ArtifactWorkspaceError("workspace path escapes its root") from error
        return path

    def _write_text(self, relative: str, content: str) -> None:
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._assert_no_symlinks()
        if path.exists() and (not path.is_file() or path.is_symlink()):
            raise ArtifactWorkspaceError("workspace target is not a regular file")
        self._atomic_write(path, content)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=".horizon-workspace-",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def apply_candidate(self, value: Mapping[str, Any]) -> ArtifactObservation:
        """Write one constrained model candidate, then verify the target artifact."""

        raw = _mapping(value, "adapter input")
        _strict_fields(raw, {"files"}, "adapter input")
        files = _list(raw.get("files"), "adapter input.files")
        if not files or len(files) > _MAX_WRITABLE_FILES:
            raise ArtifactWorkspaceError(
                "adapter input.files must contain 1 to {} files".format(_MAX_WRITABLE_FILES)
            )
        candidates: list[tuple[str, str]] = []
        for index, raw_file in enumerate(files, 1):
            item = _mapping(raw_file, "adapter input.files[{}]".format(index))
            _strict_fields(item, {"path", "content"}, "adapter input.files[{}]".format(index))
            path = _relative_path(item.get("path"), "adapter input.files[{}].path".format(index))
            if path not in self.spec.writable_paths:
                raise ArtifactWorkspaceError("adapter input writes a path outside writable_paths")
            content = _bounded_utf8(item.get("content"), "adapter input.files[{}].content".format(index))
            candidates.append((path, content))
        paths = [path for path, _content in candidates]
        if len(set(paths)) != len(paths):
            raise ArtifactWorkspaceError("adapter input.files contains duplicate paths")
        if self.spec.verifier.path not in paths:
            raise ArtifactWorkspaceError("adapter input must write the verifier path")
        for path, content in candidates:
            self._write_text(path, content)
        return self.verify(written_paths=tuple(paths))

    def verify(self, *, written_paths: tuple[str, ...] = ()) -> ArtifactObservation:
        """Evaluate only the task-owned verifier against the materialized file."""

        path = self._path(self.spec.verifier.path)
        if not path.is_file() or path.is_symlink():
            return ArtifactObservation(False, "required artifact is absent", None, written_paths)
        try:
            size = path.stat().st_size
        except OSError:
            return ArtifactObservation(False, "required artifact is unreadable", None, written_paths)
        if size > _MAX_FILE_BYTES:
            return ArtifactObservation(False, "artifact exceeds byte limit", None, written_paths)
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ArtifactObservation(False, "required artifact is unreadable", None, written_paths)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if self.spec.verifier.kind == "text_exact":
            if content == self.spec.verifier.expected:
                return ArtifactObservation(True, "artifact matches expected text", digest, written_paths)
            return ArtifactObservation(False, "artifact content does not match expected text", digest, written_paths)
        try:
            observed = json.loads(content, parse_constant=_reject_json_constant)
            canonical_observed = _canonical_json(_json_value(observed, "artifact JSON"))
        except (ArtifactWorkspaceError, json.JSONDecodeError, ValueError):
            return ArtifactObservation(False, "artifact JSON is invalid", digest, written_paths)
        if canonical_observed == _canonical_json(self.spec.verifier.expected):
            return ArtifactObservation(True, "artifact matches expected JSON", digest, written_paths)
        return ArtifactObservation(False, "artifact JSON does not match expected value", digest, written_paths)


class ArtifactWorkspaceAdapter:
    """A restart-safe, idempotent adapter bridge from a policy to one workspace."""

    def __init__(self, workspace: ArtifactWorkspace) -> None:
        if not isinstance(workspace, ArtifactWorkspace):
            raise ArtifactWorkspaceError("workspace must be an ArtifactWorkspace")
        self.workspace = workspace
        self.name = workspace.spec.adapter_name
        self._results_by_operation: dict[str, AdapterResult] = {}
        self._observations: list[ArtifactObservation] = []

    @property
    def invocation_count(self) -> int:
        return len(self._results_by_operation)

    @property
    def observations(self) -> tuple[ArtifactObservation, ...]:
        return tuple(self._observations)

    def execute(self, request: AdapterRequest) -> AdapterResult:
        operation_id = _operation_id(request.operation_id)
        cached = self._results_by_operation.get(operation_id)
        if cached is not None:
            return cached
        completed = self.workspace.operation_observation(operation_id)
        if completed is not None:
            self._observations.append(completed)
            result = self._result_for_observation(completed)
            self._results_by_operation[operation_id] = result
            return result
        try:
            adapter_input = _mapping(request.input, "adapter input")
            input_sha256 = hashlib.sha256(
                _canonical_json(_json_value(adapter_input, "adapter input")).encode("utf-8")
            ).hexdigest()
            observation = self.workspace.apply_candidate(adapter_input)
        except ArtifactWorkspaceError:
            input_sha256 = None
            observation = ArtifactObservation(False, "workspace candidate rejected", None)
        observation = self.workspace.record_operation(operation_id, input_sha256, observation)
        self._observations.append(observation)
        result = self._result_for_observation(observation)
        self._results_by_operation[operation_id] = result
        return result

    def _result_for_observation(self, observation: ArtifactObservation) -> AdapterResult:
        metadata = {
            "environment": "artifact_workspace_v1",
            "spec_sha256": self.workspace.spec.sha256,
            "verified": observation.verified,
        }
        if observation.verified:
            return AdapterResult(status="succeeded", output=observation.as_dict(), metadata=metadata)
        return AdapterResult(
            status="failed",
            output=observation.as_dict(),
            error="artifact workspace verification failed",
            metadata=metadata,
        )

    def final_observation(self) -> ArtifactObservation:
        """Read the final artifact without disclosing its contents."""

        return self.workspace.verify()


def _reject_json_constant(value: str) -> None:
    raise ValueError("invalid JSON constant {}".format(value))


__all__ = [
    "ARTIFACT_WORKSPACE_METADATA_KEY",
    "ARTIFACT_WORKSPACE_SCHEMA_VERSION",
    "ArtifactObservation",
    "ArtifactVerifier",
    "ArtifactWorkspace",
    "ArtifactWorkspaceAdapter",
    "ArtifactWorkspaceError",
    "ArtifactWorkspaceSpec",
    "WorkspaceTemplateFile",
]

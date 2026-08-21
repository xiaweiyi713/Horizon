"""Tests for the bounded artifact task environment."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from horizon_agent.adapters import AdapterRequest  # noqa: E402
from horizon_agent.artifact_workspace import (  # noqa: E402
    ARTIFACT_WORKSPACE_METADATA_KEY,
    ArtifactWorkspace,
    ArtifactWorkspaceAdapter,
    ArtifactWorkspaceError,
    ArtifactWorkspaceSpec,
)


def _metadata(*, expected: object = None) -> dict[str, object]:
    return {
        ARTIFACT_WORKSPACE_METADATA_KEY: {
            "schema_version": 1,
            "adapter_name": "artifact_workspace",
            "template_files": [
                {"path": "input/spec.json", "content": '{"mode":"strict"}\n'},
                {"path": "README.txt", "content": "Write only answer.json.\n"},
            ],
            "writable_paths": ["answer.json"],
            "verifier": {
                "kind": "json_exact",
                "path": "answer.json",
                "expected": {"answer": "durable", "version": 1} if expected is None else expected,
            },
        }
    }


class ArtifactWorkspaceTests(unittest.TestCase):
    def test_parses_a_strict_task_owned_contract_without_exposing_expected_content(self) -> None:
        spec = ArtifactWorkspaceSpec.from_metadata(_metadata())

        self.assertEqual(spec.adapter_name, "artifact_workspace")
        self.assertEqual(spec.verifier.path, "answer.json")
        self.assertEqual(spec.preflight_summary()["template_file_count"], 2)
        self.assertNotIn("durable", json.dumps(spec.preflight_summary(), ensure_ascii=False))
        with self.assertRaisesRegex(ArtifactWorkspaceError, "unrecognized fields"):
            ArtifactWorkspaceSpec.from_metadata(
                {
                    ARTIFACT_WORKSPACE_METADATA_KEY: {
                        **_metadata()[ARTIFACT_WORKSPACE_METADATA_KEY],  # type: ignore[arg-type]
                        "unexpected": True,
                    }
                }
            )

    def test_adapter_writes_only_allowlisted_files_and_verifies_objectively(self) -> None:
        spec = ArtifactWorkspaceSpec.from_metadata(_metadata())
        with tempfile.TemporaryDirectory() as temporary:
            workspace = ArtifactWorkspace.for_attempt(
                Path(temporary), run_id="run-1", task_id="task-1", spec=spec
            )
            adapter = ArtifactWorkspaceAdapter(workspace)
            result = adapter.execute(
                AdapterRequest(
                    "run-1",
                    "artifact:task-1",
                    {"files": [{"path": "answer.json", "content": '{"version":1,"answer":"durable"}'}]},
                )
            )

            self.assertEqual(result.status, "succeeded")
            self.assertTrue(result.output["verified"])
            self.assertTrue(adapter.final_observation().verified)
            self.assertEqual(
                (workspace.root / "answer.json").read_text(encoding="utf-8"),
                '{"version":1,"answer":"durable"}',
            )
            # The same durable operation ID is idempotent and cannot overwrite
            # a previously verified artifact with a later payload.
            repeated = adapter.execute(
                AdapterRequest(
                    "run-1",
                    "artifact:task-1",
                    {"files": [{"path": "answer.json", "content": "{}"}]},
                )
            )
            self.assertIs(repeated, result)
            self.assertEqual(adapter.invocation_count, 1)
            self.assertTrue(adapter.final_observation().verified)

    def test_adapter_reuses_a_terminal_operation_after_python_restart(self) -> None:
        spec = ArtifactWorkspaceSpec.from_metadata(_metadata())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = ArtifactWorkspace.for_attempt(root, run_id="run-1", task_id="task-1", spec=spec)
            first = ArtifactWorkspaceAdapter(workspace).execute(
                AdapterRequest(
                    "run-1",
                    "artifact:restart-safe",
                    {"files": [{"path": "answer.json", "content": '{"answer":"durable","version":1}'}]},
                )
            )
            self.assertEqual(first.status, "succeeded")

            # A fresh adapter represents a restarted Python process. Reusing
            # the stable operation ID must return the persisted outcome rather
            # than replacing the previously verified artifact with new input.
            restarted_workspace = ArtifactWorkspace.for_attempt(
                root, run_id="run-1", task_id="task-1", spec=spec
            )
            restarted_adapter = ArtifactWorkspaceAdapter(restarted_workspace)
            replayed = restarted_adapter.execute(
                AdapterRequest(
                    "run-1",
                    "artifact:restart-safe",
                    {"files": [{"path": "answer.json", "content": "{}"}]},
                )
            )

            self.assertEqual(replayed.status, "succeeded")
            self.assertEqual(replayed.output, first.output)
            self.assertEqual(restarted_adapter.invocation_count, 1)
            self.assertEqual(
                (restarted_workspace.root / "answer.json").read_text(encoding="utf-8"),
                '{"answer":"durable","version":1}',
            )
            receipts = (restarted_workspace.root / ".horizon-artifact-operations.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn('"answer":"durable"', receipts)

    def test_invalid_candidates_are_safe_failures_and_cannot_escape_workspace(self) -> None:
        spec = ArtifactWorkspaceSpec.from_metadata(_metadata())
        with tempfile.TemporaryDirectory() as temporary:
            workspace = ArtifactWorkspace.for_attempt(
                Path(temporary), run_id="run-2", task_id="task-2", spec=spec
            )
            adapter = ArtifactWorkspaceAdapter(workspace)
            result = adapter.execute(
                AdapterRequest(
                    "run-2",
                    "artifact:task-2",
                    {"files": [{"path": "../outside.txt", "content": "never write this"}]},
                )
            )
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.error, "artifact workspace verification failed")
            self.assertEqual(result.output["reason"], "workspace candidate rejected")
            self.assertFalse((Path(temporary) / "outside.txt").exists())
            observation = workspace.apply_candidate(
                {"files": [{"path": "answer.json", "content": "{}"}]}
            )
            self.assertFalse(observation.verified)
            self.assertEqual(observation.reason, "artifact JSON does not match expected value")

    def test_existing_workspace_rejects_different_immutable_task_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = ArtifactWorkspaceSpec.from_metadata(_metadata())
            ArtifactWorkspace.for_attempt(root, run_id="same-run", task_id="same-task", spec=first)
            second = ArtifactWorkspaceSpec.from_metadata(_metadata(expected={"answer": "different"}))

            with self.assertRaisesRegex(ArtifactWorkspaceError, "different immutable task environment"):
                ArtifactWorkspace.for_attempt(root, run_id="same-run", task_id="same-task", spec=second)


if __name__ == "__main__":
    unittest.main()

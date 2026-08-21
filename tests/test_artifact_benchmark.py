"""Tests for durable objective artifact-workspace execution."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from horizon_agent.agent import AgentConfig, RunResult  # noqa: E402
from horizon_agent.artifact_benchmark import (  # noqa: E402
    ArtifactWorkspaceJudge,
    DurableArtifactWorkspaceExecutor,
    workspace_from_root,
)
from horizon_agent.artifact_workspace import (  # noqa: E402
    ARTIFACT_WORKSPACE_METADATA_KEY,
    ArtifactWorkspace,
    ArtifactWorkspaceAdapter,
    ArtifactWorkspaceSpec,
)
from horizon_agent.benchmark import BenchmarkExecutionError  # noqa: E402


def _metadata() -> dict[str, object]:
    return {
        "initial_plan": [
            "Use artifact_workspace to write answer.json.",
            "Finish only after the adapter reports success.",
        ],
        ARTIFACT_WORKSPACE_METADATA_KEY: {
            "schema_version": 1,
            "adapter_name": "artifact_workspace",
            "template_files": [{"path": "input.txt", "content": "durable\n"}],
            "writable_paths": ["answer.json"],
            "verifier": {
                "kind": "json_exact",
                "path": "answer.json",
                "expected": {"answer": "durable"},
            },
        },
    }


def _context(*, fault_schedule: list[object] | None = None, prompt: str | None = None) -> Any:
    return SimpleNamespace(
        task=SimpleNamespace(
            task_id="artifact-01",
            category="artifact_workspace",
            goal="Write the verified answer artifact.",
            constraint="Only modify the allowed answer.json path.",
            metadata=_metadata(),
        ),
        manifest=SimpleNamespace(
            run_id="fixture-run",
            condition=SimpleNamespace(
                configuration={"agent": {"max_steps": 4, "semantic_memory_limit": 0}},
                policy_revision="fixture-policy-v1",
            ),
            checkpoint_cadence=1,
            fault_schedule=[] if fault_schedule is None else fault_schedule,
            prompt=SimpleNamespace(
                text=prompt
                or "Use invoke_adapter with adapter artifact_workspace, then finish after success."
            ),
        ),
    )


class _Client:
    def __init__(self, context: Any) -> None:
        self.context = context
        self.events_by_run: dict[str, list[dict[str, object]]] = {}

    def record_tool_invocation(self, run_id, operation_id, tool, input):
        self.events_by_run.setdefault(run_id, []).append(
            {
                "type": "tool_invoked",
                "data": {"operation_id": operation_id, "tool": tool, "input": input},
            }
        )
        return {}

    def record_tool_result(
        self,
        run_id,
        operation_id,
        tool,
        status,
        *,
        output=None,
        error=None,
        duration_ms=None,
        metadata=None,
    ):
        del error, duration_ms, metadata
        self.events_by_run.setdefault(run_id, []).append(
            {
                "type": "tool_result_recorded",
                "data": {
                    "result": {
                        "operation_id": operation_id,
                        "tool": tool,
                        "status": status,
                        "output": output,
                    }
                },
            }
        )
        return {}

    def get_run(self, run_id: str) -> dict[str, object]:
        return {
            "state": "completed",
            "cognitive": {
                "primary_goal": self.context.task.goal,
                "constraints": [{"content": self.context.task.constraint}],
            },
        }

    def events(self, run_id: str) -> list[dict[str, object]]:
        return self.events_by_run[run_id]


class ArtifactBenchmarkTests(unittest.TestCase):
    def test_executor_binds_workspace_adapter_and_uses_objective_verifier(self) -> None:
        context = _context()
        client = _Client(context)
        calls: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temporary:
            executor = DurableArtifactWorkspaceExecutor(
                client_factory=lambda _context: client,
                provider_factory=lambda _context: object(),  # type: ignore[return-value]
                workspace_factory=workspace_from_root(Path(temporary) / "workspaces"),
                agent_factory=lambda received_client, _provider, config, prompt, registry: _FixtureAgent(
                    received_client, config, prompt, registry, calls
                ),
            )
            outcome = executor(context)

            workspace_dirs = list((Path(temporary) / "workspaces").iterdir())
            self.assertEqual(len(workspace_dirs), 1)
            self.assertTrue((workspace_dirs[0] / "answer.json").is_file())

        self.assertTrue(outcome["success"])
        self.assertTrue(outcome["state_consistent"])
        self.assertEqual(outcome["constraint_violations"], 0)
        self.assertEqual(calls["config"].checkpoint_every_steps, 1)
        self.assertIn("invoke_adapter", calls["prompt"])

    def test_preflight_is_static_and_rejects_missing_prompt_contract_or_faults(self) -> None:
        calls: list[str] = []
        executor = DurableArtifactWorkspaceExecutor(
            client_factory=lambda _context: calls.append("client") or object(),
            provider_factory=lambda _context: calls.append("provider") or object(),  # type: ignore[return-value]
            workspace_factory=lambda _context, _spec: calls.append("workspace") or None,  # type: ignore[return-value]
        )
        report = executor.preflight(_context())
        self.assertEqual(calls, [])
        self.assertEqual(report["executor"], "durable_artifact_workspace_executor")
        self.assertEqual(report["artifact_environment"]["adapter_name"], "artifact_workspace")
        with self.assertRaisesRegex(BenchmarkExecutionError, "must name invoke_adapter"):
            executor.preflight(_context(prompt="do work"))
        with self.assertRaisesRegex(BenchmarkExecutionError, "fault_schedule"):
            executor.preflight(_context(fault_schedule=[{"kind": "restart"}]))

    def test_judge_rejects_a_forged_tool_event_without_a_real_workspace_invocation(self) -> None:
        context = _context()
        spec = ArtifactWorkspaceSpec.from_metadata(context.task.metadata)
        with tempfile.TemporaryDirectory() as temporary:
            workspace = ArtifactWorkspace.for_attempt(
                Path(temporary), run_id="fixture-run", task_id=context.task.task_id, spec=spec
            )
            (workspace.root / "answer.json").write_text('{"answer":"durable"}', encoding="utf-8")
            adapter = ArtifactWorkspaceAdapter(workspace)
            outcome = ArtifactWorkspaceJudge()(
                context,
                RunResult("fixture-run", "completed", 1, 3, 0),
                {
                    "state": "completed",
                    "cognitive": {
                        "primary_goal": context.task.goal,
                        "constraints": [{"content": context.task.constraint}],
                    },
                },
                [
                    {
                        "type": "tool_result_recorded",
                        "data": {
                            "result": {
                                "tool": "artifact_workspace",
                                "status": "succeeded",
                                "output": {"verified": True},
                            }
                        },
                    }
                ],
                adapter,
            )

        self.assertFalse(outcome["success"])
        self.assertFalse(outcome["state_consistent"])


class _FixtureAgent:
    def __init__(self, client, config, prompt, registry, calls) -> None:
        self.client = client
        self.config = config
        self.prompt = prompt
        self.registry = registry
        self.calls = calls

    def run(self, _goal, *, constraints, plan) -> RunResult:
        self.calls["config"] = self.config
        self.calls["prompt"] = self.prompt
        self.calls["run"] = (constraints, plan)
        result = self.registry.execute(
            self.client,
            "fixture-run",
            "artifact_workspace",
            "artifact:fixture-run",
            {"files": [{"path": "answer.json", "content": '{"answer":"durable"}'}]},
            task_id="artifact-01",
        )
        if result.status != "succeeded":
            raise AssertionError("fixture adapter must verify")
        return RunResult("fixture-run", "completed", 2, 12, 1)


if __name__ == "__main__":
    unittest.main()

"""No-network end-to-end coverage for objective artifact-workspace tasks."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.adapters import JsonlTaskAdapter  # noqa: E402
from benchmarks.horizonbench.execute_matrix import execute_matrix  # noqa: E402
from benchmarks.horizonbench.execution import load_executor  # noqa: E402
from benchmarks.horizonbench.preflight_matrix import preflight_matrix  # noqa: E402
from benchmarks.horizonbench.protocol import (  # noqa: E402
    EvaluationCondition,
    ModelProfile,
    PromptArtifact,
    TaskSetIdentity,
    build_cross_model_matrix,
    write_manifest_jsonl,
)
from benchmarks.horizonbench.score import read_jsonl, score_results  # noqa: E402
from horizon_agent import HorizonApiError, HorizonClient  # noqa: E402


FIXTURES = ROOT / "benchmarks" / "horizonbench" / "fixtures"
TASKS = FIXTURES / "artifact_workspace_tasks.jsonl"
MODELS = FIXTURES / "artifact_workspace_models.json"
CONDITIONS = FIXTURES / "artifact_workspace_conditions.json"
PROMPT = FIXTURES / "artifact_workspace_prompt.txt"
EXECUTOR_SPEC = "benchmarks.horizonbench.scripted_artifact_workspace_executor:execute"


def _free_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return "127.0.0.1:{}".format(listener.getsockname()[1])


def _horizon_binary() -> Path:
    candidate = ROOT / "target" / "debug" / "horizon"
    cargo = Path.home() / ".cargo" / "bin" / "cargo"
    subprocess.run(
        [str(cargo if cargo.exists() else "cargo"), "build", "-q", "-p", "horizon-cli"],
        cwd=ROOT,
        check=True,
    )
    if not candidate.exists():
        raise RuntimeError("cargo build did not create {}".format(candidate))
    return candidate


def _start_server(executable: Path, database: Path, address: str) -> subprocess.Popen:
    process = subprocess.Popen(
        [
            str(executable),
            "--db",
            str(database),
            "--checkpoint-every",
            "0",
            "serve",
            "--address",
            address,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = HorizonClient("http://" + address, timeout=0.25)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Horizon server exited before becoming healthy: {}".format(process.returncode))
        try:
            if client.health():
                return process
        except HorizonApiError:
            time.sleep(0.05)
    _stop_server(process)
    raise TimeoutError("Horizon server did not become healthy")


def _stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class ArtifactWorkspaceMatrixIntegrationTests(unittest.TestCase):
    def test_scripted_fixture_uses_durable_adapter_and_objective_workspace_verifier(self) -> None:
        model = ModelProfile.from_mapping(json.loads(MODELS.read_text(encoding="utf-8"))["models"][0])
        condition = EvaluationCondition.from_mapping(
            json.loads(CONDITIONS.read_text(encoding="utf-8"))["conditions"][0]
        )
        task_set = JsonlTaskAdapter("horizon-artifact-workspace-fixture", "fixture-v1").load(TASKS)
        manifests = build_cross_model_matrix(
            models=[model],
            conditions=[condition],
            task_set=TaskSetIdentity.from_task_set(task_set),
            prompt=PromptArtifact("artifact-workspace-fixture-v1", PROMPT.read_text(encoding="utf-8")),
            seeds=[7],
            runtime_revision="fixture-v1",
            checkpoint_cadence=1,
            fault_schedule=[],
        )

        with tempfile.TemporaryDirectory(prefix="horizon-artifact-matrix-") as temporary:
            directory = Path(temporary)
            matrix = directory / "matrix.jsonl"
            workspace_root = directory / "workspaces"
            write_manifest_jsonl(matrix, manifests)
            executor = load_executor(EXECUTOR_SPEC)
            environment = {
                "HORIZON_BENCH_RUNTIME_URL": "http://127.0.0.1:1",
                "HORIZON_BENCH_ARTIFACT_ROOT": str(workspace_root),
            }
            with patch.dict(os.environ, environment, clear=False):
                preflight = preflight_matrix(
                    matrix,
                    tasks_path=TASKS,
                    source_name="horizon-artifact-workspace-fixture",
                    source_revision="fixture-v1",
                    split="heldout",
                    executor=executor,
                    executor_spec=EXECUTOR_SPEC,
                )
            self.assertFalse(workspace_root.exists())
            self.assertEqual(preflight["task_count"], len(task_set.task_ids))
            self.assertTrue(
                all(
                    check["preflight"]["artifact_environment"]["adapter_name"]
                    == "artifact_workspace"
                    for check in preflight["runs"][0]["checks"]
                )
            )
            self.assertTrue(
                all(
                    check["preflight"]["artifact_environment"]["readable_template_paths"]
                    for check in preflight["runs"][0]["checks"]
                )
            )

            address = _free_address()
            server = _start_server(_horizon_binary(), directory / "horizon.db", address)
            try:
                with patch.dict(
                    os.environ,
                    {
                        "HORIZON_BENCH_RUNTIME_URL": "http://" + address,
                        "HORIZON_BENCH_ARTIFACT_ROOT": str(workspace_root),
                    },
                    clear=False,
                ):
                    execution = execute_matrix(
                        matrix,
                        tasks_path=TASKS,
                        source_name="horizon-artifact-workspace-fixture",
                        source_revision="fixture-v1",
                        split="heldout",
                        executor=executor,
                        executor_spec=EXECUTOR_SPEC,
                        results_dir=directory / "results",
                    )
                results = read_jsonl(directory / "results" / (manifests[0].run_id + ".jsonl"))
                client = HorizonClient("http://" + address)
                traces = [client.events(str(run["run_id"])) for run in client.list_runs()]
            finally:
                _stop_server(server)

            self.assertEqual(execution["runs"][0]["status"], "completed")
            self.assertEqual(len(results), len(task_set.task_ids))
            self.assertTrue(all(result.success and result.state_consistent for result in results))
            self.assertEqual(score_results(results).task_success_rate, 1.0)
            self.assertEqual(len(list(workspace_root.iterdir())), len(task_set.task_ids))
            self.assertTrue(
                all(
                    any(event["type"] == "tool_result_recorded" for event in trace)
                    for trace in traces
                )
            )
            read_contexts = [
                event["data"]["result"]["metadata"]["model_context"]
                for trace in traces
                for event in trace
                if event["type"] == "tool_result_recorded"
                and event["data"]["result"]["tool"] == "artifact_workspace"
                and event["data"]["result"]["output"].get("operation") == "read"
            ]
            self.assertEqual(len(read_contexts), len(task_set.task_ids))
            self.assertTrue(
                all(context["kind"] == "artifact_workspace_read_v1" for context in read_contexts)
            )
            self.assertTrue(
                any(
                    file["path"] == "input/release.txt" and file["content"] == "release=durable-v1\n"
                    for context in read_contexts
                    for file in context["files"]
                )
            )
            self.assertTrue(
                any(
                    [file["path"] for file in context["files"]]
                    == ["input/request.json", "input/guard.txt"]
                    for context in read_contexts
                )
            )


if __name__ == "__main__":
    unittest.main()

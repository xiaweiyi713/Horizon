"""No-network end-to-end HorizonBench durable-trace matrix integration test."""

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
TASKS = FIXTURES / "durable_trace_tasks.jsonl"
MODELS = FIXTURES / "durable_trace_models.json"
CONDITIONS = FIXTURES / "durable_trace_conditions.json"
PROMPT = FIXTURES / "durable_trace_prompt.txt"


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


class DurableTraceMatrixIntegrationTests(unittest.TestCase):
    def test_scripted_fixture_runs_through_real_http_runtime_and_objective_judge(self) -> None:
        model = ModelProfile.from_mapping(json.loads(MODELS.read_text(encoding="utf-8"))["models"][0])
        condition = EvaluationCondition.from_mapping(
            json.loads(CONDITIONS.read_text(encoding="utf-8"))["conditions"][0]
        )
        task_set = JsonlTaskAdapter("horizon-durable-trace-fixture", "fixture-v1").load(TASKS)
        manifests = build_cross_model_matrix(
            models=[model],
            conditions=[condition],
            task_set=TaskSetIdentity.from_task_set(task_set),
            prompt=PromptArtifact("fixture-v1", PROMPT.read_text(encoding="utf-8")),
            seeds=[7],
            runtime_revision="fixture-v1",
            checkpoint_cadence=2,
            fault_schedule=[],
        )

        with tempfile.TemporaryDirectory(prefix="horizon-trace-matrix-") as temporary:
            directory = Path(temporary)
            matrix = directory / "matrix.jsonl"
            write_manifest_jsonl(matrix, manifests)
            address = _free_address()
            server = _start_server(_horizon_binary(), directory / "horizon.db", address)
            try:
                with patch.dict(os.environ, {"HORIZON_BENCH_RUNTIME_URL": "http://" + address}, clear=False):
                    report = execute_matrix(
                        matrix,
                        tasks_path=TASKS,
                        source_name="horizon-durable-trace-fixture",
                        source_revision="fixture-v1",
                        split="heldout",
                        executor=load_executor(
                            "benchmarks.horizonbench.scripted_durable_trace_executor:execute"
                        ),
                        executor_spec="benchmarks.horizonbench.scripted_durable_trace_executor:execute",
                        results_dir=directory / "results",
                    )
                results = read_jsonl(directory / "results" / (manifests[0].run_id + ".jsonl"))
                receipt = json.loads(
                    (directory / "results" / (manifests[0].run_id + ".receipt.json")).read_text(
                        encoding="utf-8"
                    )
                )
                client = HorizonClient("http://" + address)
                projections = client.list_runs()
                traces = {
                    projection["run_id"]: client.events(str(projection["run_id"]))
                    for projection in projections
                }
            finally:
                _stop_server(server)

        self.assertEqual(report["runs"][0]["executed_task_ids"], list(task_set.task_ids))
        self.assertEqual(len(results), 3)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["completed_task_ids"], list(manifests[0].task_set.task_ids))
        self.assertTrue(
            all(result.success and result.state_consistent for result in results),
            {"results": results, "traces": traces},
        )
        self.assertEqual(score_results(results).task_success_rate, 1.0)
        self.assertEqual(len(projections), 3)
        self.assertTrue(all(projection["state"] == "completed" for projection in projections))


if __name__ == "__main__":
    unittest.main()

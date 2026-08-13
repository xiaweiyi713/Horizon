"""Cross-process fault-injection verification for Horizon.

This script starts the real Rust HTTP runtime, simulates a process crash while
a task is durable-but-Running, restarts from the same SQLite file, and verifies
that replay requeues the task with failure memory and a State Anchor. It also
injects a real process timeout through the Process Supervisor.

Run it after `cargo build -p horizon-cli`:

    PYTHONPATH=python python3 tests/fault_injection.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from horizon_agent.client import HorizonApiError, HorizonClient  # noqa: E402
from horizon_agent.agent import AgentConfig, DurableAgent  # noqa: E402
from horizon_agent.memory import (  # noqa: E402
    AdaptiveContextBudgetController,
    LearnedInterventionPolicy,
    LogisticDecayPredictor,
)
from horizon_agent.providers import ScriptedProvider  # noqa: E402


def free_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return "127.0.0.1:{}".format(listener.getsockname()[1])


def binary() -> Path:
    candidate = ROOT / "target" / "debug" / "horizon"
    cargo = Path.home() / ".cargo" / "bin" / "cargo"
    command = str(cargo if cargo.exists() else "cargo")
    subprocess.run([command, "build", "-q", "-p", "horizon-cli"], cwd=ROOT, check=True)
    if not candidate.exists():
        raise RuntimeError("cargo build did not create {}".format(candidate))
    return candidate


def start_server(executable: Path, db: Path, address: str) -> subprocess.Popen:
    process = subprocess.Popen(
        [
            str(executable),
            "--db",
            str(db),
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
    stop_server(process)
    raise TimeoutError("Horizon server did not become healthy")


def stop_server(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def make_executing_run(client: HorizonClient, goal: str) -> str:
    created = client.create_run(goal)
    run_id = str(created["projection"]["run_id"])
    client.transition(run_id, "planning")
    client.transition(run_id, "executing")
    return run_id


def assert_runtime_crash_recovery(client: HorizonClient, executable: Path, db: Path, address: str, server: subprocess.Popen) -> subprocess.Popen:
    run_id = make_executing_run(client, "Recover an interrupted process task")
    task = client.create_task(
        run_id,
        "durable task",
        command=["sh", "-c", "printf recovered"],
        operation_id="fault-injection-operation",
        max_retries=1,
    )
    # The CreateTask response contains a map keyed by ID; extract its sole key.
    task_id = next(iter(task["projection"]["tasks"].keys()))
    client.command(run_id, "start_task", {"task_id": task_id})
    client.checkpoint(run_id)

    # Simulated Runtime Crash: the operating process disappears after durable
    # TaskStarted + Checkpoint but before any TaskSucceeded/TaskFailed event.
    stop_server(server)
    restarted = start_server(executable, db, address)
    restarted_client = HorizonClient("http://" + address)
    recovered = restarted_client.recover(run_id)
    projection = recovered["projection"]
    assert projection["state"] == "recovering", projection["state"]
    assert projection["tasks"][task_id]["status"] == "pending", projection["tasks"][task_id]
    assert any(
        "interrupted" in item["reason"] for item in projection["cognitive"]["failed_attempts"]
    ), projection["cognitive"]["failed_attempts"]
    assert projection["last_anchor_sequence"] > 0, projection

    restarted_client.transition(run_id, "planning")
    restarted_client.transition(run_id, "executing")
    outcome = restarted_client.execute_ready_tasks(run_id)
    assert len(outcome) == 1 and "Ok" in outcome[0]["output"], outcome
    final = restarted_client.get_run(run_id)
    assert final["tasks"][task_id]["status"] == "succeeded", final["tasks"][task_id]
    return restarted


def assert_process_timeout(client: HorizonClient) -> None:
    run_id = make_executing_run(client, "Record a tool timeout as failure memory")
    task = client.create_task(
        run_id,
        "timeout task",
        command=["sh", "-c", "sleep 1"],
        timeout_ms=25,
        operation_id="timeout-operation",
        max_retries=0,
    )
    task_id = next(iter(task["projection"]["tasks"].keys()))
    outcome = client.execute_ready_tasks(run_id)
    assert len(outcome) == 1 and "Err" in outcome[0]["output"], outcome
    projection = client.get_run(run_id)
    assert projection["tasks"][task_id]["status"] == "failed", projection["tasks"][task_id]
    failures = projection["cognitive"]["failed_attempts"]
    assert any(item["operation_id"] == "timeout-operation" for item in failures), failures
    results = projection["tool_results"]
    assert any(
        item["operation_id"] == "timeout-operation" and item["status"] == "timed_out"
        for item in results
    ), results


def assert_output_cap(client: HorizonClient) -> None:
    run_id = make_executing_run(client, "Cap noisy process output without losing task state")
    task = client.create_task(
        run_id,
        "bounded output",
        command=["sh", "-c", "printf 123456789"],
        operation_id="bounded-output-operation",
        resources={"max_output_bytes": 4},
    )
    task_id = next(iter(task["projection"]["tasks"].keys()))
    outcome = client.execute_ready_tasks(run_id)
    assert len(outcome) == 1 and "Ok" in outcome[0]["output"], outcome
    projection = client.get_run(run_id)
    assert projection["tasks"][task_id]["status"] == "succeeded", projection["tasks"][task_id]
    result = next(
        item for item in projection["tool_results"] if item["operation_id"] == "bounded-output-operation"
    )
    assert result["output"]["stdout"] == "1234", result
    assert result["output"]["stdout_truncated"], result


def assert_python_policy_loop(client: HorizonClient) -> None:
    provider = ScriptedProvider(
        [
            {
                "action": {
                    "type": "open_subgoal",
                    "data": {"subgoal": {"content": "verify the Python-to-Rust boundary"}},
                }
            },
            {"action": {"type": "finish", "data": {}}},
        ]
    )
    result = DurableAgent(client, provider, config=AgentConfig(max_steps=4)).run(
        "Complete a scripted durable policy loop",
        constraints=["All state changes must be runtime commands."],
    )
    assert result.state == "completed" and result.steps == 2, result
    projection = client.get_run(result.run_id)
    assert projection["cognitive"]["open_subgoals"], projection
    events = client.events(result.run_id)
    event_types = [event["type"] for event in events]
    assert event_types.count("tool_invoked") == 2, event_types
    assert event_types.count("tool_result_recorded") == 2, event_types


def assert_learned_intervention_policy(client: HorizonClient) -> None:
    """Exercise Python prediction -> Rust assessment -> Rust anchor rendering."""
    predictor = LogisticDecayPredictor(
        # A deliberately obvious fixture model: this verifies the durable data
        # path, not a claim about empirical model quality.
        weights=(4.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        model_version="fault-fixture-v1",
    )
    policy = LearnedInterventionPolicy(
        predictor=predictor,
        controller=AdaptiveContextBudgetController(context_window_tokens=256),
    )
    result = DurableAgent(
        client,
        ScriptedProvider([{"action": {"type": "finish", "data": {}}}]),
        config=AgentConfig(
            max_steps=2,
            checkpoint_on_finish=False,
            learned_intervention_policy=policy,
        ),
    ).run("Persist learned state-decay intervention decisions")
    assert result.state == "completed" and result.anchors_injected == 1, result
    projection = client.get_run(result.run_id)
    assert projection["intervention_assessments"] == 1, projection
    assert projection["last_intervention_assessment"]["policy_id"] == "logistic_state_decay", projection
    events = client.events(result.run_id)
    event_types = [event["type"] for event in events]
    assert event_types.count("state_decay_assessed") == 1, event_types
    assert event_types.count("state_anchor_injected") == 1, event_types

    # The same HTTP path must durably record a deliberate *non*-intervention.
    low_risk_policy = LearnedInterventionPolicy(
        predictor=LogisticDecayPredictor(
            weights=(-4.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            model_version="fault-fixture-low-risk-v1",
        ),
        controller=AdaptiveContextBudgetController(context_window_tokens=256),
    )
    low_risk_result = DurableAgent(
        client,
        ScriptedProvider([{"action": {"type": "finish", "data": {}}}]),
        config=AgentConfig(
            max_steps=2,
            checkpoint_on_finish=False,
            learned_intervention_policy=low_risk_policy,
        ),
    ).run("Persist learned decisions even when no anchor is needed")
    assert low_risk_result.state == "completed" and low_risk_result.anchors_injected == 0, low_risk_result
    low_risk_events = client.events(low_risk_result.run_id)
    low_risk_assessments = [
        event for event in low_risk_events if event["type"] == "state_decay_assessed"
    ]
    assert len(low_risk_assessments) == 1, low_risk_events
    assert low_risk_assessments[0]["data"]["assessment"]["action"] == "continue", low_risk_assessments
    assert not any(event["type"] == "state_anchor_injected" for event in low_risk_events), low_risk_events


def main() -> None:
    executable = binary()
    address = free_address()
    with tempfile.TemporaryDirectory(prefix="horizon-fault-") as temporary:
        db = Path(temporary) / "horizon.db"
        server: Optional[subprocess.Popen] = None
        try:
            server = start_server(executable, db, address)
            client = HorizonClient("http://" + address)
            server = assert_runtime_crash_recovery(client, executable, db, address, server)
            restarted_client = HorizonClient("http://" + address)
            assert_process_timeout(restarted_client)
            assert_output_cap(restarted_client)
            assert_python_policy_loop(restarted_client)
            assert_learned_intervention_policy(restarted_client)
        finally:
            stop_server(server)
    print(
        "fault injection passed: crash/recovery + timeout + output cap + "
        "Python policy loop + learned intervention boundary"
    )


if __name__ == "__main__":
    main()

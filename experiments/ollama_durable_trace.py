"""Run a reproducible real-model durable-trace mechanism smoke with Ollama.

This utility is intentionally narrow. It uses a task-owned, model-readable
durable-trace mechanism suite to validate a real local model → Python policy →
Rust runtime → objective trace judge path, including frozen Anchor controls.
It is an integration/mechanism smoke, not a public benchmark or a claim that
Horizon improves a model.

The model digest is read from Ollama before planning and is embedded in every
manifest row.  The runner writes all generated artifacts below an empty output
directory and never writes credentials to them.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


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
from benchmarks.horizonbench.score_matrix import score_matrix  # noqa: E402
from horizon_agent import HorizonApiError, HorizonClient  # noqa: E402


DEFAULT_TASKS = ROOT / "experiments" / "ollama_durable_trace_tasks_v3.jsonl"
DEFAULT_CONDITIONS = ROOT / "experiments" / "ollama_durable_trace_conditions_v1.json"
DEFAULT_PROMPT = ROOT / "experiments" / "ollama_durable_trace_prompt_v2.txt"
EXECUTOR_SPEC = "benchmarks.horizonbench.openai_compatible_trace_executor:execute"


class OllamaExperimentError(RuntimeError):
    """Raised when an experiment cannot be planned or run reproducibly."""


def _parse_seeds(value: str) -> list[int]:
    parts = [item.strip() for item in value.split(",")]
    if not parts or any(not item for item in parts):
        raise OllamaExperimentError("--seeds must be a comma-separated list of integers")
    try:
        seeds = [int(item) for item in parts]
    except ValueError as error:
        raise OllamaExperimentError("--seeds must be a comma-separated list of integers") from error
    if len(set(seeds)) != len(seeds):
        raise OllamaExperimentError("--seeds must not contain duplicates")
    return seeds


def _ollama_root(api_base_url: str) -> str:
    parsed = urlsplit(api_base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise OllamaExperimentError("--api-base-url must be an absolute HTTP(S) URL")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        raise OllamaExperimentError("--api-base-url must end with /v1 for an Ollama OpenAI-compatible server")
    return urlunsplit((parsed.scheme, parsed.netloc, path[: -len("/v1")] or "/", "", "")).rstrip("/")


def ollama_model_digest(api_base_url: str, model_name: str, *, timeout: float = 10.0) -> str:
    """Return the immutable digest advertised for one locally available model."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise OllamaExperimentError("--model must be a non-empty model name")
    request = Request(_ollama_root(api_base_url) + "/api/tags", method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError) as error:
        raise OllamaExperimentError("could not read the Ollama model list: {}".format(error)) from error
    models = payload.get("models") if isinstance(payload, Mapping) else None
    if not isinstance(models, list):
        raise OllamaExperimentError("Ollama /api/tags did not return a models array")
    for model in models:
        if not isinstance(model, Mapping):
            continue
        if model.get("name") != model_name and model.get("model") != model_name:
            continue
        digest = model.get("digest")
        if isinstance(digest, str) and len(digest) == 64 and all(
            character in "0123456789abcdef" for character in digest.lower()
        ):
            return digest.lower()
        raise OllamaExperimentError("Ollama model {!r} has no valid digest".format(model_name))
    raise OllamaExperimentError("Ollama model {!r} is not installed on this host".format(model_name))


def load_conditions(path: Path) -> list[EvaluationCondition]:
    """Load a strict, JSON-safe condition artifact before the matrix is frozen."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OllamaExperimentError("could not read conditions {}: {}".format(path, error)) from error
    raw = payload.get("conditions") if isinstance(payload, Mapping) else None
    if not isinstance(raw, list) or not raw or not all(isinstance(item, Mapping) for item in raw):
        raise OllamaExperimentError("conditions file must contain a non-empty {conditions: [...]} object")
    try:
        return [EvaluationCondition.from_mapping(item) for item in raw]
    except ValueError as error:
        raise OllamaExperimentError("invalid experiment condition: {}".format(error)) from error


def _git_revision() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise OllamaExperimentError("could not inspect the Horizon git worktree")
    if status.stdout.strip():
        raise OllamaExperimentError(
            "refusing to run an experiment from a dirty worktree; commit or stash every source change first"
        )
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if completed.returncode != 0 or not revision:
        raise OllamaExperimentError("could not determine the Horizon git revision")
    return revision


def _horizon_binary(path: Path | None) -> Path:
    if path is not None:
        if not path.is_file():
            raise OllamaExperimentError("--horizon-bin is not a file: {}".format(path))
        return path
    subprocess.run(["cargo", "build", "-q", "-p", "horizon-cli"], cwd=ROOT, check=True)
    candidate = ROOT / "target" / "debug" / "horizon"
    if not candidate.is_file():
        raise OllamaExperimentError("cargo build did not create {}".format(candidate))
    return candidate


def _free_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return "127.0.0.1:{}".format(listener.getsockname()[1])


def _start_runtime(executable: Path, database: Path, address: str) -> subprocess.Popen[bytes]:
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
        stderr=subprocess.PIPE,
    )
    client = HorizonClient("http://" + address, timeout=0.25)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise OllamaExperimentError(
                "Horizon runtime exited before becoming healthy ({}): {}".format(
                    process.returncode, stderr.strip()
                )
            )
        try:
            if client.health():
                return process
        except HorizonApiError:
            time.sleep(0.05)
    _stop_runtime(process)
    raise OllamaExperimentError("Horizon runtime did not become healthy within 15 seconds")


def _stop_runtime(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise OllamaExperimentError("--output-dir is not a directory: {}".format(path))
        if any(path.iterdir()):
            raise OllamaExperimentError("--output-dir must be new or empty: {}".format(path))
    else:
        path.mkdir(parents=True)


def run_experiment(arguments: argparse.Namespace) -> dict[str, Any]:
    """Plan, preflight, execute, and score one local real-model smoke matrix."""

    output_dir = arguments.output_dir.resolve()
    _prepare_output_directory(output_dir)
    tasks_path = arguments.tasks.resolve()
    conditions_path = arguments.conditions.resolve()
    prompt_path = arguments.prompt.resolve()
    if not tasks_path.is_file() or not prompt_path.is_file():
        raise OllamaExperimentError("task source and prompt must both be regular files")
    digest = ollama_model_digest(arguments.api_base_url, arguments.model)
    conditions = load_conditions(conditions_path)
    task_set = JsonlTaskAdapter(arguments.source_name, arguments.source_revision).load(
        tasks_path, split="heldout"
    )
    model = ModelProfile(
        model_id="ollama-{}-{}".format(arguments.model.replace(":", "-"), digest[:12]),
        provider="openai-compatible",
        model=arguments.model,
        model_revision="ollama-digest:{}".format(digest),
        decoding={"temperature": 0, "top_p": 1, "max_tokens": arguments.max_tokens},
    )
    manifests = build_cross_model_matrix(
        models=[model],
        conditions=conditions,
        task_set=TaskSetIdentity.from_task_set(task_set),
        prompt=PromptArtifact(arguments.prompt_revision, prompt_path.read_text(encoding="utf-8")),
        seeds=_parse_seeds(arguments.seeds),
        runtime_revision=_git_revision(),
        checkpoint_cadence=arguments.checkpoint_cadence,
        fault_schedule=[],
        metadata={
            "experiment_kind": "real_model_durable_trace_mechanism_smoke",
            "claim_boundary": "not_a_public_benchmark_or_model_improvement_claim",
            "model_digest": digest,
        },
    )
    matrix_path = output_dir / "matrix.jsonl"
    write_manifest_jsonl(matrix_path, manifests)

    # Ollama ignores the bearer value but the generic OpenAI-compatible adapter
    # intentionally requires a non-empty credential field. Keep the local
    # marker out of every persisted artifact.
    os.environ["HORIZON_BENCH_API_BASE_URL"] = arguments.api_base_url.rstrip("/")
    os.environ["HORIZON_BENCH_API_KEY"] = arguments.api_key
    # Preflight validates only configuration, never opens this endpoint. Use a
    # benign local placeholder until the isolated runtime receives its actual
    # free port below.
    os.environ["HORIZON_BENCH_RUNTIME_URL"] = "http://127.0.0.1:8787"
    no_proxy = [item.strip() for item in os.environ.get("NO_PROXY", "").split(",") if item.strip()]
    os.environ["NO_PROXY"] = ",".join(dict.fromkeys([*no_proxy, "127.0.0.1", "localhost"]))
    executor = load_executor(EXECUTOR_SPEC)
    preflight = preflight_matrix(
        matrix_path,
        tasks_path=tasks_path,
        source_name=arguments.source_name,
        source_revision=arguments.source_revision,
        split="heldout",
        executor=executor,
        executor_spec=EXECUTOR_SPEC,
    )
    _write_json(output_dir / "preflight.json", preflight)

    runtime = None
    address = _free_address()
    try:
        runtime = _start_runtime(_horizon_binary(arguments.horizon_bin), output_dir / "horizon.db", address)
        os.environ["HORIZON_BENCH_RUNTIME_URL"] = "http://" + address
        execution = execute_matrix(
            matrix_path,
            tasks_path=tasks_path,
            source_name=arguments.source_name,
            source_revision=arguments.source_revision,
            split="heldout",
            executor=executor,
            executor_spec=EXECUTOR_SPEC,
            results_dir=output_dir / "results",
        )
        _write_json(output_dir / "execution.json", execution)
        score = score_matrix(matrix_path, output_dir / "results")
        _write_json(output_dir / "score.json", score)
    finally:
        _stop_runtime(runtime)

    summary = {
        "experiment_kind": "real_model_durable_trace_mechanism_smoke",
        "claim_boundary": "not_a_public_benchmark_or_model_improvement_claim",
        "output_dir": str(output_dir),
        "model": arguments.model,
        "model_digest": digest,
        "matrix_runs": len(manifests),
        "task_count": len(task_set.task_ids),
        "conditions": [condition.condition_id for condition in conditions],
        "preflight": "preflight.json",
        "execution": "execution.json",
        "score": "score.json",
    }
    _write_json(output_dir / "experiment.json", summary)
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real-model Horizon durable-trace mechanism smoke against local Ollama"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="new or empty artifact directory")
    parser.add_argument("--model", default="qwen2.5:7b", help="installed Ollama model name")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument(
        "--api-key",
        default="ollama-local",
        help="non-secret local marker required by the generic provider; never persisted",
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--source-name", default="horizon-ollama-durable-trace-mechanism")
    parser.add_argument("--source-revision", default="v3")
    parser.add_argument("--prompt-revision", default="ollama-durable-trace-prompt-v2")
    parser.add_argument("--seeds", default="17", help="comma-separated unique integer seeds")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--checkpoint-cadence", type=int, default=1)
    parser.add_argument("--horizon-bin", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = make_parser().parse_args(argv)
    if arguments.max_tokens < 1:
        raise SystemExit("error: --max-tokens must be positive")
    if arguments.checkpoint_cadence < 1:
        raise SystemExit("error: --checkpoint-cadence must be positive")
    try:
        summary = run_experiment(arguments)
    except (OllamaExperimentError, OSError, subprocess.SubprocessError, ValueError) as error:
        raise SystemExit("error: {}".format(error)) from error
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

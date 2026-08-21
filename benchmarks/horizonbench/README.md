# HorizonBench

HorizonBench v0.1 is a 30-task microbenchmark for durable execution state:

| Category | Tasks | Primary signal |
|---|---:|---|
| Goal retention | 5 | Original objective survives distracting work |
| Constraint retention | 5 | Long-lived guardrails remain behavioral |
| Failure avoidance | 5 | Rejected approaches are not repeated |
| Cross-session recovery | 5 | Goal, progress, and decisions restore after restart |
| Long-context degradation | 5 | Essential state survives a long trajectory |
| Fault recovery | 5 | Checkpoint/replay and idempotency survive faults |

Each agent adapter produces one JSONL row per task. The required fields are
`task_id`, `category`, `success`, `goal_retained`, and
`constraint_violations`; optional fields capture failures, recovery, tokens,
anchors, and a short non-metric `diagnostic` label. The diagnostic is capped at
512 characters and must never contain a model response, provider HTTP body,
prompt, credential, or credential-bearing URL. Score a real run with:

```bash
python3 benchmarks/horizonbench/score_runs.py results/my-agent.jsonl
```

For CI and smoke testing, the included deterministic reference harness exposes
B0–B3, the always-on-anchor control, heuristic Full Horizon, the durable
semantic-memory retrieval on/off pair, a fixed fixture model behind the real
learned/adaptive controller, its fixed-threshold control, and four mechanism
ablations:

```bash
python3 benchmarks/horizonbench/run.py --output-dir results/horizonbench
```

Those outputs are explicitly marked `synthetic_deterministic_smoke`. They test
benchmark mechanics and must not be reported as empirical LLM results. The
fixture model is not trained on HorizonBench; real learned-policy studies must
fit a separate artifact on run/model-disjoint boundary labels.

The semantic-memory pair is also a synthetic control. It models the same
State-Anchor behavior with or without a relevant durable retrieval result; it
does not claim empirical embedding or model quality. Real experiments should
retain `memory_created` and `semantic_memory_retrieved` events, score a frozen
relevant-memory set, and compare it to `semantic_memory_limit=0`.

## Held-out and cross-model runs

For an empirical run, do not reuse the synthetic `tasks.json` fixture as a
public benchmark result. Mirror the benchmark source locally as UTF-8 JSONL and
give every row these required fields:

```json
{"id":"public-001","category":"goal_retention","goal":"...","constraint":"...","split":"development"}
{"id":"heldout-001","category":"fault_recovery","goal":"...","constraint":"...","split":"heldout","metadata":{"license":"..."}}
```

Extra JSON-safe fields are retained as metadata. `JsonlTaskAdapter` rejects
blank/malformed rows, duplicate IDs even across splits, and requests for an
empty split. It records both the raw source SHA-256 and a normalized selected
task-set SHA-256, so ordinary whitespace or record ordering cannot change a
frozen held-out set.

Create `models.json` with `model_id`, `provider`, `model`, `model_revision`,
and a `decoding` object; create `conditions.json` with `condition_id`,
`policy_revision`, and a `configuration` object. Then plan, preflight, execute, and score
the matrix:

```bash
python3 benchmarks/horizonbench/plan_matrix.py \
  --tasks data/public-benchmark.jsonl \
  --source-name PUBLIC_BENCHMARK --source-revision IMMUTABLE_RELEASE \
  --models configs/models.json --conditions configs/conditions.json \
  --prompt prompts/system.txt --prompt-revision PROMPT_REVISION \
  --runtime-revision GIT_REVISION --seeds 1,2,3 \
  --output results/matrix.jsonl

# Static validation only: no runtime/model/task execution, no results, and no
# receipt files. The executor must expose execute.preflight(context).
python3 benchmarks/horizonbench/preflight_matrix.py \
  --matrix results/matrix.jsonl --tasks data/public-benchmark.jsonl \
  --source-name PUBLIC_BENCHMARK --source-revision IMMUTABLE_RELEASE \
  --executor my_experiment.executor:execute

# The executor receives a frozen manifest/task context and must return an
# environment-observed EpisodeResult, not an LLM self-assessment.
python3 benchmarks/horizonbench/execute_matrix.py \
  --matrix results/matrix.jsonl --tasks data/public-benchmark.jsonl \
  --source-name PUBLIC_BENCHMARK --source-revision IMMUTABLE_RELEASE \
  --executor my_experiment.executor:execute --results-dir results

python3 benchmarks/horizonbench/score_matrix.py \
  --matrix results/matrix.jsonl --results-dir results --output results/report.json
```

An executor is a local callable with this narrow boundary:

```python
from benchmarks.horizonbench import EpisodeResult

def execute(context):
    # Construct the configured model/runtime and run one isolated task
    # environment. `observation` must come from its verifier, tests, artifacts,
    # or other objective task boundary—not from the model claiming success.
    observation = run_task_environment(
        task=context.task,
        prompt=context.manifest.prompt.text,
        model=context.manifest.model,
        condition=context.manifest.condition,
        attempt=context.attempt,
    )
    return EpisodeResult(
        task_id=context.task.task_id,
        category=context.task.category,
        success=observation.success,
        goal_retained=observation.goal_retained,
        constraint_violations=observation.constraint_violations,
        failure_actions=observation.failure_actions,
        known_failed_approaches=observation.known_failed_approaches,
        recovery_attempted=observation.recovery_attempted,
        recovery_succeeded=observation.recovery_succeeded,
        recovery_distance=observation.recovery_distance,
        state_consistent=observation.state_consistent,
        tokens=observation.tokens,
        anchors_injected=observation.anchors_injected,
    )
```

Give the same callable a static preflight hook when it will be used through
`preflight_matrix.py`:

```python
def preflight(context):
    validate_local_task_metadata(context.task.metadata)
    validate_frozen_provider_settings(context.manifest.model)
    return {"ready": True, "provider_configured": True}

execute.preflight = preflight
```

`execute_matrix.py` writes `<run_id>.jsonl` after every task and a matching
`<run_id>.receipt.json` sidecar. It resumes only those receipt-bound outputs;
the receipt binds the result set to the manifest hash, selected task-set hash,
and executor import path. It rejects changed task sources, mismatched executors,
duplicate/unexpected outcomes, and result files without a receipt. Pass
`--overwrite` only to intentionally replace a completed or interrupted run.

Run `preflight_matrix.py` immediately before an empirical execution. It loads
the same frozen matrix and task source, validates their exact identity, and
calls `execute.preflight(context)` once for every selected manifest/task pair.
It never calls `execute(context)` and never writes results or receipts. A
custom preflight hook and its module import must stay static: validate frozen
prompt/condition/task metadata, credentials, provider controls, and local
configuration there—but do not contact a model, runtime, or task environment.
Return only non-sensitive status data: never put a key, credential-bearing URL,
or provider response into the preflight report. The built-in scripted and
OpenAI-compatible durable-trace executors provide this hook; legacy executors
can still run through `execute_matrix.py` but are intentionally rejected by
the preflight command until they add one.

For a no-network smoke test of the execution plumbing, use
`benchmarks.horizonbench.synthetic_fixture_executor:execute`. It returns
synthetic all-success outcomes and is explicitly **not** an empirical model
executor.

For a stronger no-network integration check of the new agent bridge, run:

```bash
make bench-trace
```

It starts a temporary real Rust HTTP runtime with automatic checkpoints
disabled, plans a three-task fixture matrix, runs the real Python
`DurableAgent` through `scripted_durable_trace_executor`, persists matrix
receipts, and scores the resulting immutable event/projection traces. The
fixture actions and task expectations live in `fixtures/`; it is deterministic
CI coverage, not an empirical LLM result.

## Model-backed durable-trace execution

For a narrow runtime-mechanism task set, Horizon includes the opt-in executor
`benchmarks.horizonbench.openai_compatible_trace_executor:execute`. It runs a
real `DurableAgent`, binds the manifest prompt text and checkpoint cadence to
that agent, collects its durable projection/events, and lets `DurableTraceJudge`
produce the result. A model saying `finish` is never enough to score success.
Importing or planning this executor does not contact a model endpoint.
The prompt artifact must be a complete Horizon single-action system prompt: it
is supplied verbatim, not silently prefixed with a second mutable prompt.

Each selected task must include a metadata object such as:

```json
{
  "initial_plan": ["restore checkpoint", "verify the operation ID"],
  "horizon_trace_expectation": {
    "required_event_types": ["evidence_recorded", "checkpoint_created"],
    "required_evidence_terms": ["verified checksum"],
    "required_failure_approaches": ["restart from event zero"],
    "require_checkpoint": true,
    "require_recovery": false
  }
}
```

Use an `openai-compatible` (or `openai_compatible`) model profile and put all
effective provider controls in `decoding`:

```json
{
  "model_id": "model-a",
  "provider": "openai-compatible",
  "model": "MODEL_NAME",
  "model_revision": "PROVIDER_SNAPSHOT",
  "decoding": {"temperature": 0, "top_p": 1, "max_tokens": 512}
}
```

The matrix's separate `seed` is forwarded to the provider; do not specify a
different `decoding.seed`. Put simple agent controls under a condition's
`configuration.agent`, for example
`{"agent":{"max_steps":24,"semantic_memory_limit":4}}`. To run an auditable
State Anchor ablation, also set `anchor_strategy` to `runtime_heuristic`
(default), `always`, or `disabled`. The fixed controls write durable
`fixed_state_anchor_strategy` assessments; `always` additionally writes a
`state_anchor_injected` event. Their immutable revision comes from the matrix
condition's `policy_revision`.

Start a separate benchmark runtime with automatic checkpoints disabled, so
unconfigured runtime snapshots do not add checkpoint events; the bridge then
applies the cadence frozen in the manifest:

```bash
cargo run -p horizon-cli -- --db results/durable-trace.db --checkpoint-every 0 serve

export HORIZON_BENCH_RUNTIME_URL=http://127.0.0.1:8787
export HORIZON_BENCH_API_BASE_URL=https://YOUR_COMPATIBLE_ENDPOINT/v1
export HORIZON_BENCH_API_KEY=YOUR_KEY  # or use OPENAI_API_KEY

# This checks the provider profile, effective seed/decoding, credential
# presence, trace-task expectations, and frozen runtime settings without
# sending a request or opening a runtime connection.
PYTHONPATH=python:. python3 benchmarks/horizonbench/preflight_matrix.py \
  --matrix results/matrix.jsonl --tasks data/durable-trace-tasks.jsonl \
  --source-name DURABLE_TRACE_TASKS --source-revision IMMUTABLE_RELEASE \
  --executor benchmarks.horizonbench.openai_compatible_trace_executor:execute

PYTHONPATH=python:. python3 benchmarks/horizonbench/execute_matrix.py \
  --matrix results/matrix.jsonl --tasks data/durable-trace-tasks.jsonl \
  --source-name DURABLE_TRACE_TASKS --source-revision IMMUTABLE_RELEASE \
  --executor benchmarks.horizonbench.openai_compatible_trace_executor:execute \
  --results-dir results
```

The bridge deliberately rejects a non-empty `fault_schedule`. A task-specific
environment executor must enact, observe, and judge an actual fault/restart
schedule. Likewise, the bundled cross-domain fixture does not declare trace
expectations and must use an artifact/test/environment judge; do not run it
through this trace-only executor or report its supplied observations as a
model result. Use an isolated benchmark environment for any agent allowed to
create or execute external tasks.

`score_matrix.py` refuses incomplete rows and refuses a matrix whose task set,
prompt content, runtime revision, checkpoint cadence, or fault schedule differs
between entries. To score a single result against a selected matrix row, use
`score_runs.py RESULT.jsonl --manifest-matrix results/matrix.jsonl --run-id RUN_ID`.
The profile and manifest schemas reject unrecognized top-level fields: put
provider decoding controls in `decoding` and policy-specific controls in
`configuration` so every effective setting is hashed.

## Objective artifact-workspace execution

`benchmarks.horizonbench.openai_compatible_artifact_workspace_executor:execute`
is the opt-in executor for a bounded file-artifact task. It runs a real
`DurableAgent`, but success belongs to the task environment—not to an LLM
`finish` action. The model must call the registered `artifact_workspace`
adapter, the adapter can write only task-declared relative paths, and an exact
text or semantic-JSON verifier checks the materialized artifact. The judge
also requires the retained goal/constraint, a real adapter invocation, a
durably recorded verified terminal result, and a completed runtime projection.

Each selected task includes its normal `initial_plan` plus this strict metadata
object (no unrecognized fields are accepted):

```json
{
  "horizon_artifact_workspace_v1": {
    "schema_version": 1,
    "adapter_name": "artifact_workspace",
    "template_files": [
      {"path": "input/brief.txt", "content": "Create a JSON approval artifact."}
    ],
    "writable_paths": ["answer.json"],
    "verifier": {
      "kind": "json_exact",
      "path": "answer.json",
      "expected": {"approved": true}
    }
  }
}
```

The workspace rejects absolute paths, traversal, hidden segments, symlinks,
template/writable overlaps, unknown fields, duplicate paths, and oversized
content. A policy submits the candidate via the frozen prompt contract:

```json
{
  "type": "invoke_adapter",
  "data": {
    "adapter": "artifact_workspace",
    "operation_id": "artifact:task-001:write",
    "task_id": "task-001",
    "input": {
      "files": [{"path": "answer.json", "content": "{\"approved\":true}"}]
    }
  }
}
```

The operation receipt stored beside the workspace contains a submitted-input
SHA-256 and safe verification observation, not the candidate contents. A new
Python process that repeats the same stable operation ID returns that recorded
terminal result instead of rewriting the artifact. This is still at-least-once
delivery around the boundary: a crash between the write and receipt may cause
one safe replay, and a non-empty matrix `fault_schedule` is rejected until a
task environment can genuinely enact and observe it.

This is intentionally a restricted artifact verifier, **not** a complete
process/container security sandbox. It runs no shell command and provides no
arbitrary filesystem access. In v1 the adapter only accepts candidate writes;
put instructions needed by the model in the frozen goal, constraint, and
approved plan. Never copy `verifier.expected` into those model-visible fields
for an empirical held-out task.

Start a separate runtime with automatic checkpoints disabled, then configure
the model endpoint and a persistent, caller-owned root for auditable workspace
artifacts:

```bash
cargo run -p horizon-cli -- --db results/artifact-workspace.db --checkpoint-every 0 serve

export HORIZON_BENCH_RUNTIME_URL=http://127.0.0.1:8787
export HORIZON_BENCH_API_BASE_URL=https://YOUR_COMPATIBLE_ENDPOINT/v1
export HORIZON_BENCH_API_KEY=YOUR_KEY  # or OPENAI_API_KEY
export HORIZON_BENCH_ARTIFACT_ROOT="$PWD/results/artifact-workspace-files"

# Static only: validates task schema, prompt contract, provider controls and
# root configuration. It does not create the root or contact either endpoint.
PYTHONPATH=python:. python3 benchmarks/horizonbench/preflight_matrix.py \
  --matrix results/artifact-matrix.jsonl --tasks data/artifact-tasks.jsonl \
  --source-name ARTIFACT_TASKS --source-revision IMMUTABLE_RELEASE \
  --executor benchmarks.horizonbench.openai_compatible_artifact_workspace_executor:execute

PYTHONPATH=python:. python3 benchmarks/horizonbench/execute_matrix.py \
  --matrix results/artifact-matrix.jsonl --tasks data/artifact-tasks.jsonl \
  --source-name ARTIFACT_TASKS --source-revision IMMUTABLE_RELEASE \
  --executor benchmarks.horizonbench.openai_compatible_artifact_workspace_executor:execute \
  --results-dir results
```

Retain the frozen matrix/task source, per-run result JSONL and receipt, durable
runtime database/events, and `HORIZON_BENCH_ARTIFACT_ROOT` directory together.
They provide the evidence chain from frozen task contract to observed artifact.
`fixtures/artifact_workspace_*` plus `make bench-artifact-workspace` exercise
the complete path using a scripted policy and a real Rust HTTP runtime. They
are deterministic CI coverage, not a real-model result.

## Cross-domain long-horizon workflow suite

`cross_domain_tasks.jsonl` is a deterministic held-out fixture for the scoring
path, not an empirical benchmark result. It covers four three-step workflows:
software engineering event replay, data-analysis OOS audit, research evidence
synthesis, and operations migration recovery. Each workflow names explicit
prerequisites, at least one recovery boundary, evidence-required steps, and a
minimum expected boundary count of 12.

Plan the normal matrix against this task source, execute it with an
environment-observing executor, then produce an auditable domain score and a
Markdown technical report:

```bash
python3 benchmarks/horizonbench/plan_matrix.py \
  --tasks benchmarks/horizonbench/cross_domain_tasks.jsonl \
  --source-name horizon-cross-domain-fixture --source-revision v1 \
  --models configs/models.json --conditions configs/conditions.json \
  --prompt prompts/system.txt --prompt-revision PROMPT_REVISION \
  --runtime-revision GIT_REVISION --seeds 1,2,3 \
  --output results/cross-domain-matrix.jsonl

python3 benchmarks/horizonbench/execute_matrix.py \
  --matrix results/cross-domain-matrix.jsonl \
  --tasks benchmarks/horizonbench/cross_domain_tasks.jsonl \
  --source-name horizon-cross-domain-fixture --source-revision v1 \
  --executor my_experiment.executor:execute --results-dir results

python3 benchmarks/horizonbench/cross_domain_score.py \
  --matrix results/cross-domain-matrix.jsonl --results-dir results \
  --output results/cross-domain-score.json

python3 benchmarks/horizonbench/render_report.py results/cross-domain-score.json \
  --output results/cross-domain-report.md
```

The cross-domain scorer rejects a workflow source whose selected task identity
does not exactly match the matrix. It reports both standard task metrics and
workflow completion, prerequisite integrity, recovery-boundary task success,
and evidence-task success. The generated report deliberately labels outcomes
as supplied observations; retain the raw JSONL, manifests, durable events, and
external-source license/provenance alongside it.

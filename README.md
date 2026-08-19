# Horizon

> A durable cognitive runtime for long-horizon AI agents, built with Rust for
> reliability and Python for policy intelligence.

**Project status:** v1.0 engineering milestone complete. Horizon provides a
tested, reproducible runtime and evaluation foundation. It does **not** claim
empirical model-performance gains until those claims are supported by a frozen
model/task experiment.

Long context alone does not make an agent reliable over a long task. An agent
can still lose its goal, violate an old constraint, repeat a failed approach,
or recover from a crash with an inconsistent view of progress. Horizon treats
that problem as **durable execution-state management**:

- Python decides what to do next.
- Rust validates every state change, persists an event, and updates a replayable
  projection.
- Checkpoints, replay, stable operation IDs, and structured cognitive state
  make recovery inspectable instead of transcript-dependent.

## Why Horizon?

Horizon is not another general-purpose chat agent or tool marketplace. It is a
runtime layer for agents that must survive long trajectories, task switches,
tool failures, and process restarts without treating the LLM as the source of
truth.

| Problem | Horizon mechanism |
|---|---|
| Goal or constraint drift | Durable cognitive state plus bounded State Anchors |
| Repeated failed actions | Structured failure memory with durable provenance |
| Runtime or process crash | Checkpoint, event replay, and retry-aware recovery |
| At-least-once external work | Stable logical `operation_id` for idempotent integrations |
| Context pressure | Proactive, explainable State Anchor intervention |
| Unreproducible evaluation | Frozen manifests, objective results, receipts, and preflight |

## Architecture

```text
Python policy / LLM provider
        │  JSON over localhost HTTP or optional PyO3
        ▼
┌──────────────────────────────────────────────────────────────┐
│                       Horizon Rust runtime                    │
│                                                              │
│  Command → Validate → Immutable event → Event store → Apply  │
│                                  │                           │
│                    Checkpoint / replay / audit               │
│                                  │                           │
│  Durable cognitive state + State Anchor intervention policy  │
└──────────────────────────────────────────────────────────────┘
        │
        ├── Tokio dependency-aware task scheduler
        ├── Timeout/resource-aware process supervisor
        └── SQLite by default; PostgreSQL when shared storage is needed
```

The durable projection records the primary goal, constraints, plan, subgoals,
tasks, failures, decisions, evidence, environment, semantic memories, budget,
and recovery boundaries. Python never mutates that projection directly.

## What is implemented

### Durable runtime

- Validated agent/task state machines and append-only ordered events.
- SQLite/WAL event storage, checksummed zstd snapshots, compaction, and
  schema-versioned replay; PostgreSQL is available behind the same store
  abstraction.
- Cross-process recovery of interrupted tasks with retry policy and durable
  failure memory.
- Stable `HORIZON_OPERATION_ID` delivery to process tasks for at-least-once,
  idempotent integration boundaries.
- Tokio DAG scheduling plus process timeout, output caps, CPU/memory limits,
  and an optional Docker execution backend.
- Local CLI and JSON/HTTP API, optional PyO3 native SQLite runtime, and
  post-commit OTLP/HTTP trace export.

### Cognitive-state management

- Durable goals, constraints, plans, subgoals, decisions, evidence, failures,
  environment, and budget state.
- Explainable proactive State Anchor policy based on anchor distance, context
  pressure, task switching, failures, and recovery.
- Offline-trainable state-decay predictor with an adaptive context-budget
  controller; each decision is persisted for audit.
- Source-attributed semantic-memory catalog with deterministic bounded
  `hybrid_lexical_v1` retrieval.

### Python and evaluation

- Dependency-free Python HTTP client, single-action JSON policy loop, scripted
  provider, and OpenAI-compatible provider adapter.
- HorizonBench fixture, metric scorer, baseline/ablation smoke harness, real
  cross-process fault-injection test, and cross-domain workflow scorer.
- Frozen model × condition × seed manifests, atomic per-task result receipts,
  resumable matrix execution, and objective trace judging.
- A no-execution matrix preflight that validates task identity, frozen prompt /
  condition / decoding controls, credentials, and trace expectations before a
  model or runtime is invoked.

## Quick start — no API key required

**Requirements:** Rust stable with Cargo, Python 3.9+, and `make`. The core
demos and tests use only the standard Python library.

```bash
git clone <YOUR_REPOSITORY_URL> horizon
cd horizon

make test
make demo
```

The demo creates a goal, constraint, plan, failure-memory entry, and checkpoint;
then simulates recovery and renders the restored State Anchor.

### Run a durable local DAG

```bash
cargo run -p horizon-cli -- --db example.db run examples/research-workflow.yaml

# Replace RUN_ID with the value printed by the previous command.
cargo run -p horizon-cli -- --db example.db inspect RUN_ID
cargo run -p horizon-cli -- --db example.db events RUN_ID
cargo run -p horizon-cli -- --db example.db memory RUN_ID
cargo run -p horizon-cli -- --db example.db replay RUN_ID
```

The task files use argv arrays rather than interpolated shell strings. See the
[examples](examples/README.md), including recovery, bounded process, and
research-workflow fixtures.

### Run the Python policy layer

In one terminal, start a local runtime:

```bash
cargo run -p horizon-cli -- --db horizon.db serve
```

In another, run a deterministic no-key policy:

```bash
PYTHONPATH=python python3 examples/python_scripted_demo.py
```

To use an OpenAI-compatible endpoint, configure its key and endpoint in the
environment, then use `OpenAICompatibleProvider`. The provider receives only
validated decoding controls; the Rust runtime remains the durable source of
truth. See [Python integration](docs/python.md) for the full contract.

## Recovery model

Horizon deliberately implements **at-least-once**, not pretend exactly-once,
execution. If a process is interrupted while a task is running, recovery emits
durable interruption/failure events and requeues only when the retry policy
permits it. External systems must honor the stable operation ID for idempotency.

```bash
cargo run -p horizon-cli -- --db example.db checkpoint RUN_ID
cargo run -p horizon-cli -- --db example.db resume RUN_ID --execute
```

For shared storage, pass a PostgreSQL URL instead of `--db`. Operations,
migration, Docker, telemetry, and resource-limit details are documented in
[operations](docs/operations.md).

## Evaluation and research integrity

HorizonBench covers six long-horizon categories: goal retention, constraint
retention, failure avoidance, cross-session recovery, long-context degradation,
and fault recovery.

```bash
# Deterministic fixture and ablation plumbing — not an empirical LLM claim.
make bench

# Starts a real Rust HTTP runtime and exercises the durable-trace bridge using
# scripted fixture actions; it still makes no model API call.
make bench-trace

# Validates a frozen matrix and executor configuration without writing results,
# starting a runtime, or making a model request.
make bench-preflight
```

The bundled fixture and ablation output is explicitly labeled
`synthetic_deterministic_smoke`. It proves benchmark plumbing and runtime
mechanisms; it is **not** evidence that Horizon improves an LLM.

For an empirical run, Horizon requires an objective task environment and a
frozen matrix of task source, prompt, model/revision, decoding, seed, condition,
checkpoint cadence, and fault schedule:

```bash
python3 benchmarks/horizonbench/plan_matrix.py --help
python3 benchmarks/horizonbench/preflight_matrix.py --help
python3 benchmarks/horizonbench/execute_matrix.py --help
python3 benchmarks/horizonbench/score_matrix.py --help
```

`preflight_matrix.py` rejects executors without an explicit static
`execute.preflight(context)` hook. Built-in durable-trace preflight validates
provider configuration and credential presence without logging secret values.
Public artifact/domain tasks and non-empty fault schedules need a task-specific
environment executor that actually enacts and objectively judges those events.

Read the full [HorizonBench protocol](benchmarks/horizonbench/README.md),
[evaluation guide](docs/evaluation.md), and [roadmap](docs/roadmap.md) before
reporting results.

## Verification

Run the full local verification suite before submitting a change:

```bash
make fmt
make test
make test-python
make lint
make fault
make bench
make bench-trace
make bench-preflight
```

`make fault` starts the real Rust HTTP server, terminates it after durable work,
restarts against the same SQLite database, and verifies replay, failure memory,
State Anchor injection, retry requeueing, timeout handling, semantic retrieval,
and the Python policy boundary.

## Repository map

```text
crates/
  horizon-core/       state machine, commands, events, projection
  horizon-store/      SQLite/PostgreSQL event stores and snapshots
  horizon-memory/     State Anchor, decay policy, semantic retrieval
  horizon-scheduler/  Tokio DAG scheduler
  horizon-process/    process supervision and Docker backend
  horizon-trace/      post-commit tracing bridge
  horizon-runtime/    durable orchestration, recovery, HTTP API
  horizon-py/         optional PyO3 native runtime
  horizon-cli/        CLI
python/horizon_agent/  provider adapters, policy loop, HTTP client
benchmarks/            HorizonBench, scoring, matrix runners, fixtures
experiments/           ablation-report scaffold
docs/                  architecture, API, operations, evaluation, roadmap
```

## Design boundaries

Horizon is intentionally narrow. It does not attempt to replace browser agents,
MCP ecosystems, multi-agent orchestration, dashboards, Kubernetes, or a vector
database. New integrations must preserve the command → event → projection
boundary and must not let an LLM bypass durable state validation.

## Documentation

- [Architecture](docs/architecture.md)
- [Durability and recovery](docs/durability.md)
- [HTTP API](docs/http-api.md)
- [Python integration](docs/python.md)
- [Operations](docs/operations.md)
- [Evaluation protocol](docs/evaluation.md)
- [Roadmap](docs/roadmap.md)
- [Contributing](CONTRIBUTING.md)

## License

[MIT](LICENSE)

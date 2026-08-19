# Horizon

**A durable cognitive runtime for long-horizon AI agents.**

Horizon separates intelligence from reliability:

> Python chooses what an agent should do. Rust guarantees that execution state,
> decisions, failures, and recovery boundaries are durable and auditable.

Horizon persists structured cognitive state through an append-only event stream,
snapshots it periodically, and proactively injects a compact **State Anchor**
only when state-decay risk is high.

```text
Python policy / LLM
        │ JSON over localhost HTTP
        ▼
┌─────────────────────────────────────────────────────────┐
│ Rust Horizon Runtime                                     │
│                                                         │
│ Command → Validate → Event → EventStore → Apply Projection │
│                               │                         │
│                     Checkpoint + Replay                 │
│                               │                         │
│ Structured Cognitive State + State Anchor Policy        │
└─────────────────────────────────────────────────────────┘
        │
        ├── Tokio DAG scheduler
        └── Process supervisor (limits, timeout, structured results)
```

## Why Horizon?

Long context is not long-horizon reliability. A useful agent must retain its
goal, constraints, rejected approaches, open subgoals, key decisions, evidence,
and execution progress when context is pressured, tools fail, or the process
restarts. Horizon treats this as **execution-state management**, not only vector
retrieval.

Its three core mechanisms are:

1. **Durable Cognitive State** — structured goal, constraints, plan, subgoals,
   failures, decisions, evidence, environment, and budget.
2. **Event-Sourced Runtime** — immutable ordered events reconstruct a run for
   replay, audit, and crash recovery.
3. **Proactive State Intervention** — a transparent heuristic injects an LLM
   State Anchor only at risk boundaries, rather than on every turn.

## Current capabilities

- Rust state machine with validated legal transitions; `Created → Completed` is
  rejected.
- SQLite/WAL and PostgreSQL append-only event stores with per-run sequence numbers.
- Checksummed zstd snapshots, checkpoint compaction, and schema-versioned replay.
- Cross-process recovery of interrupted `Running` tasks under retry policy.
- Stable logical `operation_id` delivered to child processes as
  `HORIZON_OPERATION_ID` for at-least-once + idempotency integration.
- Tokio dependency-aware task DAG scheduler and process supervisor with output,
  CPU, memory, and optional Docker execution controls.
- Structured failure memory, decisions, evidence, budget, environment, and
  State Anchor risk policy.
- CLI, local JSON/HTTP API, Python client/policy layer, PyO3 native runtime,
  external task adapters, and an OpenAI-compatible provider adapter.
- Structured terminal tool results and optional OTLP/HTTP spans emitted only
  after durable event commit.
- Offline-trainable state-decay predictor and adaptive context-budget controller
  whose every decision is durably auditable.
- Source-attributed semantic-memory catalog with bounded, durable and
  deterministic retrieval; policies receive ranked hits rather than the full
  catalog.
- HorizonBench 30-task fixture, baseline/ablation smoke harness, metric scorer,
  and real cross-process fault-injection test.

## Quick start

Prerequisites: Rust stable (Cargo) and Python 3.9+. The included demos need no
third-party Python packages.

```bash
git clone <YOUR_FORK_URL> horizon
cd horizon

cargo test --workspace
cargo run -p horizon-cli -- --db horizon-demo.db demo
```

The no-key demo creates a goal, constraint, plan, failure memory, and checkpoint;
then simulates a restart and shows the recovered State Anchor.

## Run a durable local DAG

```bash
cargo run -p horizon-cli -- --db example.db run examples/research-workflow.yaml

# Copy the printed run id into these commands.
cargo run -p horizon-cli -- --db example.db inspect RUN_ID
cargo run -p horizon-cli -- --db example.db events RUN_ID
cargo run -p horizon-cli -- --db example.db memory RUN_ID
cargo run -p horizon-cli -- --db example.db replay RUN_ID
```

The task file uses argv arrays, not interpolated shell strings. See
[`examples/research-workflow.yaml`](examples/research-workflow.yaml) and
[`examples/fault-recovery.yaml`](examples/fault-recovery.yaml). For resource
limits, see [`examples/bounded-process.yaml`](examples/bounded-process.yaml).

## Operations

SQLite remains the default local-first store. For shared durable state, use a
PostgreSQL URL instead of `--db`:

```bash
cargo run -p horizon-cli -- \
  --database-url postgresql://horizon:secret@127.0.0.1:5432/horizon status
```

Snapshots are zstd-compressed and checksummed by default. Keep or compact only
snapshot blobs without touching immutable history:

```bash
cargo run -p horizon-cli -- --db example.db compact RUN_ID --retain-latest 8
```

Optional OTLP export is built explicitly and streams only already-committed
events:

```bash
cargo run -p horizon-cli --features otel -- \
  --otlp-endpoint http://127.0.0.1:4318 --db example.db demo
```

See [operations](docs/operations.md) for migration, Docker, resource-limit, and
telemetry behavior.

## Recover a run

```bash
cargo run -p horizon-cli -- --db example.db checkpoint RUN_ID
cargo run -p horizon-cli -- --db example.db resume RUN_ID --execute
```

When recovery finds a task left in `Running`, it emits durable interruption and
failure-memory events. If retry budget remains it becomes `Pending`; otherwise
it remains `Failed`. Horizon does not infer that an interrupted external
operation succeeded. It provides at-least-once execution: integrations must
honor the stable operation ID to be idempotent.

## Python policy layer

Start the local runtime:

```bash
cargo run -p horizon-cli -- --db horizon.db serve
```

Then run a no-key scripted policy:

```bash
PYTHONPATH=python python3 examples/python_scripted_demo.py
```

For an OpenAI-compatible endpoint:

```python
from horizon_agent import AgentConfig, DurableAgent, HorizonClient
from horizon_agent.providers import OpenAICompatibleProvider

agent = DurableAgent(
    HorizonClient(),
    OpenAICompatibleProvider(model="YOUR_MODEL"),
    config=AgentConfig(max_steps=20, token_budget=20_000),
)
result = agent.run(
    "Create a reproducible experiment plan",
    constraints=["Do not retry known non-retryable failures."],
)
print(result)
```

The LLM returns one JSON action at a time. Rust validates it and creates events;
Python never directly mutates agent state. See [Python](docs/python.md) and the
[HTTP API](docs/http-api.md).

## State Anchor protocol

Horizon represents behavioral state directly instead of reinjecting all history:

```text
STATE ANCHOR
Primary Goal: ...
Constraints: ...
Current Plan: ...
Open Subgoals: ...
Important Decisions: ...
Failed Approaches: ...
Evidence: ...
Budget: ...
```

The default interpretable risk function is:

```text
R = w_distance·L + w_context·C + w_switch·S + w_failure·F + w_recovery·X
```

It considers distance since the last anchor, context pressure, subgoal switching,
recent failures, and session recovery. A trigger produces a durable
`StateAnchorInjected` event, making interventions observable and ablatable.

## Evaluation

HorizonBench contains 30 microtasks across goal retention, constraint retention,
failure avoidance, cross-session recovery, long-context degradation, and fault
recovery.

```bash
# Deterministic CI smoke harness: not an empirical model claim.
python3 benchmarks/horizonbench/run.py --output-dir results/horizonbench

# Planned full/ablation report scaffold.
python3 experiments/run_ablation.py

# Score observed adapter/LLM episode JSONL.
python3 benchmarks/horizonbench/score_runs.py results/my-agent.jsonl
```

The scorer reports task success, goal retention, constraint violations,
repeated-failure rate, recovery success/distance, state consistency, token use,
tokens per success, and anchors injected. See [HorizonBench](benchmarks/horizonbench/README.md).

## Verification

```bash
make fmt
make test
make test-python
make fault
make bench
```

`make fault` starts the real Rust HTTP server, terminates it after a durable
`TaskStarted`, restarts it against the same SQLite file, and checks replay,
failure memory, State Anchor injection, retry requeueing, and a real process
timeout.

## Workspace

```text
crates/
  horizon-core/       state machine, command/event model, projection
  horizon-store/      SQLite/WAL event store and snapshots
  horizon-memory/     state-decay risk, State Anchor, and semantic retrieval policies
  horizon-scheduler/  Tokio DAG scheduling primitives
  horizon-process/    timeout-aware process supervision
  horizon-trace/      structured tracing bridge for durable events
  horizon-runtime/    durable orchestration, recovery, HTTP API
  horizon-py/         optional PyO3 native SQLite bindings
  horizon-cli/        local CLI
python/horizon_agent/ LLM providers, policy loop, HTTP client
benchmarks/           HorizonBench fixtures and scorer
experiments/          ablation-report scaffold
```

## Architecture decisions

- **SQLite first, PostgreSQL when needed:** one inspectable file remains the
  default; the same `EventStore` contract supports shared deployments.
- **HTTP plus PyO3:** HTTP stays the default integration contract. Native
  bindings expose the same command/projection JSON model locally.
- **At-least-once, not pretend exactly-once:** stable operation IDs and audit
  make effects safe only when integrations honor idempotency.
- **No vector database yet:** structured execution state is the first problem.
  Semantic retrieval remains future optional work.
- **No browser, multi-agent, full sandbox, Kubernetes, or dashboard:** those
  remain intentionally outside the durable-runtime thesis.

## Status and roadmap

Horizon is a durable runtime and evaluation foundation. Its research-control
baseline now includes an offline-trained intervention policy, adaptive context
budgets, and deterministic semantic-memory retrieval; planned extensions are
external benchmark adapters and cross-model/cross-domain experiments. See
[roadmap](docs/roadmap.md).

## License

MIT. See [LICENSE](LICENSE).

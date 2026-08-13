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
│ Command → Validate → Event → SQLite → Apply Projection │
│                               │                         │
│                     Checkpoint + Replay                 │
│                               │                         │
│ Structured Cognitive State + State Anchor Policy        │
└─────────────────────────────────────────────────────────┘
        │
        ├── Tokio DAG scheduler
        └── Process supervisor (timeout, stdout/stderr, retry metadata)
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

## Implemented v0.1

- Rust state machine with validated legal transitions; `Created → Completed` is
  rejected.
- SQLite/WAL append-only event store with per-run sequence numbers.
- Snapshot checkpoints and checkpoint-plus-suffix replay.
- Cross-process recovery of interrupted `Running` tasks under retry policy.
- Stable logical `operation_id` delivered to child processes as
  `HORIZON_OPERATION_ID` for at-least-once + idempotency integration.
- Tokio dependency-aware task DAG scheduler and timeout-aware process
  supervisor.
- Structured failure memory, decisions, evidence, budget, environment, and
  State Anchor risk policy.
- CLI, local JSON/HTTP API, dependency-free Python client/policy layer, and
  OpenAI-compatible provider adapter.
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
[`examples/fault-recovery.yaml`](examples/fault-recovery.yaml).

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
  horizon-memory/     state-decay risk and State Anchor policy
  horizon-scheduler/  Tokio DAG scheduling primitives
  horizon-process/    timeout-aware process supervision
  horizon-trace/      structured tracing bridge for durable events
  horizon-runtime/    durable orchestration, recovery, HTTP API
  horizon-cli/        local CLI
python/horizon_agent/ LLM providers, policy loop, HTTP client
benchmarks/           HorizonBench fixtures and scorer
experiments/          ablation-report scaffold
```

## Architecture decisions

- **SQLite first:** one inspectable file, no service dependency. The
  `EventStore` trait isolates a future PostgreSQL backend.
- **HTTP before PyO3:** Python iteration stays simple while runtime contracts
  stabilize. Native bindings are a future enhancement.
- **At-least-once, not pretend exactly-once:** stable operation IDs and audit
  make effects safe only when integrations honor idempotency.
- **No vector database in v0.1:** structured execution state is the first
  problem. Semantic retrieval is future optional work.
- **No browser, multi-agent, sandbox, Kubernetes, or dashboard in v0.1:** those
  are intentionally outside the durable-runtime thesis.

## Status and roadmap

v0.1 is a local-first durable runtime and evaluation foundation. Planned research
extensions include a learned intervention policy, adaptive context budgets,
semantic-memory retrieval, PyO3 bindings, external benchmark adapters, and
cross-model/cross-domain experiments. See [roadmap](docs/roadmap.md).

## License

MIT. See [LICENSE](LICENSE).

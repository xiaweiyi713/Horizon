# Architecture

## Responsibility boundary

Horizon does not try to become a browser agent, coding agent, or model gateway.
Its split is deliberate:

| Python research layer | Rust runtime layer |
|---|---|
| LLM provider and prompts | State-machine validation |
| Policy and action selection | Event append and projection replay |
| External task adapters | SQLite/PostgreSQL checkpoints and recovery |
| Memory scoring inputs | Tokio task DAG and process supervision |
| Benchmark orchestration | Telemetry and audit trail |

## Durable command path

```text
External policy / CLI
          │ RuntimeCommand
          ▼
  validate current projection
          │
          ▼
  construct EventKind
          │
          ▼
  append immutable event to EventStore
          │ assigns next per-run sequence
          ▼
  apply event to materialized RunProjection
          │
          ├── automatic checkpoint when cadence is reached
          └── return projection to caller
```

No public runtime method exposes mutable agent state. `RunProjection` is a
materialized view, never a competing source of truth.

## Event sourcing and snapshots

`horizon_events` stores every state-relevant event ordered per run.
`horizon_checkpoints` stores serialized `RunProjection` snapshots. Recovery:

```text
latest checkpoint
      │
      ├── deserialize projection at checkpoint sequence N
      └── load and apply event sequence N+1…latest
                          │
                          ▼
                    recovered projection
```

SQLite uses WAL mode. PostgreSQL is a drop-in `EventStore` alternative whose
per-run sequence row is locked inside the append transaction. A per-process
mutation gate serializes commands in one runtime, while each backend's atomic
expected-sequence check protects durable ordering across independent runtimes.

Runtime writes include the projection's expected sequence. If another process
commits first, SQLite rejects the stale command instead of allowing an event
whose `from` state no longer matches replay. The caller reloads and decides again.

When one command produces coupled events (for example `TaskFailed` and
`FailureRemembered`), Horizon commits them in a single SQLite transaction. A
crash cannot leave the event log with only half of that logical command.

Snapshots store explicit encoding/schema metadata, an uncompressed size, and a
checksum. Current snapshots use zstd-compressed JSON by default; legacy JSON
snapshots and older event payload schemas remain readable through explicit
decode/upcast boundaries. Snapshot compaction removes only redundant snapshots,
never immutable events.

After a successful event commit, `horizon-trace` emits a structured `tracing`
record containing the run ID, sequence, timestamp, and event type. Its optional
OTLP/HTTP feature additionally streams one trace span per committed event. The
immutable event store remains the source of truth; trace exporters are
operational views.

## Cognitive state

The projection includes a `CognitiveState`, separate from raw chat history:

```text
primary_goal
constraints
current_plan
open_subgoals / completed_subgoals
failed_attempts
decisions
evidence
environment
budget
```

The memory crate renders this into a compact State Anchor. Both full Anchors
and ordinary compact policy context label the task-owned steps as an approved
plan, then expose bounded counts for decisions, evidence, and remembered
failures before their detailed sections. A policy can therefore determine
progress without inferring it from prose. A State Anchor is persisted as an
event, so its trigger and contents are auditable.

## Scheduler and processes

`horizon-scheduler` validates a dependency DAG, selects tasks whose dependencies
have succeeded, orders by priority, and executes independent work under a Tokio
semaphore. It never mutates durable state; only `horizon-runtime` writes task
events. Tasks may also retain an optional parent task ID for hierarchical plan
provenance; readiness is still defined solely by explicit dependencies.

`horizon-process` receives argv vectors, a working directory, environment,
timeout, operation ID, resource limits, and an explicit local/Docker backend.
It captures stdout/stderr with bounded retention, records truncation alongside
exit/timeout metadata, and kills timed-out children. Docker defaults to no
network and a read-only root filesystem; neither executor mode is a complete
security sandbox.

Python-owned external systems use `AdapterRegistry`. The registry writes
`ToolInvoked` before a side effect and one normalized `ToolResultRecorded` after
it, using the same stable operation ID as recovery/idempotency logic. This keeps
adapters outside Rust's policy boundary without losing durable audit semantics.
The bounded `ArtifactWorkspaceAdapter` additionally stores a task-owned
operation receipt beside its verifier workspace, so a restarted Python process
can return the prior terminal observation for the same operation ID without
rewriting the artifact. It can expose only declared immutable template files as
a bounded durable `artifact_workspace_read_v1` context; the policy renders that
context as explicitly untrusted task data while all other external tool output
remains hidden. It is an exact artifact-verification primitive, not a general
filesystem or process sandbox.

For objective artifact experiments, the task executor can enact exactly one
frozen `policy_restart` after a declared number of model calls. It deliberately
reconstructs the Python provider/client/workspace/adapter boundary and calls
`run_existing(..., recovered_session=True)` against the same Rust run; durable
LLM policy operation IDs continue from the prior terminal result. This is a
policy-worker recovery control, not a substitute for crashing or recovering the
Rust runtime process itself. If a later failed policy result would otherwise
hide a prior verified artifact result, compact context retains only a safe
verified-status/operation-ID summary, never candidate artifact contents.

## Learned intervention boundary

`horizon_agent.memory` can train a small offline logistic state-decay predictor
and combine its risk score with an adaptive context-budget controller. The
controller runs in Python because it is experimental policy logic. It sends a
typed `InterventionAssessment` to Rust, which validates score/threshold ranges,
persists `StateDecayAssessed`, and renders the State Anchor from the current
projection only when the requested action is valid. This records both action and
inaction without granting a model mutable access to cognitive state.

## Semantic-memory retrieval boundary

`memory_created` events build a separately indexed catalog of source-attributed
`MemoryItem` records. A policy does not receive that catalog wholesale. It
submits a compact query and a bounded limit through `RetrieveSemanticMemory`;
Rust runs the deterministic local `hybrid_lexical_v1` ranker against the
replayed projection, then persists `SemanticMemoryRetrieved` with the exact
query, algorithm, copied hit content, kinds, and integer scores. This makes a
retrieval result durable evidence of what was presented at that boundary, not
an opaque cache or mutable vector-store response.

The v0.4 ranker is dependency-free and deliberately local: normalized token
overlap plus character-trigram similarity, with importance/confidence only as
secondary quality weighting. It is a reproducible semantic-memory baseline, not
a claim that Horizon ships a neural embedding service. Python appends only the
returned bounded hits to the next policy context and reloads the returned
projection before any subsequent intervention decision.

## Python bindings

HTTP remains the default integration contract. The optional `horizon_native`
PyO3 module exposes the same JSON command/projection boundary directly against
a local SQLite runtime. Its Python wrapper implements the same client methods,
so policy code can choose HTTP or native mode without duplicating validation or
projection behavior.

## Idempotency model

Horizon provides **at-least-once execution**. A task's stable `operation_id` is
reused through retries and passed to child processes as `HORIZON_OPERATION_ID`.
An external integration should make that identifier its idempotency key. Horizon
does not claim distributed exactly-once behavior.

## Deliberate non-goals

- Distributed workflow coordination beyond the PostgreSQL event backend
- Complete container sandboxing and orchestration
- Browser and GUI automation
- Multi-agent orchestration

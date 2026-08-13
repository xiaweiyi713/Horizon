# Architecture

## Responsibility boundary

Horizon does not try to become a browser agent, coding agent, or model gateway.
Its split is deliberate:

| Python research layer | Rust runtime layer |
|---|---|
| LLM provider and prompts | State-machine validation |
| Policy and action selection | Event append and projection replay |
| Experiment adapters | SQLite checkpoints and recovery |
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
  append immutable event to SQLite
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

Events use SQLite WAL mode. A per-process mutation gate serializes runtime
commands; an atomic per-run SQLite sequence table protects durable ordering
across independent runtime instances.

Runtime writes include the projection's expected sequence. If another process
commits first, SQLite rejects the stale command instead of allowing an event
whose `from` state no longer matches replay. The caller reloads and decides again.

When one command produces coupled events (for example `TaskFailed` and
`FailureRemembered`), Horizon commits them in a single SQLite transaction. A
crash cannot leave the event log with only half of that logical command.

After a successful event commit, `horizon-trace` emits a structured `tracing`
record containing the run ID, sequence, timestamp, and event type. The immutable
event store remains the source of truth; trace exporters are operational views.

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

The memory crate renders this into a compact State Anchor. A State Anchor is
persisted as an event, so its trigger and contents are auditable.

## Scheduler and processes

`horizon-scheduler` validates a dependency DAG, selects tasks whose dependencies
have succeeded, orders by priority, and executes independent work under a Tokio
semaphore. It never mutates durable state; only `horizon-runtime` writes task
events.

`horizon-process` receives argv vectors, a working directory, environment,
timeout, and operation ID. It captures stdout/stderr, records an exit code or
timeout, and kills timed-out children. It is not a sandbox.

## Idempotency model

Horizon provides **at-least-once execution**. A task's stable `operation_id` is
reused through retries and passed to child processes as `HORIZON_OPERATION_ID`.
An external integration should make that identifier its idempotency key. Horizon
does not claim distributed exactly-once behavior.

## Deliberate non-goals in v0.1

- Distributed execution / PostgreSQL
- Container sandboxing and resource cgroups
- Browser and GUI automation
- Multi-agent orchestration
- Vector database / semantic-memory retrieval
- Learned intervention policy
- PyO3 bridge (HTTP is used until runtime contracts stabilize)

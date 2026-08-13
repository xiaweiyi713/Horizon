# Durability and recovery semantics

## Guarantees

For state mutations that reach the runtime:

1. The runtime validates the command against the current projection.
2. It appends an ordered immutable event to SQLite before applying that event.
3. A projection can be reconstructed from a snapshot plus its event suffix.
4. Checkpoint metadata and snapshot sequence must agree; malformed checkpoint
   state is rejected rather than silently used.
5. A command that produces multiple causally coupled events is persisted as one
   SQLite batch; task failure and failure memory are not split by a crash.
6. Runtime writes use optimistic expected-sequence checks, so a stale
   cross-process command is rejected rather than making replay inconsistent.

## Crash during a task

The critical boundary is a task left in `Running`:

```text
TaskStarted(operation_id = stable-key)
               │
               ├── process/runtime crash
               │
               ▼
restart → Recovering → AgentRecovered
               │
               ├── TaskFailed("interrupted by runtime recovery")
               ├── FailureRemembered(stable-key)
               └── TaskRetried → Pending, if retry budget remains
```

The runtime never infers that an external operation succeeded merely because it
was running. If a tool can query or deduplicate its stable key, a resumed
executor can safely mark it as idempotently completed; otherwise it follows the
retry policy.

Operation IDs are unique within a run. Horizon rejects a second task that tries
to reuse another task's stable key, preventing accidental idempotency collisions.

## Checkpoint cadence

`--checkpoint-every N` creates a snapshot after every `N` domain events since
the last checkpoint. A checkpoint event is part of the immutable log and the
snapshot includes that event's sequence. Use `0` to disable automatic snapshots
and issue `horizon checkpoint RUN_ID` manually.

## Operational notes

- SQLite is a local single-file backend in v0.1. Back up `.db`, `-wal`, and
  `-shm` together while active, or use SQLite's online backup API.
- The HTTP server is intended for localhost and has no authentication. Use an
  authenticated reverse proxy before exposing it remotely.
- The process supervisor is not a sandbox. Tasks run with the runtime user's
  permissions; use trusted task specifications only.

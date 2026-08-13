# Durability and recovery semantics

## Guarantees

For state mutations that reach the runtime:

1. The runtime validates the command against the current projection.
2. It appends an ordered immutable event to the configured `EventStore` before
   applying that event.
3. A projection can be reconstructed from a snapshot plus its event suffix.
4. Checkpoint metadata and snapshot sequence must agree; malformed checkpoint
   state is rejected rather than silently used.
5. A command that produces multiple causally coupled events is persisted as one
   SQLite batch; task failure and failure memory are not split by a crash.
6. Runtime writes use optimistic expected-sequence checks, so a stale
   cross-process command is rejected rather than making replay inconsistent.
7. Snapshot decoding verifies encoding/schema metadata and a checksum when one
   is present; corrupt checkpoints are rejected rather than silently used.

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

Snapshots use zstd-compressed JSON by default. A snapshot's encoding is stored
with the blob, so an operator may switch new writes to `--snapshot-encoding json`
without invalidating older snapshots. `horizon compact RUN_ID --retain-latest N`
deletes only older snapshot blobs; the complete event stream remains available
for replay and audit.

## Schema evolution

Database migrations are additive and recorded in `horizon_schema_migrations`.
Every event carries a payload schema version and is decoded through a versioned
upcast boundary. This permits future releases to understand older immutable
events without rewriting historical records. PostgreSQL uses the same schema and
event contract as SQLite, including expected-sequence conflict detection.

The current event vocabulary is schema v2: v1 histories are still decoded
through an explicit identity upcast, while v2 adds durable
`state_decay_assessed` records for learned-policy research. Snapshot projection
fields for those assessments use Serde defaults, so v0.2 snapshots remain
readable and replay their v1 event suffix correctly.

## Operational notes

- SQLite is a local single-file backend. Back up `.db`, `-wal`, and
  `-shm` together while active, or use SQLite's online backup API.
- PostgreSQL is selected through `--database-url`; back it up with your normal
  database operations process. Horizon never uses it to delete event history.
- The HTTP server is intended for localhost and has no authentication. Use an
  authenticated reverse proxy before exposing it remotely.
- Process resource controls cap output everywhere and use host/Docker limits
  where supported. They are not a security sandbox; tasks run with the runtime
  user's permissions or the Docker daemon's policy, so use trusted specs only.

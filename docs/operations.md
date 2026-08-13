# Operations and deployment

## Storage backends

Horizon uses the same `EventStore` contract for both supported stores:

- **SQLite** is the default local-first backend. It uses WAL mode and stores
  everything in one inspectable database file.
- **PostgreSQL** is selected with `--database-url`; it is appropriate when
  multiple Horizon processes need one durable run history.

```bash
horizon --database-url postgresql://horizon:secret@db:5432/horizon status
```

PostgreSQL assigns per-run sequence numbers under a row lock and retains the
same optimistic expected-sequence behavior as SQLite. Both stores apply
additive schema migrations and track them in `horizon_schema_migrations`.
Set `HORIZON_POSTGRES_URL` when running the opt-in PostgreSQL integration test:

```bash
HORIZON_POSTGRES_URL=postgresql://horizon:secret@127.0.0.1:5432/horizon \
  cargo test -p horizon-store round_trips_against_configured_postgres
```

## Checkpoints, schemas, and compaction

New snapshots are JSON compressed with zstd by default and carry a snapshot
schema version, uncompressed size, and BLAKE3 checksum. Existing JSON snapshots
remain readable. Event payloads also carry their own schema version, so a future
runtime can upcast old immutable event payloads without rewriting audit history.

```bash
# Use plain JSON snapshots only when inspectability outweighs storage cost.
horizon --snapshot-encoding json --retain-checkpoints 0 run task.yaml

# Keep only the eight newest snapshots for one run. Events are never deleted.
horizon compact RUN_ID --retain-latest 8
```

`--retain-checkpoints 0` keeps every automatic snapshot. `compact` removes only
older snapshot blobs; replay and audit always retain the complete event log.

## Process resource controls

Each process task can declare output, CPU, and memory limits alongside an
explicit execution backend:

```yaml
tasks:
  - name: bounded-report
    command: ["sh", "-c", "python report.py"]
    timeout_ms: 300000
    resources:
      max_output_bytes: 1048576
      max_cpu_time_ms: 120000
      max_memory_bytes: 1073741824
    executor:
      kind: docker
      image: python:3.12-slim
      allow_network: false
      read_only: true
```

`max_output_bytes` is enforced separately while stdout and stderr are drained,
and records explicit truncation flags in the durable tool result. On Unix, local CPU limits use
`setrlimit`; Linux/other compatible Unix hosts also receive an address-space
memory limit. macOS refuses local memory limits rather than claiming to enforce
them in a restricted child process—use Docker there. Docker runs with networking
disabled and a read-only root filesystem by default, translates memory/CPU
limits to Docker flags, and mounts a requested working directory at
`/workspace`.

These controls reduce accidental resource use; neither local execution nor the
Docker mode is presented as a complete security sandbox. Run only trusted task
specifications and add deployment-specific isolation for untrusted code.

## OTLP trace export

The default binary emits local `tracing` records only. Build the optional OTLP
feature to stream one trace span for each event *after it commits*:

```bash
cargo run -p horizon-cli --features otel -- \
  --otlp-endpoint http://127.0.0.1:4318 \
  --otel-service-name horizon-demo \
  --db horizon.db demo
```

The endpoint is an OTLP/HTTP collector base URL; the exporter uses HTTP protobuf
and the trace path. A collector outage never changes the EventStore guarantee:
the immutable event is still the source of truth, while telemetry is an
operational view. See the [OpenTelemetry Rust exporter guide](https://opentelemetry.io/docs/languages/rust/exporters/)
for collector configuration.

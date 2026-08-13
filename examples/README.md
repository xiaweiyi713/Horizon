# Examples

```bash
# Execute a four-node DAG in local SQLite.
cargo run -p horizon-cli -- --db example.db run examples/research-workflow.yaml

# Inspect the resulting durable state and immutable event history.
cargo run -p horizon-cli -- --db example.db inspect RUN_ID
cargo run -p horizon-cli -- --db example.db events RUN_ID
```

`fault-recovery.yaml` demonstrates an explicitly stable `operation_id` passed to
the child process as `HORIZON_OPERATION_ID`. See `tests/fault_injection.py` for
a real cross-process restart test.

`bounded-process.yaml` demonstrates task-level `max_output_bytes` and a local
CPU budget. See [operations](../docs/operations.md) for Docker and memory-limit
semantics.

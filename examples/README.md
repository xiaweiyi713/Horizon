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

`state-decay-labels.jsonl` is a deliberately tiny **illustrative** label file
for the offline state-decay trainer. It validates the model artifact workflow;
it is not an empirical training or benchmark dataset. See
[Python research controls](../docs/python.md#learned-state-decay-policy) for
the required run/model-disjoint evaluation protocol.

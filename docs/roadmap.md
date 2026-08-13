# Roadmap

## v0.1 — implemented

- Event-sourced local runtime, SQLite checkpoints, and replay
- Agent/task state machines and Tokio DAG primitives
- Process timeout, retry metadata, stable operation IDs
- Structured cognitive state and heuristic proactive State Anchor
- CLI, localhost HTTP API, Python policy/client layer
- HorizonBench fixture/scorer, ablation scaffold, fault injection

## v0.2 — implemented

- PostgreSQL event store behind `EventStore`, with an opt-in integration test
- Snapshot compaction, zstd/checksummed snapshots, and event-schema boundaries
- Process resource limits and an explicit Docker execution backend
- Feature-gated OTLP/HTTP trace export after durable event commit
- PyO3 native SQLite runtime module alongside the stable HTTP contract
- Python external task adapter registry and normalized tool-result schemas

## v0.3 — implemented research-control baseline

- Offline-trainable logistic state-decay predictor with serializable feature schema
- Adaptive context-budget controller that accounts for immediate anchor cost
- Durable `state_decay_assessed` events for both skipped and injected decisions
- HTTP/native/Python policy integration plus held-out calibration metrics
- HorizonBench learned-adaptive versus fixed-threshold research controls

## Research version

- Semantic-memory layer and retrieval ablations
- Cross-model evaluation and held-out public benchmark adapters
- Cross-domain long-horizon workflows and a technical report

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

## Research version

- Learned state-decay predictor and learned intervention policy
- Adaptive context-budget controller
- Semantic-memory layer and retrieval ablations
- Cross-model evaluation and held-out public benchmark adapters
- Cross-domain long-horizon workflows and a technical report

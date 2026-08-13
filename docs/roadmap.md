# Roadmap

## v0.1 — implemented

- Event-sourced local runtime, SQLite checkpoints, and replay
- Agent/task state machines and Tokio DAG primitives
- Process timeout, retry metadata, stable operation IDs
- Structured cognitive state and heuristic proactive State Anchor
- CLI, localhost HTTP API, Python policy/client layer
- HorizonBench fixture/scorer, ablation scaffold, fault injection

## v0.2 — next engineering work

- PostgreSQL event store behind `EventStore`
- Snapshot compaction and event-schema migration tooling
- Process resource limits / optional container executor
- Streaming trace export via OpenTelemetry
- PyO3 native module after HTTP contract stabilizes
- External task adapters and richer tool-result schemas

## Research version

- Learned state-decay predictor and learned intervention policy
- Adaptive context-budget controller
- Semantic-memory layer and retrieval ablations
- Cross-model evaluation and held-out public benchmark adapters
- Cross-domain long-horizon workflows and a technical report

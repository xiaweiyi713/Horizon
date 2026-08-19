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

## v0.4 — implemented semantic-memory baseline

- Source-attributed durable semantic-memory catalog in `RunProjection`
- Deterministic bounded `hybrid_lexical_v1` retrieval with replayable hits
- Python compact-context integration and retrieval-off configuration control
- Retrieval audit protocol and schema-v3 compatibility boundaries

## v0.5 — implemented held-out cross-model evaluation protocol

- Strict dependency-free JSONL adapter for local mirrors of public benchmark splits
- Source-byte and normalized selected-task fingerprints, with duplicate/split validation
- Self-validating run manifests that freeze task IDs, prompt, model/provider revision,
  decoding, seed, policy condition, runtime controls, and fault schedule
- Cross-model matrix planner plus manifest-bound per-run and aggregate scorers

## v0.6 — implemented cross-domain workflow evaluation tooling

- Four-domain, 12-task held-out long-horizon workflow fixture with explicit
  prerequisites, recovery boundaries, evidence requirements, and immutable audit identity
- Manifest-bound cross-domain scorer for workflow completion and dependency integrity
- Deterministic Markdown technical-report generator with provenance and interpretation caveats

## v0.7 — implemented resumable experiment execution

- Executor contract binding each observed episode outcome to one frozen manifest/task attempt
- Matrix execution CLI with dynamic local executor loading and exact task-source identity checks
- Atomic per-task outcome persistence plus manifest/executor-bound sidecar receipts
- Safe resume after a failed task, with rejected unreceipted or incompatible result artifacts
- Synthetic fixture executor for plumbing tests only; it cannot be used as an empirical model claim

## v0.8 — implemented model-backed durable-trace execution

- `DurableAgent` bridge from a frozen matrix task to a separately objective judge
- Frozen manifest prompt and checkpoint cadence made effective in the agent loop
- Strict durable-trace expectation judge for runtime-mechanism tasks, with no model self-report scoring
- OpenAI-compatible opt-in executor that forwards validated decoding controls and matrix seed
- Explicit rejection of fault schedules that the generic bridge cannot actually enact

## v0.9 — implemented end-to-end durable-trace integration fixture

- Three immutable scripted durable-trace tasks plus model/condition/prompt fixtures
- No-network executor that deeply detaches frozen task metadata before policy serialization
- Real Rust HTTP runtime matrix integration test covering agent actions, receipts, and objective trace scoring
- `make bench-trace` reproducibility entry point for the complete bridge

## v1.0 — implemented execution preflight

- Separate `preflight_matrix.py` command that verifies frozen matrix/task
  identity and performs no execution, result writing, or receipt creation
- Explicit JSON-safe `execute.preflight(context)` executor protocol, so a
  matrix is rejected before the normal executor, provider, runtime, or task
  environment can be invoked
- Static durable-trace validation for task expectations, agent controls,
  checkpoint cadence, model profile, decoding/seed, endpoint configuration,
  and credential presence without exposing credential values
- No-network scripted-fixture CLI regression and `make bench-preflight`
  reproducibility entry point

## Research execution

- Configure task-environment executors for selected model backbones, add their
  static preflight hooks, collect objective held-out outcomes, and publish the
  resulting technical report. Public artifact/domain tasks and fault schedules
  still require their own environment-specific judge.

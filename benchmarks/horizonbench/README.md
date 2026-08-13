# HorizonBench

HorizonBench v0.1 is a 30-task microbenchmark for durable execution state:

| Category | Tasks | Primary signal |
|---|---:|---|
| Goal retention | 5 | Original objective survives distracting work |
| Constraint retention | 5 | Long-lived guardrails remain behavioral |
| Failure avoidance | 5 | Rejected approaches are not repeated |
| Cross-session recovery | 5 | Goal, progress, and decisions restore after restart |
| Long-context degradation | 5 | Essential state survives a long trajectory |
| Fault recovery | 5 | Checkpoint/replay and idempotency survive faults |

Each agent adapter produces one JSONL row per task. The required fields are
`task_id`, `category`, `success`, `goal_retained`, and
`constraint_violations`; optional fields capture failures, recovery, tokens,
and anchors. Score a real run with:

```bash
python3 benchmarks/horizonbench/score_runs.py results/my-agent.jsonl
```

For CI and smoke testing, the included deterministic reference harness exposes
B0–B3, the always-on-anchor control, Full Horizon, and four ablations:

```bash
python3 benchmarks/horizonbench/run.py --output-dir results/horizonbench
```

Those outputs are explicitly marked `synthetic_deterministic_smoke`. They test
benchmark mechanics and must not be reported as empirical LLM results.

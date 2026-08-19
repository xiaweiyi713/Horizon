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
B0–B3, the always-on-anchor control, heuristic Full Horizon, the durable
semantic-memory retrieval on/off pair, a fixed fixture model behind the real
learned/adaptive controller, its fixed-threshold control, and four mechanism
ablations:

```bash
python3 benchmarks/horizonbench/run.py --output-dir results/horizonbench
```

Those outputs are explicitly marked `synthetic_deterministic_smoke`. They test
benchmark mechanics and must not be reported as empirical LLM results. The
fixture model is not trained on HorizonBench; real learned-policy studies must
fit a separate artifact on run/model-disjoint boundary labels.

The semantic-memory pair is also a synthetic control. It models the same
State-Anchor behavior with or without a relevant durable retrieval result; it
does not claim empirical embedding or model quality. Real experiments should
retain `memory_created` and `semantic_memory_retrieved` events, score a frozen
relevant-memory set, and compare it to `semantic_memory_limit=0`.

## Held-out and cross-model runs

For an empirical run, do not reuse the synthetic `tasks.json` fixture as a
public benchmark result. Mirror the benchmark source locally as UTF-8 JSONL and
give every row these required fields:

```json
{"id":"public-001","category":"goal_retention","goal":"...","constraint":"...","split":"development"}
{"id":"heldout-001","category":"fault_recovery","goal":"...","constraint":"...","split":"heldout","metadata":{"license":"..."}}
```

Extra JSON-safe fields are retained as metadata. `JsonlTaskAdapter` rejects
blank/malformed rows, duplicate IDs even across splits, and requests for an
empty split. It records both the raw source SHA-256 and a normalized selected
task-set SHA-256, so ordinary whitespace or record ordering cannot change a
frozen held-out set.

Create `models.json` with `model_id`, `provider`, `model`, `model_revision`,
and a `decoding` object; create `conditions.json` with `condition_id`,
`policy_revision`, and a `configuration` object. Then plan, run, and score the
matrix:

```bash
python3 benchmarks/horizonbench/plan_matrix.py \
  --tasks data/public-benchmark.jsonl \
  --source-name PUBLIC_BENCHMARK --source-revision IMMUTABLE_RELEASE \
  --models configs/models.json --conditions configs/conditions.json \
  --prompt prompts/system.txt --prompt-revision PROMPT_REVISION \
  --runtime-revision GIT_REVISION --seeds 1,2,3 \
  --output results/matrix.jsonl

# A provider runner writes results/<run_id>.jsonl, one EpisodeResult per task.
python3 benchmarks/horizonbench/score_matrix.py \
  --matrix results/matrix.jsonl --results-dir results --output results/report.json
```

`score_matrix.py` refuses incomplete rows and refuses a matrix whose task set,
prompt content, runtime revision, checkpoint cadence, or fault schedule differs
between entries. To score a single result against a selected matrix row, use
`score_runs.py RESULT.jsonl --manifest-matrix results/matrix.jsonl --run-id RUN_ID`.
The profile and manifest schemas reject unrecognized top-level fields: put
provider decoding controls in `decoding` and policy-specific controls in
`configuration` so every effective setting is hashed.

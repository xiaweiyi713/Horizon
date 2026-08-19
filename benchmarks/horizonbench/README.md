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
`policy_revision`, and a `configuration` object. Then plan, execute, and score
the matrix:

```bash
python3 benchmarks/horizonbench/plan_matrix.py \
  --tasks data/public-benchmark.jsonl \
  --source-name PUBLIC_BENCHMARK --source-revision IMMUTABLE_RELEASE \
  --models configs/models.json --conditions configs/conditions.json \
  --prompt prompts/system.txt --prompt-revision PROMPT_REVISION \
  --runtime-revision GIT_REVISION --seeds 1,2,3 \
  --output results/matrix.jsonl

# The executor receives a frozen manifest/task context and must return an
# environment-observed EpisodeResult, not an LLM self-assessment.
python3 benchmarks/horizonbench/execute_matrix.py \
  --matrix results/matrix.jsonl --tasks data/public-benchmark.jsonl \
  --source-name PUBLIC_BENCHMARK --source-revision IMMUTABLE_RELEASE \
  --executor my_experiment.executor:execute --results-dir results

python3 benchmarks/horizonbench/score_matrix.py \
  --matrix results/matrix.jsonl --results-dir results --output results/report.json
```

An executor is a local callable with this narrow boundary:

```python
from benchmarks.horizonbench import EpisodeResult

def execute(context):
    # Construct the configured model/runtime and run one isolated task
    # environment. `observation` must come from its verifier, tests, artifacts,
    # or other objective task boundary—not from the model claiming success.
    observation = run_task_environment(
        task=context.task,
        prompt=context.manifest.prompt.text,
        model=context.manifest.model,
        condition=context.manifest.condition,
        attempt=context.attempt,
    )
    return EpisodeResult(
        task_id=context.task.task_id,
        category=context.task.category,
        success=observation.success,
        goal_retained=observation.goal_retained,
        constraint_violations=observation.constraint_violations,
        failure_actions=observation.failure_actions,
        known_failed_approaches=observation.known_failed_approaches,
        recovery_attempted=observation.recovery_attempted,
        recovery_succeeded=observation.recovery_succeeded,
        recovery_distance=observation.recovery_distance,
        state_consistent=observation.state_consistent,
        tokens=observation.tokens,
        anchors_injected=observation.anchors_injected,
    )
```

`execute_matrix.py` writes `<run_id>.jsonl` after every task and a matching
`<run_id>.receipt.json` sidecar. It resumes only those receipt-bound outputs;
the receipt binds the result set to the manifest hash, selected task-set hash,
and executor import path. It rejects changed task sources, mismatched executors,
duplicate/unexpected outcomes, and result files without a receipt. Pass
`--overwrite` only to intentionally replace a completed or interrupted run.

For a no-network smoke test of the execution plumbing, use
`benchmarks.horizonbench.synthetic_fixture_executor:execute`. It returns
synthetic all-success outcomes and is explicitly **not** an empirical model
executor.

`score_matrix.py` refuses incomplete rows and refuses a matrix whose task set,
prompt content, runtime revision, checkpoint cadence, or fault schedule differs
between entries. To score a single result against a selected matrix row, use
`score_runs.py RESULT.jsonl --manifest-matrix results/matrix.jsonl --run-id RUN_ID`.
The profile and manifest schemas reject unrecognized top-level fields: put
provider decoding controls in `decoding` and policy-specific controls in
`configuration` so every effective setting is hashed.

## Cross-domain long-horizon workflow suite

`cross_domain_tasks.jsonl` is a deterministic held-out fixture for the scoring
path, not an empirical benchmark result. It covers four three-step workflows:
software engineering event replay, data-analysis OOS audit, research evidence
synthesis, and operations migration recovery. Each workflow names explicit
prerequisites, at least one recovery boundary, evidence-required steps, and a
minimum expected boundary count of 12.

Plan the normal matrix against this task source, execute it with an
environment-observing executor, then produce an auditable domain score and a
Markdown technical report:

```bash
python3 benchmarks/horizonbench/plan_matrix.py \
  --tasks benchmarks/horizonbench/cross_domain_tasks.jsonl \
  --source-name horizon-cross-domain-fixture --source-revision v1 \
  --models configs/models.json --conditions configs/conditions.json \
  --prompt prompts/system.txt --prompt-revision PROMPT_REVISION \
  --runtime-revision GIT_REVISION --seeds 1,2,3 \
  --output results/cross-domain-matrix.jsonl

python3 benchmarks/horizonbench/execute_matrix.py \
  --matrix results/cross-domain-matrix.jsonl \
  --tasks benchmarks/horizonbench/cross_domain_tasks.jsonl \
  --source-name horizon-cross-domain-fixture --source-revision v1 \
  --executor my_experiment.executor:execute --results-dir results

python3 benchmarks/horizonbench/cross_domain_score.py \
  --matrix results/cross-domain-matrix.jsonl --results-dir results \
  --output results/cross-domain-score.json

python3 benchmarks/horizonbench/render_report.py results/cross-domain-score.json \
  --output results/cross-domain-report.md
```

The cross-domain scorer rejects a workflow source whose selected task identity
does not exactly match the matrix. It reports both standard task metrics and
workflow completion, prerequisite integrity, recovery-boundary task success,
and evidence-task success. The generated report deliberately labels outcomes
as supplied observations; retain the raw JSONL, manifests, durable events, and
external-source license/provenance alongside it.

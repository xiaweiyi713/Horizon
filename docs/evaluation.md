# Evaluation protocol

## Core metrics

HorizonBench implements these metrics:

| Metric | Definition |
|---|---|
| Task success rate | Successful tasks / all tasks |
| Goal retention | Tasks retaining original goal / all tasks |
| Constraint violation rate | Violations / task |
| Repeated failure rate | Repeated known failed actions / all failed actions |
| Recovery success | Successful recoveries / recovery attempts |
| Recovery distance | Extra steps to regain pre-fault progress |
| State consistency | Recovery agrees on goal, plan, failures, progress |
| Token consumption | Prompt + completion tokens reported by provider |
| Tokens per successful task | Total tokens / successful tasks |

## Conditions

| ID | Condition |
|---|---|
| B0 | Raw context / truncation |
| B1 | Periodic summary |
| B2 | Vector retrieval |
| B3 | Structured passive memory |
| H | Proactive durable cognitive state |
| S | Durable semantic-memory retrieval (`hybrid_lexical_v1`) |
| S-off | Same policy with `semantic_memory_limit=0` |
| L | Learned state-decay predictor + adaptive context-budget controller |
| L-fixed | Same learned predictor with a fixed intervention threshold |

Include `Horizon_always_on_anchor` as a control for the token/performance
tradeoff. Compare `L` and `L-fixed` on the same frozen model artifact to isolate
adaptive threshold/cost control. The planned ablations are State Anchor, failure
memory, checkpoint recovery, and structured cognitive state. Add `S` versus
`S-off` on the same frozen model, prompts, and memory catalog to isolate the
behavioral value and token cost of retrieval.

## Reproducibility record

Every model-backed run should save:

- task fixture hash and task IDs;
- model/provider version and decoding settings;
- system prompt and policy revision;
- random seed where supported;
- checkpoint cadence and anchor-policy weights;
- fault schedule and expected recovery behavior;
- raw JSONL episode outcomes plus aggregate score.

The built-in synthetic smoke suite validates this data path but is not an
empirical LLM result.

## Held-out cross-model protocol

Use `benchmarks/horizonbench/adapters.py` to load a local UTF-8 JSONL mirror of
a public benchmark. A task record must have nonempty `id`, `category`, `goal`,
`constraint`, and `split` fields. Keep development and held-out rows in the
same immutable source when possible: the adapter rejects duplicate identifiers
across the whole source, fingerprints the raw bytes, and fingerprints the
normalized selected split independently of whitespace and input ordering.

Before invoking a model, use `plan_matrix.py` to produce a manifest row for
every model × condition × seed. Each row includes the selected task IDs and
hashes, complete prompt text/hash and revision, provider/model revision,
decoding controls, seed (or explicit `null` when unsupported), policy revision,
checkpoint cadence, fault schedule, runtime revision, and optional JSON-safe
metadata such as a frozen memory-catalog hash. The manifest itself has a
content SHA-256, so a changed setting cannot silently reuse an old run ID.

Provider adapters remain outside Horizon; they must emit exactly one observed
`EpisodeResult` per frozen task into `results/<run_id>.jsonl`. Score a single
row with `score_runs.py --manifest-matrix MATRIX --run-id RUN_ID`, or score the
entire matrix with `score_matrix.py`. The aggregate scorer verifies that every
row shares the same held-out task set, prompt content, runtime revision,
checkpoint cadence, and fault schedule before calculating per-run metrics and
mean/sample-standard-deviation summaries by model and condition. A rejected
check is a protocol failure, not a missing metric.

For durable-runtime mechanism tasks, `DurableAgentEpisodeExecutor` provides a
model-backed bridge that actually supplies the frozen prompt to the policy and
applies the frozen checkpoint cadence. The OpenAI-compatible trace plugin also
forwards validated manifest decoding controls and the matrix seed. Its
`DurableTraceJudge` verifies immutable event/projection facts, not an LLM
self-assessment. It rejects non-empty fault schedules: a public task must use
an environment executor that performs the declared fault and judges the
resulting artifact/test outcome objectively.

`make bench-trace` is the no-network end-to-end regression for this bridge. It
starts the actual Rust HTTP runtime and uses a frozen scripted fixture to prove
the matrix → agent → receipt → objective-judge path. Its outcomes are CI
coverage only, never empirical model measurements.

## Cross-domain long-horizon workflow protocol

The bundled `cross_domain_tasks.jsonl` fixture exercises the scoring path across
software engineering, data analysis, research synthesis, and operations. It is
not a substitute for externally collected benchmark data. Each task includes a
workflow ID/title, consecutive step index, expected durable-boundary count,
earlier-task dependencies, a recovery-boundary flag, and an evidence-required
flag. The loader rejects an incomplete sequence, a cross-workflow/later
dependency, inconsistent workflow metadata, or a workflow without recovery.

Use the normal matrix planner with this task source, then run
`cross_domain_score.py` on the same matrix/results directory. It first verifies
the source/task identity against every manifest, so changing a fixture or
source revision after planning cannot be scored as the same experiment. In
addition to the core metric set, report per-domain and per-workflow task
completion, whole-workflow completion, prerequisite integrity, recovery
boundary task success, evidence-task success, and expected-boundary budget.

`render_report.py` turns the JSON score into a deterministic Markdown technical
report. It includes source and task hashes, prompt/runtime identity, all
model/condition aggregates, and a per-run workflow ledger. The renderer adds
an explicit non-causality caveat; do not remove it or present a deterministic
fixture result as an empirical cross-model conclusion.

## Learned state-decay predictor protocol

For the learned intervention condition, emit one labeled boundary row with the
five `signals` fields shown in [Python](python.md#learned-state-decay-policy)
and a `decayed` label. Define the label before data collection, for example as a
state inconsistency, repeated known failure, or constraint/goal miss observed
within a fixed number of subsequent boundaries. Do not label an anchor merely
because the same policy chose to inject one.

Split at least by run and preferably by model/backbone: a boundary from one
trajectory must never appear in both training and held-out evaluation. Report
the predictor's held-out accuracy, precision, recall, and Brier score alongside
the existing task/recovery/token metrics. The adaptive controller's threshold,
estimated compact/anchor token counts, and run-budget pressure are persisted in
`state_decay_assessed` metadata, making threshold and cost decisions auditable.

The synthetic examples and smoke tests validate schema and runtime plumbing;
they are not evidence that the learned predictor improves any benchmark.

## Semantic-memory retrieval protocol

For every memory-enabled episode, retain the immutable `memory_created` source
events and the resulting `semantic_memory_retrieved` events. Report retrieval
precision@k and recall@k against a predeclared relevant-memory set, empty-query
rate, mean selected-memory tokens, and the existing task/recovery metrics. The
retrieval-on/off comparison must use the same source catalog, run seeds, model,
and prompt. Do not score a record as relevant only because the retrieval policy
selected it.

The built-in `hybrid_lexical_v1` implementation is a local deterministic
baseline, not a claim of neural embedding quality. If a later embedding backend
is compared, persist a distinct algorithm/revision identifier and hold the
catalog and evaluation labels fixed.

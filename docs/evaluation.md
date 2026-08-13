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
| L | Learned state-decay predictor + adaptive context-budget controller |
| L-fixed | Same learned predictor with a fixed intervention threshold |

Include `Horizon_always_on_anchor` as a control for the token/performance
tradeoff. Compare `L` and `L-fixed` on the same frozen model artifact to isolate
adaptive threshold/cost control. The planned ablations are State Anchor, failure
memory, checkpoint recovery, and structured cognitive state.

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

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

Include `Horizon_always_on_anchor` as a control for the token/performance
tradeoff. The planned ablations are State Anchor, failure memory, checkpoint
recovery, and structured cognitive state.

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

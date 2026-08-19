# Experiments

`run_ablation.py` generates a deterministic smoke report for the planned
ablations:

```bash
python3 experiments/run_ablation.py
```

It writes ignored artifacts under `results/synthetic-ablation/`. The harness is
useful for CI and validating score plumbing. It is not an LLM experiment and
must not be cited as one.

For an actual study:

1. Use a local JSONL mirror of a public source with a `heldout` split; do not
   treat the deterministic fixture as a model result.
2. Generate a frozen model × condition × seed matrix with
   `benchmarks/horizonbench/plan_matrix.py`.
3. Emit exactly one observed JSONL `EpisodeResult` per task at
   `results/<run_id>.jsonl`.
4. Score the complete run set with `benchmarks/horizonbench/score_matrix.py`.

The matrix records the source/task hashes, task IDs, model/provider and
decoding revision, complete prompt hash/text, seed, policy/runtime revision,
checkpoint cadence, and fault schedule. The scorer rejects a missing task or a
mixed task/prompt/runtime control rather than silently averaging incomparable
runs. See [HorizonBench](../benchmarks/horizonbench/README.md) for the exact
JSONL and command-line contract.

## Learned intervention calibration

Collect durable boundary labels separately from HorizonBench episode scores,
then fit the dependency-free logistic predictor:

```bash
PYTHONPATH=python python3 -m horizon_agent.memory.train_decay \
  results/train-boundaries.jsonl \
  --validation-jsonl results/heldout-boundaries.jsonl \
  --output results/decay-model.json \
  --model-version MODEL-POLICY-DATASET-REVISION
```

The validation file must be run/model-disjoint from training. The command emits
accuracy, precision, recall, and Brier score; report those with HorizonBench
outcomes, not in place of them. The `examples/state-decay-labels.jsonl` file is
only a schema smoke fixture.

## Semantic-memory retrieval

For the retrieval-on/off condition, freeze the `memory_created` catalog before
running an episode and retain every resulting `semantic_memory_retrieved` event.
Predeclare the relevant memory IDs for each query, then report precision@k,
recall@k, mean selected-memory tokens, empty-query rate, and HorizonBench
outcomes for the same model, seed, prompts, and catalog. The built-in
`hybrid_lexical_v1` ranker is a deterministic local baseline; do not represent
its synthetic smoke result as an embedding or LLM retrieval result.

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

1. Run each agent/memory condition on the same `HorizonBench` task fixture.
2. Emit one observed JSONL `EpisodeResult` per task.
3. Score it with `benchmarks/horizonbench/score_runs.py`.
4. Record model, provider version, prompt, seed, task fixture hash, and failure
   injection configuration alongside the outputs.

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

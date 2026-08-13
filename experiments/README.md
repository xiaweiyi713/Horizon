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

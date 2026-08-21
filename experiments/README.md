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

## Windows GPU 上的真实模型机制 smoke

`ollama_durable_trace.py` 是面向本地 Ollama 的可复现实验入口。默认任务集为
`ollama_durable_trace_tasks_v2.jsonl`；它会先从 `/api/tags` 读取所选模型的不可变
digest，再冻结 model × condition × seed matrix，运行无副作用 preflight，启动独立的 Rust
HTTP runtime，并保存 receipt、原始 EpisodeResult、评分和实验摘要。它使用模型可读的
durable-trace mechanism suite 来验证
真实模型到运行时的完整链路；该 suite 的 Approved plan 明确给出每一个持久化动作，因而
它**不是**公共 benchmark，也不能用来声称 Horizon 改善了某个模型。

Windows / WSL 中先启动指向本地权重目录的 Ollama（示例使用现有的 Qwen2.5 7B）：

```bash
tmux new-session -d -s horizon-ollama \
  'env OLLAMA_MODELS=/mnt/d/FAR-models/ollama OLLAMA_HOST=127.0.0.1:11434 \
  /mnt/d/FAR-runtime/ollama/bin/ollama serve'
```

推荐由 Mac 编排实验、Windows 仅承担 GPU 推理。这样无需在 Windows 额外安装 Rust
toolchain，Horizon 的隔离 runtime 仍会在 Mac 上按冻结配置启动。Mac 上开一个 SSH
本地端口转发：

```bash
ssh -f -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:11435:127.0.0.1:11434 windows-gpu
```

然后在 Mac 的 Horizon 仓库目录运行：

```bash
PYTHONPATH=python:. python3 experiments/ollama_durable_trace.py \
  --output-dir results/ollama-durable-trace/qwen2.5-7b-seed17 \
  --model qwen2.5:7b --seeds 17 \
  --api-base-url http://127.0.0.1:11435/v1
```

输出目录必须是新目录或空目录，避免将不同运行混在一起。默认条件是
`Horizon_anchor_off`、`Horizon_anchor_always` 与 `Horizon_runtime_heuristic`；它们的
`anchor_strategy`、policy revision、模型 digest、prompt、seed 与 checkpoint cadence
都会写入 manifest。运行时会停止其临时 Horizon server，但不会停止 `horizon-ollama`
服务或 SSH 转发。启动前实验入口会拒绝 dirty Git worktree，避免将未提交代码错误标记为
某个 commit 的实验结果。实验结束后可以在 Mac 用 `ps -ax | rg '127.0.0.1:11435'`
找到对应 SSH 进程后再结束它。

查看结果时先读取 `experiment.json`、`preflight.json` 和 `score.json`，并保留
`matrix.jsonl`、`results/*.jsonl`、`*.receipt.json` 和 `horizon.db`。若要继续用 SSH
观察 GPU：

```bash
watch -n 1 /usr/lib/wsl/lib/nvidia-smi
```

每一条 `EpisodeResult` 可选带有受限的 `diagnostic` 运行类别，用于区分例如
`agent reached max_steps`、策略调用失败和被拒绝的策略动作。它不是评分指标；真实模型
响应、provider HTTP 正文、prompt、密钥或带凭据 URL 都不得写入此字段。

## Cross-domain workflow study

For the cross-domain fixture or a compatible public source, retain its workflow
metadata (`workflow_id`, prerequisites, recovery boundary, evidence flag, and
expected boundaries) in the immutable JSONL task source. After the ordinary
matrix run, execute:

```bash
python3 benchmarks/horizonbench/cross_domain_score.py \
  --matrix results/cross-domain-matrix.jsonl --results-dir results \
  --output results/cross-domain-score.json
python3 benchmarks/horizonbench/render_report.py results/cross-domain-score.json \
  --output results/cross-domain-report.md
```

Treat the report as an audit artifact, not a paper result by itself. Attach the
provider's raw outcomes, source license/provenance, manifests, seeds, prompt,
and durable event traces before making any empirical claim.

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

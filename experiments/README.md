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
`ollama_durable_trace_tasks_v3.jsonl`；它会先从 `/api/tags` 读取所选模型的不可变
digest，再冻结 model × condition × seed matrix，运行无副作用 preflight，启动独立的 Rust
HTTP runtime，并保存 receipt、原始 EpisodeResult、评分和实验摘要。它使用模型可读的
durable-trace mechanism suite 来验证
真实模型到运行时的完整链路；该 suite 的 Approved plan 明确给出每一个持久化动作，因而
它**不是**公共 benchmark，也不能用来声称 Horizon 改善了某个模型。早期 v2 任务集
仍保留用于重放历史工件；v3 只将一条自然语言证据的句末标点从精确匹配范围中移除，
避免把 tokenizer 表面差异误记为持久化机制失败。

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

`--model` 可以重复指定；每个模型会先冻结自身的 Ollama digest，再与相同的条件、任务、
prompt 和 seed 组成独立 manifest 行。例如：

```bash
PYTHONPATH=python:. python3 experiments/ollama_durable_trace.py \
  --output-dir results/ollama-durable-trace/qwen-llama-seed17 \
  --model qwen2.5:7b --model llama3.1:8b --seeds 17 \
  --api-base-url http://127.0.0.1:11435/v1
```

该入口会按 matrix 顺序串行执行模型，避免两套权重争用单张 GPU 显存。它仍是机制 smoke，
不是跨模型性能排行。

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

### 生成可复核的中文报告

新的 `ollama_durable_trace.py` 在一次完整运行结束时会自动写出根目录的 `report.md`。
报告不会只信任汇总分数：它会重新读取并核对 `matrix.jsonl` 的自校验 manifest、静态
preflight、每个 completed receipt、原始 `EpisodeResult` JSONL 与 `score.json`。任何任务集、
模型 digest、运行 ID、结果文件、receipt 或重算分数不一致都会拒绝生成报告。

历史工件或拷贝到另一台机器后的完整工件目录，也可以单独重建报告：

```bash
PYTHONPATH=python:. python3 experiments/render_ollama_durable_trace_report.py \
  results/ollama-durable-trace/qwen-llama-seed17 \
  --output results/ollama-durable-trace/qwen-llama-seed17/report.md
```

报告明确标注 mechanism-smoke 的研究边界，展示任务／prompt／runtime 身份、模型 digest、
完整性检查和按模型 × 条件的观察值。诊断表只输出 allowlist 中的短运行类别；未识别的
原始诊断文本会被抑制，避免意外泄露模型响应、provider 正文或凭据。

## 真实模型的客观 artifact 任务

若要从 mechanism smoke 进入真实任务结果，请不要把
`ollama_durable_trace.py` 的 scripted Approved plan 当作 benchmark。应另行冻结包含
heldout artifact task 的 JSONL、模型／条件／seed matrix 与完整 prompt，并使用
`openai_compatible_artifact_workspace_executor`。该 executor 只允许模型通过
`artifact_workspace` adapter 写入任务白名单中的文件，最终由精确文本／JSON verifier 判定；
模型自己返回 `finish` 或伪造 `record_tool_result` 都不会得到成功分数。

启动独立 runtime 与模型 endpoint 后，配置持久化工件根目录，再先跑静态 preflight：

```bash
export HORIZON_BENCH_RUNTIME_URL=http://127.0.0.1:8787
export HORIZON_BENCH_API_BASE_URL=http://127.0.0.1:11435/v1
export HORIZON_BENCH_API_KEY=ollama-local
export HORIZON_BENCH_ARTIFACT_ROOT="$PWD/results/artifact-workspaces"

PYTHONPATH=python:. python3 benchmarks/horizonbench/preflight_matrix.py \
  --matrix results/artifact-matrix.jsonl --tasks data/artifact-tasks.jsonl \
  --source-name ARTIFACT_TASKS --source-revision IMMUTABLE_RELEASE \
  --executor benchmarks.horizonbench.openai_compatible_artifact_workspace_executor:execute
```

随后用相同 matrix／task 参数调用 `execute_matrix.py`，并保留结果 JSONL、receipt、
runtime 数据库／事件与 `HORIZON_BENCH_ARTIFACT_ROOT`。详细 schema 与完整命令见
[HorizonBench 的 artifact workspace 协议](../benchmarks/horizonbench/README.md#objective-artifact-workspace-execution)。
这是受限 verifier 环境而非通用 shell/Docker sandbox；当前也会拒绝非空 fault schedule。

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

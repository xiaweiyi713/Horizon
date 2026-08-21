# Horizon

> 面向长时程 AI Agent 的持久化认知运行时：Python 负责策略智能，Rust 负责可靠性。

**项目状态：** v1.2 工程里程碑已完成。Horizon 提供经过测试、可复现的运行时与评估基础；在没有冻结模型、任务与客观环境实验之前，项目**不会**宣称自己带来了真实模型性能提升。

长上下文并不等于长时程可靠性。Agent 仍可能遗忘目标、违反早期约束、重复已经失败的方法，或在崩溃后以不一致的进度继续执行。Horizon 将这个问题定义为**持久化执行状态管理**：

- Python 决定下一步应该做什么。
- Rust 验证每一次状态变更，持久化事件，并更新可重放的投影状态。
- Checkpoint、Replay、稳定操作 ID 与结构化认知状态，让恢复过程可审计，而不是依赖重新阅读完整对话轨迹。

## 为什么是 Horizon？

Horizon 不是通用聊天 Agent，也不是工具市场。它是一层可靠性运行时，用于支撑需要跨越长轨迹、任务切换、工具失败和进程重启的 Agent；LLM 不会被当作系统事实的唯一来源。

| 问题 | Horizon 机制 |
|---|---|
| 目标或约束漂移 | 持久化认知状态与有边界的 State Anchor |
| 重复已失败的操作 | 带来源记录的结构化失败记忆 |
| 运行时或进程崩溃 | Checkpoint、事件重放与遵守重试策略的恢复 |
| 至少一次的外部执行 | 稳定的 `operation_id` 支持幂等集成 |
| 上下文压力 | 可解释、主动的 State Anchor 干预 |
| 难以复现实验 | 冻结 manifest、客观结果、receipt 与 preflight |

## 架构

```text
Python 策略 / LLM Provider
        │  localhost JSON/HTTP 或可选 PyO3
        ▼
┌──────────────────────────────────────────────────────────────┐
│                       Horizon Rust Runtime                     │
│                                                              │
│  Command → Validate → Immutable Event → Event Store → Apply  │
│                                  │                           │
│                    Checkpoint / Replay / Audit               │
│                                  │                           │
│   Durable Cognitive State + State Anchor Intervention Policy  │
└──────────────────────────────────────────────────────────────┘
        │
        ├── 基于 Tokio 的依赖感知任务调度器
        ├── 支持超时与资源限制的进程监督器
        └── 默认 SQLite；需要共享存储时可使用 PostgreSQL
```

持久化投影会记录主目标、约束、计划、子目标、任务、失败、决策、证据、环境、语义记忆、预算与恢复边界。Python 不会直接修改这些投影状态。

## 已实现能力

### 持久化运行时

- 经验证的 Agent / Task 状态机与追加式有序事件流。
- SQLite/WAL 事件存储、带校验和的 zstd 快照、压缩清理与带 schema 版本的重放；PostgreSQL 通过同一存储抽象提供支持。
- 中断任务的跨进程恢复、重试策略与持久化失败记忆。
- 向进程任务传递稳定的 `HORIZON_OPERATION_ID`，为至少一次执行提供幂等集成边界。
- Tokio DAG 调度、进程超时、输出上限、CPU/内存限制，以及可选 Docker 执行后端。
- 本地 CLI 与 JSON/HTTP API、可选 PyO3 原生 SQLite runtime，以及仅在事件持久化后发送的 OTLP/HTTP Trace。

### 认知状态管理

- 持久化目标、约束、计划、子目标、决策、证据、失败、环境与预算。
- 基于 Anchor 距离、上下文压力、任务切换、失败与恢复信号的可解释主动 State Anchor 策略。
- 可离线训练的 state-decay 预测器与自适应上下文预算控制器；每次决策都会被持久化以便审计。
- 带来源记录、确定性且有上限的 `hybrid_lexical_v1` 语义记忆检索。

### Python 与评估

- 无第三方依赖的 Python HTTP Client、单动作 JSON Policy Loop、Scripted Provider 与 OpenAI-compatible Provider Adapter。
- HorizonBench fixture、指标评分器、baseline / ablation smoke harness、真实跨进程故障注入测试与跨领域工作流评分器。
- 冻结的模型 × 条件 × 随机种子 manifest、按任务原子持久化的 result receipt、可恢复矩阵执行与客观 trace judge。
- 不执行任务的 matrix preflight：在调用模型或运行时之前，校验任务身份、冻结的 prompt / condition / decoding 控制、凭据是否存在与 trace expectation。
- 受限的 artifact workspace 任务环境：模型只能经由已注册的 adapter 写入任务白名单文件，精确文本／JSON verifier 独立判定结果；adapter 操作 receipt 会随工作区保存，使稳定操作 ID 在 Python 重启后仍不会重复改写已完成工件。

## 快速开始：无需 API Key

**环境要求：** Rust stable（含 Cargo）、Python 3.9+ 与 `make`。核心 demo 和测试只使用 Python 标准库。

```bash
git clone https://github.com/xiaweiyi713/Horizon.git horizon
cd horizon

make test
make demo
```

Demo 会创建目标、约束、计划、失败记忆和 checkpoint，然后模拟恢复并输出还原后的 State Anchor。

### 运行一个持久化本地 DAG

```bash
cargo run -p horizon-cli -- --db example.db run examples/research-workflow.yaml

# 将 RUN_ID 替换为上一个命令输出的值。
cargo run -p horizon-cli -- --db example.db inspect RUN_ID
cargo run -p horizon-cli -- --db example.db events RUN_ID
cargo run -p horizon-cli -- --db example.db memory RUN_ID
cargo run -p horizon-cli -- --db example.db replay RUN_ID
```

任务文件使用 argv 数组，而不是插值后的 shell 字符串。请查看 [examples](examples/README.md)，其中包含恢复、资源受限进程和研究工作流 fixture。

### 运行 Python 策略层

在第一个终端启动本地 runtime：

```bash
cargo run -p horizon-cli -- --db horizon.db serve
```

在另一个终端运行确定性的无 Key 策略：

```bash
PYTHONPATH=python python3 examples/python_scripted_demo.py
```

如需接入 OpenAI-compatible endpoint，请通过环境变量配置其 Key 和地址，再使用 `OpenAICompatibleProvider`。Provider 只能接收已经验证的 decoding 控制；Rust runtime 仍是持久化事实的来源。完整约定见 [Python 集成文档](docs/python.md)。

## 恢复模型

Horizon 有意识地实现**至少一次（at-least-once）**语义，而不是假装提供 exactly-once。若某任务运行期间进程中断，恢复流程会生成持久化的中断 / 失败事件，并仅在重试策略允许时重新入队。外部系统必须使用稳定操作 ID 来保证幂等。

```bash
cargo run -p horizon-cli -- --db example.db checkpoint RUN_ID
cargo run -p horizon-cli -- --db example.db resume RUN_ID --execute
```

若需共享存储，可传入 PostgreSQL URL 代替 `--db`。迁移、Docker、Telemetry 与资源限制请查阅 [运维文档](docs/operations.md)。

## 评估与研究诚信

HorizonBench 覆盖六类长时程能力：目标保持、约束保持、失败规避、跨会话恢复、长上下文退化与故障恢复。

```bash
# 确定性 fixture 与 ablation plumbing；不是实证 LLM 结论。
make bench

# 启动真实 Rust HTTP runtime，并用 scripted fixture action 运行 durable-trace bridge；
# 仍然不会调用模型 API。
make bench-trace

# 使用真实 Rust runtime、scripted policy、受限文件工作区和确定性 verifier 的端到端回归；
# 同样不会调用模型 API，也不能被当作模型实验结果。
make bench-artifact-workspace

# 校验冻结矩阵与执行器配置；不写入结果、不启动 runtime、不发起模型请求。
make bench-preflight
```

内置 fixture 与 ablation 输出明确标记为 `synthetic_deterministic_smoke`。它们证明 benchmark plumbing 和 runtime 机制可用，**不能**作为“Horizon 改善了某个 LLM”的证据。

真实实验要求客观的任务环境，以及冻结任务源、prompt、模型及版本、decoding、随机种子、条件、checkpoint cadence 和 fault schedule 的矩阵：

```bash
python3 benchmarks/horizonbench/plan_matrix.py --help
python3 benchmarks/horizonbench/preflight_matrix.py --help
python3 benchmarks/horizonbench/execute_matrix.py --help
python3 benchmarks/horizonbench/score_matrix.py --help
```

`preflight_matrix.py` 会拒绝没有静态 `execute.preflight(context)` hook 的执行器。内置 durable-trace preflight 会验证 Provider 配置和凭据存在性，但不会记录任何敏感值。对于只需受限文件产物的任务，可使用 `openai_compatible_artifact_workspace_executor`：它要求显式配置 `HORIZON_BENCH_ARTIFACT_ROOT`，并由任务拥有的精确 verifier 判定结果。该环境不执行 shell，也不是通用安全沙箱；公共 domain 任务和非空 fault schedule 仍需要能真正注入并客观评判这些事件的专用环境。

在报告结果前，请阅读完整的 [HorizonBench 协议](benchmarks/horizonbench/README.md)、[评估指南](docs/evaluation.md) 和 [路线图](docs/roadmap.md)。

## 验证

提交变更前，请运行完整的本地验证套件：

```bash
make fmt
make test
make test-python
make lint
make fault
make bench
make bench-trace
make bench-artifact-workspace
make bench-preflight
```

`make fault` 会启动真实 Rust HTTP Server，在持久化工作之后终止它，并对同一个 SQLite 数据库重启；它会验证 replay、失败记忆、State Anchor 注入、重试重新入队、超时处理、语义检索与 Python policy 边界。

## 仓库结构

```text
crates/
  horizon-core/       状态机、命令、事件与投影
  horizon-store/      SQLite/PostgreSQL 事件存储与快照
  horizon-memory/     State Anchor、状态衰减策略、语义检索
  horizon-scheduler/  Tokio DAG 调度器
  horizon-process/    进程监督与 Docker 后端
  horizon-trace/      事件提交后的 Trace Bridge
  horizon-runtime/    持久化编排、恢复与 HTTP API
  horizon-py/         可选 PyO3 原生 runtime
  horizon-cli/        CLI
python/horizon_agent/  Provider Adapter、Policy Loop 与 HTTP Client
benchmarks/            HorizonBench、评分、矩阵运行器与 fixture
experiments/           ablation report scaffold
docs/                  架构、API、运维、评估与路线图
```

## 设计边界

Horizon 有意保持聚焦：它不试图替代 Browser Agent、MCP ecosystem、多 Agent 编排、Dashboard、Kubernetes 或 Vector Database。任何新集成都必须保留“命令 → 事件 → 投影”边界，且不能让 LLM 绕过持久化状态校验。

## 文档

- [架构](docs/architecture.md)
- [持久性与恢复](docs/durability.md)
- [HTTP API](docs/http-api.md)
- [Python 集成](docs/python.md)
- [运维](docs/operations.md)
- [评估协议](docs/evaluation.md)
- [路线图](docs/roadmap.md)
- [贡献指南](CONTRIBUTING.md)

## 许可证

[MIT](LICENSE)

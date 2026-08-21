# Python policy layer

`python/horizon_agent` is intentionally small and provider-agnostic.

## Provider protocol

Implement one method:

```python
def complete(messages, *, json_mode=True) -> ProviderResponse:
    ...
```

Horizon ships `ScriptedProvider` for deterministic demos/tests and
`OpenAICompatibleProvider` for `/chat/completions` endpoints using only the
standard library.

`OpenAICompatibleProvider` accepts an optional `decoding` mapping. It forwards
only `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `seed`,
`stop`, `presence_penalty`, and `frequency_penalty`; invalid, non-finite, or
reserved request fields fail before any network request. If `decoding` includes
`temperature`, it overrides the constructor's compatibility default. This makes
the settings frozen by a HorizonBench model manifest effective at the provider
boundary rather than merely descriptive metadata.

## Action protocol

The policy returns exactly one JSON action per boundary:

```json
{
  "thought": "A failed attempt makes a State Anchor useful.",
  "action": {
    "type": "record_decision",
    "data": {
      "decision": {
        "id": "regime-oos",
        "decision": "Use regime-specific OOS comparison",
        "rationale": "Full-sample Sharpe cannot identify regime degradation"
      }
    }
  }
}
```

`DurableAgent` forwards every state-changing action to Rust, then reads the next
materialized projection. Normal turns get a small execution summary; a full State
Anchor is supplied only after the Rust policy signals an intervention.

Each provider call is recorded as a durable `tool_invoked` event followed by a
normalized `tool_result_recorded` event. A result contains a terminal status
(`succeeded`, `failed`, `timed_out`, or `cancelled`), JSON output, error,
duration, and metadata. Horizon intentionally records token/count metadata
rather than the full prompt or response, keeping the audit trail useful without
making it a second transcript store.

## State Anchor 对照策略

`AgentConfig.anchor_strategy` 默认为 `"runtime_heuristic"`，即由 Rust 的可解释
启发式决定何时注入 Anchor。研究矩阵还可以使用两个固定且可审计的控制：

```python
AgentConfig(anchor_strategy="always", anchor_policy_revision="fixed-anchor-control-v1")
AgentConfig(anchor_strategy="disabled", anchor_policy_revision="fixed-anchor-control-v1")
```

`always` 在每个模型决策边界注入 Rust 渲染的 Anchor，`disabled` 始终使用 compact
context。两者都会通过 `apply_intervention` 写入 `state_decay_assessed`，而不是在
Python 中悄悄改写 prompt；`always` 还会产生 `state_anchor_injected`。在
HorizonBench 中，把 `anchor_strategy` 放进冻结的 `configuration.agent`，系统会使用
该条件的 `policy_revision` 作为 assessment 版本。固定策略不能和
`learned_intervention_policy` 同时配置。

无论注入完整 Anchor 还是使用 compact context，策略都会看到同一行受限进度摘要：
`Durable records: decisions=N, evidence=N, remembered failures=N`。完整 Anchor 随后才
展示相应记录的细节，避免模型靠重复文本推断计划中的计数条件。

## Manifest-bound HorizonBench episodes

`DurableAgentEpisodeExecutor` is the model-backed bridge for a frozen
HorizonBench matrix row. It creates one isolated durable run for the task,
passes the matrix's immutable prompt text as the policy system prompt, applies
the manifest checkpoint cadence after model decision boundaries, and gives the
completed runtime projection plus immutable event trace to an objective judge.
An LLM's own `finish` action is never a benchmark success signal by itself.

`DurableTraceJudge` is the built-in judge for narrow runtime-mechanism tasks.
Those task records must declare their observable requirements in metadata:

```json
{
  "initial_plan": ["restore the checkpoint", "verify the operation ID"],
  "horizon_trace_expectation": {
    "required_event_types": ["evidence_recorded", "checkpoint_created"],
    "required_evidence_terms": ["verified checksum"],
    "require_checkpoint": true
  }
}
```

It checks durable facts only (goal/constraint retention, event types,
checkpoints, recovery, evidence and remembered failures), and leaves recovery
distance unset rather than inventing a progress oracle. It is not an
artifact-, test-, or environment-based judge for a public domain benchmark;
provide a task-specific `ObjectiveEpisodeJudge`/executor for those tasks.
The generic bridge intentionally rejects a non-empty manifest fault schedule,
because only a task environment can safely enact and observe an arbitrary
crash/restart protocol.

## External tools

For tools outside the Rust Process Supervisor, use `AdapterRegistry`. It writes
the durable invocation before the Python side effect and the normalized result
afterward, using a stable operation ID:

```python
from horizon_agent import AdapterRegistry, AdapterResult, FunctionAdapter, HorizonClient

registry = AdapterRegistry([
    FunctionAdapter("ticketing", lambda request: AdapterResult(
        output={"ticket": "T-42", "operation_id": request.operation_id}
    ))
])
result = registry.execute(
    HorizonClient(), run_id="RUN_ID", adapter_name="ticketing",
    operation_id="create-ticket:RUN_ID", input={"title": "Investigate drift"},
)
```

If a process dies after the external side effect but before its result is
stored, recovery retains the same operation ID for the integration to
deduplicate or query. This is intentionally at-least-once behavior. Pair known
non-retryable failures with `remember_failure` when policy behavior should be
blocked as well.

## Optional native runtime

The default client remains HTTP. For an in-process local SQLite runtime, build
the optional PyO3 module with [Maturin](https://www.maturin.rs/):

```bash
python3 -m pip install maturin
maturin develop --manifest-path crates/horizon-py/Cargo.toml

PYTHONPATH=python python3 - <<'PY'
from horizon_agent import NativeHorizonRuntime

runtime = NativeHorizonRuntime("native-horizon.db")
created = runtime.create_run("Keep execution state durable")
print(created["projection"]["run_id"])
PY
```

`NativeHorizonRuntime` implements the same command-oriented methods used by
`HorizonClient`, including `DurableAgent` compatibility. It does not bypass the
Rust state machine or event store; it merely removes the localhost HTTP hop.

## Semantic-memory retrieval

Store source-attributed records with `HorizonClient.create_memory`, then let a
`DurableAgent` add only a bounded ranked subset to its regular compact context:

```python
from uuid import uuid4

from horizon_agent import AgentConfig, DurableAgent, HorizonClient

client = HorizonClient()
client.create_memory(
    run_id,
    "Recover PostgreSQL from a verified checkpoint before retrying the migration.",
    kind="episodic",
    source_event_id=str(uuid4()),  # durable/external provenance reference
    importance=0.9,
    confidence=0.95,
)
agent = DurableAgent(client, provider, config=AgentConfig(semantic_memory_limit=4))
```

The current local `hybrid_lexical_v1` ranker combines normalized token overlap
and character-trigram similarity. It is deterministic, requires no vector
database or network service, and treats importance/confidence only as secondary
weighting; irrelevant high-quality records cannot be selected. Every lookup is
persisted as `semantic_memory_retrieved` with the query and exact hits. Set
`semantic_memory_limit=0` for the retrieval-off ablation. The command allows at
most eight hits, and agents do not query at all when a run has no memory catalog.

`source_event_id` is deliberately required by the helper: retain the durable or
external origin of a summary rather than creating anonymous memory. Retrieval
is not a replacement for the State Anchor; if an anchor follows a lookup, the
agent preserves the bounded retrieval context alongside it.

## Learned state-decay policy

The research layer includes a dependency-free logistic predictor over the
durable decay signals: anchor distance, context pressure, subgoal switching,
recent failures, and recovery. It is deliberately **offline-trained**. Do not
call a fitted model "learned" without keeping its training labels, revision,
and a run/model-disjoint validation set.

Each JSONL training row has this shape:

```json
{
  "signals": {
    "steps_since_anchor": 11,
    "context_pressure": 0.82,
    "subgoal_switches": 2,
    "recent_failures": 1,
    "recovered_session": false
  },
  "decayed": true
}
```

Train a reproducible artifact and, when available, evaluate it on separately
collected labels:

```bash
PYTHONPATH=python python3 -m horizon_agent.memory.train_decay \
  examples/state-decay-labels.jsonl \
  --output results/decay-model.json \
  --model-version experiment-2026-08-13
```

The bundled label file only demonstrates the artifact format; it is not an
empirical dataset. Load a real held-out model into a `DurableAgent` as follows:

```python
from pathlib import Path

from horizon_agent import AgentConfig, DurableAgent, HorizonClient
from horizon_agent.memory import (
    AdaptiveContextBudgetController,
    LearnedInterventionPolicy,
    LogisticDecayPredictor,
)

predictor = LogisticDecayPredictor.load(Path("results/decay-model.json"))
agent = DurableAgent(
    HorizonClient(),
    provider,
    config=AgentConfig(
        learned_intervention_policy=LearnedInterventionPolicy(
            predictor=predictor,
            controller=AdaptiveContextBudgetController(context_window_tokens=8192),
        )
    ),
)
```

At every learned-policy boundary, Python estimates compact-versus-anchor token
cost and submits a complete assessment to Rust. Rust validates it and persists
`state_decay_assessed`; if the action is `inject_anchor`, it commits the
assessment and a Rust-rendered `state_anchor_injected` event together. Thus a
skipped intervention is just as auditable as an injected one, and Python never
writes anchor content directly. The persisted action must agree with the
assessment's risk/threshold comparison, preventing an audit record from claiming
to skip a boundary that its own policy threshold says should inject.

## Budgets

The Python loop records observed token/wall-time usage in durable state and
suspends at a configured next boundary. Actual post-call usage is intentionally
recorded even if it crosses a budget so audit/recovery can see the overrun.

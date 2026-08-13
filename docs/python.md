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

## Budgets

The Python loop records observed token/wall-time usage in durable state and
suspends at a configured next boundary. Actual post-call usage is intentionally
recorded even if it crosses a budget so audit/recovery can see the overrun.

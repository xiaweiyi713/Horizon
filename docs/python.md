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

## External tools

For tools outside the Rust Process Supervisor, record durable audit events with
`record_tool_invocation`, `record_tool_success`, or `record_tool_failure`; pair
non-retryable failures with `remember_failure`. Use stable operation IDs for
idempotent integrations.

## Budgets

The Python loop records observed token/wall-time usage in durable state and
suspends at a configured next boundary. Actual post-call usage is intentionally
recorded even if it crosses a budget so audit/recovery can see the overrun.

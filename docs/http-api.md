# HTTP API

The runtime provides a local JSON API for Python policies:

```bash
horizon --db horizon.db serve --address 127.0.0.1:8787
```

Errors use JSON `{ "error": "..." }`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Returns `ok` |
| `POST` | `/v1/runs` | Create a run from `{ "goal": "..." }` |
| `GET` | `/v1/runs` | List projections |
| `GET` | `/v1/runs/{id}` | Get current materialized projection |
| `GET` | `/v1/runs/{id}/events` | Get immutable event log |
| `GET` | `/v1/runs/{id}/anchor` | Render current State Anchor |
| `POST` | `/v1/runs/{id}/commands` | Submit a `RuntimeCommand` |
| `POST` | `/v1/runs/{id}/checkpoint` | Create explicit checkpoint |
| `POST` | `/v1/runs/{id}/recover` | Replay and record recovery boundary |
| `POST` | `/v1/runs/{id}/interventions` | Ask policy to inject an anchor |
| `POST` | `/v1/runs/{id}/tasks/execute` | Run ready process tasks |

## Command shape

```json
{
  "type": "add_constraint",
  "data": {
    "constraint": {
      "id": "no-repeat",
      "content": "Do not retry non-idempotent work without an operation ID.",
      "source": "operator"
    }
  }
}
```

Rust converts a valid command into one or more persisted events. It rejects
invalid lifecycle and task transitions with HTTP 422. A stale cross-process
command receives HTTP 409 so a policy can reload the projection and retry its
decision safely.

Task declarations may include `resources` (`max_memory_bytes`,
`max_cpu_time_ms`, `max_output_bytes`) and an `executor` object such as
`{"kind":"local"}` or `{"kind":"docker","image":"python:3.12-slim"}`.
Process output and external integrations should use the `record_tool_result`
command for a structured terminal result:

```json
{
  "type": "record_tool_result",
  "data": {
    "result": {
      "operation_id": "weather:request-42",
      "tool": "weather",
      "status": "succeeded",
      "output": {"temperature_c": 21},
      "error": null,
      "duration_ms": 87,
      "metadata": {"provider": "example"}
    }
  }
}
```

## Semantic-memory commands

Create a source-attributed memory record before it can be retrieved. Quality
values are finite normalized values; Rust rejects blank content, invalid quality
values, duplicate IDs, and oversized records.

```json
{
  "type": "create_memory",
  "data": {
    "memory": {
      "id": "5c4dddc8-971b-48a6-a147-49e07c51e4d5",
      "kind": "episodic",
      "content": "Recover PostgreSQL from a verified checkpoint before retrying the migration.",
      "source_event": "5ba9b609-ed29-4e65-af43-bd9d1bcdb539",
      "importance": 0.9,
      "confidence": 0.95,
      "created_at": "2026-08-19T00:00:00Z"
    }
  }
}
```

Retrieve a bounded subset through the normal command endpoint:

```json
{
  "type": "retrieve_semantic_memory",
  "data": {
    "query": "recover postgres checkpoint",
    "limit": 4
  }
}
```

The accepted command emits one `semantic_memory_retrieved` event. Its payload
contains Rust-selected hits and `score_milli` values, not caller-supplied
results. Limits are 1–8 and query length is bounded. Empty hit lists are still
persisted, making retrieval-specific ablations and audit possible.

## Learned intervention command

An offline-trained Python policy submits each decision through
`apply_intervention`. Rust rejects malformed scores and renders any anchor from
the durable projection itself:

```json
{
  "type": "apply_intervention",
  "data": {
    "assessment": {
      "policy_id": "logistic_state_decay",
      "policy_version": "experiment-42",
      "risk_score_milli": 810,
      "threshold_milli": 640,
      "signals": {
        "steps_since_anchor": 8,
        "context_pressure": 0.72,
        "subgoal_switches": 2,
        "recent_failures": 1,
        "recovered_session": false
      },
      "action": "inject_anchor",
      "reason": "learned risk crossed the adaptive budget threshold",
      "metadata": {
        "estimated_anchor_tokens": 120,
        "context_window_tokens": 8192
      }
    }
  }
}
```

Every accepted assessment emits `state_decay_assessed`. An `inject_anchor`
decision additionally emits `state_anchor_injected` in the same durable command
batch. Its action must agree with the declared risk/threshold comparison; an
inconsistent request is rejected with HTTP 422. Rust also verifies that
`steps_since_anchor` equals the current durable event distance, so policies must
reload and reassess after a concurrent mutation rather than submit stale input.

## Intervention request

```json
{
  "steps_since_anchor": 8,
  "context_pressure": 0.82,
  "subgoal_switches": 2,
  "recent_failures": 1,
  "recovered_session": false
}
```

The response is either `null` (no intervention) or an outcome containing a
persisted `state_anchor_injected` event.

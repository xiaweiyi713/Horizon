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

# Contributing

## Setup

```bash
cargo test --workspace
PYTHONPATH=python:. python3 -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=python python3 tests/fault_injection.py
```

## Runtime invariants

- Do not mutate `RunProjection` outside event application.
- Add a command only when it validates to immutable events.
- Preserve event sequence/replay compatibility when evolving payloads.
- Keep external side effects behind stable operation IDs; Horizon is
  at-least-once, not exactly-once.
- Add fault/replay tests whenever changing checkpoint or task lifecycle logic.

## Scope discipline

Horizon v0.1 optimizes for durable runtime correctness. New browser agents,
vector databases, dashboards, or multi-agent features must not bypass the
event/state boundary or obscure the core reliability thesis.

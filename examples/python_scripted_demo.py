"""No-key Python durable agent demo.

First start the Rust server in another terminal:

    cargo run -p horizon-cli -- --db horizon.db serve

Then run:

    PYTHONPATH=python python3 examples/python_scripted_demo.py
"""

from horizon_agent import AgentConfig, DurableAgent, HorizonClient
from horizon_agent.providers import ScriptedProvider


provider = ScriptedProvider(
    [
        {
            "thought": "Track the next durable deliverable.",
            "action": {
                "type": "open_subgoal",
                "data": {"subgoal": {"content": "verify State Anchor and replay behavior"}},
            },
        },
        {
            "thought": "The fixed demo is complete.",
            "action": {"type": "finish", "data": {}},
        },
    ]
)

agent = DurableAgent(
    HorizonClient(),
    provider,
    config=AgentConfig(max_steps=4),
)
result = agent.run(
    "Demonstrate Python intelligence over a durable Rust runtime",
    constraints=["Persist every cognitive-state change as an event."],
)
print(result)

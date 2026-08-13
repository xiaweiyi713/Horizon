"""Run an OpenAI-compatible LLM policy against the local Horizon runtime.

Set OPENAI_API_KEY (or pass `api_key` directly) before use. The provider accepts
any compatible `/chat/completions` service, including a local one.
"""

from horizon_agent import AgentConfig, DurableAgent, HorizonClient
from horizon_agent.providers import OpenAICompatibleProvider


agent = DurableAgent(
    HorizonClient("http://127.0.0.1:8787"),
    OpenAICompatibleProvider(model="YOUR_MODEL_NAME"),
    config=AgentConfig(max_steps=20, token_budget=20_000),
)
result = agent.run(
    "Create a concise, reproducible experiment plan",
    constraints=[
        "Do not retry a known non-retryable failure.",
        "Record important decisions and evidence before completing.",
    ],
)
print(result)

"""Command-line entry point for an OpenAI-compatible Horizon policy loop."""

from __future__ import annotations

import argparse
import json
import sys

from .agent import AgentConfig, DurableAgent
from .client import HorizonClient
from .providers import OpenAICompatibleProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Python LLM policy against Horizon's local runtime")
    parser.add_argument("--server", default="http://127.0.0.1:8787", help="Horizon HTTP base URL")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--plan-step", action="append", default=[])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1", help="LLM-compatible API base URL")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--token-budget", type=int, default=None)
    arguments = parser.parse_args()

    agent = DurableAgent(
        HorizonClient(arguments.server),
        OpenAICompatibleProvider(
            arguments.model,
            api_key=arguments.api_key,
            base_url=arguments.base_url,
        ),
        config=AgentConfig(max_steps=arguments.max_steps, token_budget=arguments.token_budget),
    )
    try:
        result = agent.run(arguments.goal, constraints=arguments.constraint, plan=arguments.plan_step)
    except Exception as error:
        print("horizon-agent failed: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Python research and LLM-policy layer for Horizon.

Horizon deliberately keeps reliability in Rust and intelligence in Python:
this package calls the local runtime's JSON/HTTP API rather than duplicating
state-machine or persistence logic in Python.
"""

from .agent import AgentConfig, DurableAgent, RunResult
from .client import HorizonApiError, HorizonClient

__all__ = [
    "AgentConfig",
    "DurableAgent",
    "HorizonApiError",
    "HorizonClient",
    "RunResult",
]

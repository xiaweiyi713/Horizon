"""Python research and LLM-policy layer for Horizon.

Horizon deliberately keeps reliability in Rust and intelligence in Python:
this package calls the local runtime's JSON/HTTP API rather than duplicating
state-machine or persistence logic in Python.
"""

from .agent import AgentConfig, DurableAgent, RunResult
from .adapters import AdapterError, AdapterRegistry, AdapterRequest, AdapterResult, FunctionAdapter
from .client import HorizonApiError, HorizonClient
from .memory import AdaptiveContextBudgetController, LearnedInterventionPolicy, LogisticDecayPredictor
from .native import NativeHorizonRuntime, NativeRuntimeUnavailable

__all__ = [
    "AgentConfig",
    "AdapterError",
    "AdapterRegistry",
    "AdapterRequest",
    "AdapterResult",
    "AdaptiveContextBudgetController",
    "DurableAgent",
    "FunctionAdapter",
    "HorizonApiError",
    "HorizonClient",
    "LearnedInterventionPolicy",
    "LogisticDecayPredictor",
    "NativeHorizonRuntime",
    "NativeRuntimeUnavailable",
    "RunResult",
]

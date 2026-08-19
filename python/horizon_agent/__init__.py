"""Python research and LLM-policy layer for Horizon.

Horizon deliberately keeps reliability in Rust and intelligence in Python:
this package calls the local runtime's JSON/HTTP API rather than duplicating
state-machine or persistence logic in Python.
"""

from .agent import AgentConfig, DurableAgent, RunResult
from .adapters import AdapterError, AdapterRegistry, AdapterRequest, AdapterResult, FunctionAdapter
from .benchmark import (
    BenchmarkExecutionError,
    DurableAgentEpisodeExecutor,
    DurableTraceExpectation,
    DurableTraceJudge,
    ObjectiveEpisodeJudge,
    agent_config_from_condition,
    initial_plan_from_task,
    system_prompt_from_manifest,
)
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
    "BenchmarkExecutionError",
    "DurableAgent",
    "DurableAgentEpisodeExecutor",
    "DurableTraceExpectation",
    "DurableTraceJudge",
    "FunctionAdapter",
    "HorizonApiError",
    "HorizonClient",
    "LearnedInterventionPolicy",
    "LogisticDecayPredictor",
    "NativeHorizonRuntime",
    "NativeRuntimeUnavailable",
    "ObjectiveEpisodeJudge",
    "RunResult",
    "agent_config_from_condition",
    "initial_plan_from_task",
    "system_prompt_from_manifest",
]

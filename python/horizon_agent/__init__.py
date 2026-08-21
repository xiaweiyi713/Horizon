"""Python research and LLM-policy layer for Horizon.

Horizon deliberately keeps reliability in Rust and intelligence in Python:
this package calls the local runtime's JSON/HTTP API rather than duplicating
state-machine or persistence logic in Python.
"""

from .agent import AgentConfig, DurableAgent, RunResult
from .adapters import AdapterError, AdapterRegistry, AdapterRequest, AdapterResult, FunctionAdapter
from .artifact_benchmark import ArtifactWorkspaceJudge, DurableArtifactWorkspaceExecutor, workspace_from_root
from .artifact_workspace import (
    ARTIFACT_WORKSPACE_METADATA_KEY,
    ARTIFACT_WORKSPACE_SCHEMA_VERSION,
    ArtifactObservation,
    ArtifactVerifier,
    ArtifactWorkspace,
    ArtifactWorkspaceAdapter,
    ArtifactWorkspaceError,
    ArtifactWorkspaceSpec,
    WorkspaceTemplateFile,
)
from .benchmark import (
    BenchmarkExecutionError,
    DurableAgentEpisodeExecutor,
    DurableTraceExpectation,
    DurableTraceJudge,
    ObjectiveEpisodeJudge,
    agent_config_from_condition,
    bounded_run_diagnostic,
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
    "ARTIFACT_WORKSPACE_METADATA_KEY",
    "ARTIFACT_WORKSPACE_SCHEMA_VERSION",
    "ArtifactObservation",
    "ArtifactVerifier",
    "ArtifactWorkspace",
    "ArtifactWorkspaceAdapter",
    "ArtifactWorkspaceError",
    "ArtifactWorkspaceJudge",
    "ArtifactWorkspaceSpec",
    "BenchmarkExecutionError",
    "bounded_run_diagnostic",
    "DurableAgent",
    "DurableAgentEpisodeExecutor",
    "DurableArtifactWorkspaceExecutor",
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
    "WorkspaceTemplateFile",
    "agent_config_from_condition",
    "initial_plan_from_task",
    "system_prompt_from_manifest",
    "workspace_from_root",
]

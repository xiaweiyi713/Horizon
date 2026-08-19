"""HorizonBench: fixtures and scoring for long-horizon agent state management."""

from .adapters import BenchmarkTask, BenchmarkTaskSet, JsonlTaskAdapter
from .protocol import (
    EvaluationCondition,
    ModelProfile,
    PromptArtifact,
    RunManifest,
    TaskSetIdentity,
    build_cross_model_matrix,
)
from .cross_domain_score import score_cross_domain
from .execution import EpisodeExecutor, ExecutionContext, ExecutionValidationError, load_executor
from .execute_matrix import MatrixExecutionError, execute_manifest, execute_matrix
from .render_report import render_technical_report
from .score import BenchmarkMetrics, EpisodeResult, score_results
from .workflows import (
    CrossDomainWorkflowSuite,
    DEFAULT_CROSS_DOMAIN_TASKS,
    WorkflowDefinition,
    WorkflowTask,
    WorkflowValidationError,
    load_cross_domain_workflows,
)

__all__ = [
    "BenchmarkMetrics",
    "BenchmarkTask",
    "BenchmarkTaskSet",
    "CrossDomainWorkflowSuite",
    "DEFAULT_CROSS_DOMAIN_TASKS",
    "EpisodeExecutor",
    "EpisodeResult",
    "EvaluationCondition",
    "ExecutionContext",
    "ExecutionValidationError",
    "JsonlTaskAdapter",
    "MatrixExecutionError",
    "ModelProfile",
    "PromptArtifact",
    "RunManifest",
    "TaskSetIdentity",
    "WorkflowDefinition",
    "WorkflowTask",
    "WorkflowValidationError",
    "build_cross_model_matrix",
    "execute_manifest",
    "execute_matrix",
    "load_executor",
    "load_cross_domain_workflows",
    "render_technical_report",
    "score_cross_domain",
    "score_results",
]

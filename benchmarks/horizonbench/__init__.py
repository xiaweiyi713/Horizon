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
    "EpisodeResult",
    "EvaluationCondition",
    "JsonlTaskAdapter",
    "ModelProfile",
    "PromptArtifact",
    "RunManifest",
    "TaskSetIdentity",
    "WorkflowDefinition",
    "WorkflowTask",
    "WorkflowValidationError",
    "build_cross_model_matrix",
    "load_cross_domain_workflows",
    "render_technical_report",
    "score_cross_domain",
    "score_results",
]

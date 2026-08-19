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
from .score import BenchmarkMetrics, EpisodeResult, score_results

__all__ = [
    "BenchmarkMetrics",
    "BenchmarkTask",
    "BenchmarkTaskSet",
    "EpisodeResult",
    "EvaluationCondition",
    "JsonlTaskAdapter",
    "ModelProfile",
    "PromptArtifact",
    "RunManifest",
    "TaskSetIdentity",
    "build_cross_model_matrix",
    "score_results",
]

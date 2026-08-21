"""Durable, objectively judged execution for bounded artifact workspaces.

The generic durable-trace bridge can judge only runtime facts.  This module
binds a real :class:`~horizon_agent.agent.DurableAgent` to the task-owned
``artifact_workspace_v1`` environment: the policy must invoke its registered
adapter, the adapter materializes a constrained candidate artifact, and a
deterministic verifier owns success.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from .adapters import AdapterRegistry
from .agent import AgentConfig, DurableAgent, RunResult, validate_agent_config
from .artifact_workspace import (
    ArtifactObservation,
    ArtifactWorkspace,
    ArtifactWorkspaceAdapter,
    ArtifactWorkspaceError,
    ArtifactWorkspaceSpec,
)
from .benchmark import (
    BenchmarkExecutionError,
    agent_config_from_condition,
    bounded_run_diagnostic,
    initial_plan_from_task,
    system_prompt_from_manifest,
)
from .providers.base import LlmProvider


WorkspaceFactory = Callable[[Any, ArtifactWorkspaceSpec], ArtifactWorkspace]
ClientFactory = Callable[[Any], Any]
ProviderFactory = Callable[[Any], LlmProvider]
AgentConfigFactory = Callable[[Any], AgentConfig]
InitialPlanFactory = Callable[[Any], Sequence[str]]
SystemPromptFactory = Callable[[Any], str]
ArtifactAgentFactory = Callable[[Any, LlmProvider, AgentConfig, str, AdapterRegistry], Any]


def _default_agent_factory(
    client: Any,
    provider: LlmProvider,
    config: AgentConfig,
    system_prompt: str,
    adapters: AdapterRegistry,
) -> DurableAgent:
    return DurableAgent(client, provider, config=config, system_prompt=system_prompt, adapters=adapters)


def _task(context: Any) -> Any:
    task = getattr(context, "task", None)
    if task is None:
        raise BenchmarkExecutionError("execution context must expose task")
    for field in ("task_id", "category", "goal", "constraint", "metadata"):
        if not hasattr(task, field):
            raise BenchmarkExecutionError("execution context task is missing {}".format(field))
    if not isinstance(task.task_id, str) or not task.task_id.strip():
        raise BenchmarkExecutionError("task.task_id must be a non-empty string")
    if not isinstance(task.category, str) or not task.category.strip():
        raise BenchmarkExecutionError("task.category must be a non-empty string")
    if not isinstance(task.goal, str) or not task.goal.strip():
        raise BenchmarkExecutionError("task.goal must be a non-empty string")
    if not isinstance(task.constraint, str) or not task.constraint.strip():
        raise BenchmarkExecutionError("task.constraint must be a non-empty string")
    if not isinstance(task.metadata, Mapping):
        raise BenchmarkExecutionError("task.metadata must be an object")
    return task


def _checkpoint_cadence(context: Any) -> int:
    value = getattr(getattr(context, "manifest", None), "checkpoint_cadence", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BenchmarkExecutionError("manifest.checkpoint_cadence must be a positive integer")
    return value


def _reject_fault_schedule(context: Any) -> None:
    schedule = getattr(getattr(context, "manifest", None), "fault_schedule", None)
    if isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence):
        raise BenchmarkExecutionError("manifest.fault_schedule must be an array")
    if schedule:
        raise BenchmarkExecutionError(
            "artifact workspace executor cannot enact manifest.fault_schedule; use a fault-specific environment"
        )


def _projection_state(projection: Mapping[str, Any], task: Any) -> tuple[bool, bool]:
    cognitive = projection.get("cognitive")
    if not isinstance(cognitive, Mapping):
        return False, False
    goal_retained = cognitive.get("primary_goal") == task.goal
    constraints = cognitive.get("constraints")
    constraint_retained = isinstance(constraints, Sequence) and not isinstance(
        constraints, (str, bytes)
    ) and any(
        isinstance(item, Mapping) and item.get("content") == task.constraint for item in constraints
    )
    return goal_retained, constraint_retained


def _has_durable_adapter_result(
    events: Sequence[Mapping[str, Any]],
    *,
    adapter_name: str,
) -> bool:
    """Require a real adapter-created terminal event, not a model self-report."""

    for event in events:
        if event.get("type") != "tool_result_recorded":
            continue
        data = event.get("data")
        if not isinstance(data, Mapping):
            continue
        result = data.get("result")
        if not isinstance(result, Mapping):
            continue
        output = result.get("output")
        if (
            result.get("tool") == adapter_name
            and result.get("status") == "succeeded"
            and isinstance(output, Mapping)
            and output.get("verified") is True
        ):
            return True
    return False


class ArtifactWorkspaceJudge:
    """Turn a constrained workspace observation into an objective EpisodeResult."""

    def __call__(
        self,
        context: Any,
        run_result: RunResult,
        projection: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        adapter: ArtifactWorkspaceAdapter,
    ) -> dict[str, Any]:
        task = _task(context)
        if not isinstance(run_result, RunResult):
            raise BenchmarkExecutionError("artifact judge requires a RunResult")
        if not isinstance(projection, Mapping):
            raise BenchmarkExecutionError("artifact judge requires a projection object")
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            raise BenchmarkExecutionError("artifact judge requires an event sequence")
        if not isinstance(adapter, ArtifactWorkspaceAdapter):
            raise BenchmarkExecutionError("artifact judge requires an ArtifactWorkspaceAdapter")
        typed_events = [event for event in events if isinstance(event, Mapping)]
        if len(typed_events) != len(events):
            raise BenchmarkExecutionError("artifact judge event sequence must contain objects")
        observation = adapter.final_observation()
        if not isinstance(observation, ArtifactObservation):  # Defensive against adapter changes.
            raise BenchmarkExecutionError("artifact adapter returned an invalid final observation")
        goal_retained, constraint_retained = _projection_state(projection, task)
        durable_adapter_result = _has_durable_adapter_result(
            typed_events, adapter_name=adapter.name
        )
        state_completed = projection.get("state") == "completed" and run_result.state == "completed"
        workspace_executed = adapter.invocation_count > 0
        success = bool(
            state_completed
            and goal_retained
            and constraint_retained
            and workspace_executed
            and durable_adapter_result
            and observation.verified
        )
        return {
            "task_id": task.task_id,
            "category": task.category,
            "success": success,
            "goal_retained": goal_retained,
            "constraint_violations": int(not constraint_retained) + int(not observation.verified),
            "failure_actions": (),
            "known_failed_approaches": (),
            "recovery_attempted": False,
            "recovery_succeeded": False,
            "recovery_distance": None,
            "state_consistent": bool(
                goal_retained and constraint_retained and workspace_executed and durable_adapter_result
            ),
            "tokens": run_result.tokens_used,
            "anchors_injected": run_result.anchors_injected,
            "diagnostic": bounded_run_diagnostic(run_result.error),
        }


class DurableArtifactWorkspaceExecutor:
    """Run one frozen task through a durable policy and its artifact environment."""

    def __init__(
        self,
        *,
        client_factory: ClientFactory,
        provider_factory: ProviderFactory,
        workspace_factory: WorkspaceFactory,
        judge: ArtifactWorkspaceJudge = ArtifactWorkspaceJudge(),
        agent_config_factory: AgentConfigFactory = agent_config_from_condition,
        initial_plan_factory: InitialPlanFactory = initial_plan_from_task,
        system_prompt_factory: SystemPromptFactory = system_prompt_from_manifest,
        agent_factory: ArtifactAgentFactory = _default_agent_factory,
    ) -> None:
        for name, value in (
            ("client_factory", client_factory),
            ("provider_factory", provider_factory),
            ("workspace_factory", workspace_factory),
            ("judge", judge),
            ("agent_config_factory", agent_config_factory),
            ("initial_plan_factory", initial_plan_factory),
            ("system_prompt_factory", system_prompt_factory),
            ("agent_factory", agent_factory),
        ):
            if not callable(value):
                raise BenchmarkExecutionError("{} must be callable".format(name))
        self._client_factory = client_factory
        self._provider_factory = provider_factory
        self._workspace_factory = workspace_factory
        self._judge = judge
        self._agent_config_factory = agent_config_factory
        self._initial_plan_factory = initial_plan_factory
        self._system_prompt_factory = system_prompt_factory
        self._agent_factory = agent_factory

    def __call__(self, context: Any) -> dict[str, Any]:
        task, spec, config, plan, system_prompt = self._static_inputs(context)
        workspace = self._workspace_factory(context, spec)
        if not isinstance(workspace, ArtifactWorkspace):
            raise BenchmarkExecutionError("workspace_factory must return an ArtifactWorkspace")
        adapter = ArtifactWorkspaceAdapter(workspace)
        registry = AdapterRegistry([adapter])
        client = self._client_factory(context)
        provider = self._provider_factory(context)
        agent = self._agent_factory(client, provider, config, system_prompt, registry)
        run = getattr(agent, "run", None)
        if not callable(run):
            raise BenchmarkExecutionError("agent_factory result must expose run(goal, constraints, plan)")
        run_result = run(task.goal, constraints=(task.constraint,), plan=plan)
        if not isinstance(run_result, RunResult):
            raise BenchmarkExecutionError("agent.run must return RunResult")
        get_run = getattr(client, "get_run", None)
        events_method = getattr(client, "events", None)
        if not callable(get_run) or not callable(events_method):
            raise BenchmarkExecutionError("client must expose get_run(run_id) and events(run_id)")
        projection = get_run(run_result.run_id)
        events = events_method(run_result.run_id)
        if not isinstance(projection, Mapping):
            raise BenchmarkExecutionError("client.get_run must return a projection object")
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            raise BenchmarkExecutionError("client.events must return an event sequence")
        return self._judge(context, run_result, projection, events, adapter)

    def preflight(self, context: Any) -> dict[str, Any]:
        """Validate immutable task/agent inputs without creating a workspace or provider."""

        task, spec, config, plan, _system_prompt = self._static_inputs(context)
        return {
            "executor": "durable_artifact_workspace_executor",
            "task_id": task.task_id,
            "category": task.category,
            "checkpoint_cadence": config.checkpoint_every_steps,
            "initial_plan_steps": len(plan),
            "agent": {
                "max_steps": config.max_steps,
                "token_budget": config.token_budget,
                "wall_time_budget_ms": config.wall_time_budget_ms,
                "checkpoint_on_finish": config.checkpoint_on_finish,
                "checkpoint_every_steps": config.checkpoint_every_steps,
                "semantic_memory_limit": config.semantic_memory_limit,
                "learned_intervention_policy_configured": config.learned_intervention_policy is not None,
                "anchor_strategy": config.anchor_strategy,
                "anchor_policy_revision": config.anchor_policy_revision,
            },
            "artifact_environment": spec.preflight_summary(),
        }

    def _static_inputs(
        self, context: Any
    ) -> tuple[Any, ArtifactWorkspaceSpec, AgentConfig, tuple[str, ...], str]:
        task = _task(context)
        _reject_fault_schedule(context)
        try:
            spec = ArtifactWorkspaceSpec.from_metadata(task.metadata)
        except ArtifactWorkspaceError as error:
            raise BenchmarkExecutionError("invalid artifact workspace task metadata: {}".format(error)) from error
        checkpoint_cadence = _checkpoint_cadence(context)
        plan = tuple(self._initial_plan_factory(context))
        if not plan or not all(isinstance(item, str) and item.strip() for item in plan):
            raise BenchmarkExecutionError("artifact workspace tasks require a non-empty initial plan")
        system_prompt = self._system_prompt_factory(context)
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise BenchmarkExecutionError("system_prompt_factory must return a non-empty string")
        if "invoke_adapter" not in system_prompt or spec.adapter_name not in system_prompt:
            raise BenchmarkExecutionError(
                "frozen artifact workspace prompt must name invoke_adapter and the task adapter"
            )
        config = self._agent_config_factory(context)
        if not isinstance(config, AgentConfig):
            raise BenchmarkExecutionError("agent_config_factory must return AgentConfig")
        config = replace(config, checkpoint_every_steps=checkpoint_cadence)
        try:
            validate_agent_config(config)
        except ValueError as error:
            raise BenchmarkExecutionError("invalid agent configuration: {}".format(error)) from error
        return task, spec, config, plan, system_prompt


def workspace_from_root(root: Path) -> WorkspaceFactory:
    """Create a factory binding artifacts to a caller-owned persistent root."""

    if not isinstance(root, Path):
        raise BenchmarkExecutionError("artifact workspace root must be a pathlib.Path")

    def build(context: Any, spec: ArtifactWorkspaceSpec) -> ArtifactWorkspace:
        manifest = getattr(context, "manifest", None)
        run_id = getattr(manifest, "run_id", None)
        task = _task(context)
        if not isinstance(run_id, str) or not run_id.strip():
            raise BenchmarkExecutionError("execution context manifest.run_id must be a non-empty string")
        try:
            return ArtifactWorkspace.for_attempt(root, run_id=run_id, task_id=task.task_id, spec=spec)
        except ArtifactWorkspaceError as error:
            raise BenchmarkExecutionError("could not prepare artifact workspace: {}".format(error)) from error

    return build


__all__ = [
    "ArtifactWorkspaceJudge",
    "DurableArtifactWorkspaceExecutor",
    "workspace_from_root",
]

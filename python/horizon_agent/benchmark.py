"""Model-backed, objectively judged HorizonBench executor building blocks.

The matrix runner deliberately knows nothing about model providers or task
environments.  This module supplies the complementary bridge: it runs one
``DurableAgent`` against a frozen benchmark task, collects the durable
projection/event trace, and hands that trace to an objective judge.  The judge
is responsible for task success; an LLM declaring itself finished is never
treated as a benchmark result by this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from .agent import AgentConfig, DurableAgent, RunResult, validate_agent_config
from .providers.base import LlmProvider


Json = Any


class BenchmarkExecutionError(ValueError):
    """Raised for an invalid agent-executor configuration or durable trace."""


class ObjectiveEpisodeJudge(Protocol):
    """Turn a durable agent trace into a benchmark outcome.

    ``context`` is the frozen HorizonBench execution context. ``run_result``
    comes from the Python agent loop, while ``projection`` and ``events`` come
    from the Rust runtime after that loop ends. A judge may return an
    ``EpisodeResult`` or its JSON-compatible mapping; the matrix runner checks
    its task/category identity at the outer boundary.
    """

    def __call__(
        self,
        context: Any,
        run_result: RunResult,
        projection: Mapping[str, Json],
        events: Sequence[Mapping[str, Json]],
    ) -> Json:
        """Evaluate one completed (or interrupted) durable agent episode."""


AgentFactory = Callable[[Any, LlmProvider, AgentConfig, str], Any]
ClientFactory = Callable[[Any], Any]
ProviderFactory = Callable[[Any], LlmProvider]
AgentConfigFactory = Callable[[Any], AgentConfig]
InitialPlanFactory = Callable[[Any], Sequence[str]]
SystemPromptFactory = Callable[[Any], str]


def _default_agent_factory(
    client: Any,
    provider: LlmProvider,
    config: AgentConfig,
    system_prompt: str,
) -> DurableAgent:
    return DurableAgent(client, provider, config=config, system_prompt=system_prompt)


def agent_config_from_condition(context: Any) -> AgentConfig:
    """Build the JSON-configurable part of :class:`AgentConfig` from a condition.

    A matrix condition can carry an ``agent`` object with only the primitive
    controls below. Learned policy objects are deliberately not deserialized
    from arbitrary JSON; experiments that use one should provide a custom
    ``agent_config_factory`` which loads a separately versioned artifact.
    """

    condition = getattr(getattr(context, "manifest", None), "condition", None)
    configuration = getattr(condition, "configuration", None)
    if not isinstance(configuration, Mapping):
        raise BenchmarkExecutionError("execution context must expose manifest.condition.configuration")
    # v0.5/v0.6 matrix conditions stored simple controls directly in
    # ``configuration``. Keep those effective while allowing later experiments
    # to put the full agent subsection under an explicit key.
    raw = configuration.get("agent")
    if raw is None:
        raw = {
            key: configuration[key]
            for key in (
                "max_steps",
                "token_budget",
                "wall_time_budget_ms",
                "checkpoint_on_finish",
                "semantic_memory_limit",
                "anchor_strategy",
            )
            if key in configuration
        }
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise BenchmarkExecutionError("condition configuration field agent must be an object")
    allowed = {
        "max_steps",
        "token_budget",
        "wall_time_budget_ms",
        "checkpoint_on_finish",
        "semantic_memory_limit",
        "anchor_strategy",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise BenchmarkExecutionError(
            "condition configuration.agent has unrecognized fields: {}".format(", ".join(unknown))
        )
    values: dict[str, Any] = {}
    for name in ("max_steps", "token_budget", "wall_time_budget_ms", "semantic_memory_limit"):
        if name not in raw:
            continue
        value = raw[name]
        if value is None and name in {"token_budget", "wall_time_budget_ms"}:
            values[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise BenchmarkExecutionError("condition configuration.agent.{} must be an integer".format(name))
        if name == "max_steps" and value < 1:
            raise BenchmarkExecutionError("condition configuration.agent.max_steps must be positive")
        if name != "max_steps" and value < 0:
            raise BenchmarkExecutionError("condition configuration.agent.{} cannot be negative".format(name))
        values[name] = value
    if "checkpoint_on_finish" in raw:
        value = raw["checkpoint_on_finish"]
        if not isinstance(value, bool):
            raise BenchmarkExecutionError("condition configuration.agent.checkpoint_on_finish must be boolean")
        values["checkpoint_on_finish"] = value
    if "anchor_strategy" in raw:
        value = raw["anchor_strategy"]
        if not isinstance(value, str):
            raise BenchmarkExecutionError("condition configuration.agent.anchor_strategy must be a string")
        values["anchor_strategy"] = value
        if value != "runtime_heuristic":
            policy_revision = getattr(condition, "policy_revision", None)
            if not isinstance(policy_revision, str) or not policy_revision.strip():
                raise BenchmarkExecutionError(
                    "fixed anchor_strategy requires a non-empty manifest.condition.policy_revision"
                )
            values["anchor_policy_revision"] = policy_revision
    try:
        return AgentConfig(**values)
    except ValueError as error:
        raise BenchmarkExecutionError("invalid agent configuration: {}".format(error)) from error


def initial_plan_from_task(context: Any) -> tuple[str, ...]:
    """Use optional ``metadata.initial_plan`` without making hidden prompt state."""

    task = _task_from_context(context)
    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise BenchmarkExecutionError("benchmark task metadata must be an object")
    raw = metadata.get("initial_plan", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise BenchmarkExecutionError("benchmark task metadata.initial_plan must be an array of strings")
    plan = tuple(raw)
    if not all(isinstance(item, str) and item.strip() for item in plan):
        raise BenchmarkExecutionError("benchmark task metadata.initial_plan must contain non-empty strings")
    return plan


def system_prompt_from_manifest(context: Any) -> str:
    """Return the immutable prompt that is hashed into this matrix row."""

    prompt = getattr(getattr(context, "manifest", None), "prompt", None)
    text = getattr(prompt, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise BenchmarkExecutionError("execution context must expose manifest.prompt.text")
    return text


class DurableAgentEpisodeExecutor:
    """Run a real durable agent once, then delegate success to an objective judge.

    Factories make the class usable both with an HTTP ``HorizonClient`` and an
    optional native client, while keeping provider credentials and task
    environment details out of the frozen matrix itself.  One executor object
    is normally exposed by a local ``module:callable`` plugin to
    ``execute_matrix.py``.
    """

    def __init__(
        self,
        *,
        client_factory: ClientFactory,
        provider_factory: ProviderFactory,
        judge: ObjectiveEpisodeJudge,
        agent_config_factory: AgentConfigFactory = agent_config_from_condition,
        initial_plan_factory: InitialPlanFactory = initial_plan_from_task,
        system_prompt_factory: SystemPromptFactory = system_prompt_from_manifest,
        agent_factory: AgentFactory = _default_agent_factory,
    ) -> None:
        for name, value in (
            ("client_factory", client_factory),
            ("provider_factory", provider_factory),
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
        self._judge = judge
        self._agent_config_factory = agent_config_factory
        self._initial_plan_factory = initial_plan_factory
        self._system_prompt_factory = system_prompt_factory
        self._agent_factory = agent_factory

    def __call__(self, context: Any) -> Json:
        task, checkpoint_cadence, plan, system_prompt = self._static_inputs(context)
        client = self._client_factory(context)
        provider = self._provider_factory(context)
        config = self._effective_agent_config(context, checkpoint_cadence)
        agent = self._agent_factory(client, provider, config, system_prompt)
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
        projection_snapshot = _projection_snapshot(projection)
        event_snapshots = _event_snapshots(events)
        return self._judge(context, run_result, projection_snapshot, event_snapshots)

    def preflight(self, context: Any) -> dict[str, Json]:
        """Validate frozen agent inputs without constructing runtime/model clients."""

        task, checkpoint_cadence, plan, _system_prompt = self._static_inputs(context)
        config = self._effective_agent_config(context, checkpoint_cadence)
        return {
            "executor": "durable_agent_episode_executor",
            "task_id": task.task_id,
            "category": task.category,
            "checkpoint_cadence": checkpoint_cadence,
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
        }

    def _static_inputs(self, context: Any) -> tuple[Any, int, tuple[str, ...], str]:
        """Build prompt/task inputs shared by execution and static preflight."""

        task = _task_from_context(context)
        checkpoint_cadence = _checkpoint_cadence_from_manifest(context)
        plan = tuple(self._initial_plan_factory(context))
        if not all(isinstance(item, str) and item.strip() for item in plan):
            raise BenchmarkExecutionError("initial_plan_factory must return non-empty strings")
        system_prompt = self._system_prompt_factory(context)
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise BenchmarkExecutionError("system_prompt_factory must return a non-empty string")
        return task, checkpoint_cadence, plan, system_prompt

    def _effective_agent_config(self, context: Any, checkpoint_cadence: int) -> AgentConfig:
        """Validate the condition-derived agent controls at one shared boundary."""

        config = self._agent_config_factory(context)
        if not isinstance(config, AgentConfig):
            raise BenchmarkExecutionError("agent_config_factory must return AgentConfig")
        config = replace(config, checkpoint_every_steps=checkpoint_cadence)
        try:
            validate_agent_config(config)
        except ValueError as error:
            raise BenchmarkExecutionError("invalid agent configuration: {}".format(error)) from error
        return config


@dataclass(frozen=True)
class DurableTraceExpectation:
    """Declarative checks for a runtime-trace judge, stored in task metadata."""

    required_event_types: tuple[str, ...] = ()
    forbidden_event_types: tuple[str, ...] = ()
    required_evidence_terms: tuple[str, ...] = ()
    required_failure_approaches: tuple[str, ...] = ()
    require_checkpoint: bool = False
    require_recovery: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Json]) -> "DurableTraceExpectation":
        allowed = {
            "required_event_types",
            "forbidden_event_types",
            "required_evidence_terms",
            "required_failure_approaches",
            "require_checkpoint",
            "require_recovery",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise BenchmarkExecutionError(
                "durable trace expectation has unrecognized fields: {}".format(", ".join(unknown))
            )
        return cls(
            required_event_types=_string_tuple(value.get("required_event_types", ()), "required_event_types"),
            forbidden_event_types=_string_tuple(value.get("forbidden_event_types", ()), "forbidden_event_types"),
            required_evidence_terms=_string_tuple(
                value.get("required_evidence_terms", ()), "required_evidence_terms"
            ),
            required_failure_approaches=_string_tuple(
                value.get("required_failure_approaches", ()), "required_failure_approaches"
            ),
            require_checkpoint=_strict_bool(value.get("require_checkpoint", False), "require_checkpoint"),
            require_recovery=_strict_bool(value.get("require_recovery", False), "require_recovery"),
        )


class DurableTraceJudge:
    """Score durable runtime facts declared by ``horizon_trace_expectation``.

    It objectively validates the projection and immutable event trace. It is a
    useful judge for runtime-mechanism tasks, but it is not a substitute for an
    artifact/test-based judge on a public domain benchmark.
    """

    def __init__(self, *, metadata_key: str = "horizon_trace_expectation") -> None:
        if not isinstance(metadata_key, str) or not metadata_key.strip():
            raise BenchmarkExecutionError("metadata_key must be a non-empty string")
        self.metadata_key = metadata_key

    def __call__(
        self,
        context: Any,
        run_result: RunResult,
        projection: Mapping[str, Json],
        events: Sequence[Mapping[str, Json]],
    ) -> dict[str, Json]:
        task = _task_from_context(context)
        expectation = self._expectation(task)
        if not isinstance(run_result, RunResult):
            raise BenchmarkExecutionError("judge requires a RunResult")
        if not isinstance(projection, Mapping):
            raise BenchmarkExecutionError("judge projection must be an object")
        event_types = _event_types(events)
        cognitive = projection.get("cognitive")
        if not isinstance(cognitive, Mapping):
            cognitive = {}
        goal_retained = cognitive.get("primary_goal") == task.goal
        constraint_retained = task.constraint in _record_contents(cognitive.get("constraints"))
        evidence = _record_contents(cognitive.get("evidence"))
        failure_approaches = _record_contents(cognitive.get("failed_attempts"), field="approach")
        missing_events = [item for item in expectation.required_event_types if item not in event_types]
        missing_evidence = _missing_terms(expectation.required_evidence_terms, evidence)
        missing_failures = _missing_terms(expectation.required_failure_approaches, failure_approaches)
        checkpoint_seen = "checkpoint_created" in event_types
        recovery_attempted = "agent_recovered" in event_types
        forbidden_count = sum(event_types.count(item) for item in expectation.forbidden_event_types)
        expectation_met = (
            not missing_events
            and not missing_evidence
            and not missing_failures
            and (not expectation.require_checkpoint or checkpoint_seen)
            and (not expectation.require_recovery or recovery_attempted)
            and forbidden_count == 0
        )
        terminal_completed = run_result.state == "completed" and projection.get("state") == "completed"
        state_consistent = goal_retained and constraint_retained and expectation_met
        success = terminal_completed and state_consistent
        recovery_succeeded = recovery_attempted and terminal_completed and state_consistent
        violations = forbidden_count + int(not constraint_retained)
        return {
            "task_id": task.task_id,
            "category": task.category,
            "success": success,
            "goal_retained": goal_retained,
            "constraint_violations": violations,
            "failure_actions": tuple(failure_approaches),
            "known_failed_approaches": (),
            "recovery_attempted": recovery_attempted,
            "recovery_succeeded": recovery_succeeded,
            # A trace alone cannot measure the benchmark's "extra steps to
            # regain pre-fault progress" without inventing a progress oracle.
            # Leave it absent rather than substitute event count; an
            # environment-specific judge can report the real distance.
            "recovery_distance": None,
            "state_consistent": state_consistent,
            "tokens": _non_negative_int(run_result.tokens_used, "run_result.tokens_used"),
            "anchors_injected": _non_negative_int(
                run_result.anchors_injected, "run_result.anchors_injected"
            ),
            "diagnostic": bounded_run_diagnostic(run_result.error),
        }

    def preflight(self, context: Any) -> dict[str, Json]:
        """Validate a task's immutable trace expectations before execution."""

        task = _task_from_context(context)
        expectation = self._expectation(task)
        return {
            "required_event_types": list(expectation.required_event_types),
            "forbidden_event_types": list(expectation.forbidden_event_types),
            "required_evidence_terms": list(expectation.required_evidence_terms),
            "required_failure_approaches": list(expectation.required_failure_approaches),
            "require_checkpoint": expectation.require_checkpoint,
            "require_recovery": expectation.require_recovery,
        }

    def _expectation(self, task: Any) -> DurableTraceExpectation:
        metadata = getattr(task, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise BenchmarkExecutionError("benchmark task metadata must be an object")
        value = metadata.get(self.metadata_key)
        if not isinstance(value, Mapping):
            raise BenchmarkExecutionError(
                "benchmark task metadata.{} must be an object for DurableTraceJudge".format(
                    self.metadata_key
                )
            )
        return DurableTraceExpectation.from_mapping(value)


def _task_from_context(context: Any) -> Any:
    task = getattr(context, "task", None)
    for field in ("task_id", "category", "goal", "constraint"):
        value = getattr(task, field, None)
        if not isinstance(value, str) or not value.strip():
            raise BenchmarkExecutionError("execution context task must expose non-empty {}".format(field))
    return task


def _checkpoint_cadence_from_manifest(context: Any) -> int:
    """Bind only execution controls the generic agent bridge can truly enact.

    The runtime can checkpoint after model decision boundaries, but it cannot
    safely invent a task-environment crash/restart protocol from arbitrary
    JSON. Rejecting a non-empty fault schedule is intentional: callers must
    supply a task-specific executor that performs and observes those faults.
    """

    manifest = getattr(context, "manifest", None)
    cadence = getattr(manifest, "checkpoint_cadence", None)
    if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 1:
        raise BenchmarkExecutionError("execution context must expose positive manifest.checkpoint_cadence")
    schedule = getattr(manifest, "fault_schedule", None)
    if isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence):
        raise BenchmarkExecutionError("execution context must expose manifest.fault_schedule as an array")
    if schedule:
        raise BenchmarkExecutionError(
            "DurableAgentEpisodeExecutor cannot enact a non-empty manifest.fault_schedule; "
            "use a task-specific environment executor"
        )
    return cadence


def _projection_snapshot(value: Any) -> Mapping[str, Json]:
    if not isinstance(value, Mapping):
        raise BenchmarkExecutionError("client get_run(run_id) must return an object")
    return dict(value)


def _event_snapshots(value: Any) -> tuple[Mapping[str, Json], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BenchmarkExecutionError("client events(run_id) must return an array of objects")
    snapshots: list[Mapping[str, Json]] = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, Mapping):
            raise BenchmarkExecutionError("event {} must be an object".format(index))
        snapshots.append(dict(item))
    return tuple(snapshots)


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BenchmarkExecutionError("durable trace expectation.{} must be an array of strings".format(field))
    items = tuple(value)
    if not all(isinstance(item, str) and item.strip() for item in items):
        raise BenchmarkExecutionError(
            "durable trace expectation.{} must contain non-empty strings".format(field)
        )
    if len(set(items)) != len(items):
        raise BenchmarkExecutionError("durable trace expectation.{} contains duplicates".format(field))
    return items


def _strict_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkExecutionError("durable trace expectation.{} must be boolean".format(field))
    return value


def _event_types(events: Sequence[Mapping[str, Json]]) -> list[str]:
    event_types: list[str] = []
    for index, event in enumerate(events, 1):
        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise BenchmarkExecutionError("event {} has no non-empty type".format(index))
        event_types.append(event_type)
    return event_types


def _record_contents(value: Any, *, field: str = "content") -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    contents: list[str] = []
    for record in value:
        if not isinstance(record, Mapping):
            continue
        item = record.get(field)
        if isinstance(item, str) and item.strip():
            contents.append(item)
    return tuple(contents)


def _missing_terms(required: Sequence[str], observed: Sequence[str]) -> list[str]:
    searchable = "\n".join(observed).casefold()
    return [term for term in required if term.casefold() not in searchable]


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkExecutionError("{} must be a non-negative integer".format(label))
    return value


def bounded_run_diagnostic(value: Any) -> Optional[str]:
    """Return a safe operational class rather than provider/runtime error text."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise BenchmarkExecutionError("run_result.error must be a string or null")
    normalized = " ".join(value.split())
    if not normalized:
        return None
    if normalized in {"configured budget exhausted", "agent reached max_steps"}:
        return normalized
    if normalized.startswith("LLM policy call failed:"):
        return "LLM policy call failed"
    if normalized.startswith("action `") and " rejected:" in normalized:
        return "policy action rejected"
    return "agent stopped with an unclassified error"


# Kept as a private compatibility alias for existing callers/tests while new
# task-environment judges use the explicit public name.
_bounded_run_diagnostic = bounded_run_diagnostic


__all__ = [
    "AgentConfigFactory",
    "AgentFactory",
    "BenchmarkExecutionError",
    "ClientFactory",
    "DurableAgentEpisodeExecutor",
    "DurableTraceExpectation",
    "DurableTraceJudge",
    "InitialPlanFactory",
    "ObjectiveEpisodeJudge",
    "ProviderFactory",
    "SystemPromptFactory",
    "agent_config_from_condition",
    "bounded_run_diagnostic",
    "initial_plan_from_task",
    "system_prompt_from_manifest",
]

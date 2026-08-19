//! Horizon's durable cognitive runtime.
//!
//! The core invariant is deliberately visible in [`HorizonRuntime::dispatch`]:
//!
//! ```text
//! Command -> validate -> event -> persist -> apply projection
//! ```
//!
//! No public method exposes mutable agent state. A projection is recovered from
//! a checkpoint plus its immutable event suffix, making every run auditable and
//! restartable.

mod http;

use std::{collections::BTreeMap, sync::Arc};

use chrono::Utc;
use horizon_core::{
    AgentState, CURRENT_EVENT_SCHEMA_VERSION, CommandError, EventId, EventKind, EventRecord,
    FailureRecord, InterventionAssessment, NewEvent, ProjectionError, RunId, RunProjection,
    RuntimeCommand, SemanticMemoryHit, SemanticMemoryRetrieval, TaskId, TaskRecord,
    TaskSpecValidationError, TaskStatus, ToolResult, ToolResultStatus,
};
use horizon_memory::{
    DecaySignals, SEMANTIC_RETRIEVAL_ALGORITHM, SemanticMemoryRetrievalPolicy, StateAnchorPolicy,
    matching_failure,
};
use horizon_process::{ProcessOutput, ProcessRequest, ProcessSupervisor};
use horizon_scheduler::{AsyncTaskScheduler, TaskDag, TaskExecutionResult, TaskExecutor};
use horizon_store::{
    CheckpointCompaction, CheckpointRecord, EventStore, SnapshotEncoding, StoreError,
    decode_snapshot, encode_snapshot,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{debug, info};

pub use http::{HttpServerError, serve};

/// Bound the size of one audit event and one LLM retrieval injection. Callers
/// may request fewer hits, but never make a single boundary unbounded.
const MAX_SEMANTIC_MEMORY_RETRIEVAL_HITS: usize = 8;
const MAX_SEMANTIC_MEMORY_QUERY_CHARS: usize = 4_096;
const MAX_SEMANTIC_MEMORY_CONTENT_CHARS: usize = 8_192;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Snapshot cadence in appended domain events. Set to zero to disable
    /// automatic checkpoints; manual checkpoints remain available.
    pub checkpoint_every_events: u64,
    pub state_anchor_policy: StateAnchorPolicy,
    /// Maximum concurrently running process-backed tasks.
    pub task_concurrency: usize,
    /// Per-checkpoint payload encoding. Stored metadata makes this safe to
    /// change across runtime upgrades.
    pub snapshot_encoding: SnapshotEncoding,
    /// Bound retained checkpoint snapshots after each successful snapshot. The
    /// immutable event log is never compacted. `None` preserves every snapshot.
    pub checkpoint_retention: Option<usize>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            checkpoint_every_events: 12,
            state_anchor_policy: StateAnchorPolicy::default(),
            task_concurrency: 4,
            snapshot_encoding: SnapshotEncoding::ZstdJson,
            checkpoint_retention: Some(8),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CommandOutcome {
    pub events: Vec<EventRecord>,
    pub projection: RunProjection,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecoveryOutcome {
    pub restored_from_checkpoint: Option<u64>,
    pub events: Vec<EventRecord>,
    pub projection: RunProjection,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReadyTaskOutcome {
    pub task_id: TaskId,
    pub skipped_as_idempotent: bool,
    pub output: Result<String, String>,
}

/// Generic runtime so storage can be swapped without rewriting domain code.
pub struct HorizonRuntime<S: EventStore + ?Sized> {
    store: Arc<S>,
    config: RuntimeConfig,
    supervisor: ProcessSupervisor,
    /// Process tasks may run concurrently, while every state mutation observes
    /// one serial event history.
    mutation_gate: Mutex<()>,
}

impl<S: EventStore + ?Sized> HorizonRuntime<S> {
    #[must_use]
    pub fn new(store: Arc<S>) -> Self {
        Self::with_config(store, RuntimeConfig::default())
    }

    #[must_use]
    pub fn with_config(store: Arc<S>, mut config: RuntimeConfig) -> Self {
        // The CLI and native binding both use zero as the user-friendly
        // spelling for "keep every snapshot". Preserve that behavior for
        // callers constructing RuntimeConfig directly instead of allowing a
        // later automatic checkpoint to fail after it was already persisted.
        if config.checkpoint_retention == Some(0) {
            config.checkpoint_retention = None;
        }
        Self { store, config, supervisor: ProcessSupervisor, mutation_gate: Mutex::new(()) }
    }

    #[must_use]
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    #[must_use]
    pub fn store(&self) -> &Arc<S> {
        &self.store
    }

    /// Creates a run at the `Created` state. A policy must explicitly move it to
    /// `Planning`; this prevents accidental direct execution.
    pub async fn create_run(
        &self,
        goal: impl Into<String>,
    ) -> Result<CommandOutcome, RuntimeError> {
        let _gate = self.mutation_gate.lock().await;
        let goal = goal.into();
        if goal.trim().is_empty() {
            return Err(RuntimeError::Command(CommandError::EmptyGoal));
        }
        let run_id = RunId::new();
        let mut projection = RunProjection::empty(run_id);
        let event = self.append_apply(&mut projection, EventKind::RunCreated { goal }).await?;
        Ok(CommandOutcome { events: vec![event], projection })
    }

    /// Rebuilds a projection without changing a run. Read operations must use
    /// this rather than `recover`, which intentionally records recovery events.
    pub async fn projection(&self, run_id: RunId) -> Result<RunProjection, RuntimeError> {
        let _gate = self.mutation_gate.lock().await;
        Ok(self.load_projection_with_origin(run_id).await?.projection)
    }

    pub async fn events(&self, run_id: RunId) -> Result<Vec<EventRecord>, RuntimeError> {
        let _gate = self.mutation_gate.lock().await;
        Ok(self.store.load_events(run_id, 0).await?)
    }

    /// Safely discard older *checkpoint snapshots* while preserving every
    /// immutable event. This is an operational storage control, not history
    /// deletion, and does not change the recovered projection.
    pub async fn compact_checkpoints(
        &self,
        run_id: RunId,
        retain_latest: usize,
    ) -> Result<CheckpointCompaction, RuntimeError> {
        let _gate = self.mutation_gate.lock().await;
        let projection = self.load_projection_with_origin(run_id).await?.projection;
        if projection.sequence == 0 {
            return Err(RuntimeError::UnknownRun(run_id));
        }
        Ok(self.store.compact_checkpoints(run_id, retain_latest).await?)
    }

    pub async fn list_runs(&self) -> Result<Vec<RunProjection>, RuntimeError> {
        let _gate = self.mutation_gate.lock().await;
        let run_ids = self.store.list_run_ids().await?;
        let mut runs = Vec::with_capacity(run_ids.len());
        for run_id in run_ids {
            runs.push(self.load_projection_with_origin(run_id).await?.projection);
        }
        Ok(runs)
    }

    /// Converts a runtime command to one or more immutable events.
    pub async fn dispatch(
        &self,
        run_id: RunId,
        command: RuntimeCommand,
    ) -> Result<CommandOutcome, RuntimeError> {
        let _gate = self.mutation_gate.lock().await;
        let mut projection = self.load_projection_with_origin(run_id).await?.projection;
        if projection.sequence == 0 {
            return Err(RuntimeError::UnknownRun(run_id));
        }
        let event_kinds = self.validate_command(&projection, command)?;
        let events =
            self.append_apply_many_without_auto_checkpoint(&mut projection, event_kinds).await?;
        self.maybe_automatic_checkpoint(&mut projection).await?;
        Ok(CommandOutcome { events, projection })
    }

    /// Stores a snapshot after recording an audit event. The snapshot includes
    /// that event, so loading it never leaves an unexplained sequence gap.
    pub async fn checkpoint(&self, run_id: RunId) -> Result<CommandOutcome, RuntimeError> {
        let _gate = self.mutation_gate.lock().await;
        let mut projection = self.load_projection_with_origin(run_id).await?.projection;
        if projection.sequence == 0 {
            return Err(RuntimeError::UnknownRun(run_id));
        }
        let checkpoint_sequence = projection.sequence;
        let event = self
            .append_apply_without_auto_checkpoint(
                &mut projection,
                EventKind::CheckpointCreated { sequence: checkpoint_sequence + 1 },
            )
            .await?;
        self.persist_snapshot(&projection).await?;
        Ok(CommandOutcome { events: vec![event], projection })
    }

    /// Reconstructs from the latest checkpoint plus event suffix and records a
    /// recovery boundary. A subsequent policy can transition `Recovering` to
    /// `Planning` or `Executing` explicitly.
    pub async fn recover(&self, run_id: RunId) -> Result<RecoveryOutcome, RuntimeError> {
        let _gate = self.mutation_gate.lock().await;
        let loaded = self.load_projection_with_origin(run_id).await?;
        let mut projection = loaded.projection;
        if projection.sequence == 0 {
            return Err(RuntimeError::UnknownRun(run_id));
        }
        if projection.state.is_terminal() {
            return Ok(RecoveryOutcome {
                restored_from_checkpoint: loaded.checkpoint_sequence,
                events: Vec::new(),
                projection,
            });
        }

        let from_sequence = projection.sequence;
        let mut event_kinds = Vec::new();
        if projection.state != AgentState::Recovering {
            projection.state.validate_transition(AgentState::Recovering)?;
            event_kinds.push(EventKind::StateTransitioned {
                from: projection.state,
                to: AgentState::Recovering,
            });
        }
        event_kinds.push(EventKind::AgentRecovered { from_sequence });

        // Snapshot interruption candidates before simulating any recovery event.
        let interrupted: Vec<_> = projection
            .tasks
            .values()
            .filter(|task| task.status == TaskStatus::Running)
            .map(|task| {
                (
                    task.spec.id,
                    task.spec.title.clone(),
                    task.spec
                        .operation_id
                        .clone()
                        .unwrap_or_else(|| format!("task:{}", task.spec.id)),
                    task.attempt,
                    task.spec.max_retries,
                )
            })
            .collect();
        for (task_id, title, operation_id, attempt, max_retries) in interrupted {
            if projection.is_operation_completed(&operation_id) {
                event_kinds.push(EventKind::TaskSucceeded {
                    task_id,
                    output: format!("recovered completed idempotent operation {operation_id}"),
                });
                continue;
            }
            let retryable = attempt <= max_retries;
            let error = "task interrupted by runtime recovery".to_owned();
            event_kinds.push(EventKind::TaskFailed {
                task_id,
                error: error.clone(),
                retryable,
                operation_id: Some(operation_id.clone()),
            });
            event_kinds.push(EventKind::FailureRemembered {
                failure: FailureRecord {
                    id: format!("recovery:{task_id}:attempt:{attempt}"),
                    approach: title,
                    reason: error,
                    operation_id: Some(operation_id),
                    retryable,
                },
            });
            if retryable {
                event_kinds.push(EventKind::TaskRetried { task_id, attempt });
            }
        }

        // Evaluate the State Anchor against the cognitive state that recovery
        // will produce, then commit the entire recovery boundary atomically.
        let predicted = simulate_events(&projection, &event_kinds)?;
        if let Some(anchor) = self.config.state_anchor_policy.decide(
            &predicted.cognitive,
            &DecaySignals {
                steps_since_anchor: predicted
                    .sequence
                    .saturating_sub(predicted.last_anchor_sequence),
                recovered_session: true,
                ..Default::default()
            },
        ) {
            event_kinds.push(EventKind::StateAnchorInjected { anchor });
        }
        let events =
            self.append_apply_many_without_auto_checkpoint(&mut projection, event_kinds).await?;
        self.maybe_automatic_checkpoint(&mut projection).await?;
        Ok(RecoveryOutcome {
            restored_from_checkpoint: loaded.checkpoint_sequence,
            events,
            projection,
        })
    }

    /// Decides whether current signals warrant an explicit cognitive-state
    /// injection. It does not force an anchor at every execution boundary.
    pub async fn intervene_if_needed(
        &self,
        run_id: RunId,
        mut signals: DecaySignals,
    ) -> Result<Option<CommandOutcome>, RuntimeError> {
        let _gate = self.mutation_gate.lock().await;
        let mut projection = self.load_projection_with_origin(run_id).await?.projection;
        if projection.sequence == 0 {
            return Err(RuntimeError::UnknownRun(run_id));
        }
        if signals.steps_since_anchor == 0 {
            signals.steps_since_anchor =
                projection.sequence.saturating_sub(projection.last_anchor_sequence);
        }
        let Some(anchor) = self.config.state_anchor_policy.decide(&projection.cognitive, &signals)
        else {
            return Ok(None);
        };
        let event =
            self.append_apply(&mut projection, EventKind::StateAnchorInjected { anchor }).await?;
        Ok(Some(CommandOutcome { events: vec![event], projection }))
    }

    /// Persist an externally selected state-decay assessment and, when the
    /// action requests it, inject a State Anchor rendered from this runtime's
    /// durable projection. Python may own model training/inference, but it
    /// cannot mutate state or fabricate anchor content outside this boundary.
    pub async fn apply_intervention(
        &self,
        run_id: RunId,
        assessment: InterventionAssessment,
    ) -> Result<CommandOutcome, RuntimeError> {
        self.dispatch(run_id, RuntimeCommand::ApplyIntervention { assessment }).await
    }

    /// Select and durably record a compact semantic-memory subset. The caller
    /// supplies only its query and limit; Rust ranks the projection's immutable
    /// memory catalog and records the exact result for replay/audit.
    pub async fn retrieve_semantic_memory(
        &self,
        run_id: RunId,
        query: impl Into<String>,
        limit: usize,
    ) -> Result<CommandOutcome, RuntimeError> {
        self.dispatch(run_id, RuntimeCommand::RetrieveSemanticMemory { query: query.into(), limit })
            .await
    }

    /// Returns a remembered non-retryable failure, allowing an external agent
    /// policy to block a known-bad strategy before it creates a side effect.
    pub async fn known_failure(
        &self,
        run_id: RunId,
        operation_or_approach: &str,
    ) -> Result<Option<FailureRecord>, RuntimeError> {
        let projection = self.projection(run_id).await?;
        Ok(matching_failure(&projection.cognitive.failed_attempts, operation_or_approach).cloned())
    }

    /// Executes currently ready process-backed tasks concurrently. Commands are
    /// still recorded through `dispatch`; the scheduler never mutates state.
    pub async fn execute_ready_tasks(
        self: &Arc<Self>,
        run_id: RunId,
    ) -> Result<Vec<ReadyTaskOutcome>, RuntimeError> {
        let projection = self.projection(run_id).await?;
        let dag = TaskDag::from_records(projection.tasks.values().cloned())
            .map_err(RuntimeError::Scheduler)?;
        let ready: Vec<_> = dag
            .ready()
            .into_iter()
            .filter(|task| task.command.as_ref().is_some_and(|command| !command.is_empty()))
            .collect();
        if ready.is_empty() {
            return Ok(Vec::new());
        }
        let scheduler = AsyncTaskScheduler::new(self.config.task_concurrency);
        let executor = Arc::new(RuntimeTaskExecutor { runtime: Arc::clone(self), run_id });
        let results = scheduler.execute(ready, executor).await;
        Ok(results
            .into_iter()
            .map(|result| ReadyTaskOutcome {
                task_id: result.task_id,
                skipped_as_idempotent: result
                    .output
                    .as_ref()
                    .is_ok_and(|message| message.starts_with("[idempotent]")),
                output: result.output,
            })
            .collect())
    }

    fn validate_command(
        &self,
        projection: &RunProjection,
        command: RuntimeCommand,
    ) -> Result<Vec<EventKind>, RuntimeError> {
        match command {
            RuntimeCommand::Transition { to } => {
                projection.state.validate_transition(to)?;
                Ok(vec![EventKind::StateTransitioned { from: projection.state, to }])
            }
            RuntimeCommand::RegisterGoal { goal } => {
                if goal.trim().is_empty() {
                    return Err(CommandError::EmptyGoal.into());
                }
                Ok(vec![EventKind::GoalRegistered { goal }])
            }
            RuntimeCommand::AddConstraint { constraint } => {
                Ok(vec![EventKind::ConstraintAdded { constraint }])
            }
            RuntimeCommand::SetPlan { steps } => {
                if steps.is_empty() || steps.iter().any(|step| step.trim().is_empty()) {
                    return Err(CommandError::EmptyPlan.into());
                }
                Ok(vec![EventKind::PlanCreated { steps }])
            }
            RuntimeCommand::OpenSubgoal { subgoal } => {
                if subgoal.id.trim().is_empty() || subgoal.content.trim().is_empty() {
                    return Err(CommandError::EmptySubgoal.into());
                }
                if projection
                    .cognitive
                    .open_subgoals
                    .iter()
                    .chain(projection.cognitive.completed_subgoals.iter())
                    .any(|existing| existing.id == subgoal.id)
                {
                    return Err(CommandError::SubgoalAlreadyExists(subgoal.id).into());
                }
                Ok(vec![EventKind::SubgoalOpened { subgoal }])
            }
            RuntimeCommand::CompleteSubgoal { id } => {
                if !projection.cognitive.open_subgoals.iter().any(|subgoal| subgoal.id == id) {
                    return Err(CommandError::UnknownSubgoal(id).into());
                }
                Ok(vec![EventKind::SubgoalCompleted { id }])
            }
            RuntimeCommand::CreateTask { mut task } => {
                if task
                    .operation_id
                    .as_deref()
                    .is_none_or(|operation_id| operation_id.trim().is_empty())
                {
                    task.operation_id = Some(format!("task:{}", task.id));
                }
                if projection.tasks.contains_key(&task.id) {
                    return Err(CommandError::TaskAlreadyExists(task.id).into());
                }
                task.validate_execution()?;
                let operation_id =
                    task.operation_id.as_deref().expect("normalized immediately above");
                if projection
                    .tasks
                    .values()
                    .any(|existing| existing.spec.operation_id.as_deref() == Some(operation_id))
                {
                    return Err(CommandError::DuplicateOperationId(operation_id.to_owned()).into());
                }
                let mut records = projection.tasks.values().cloned().collect::<Vec<_>>();
                records.push(TaskRecord::from_spec(task.clone(), Utc::now()));
                TaskDag::from_records(records).map_err(RuntimeError::Scheduler)?;
                Ok(vec![EventKind::TaskCreated { task }])
            }
            RuntimeCommand::StartTask { task_id } => {
                let task = task(projection, task_id)?;
                if !dependencies_succeeded(projection, task) {
                    return Err(CommandError::TaskNotReady(task_id).into());
                }
                task.status.validate_transition(TaskStatus::Running)?;
                Ok(vec![EventKind::TaskStarted { task_id, attempt: task.attempt + 1 }])
            }
            RuntimeCommand::BlockTask { task_id, reason } => {
                let task = task(projection, task_id)?;
                task.status.validate_transition(TaskStatus::Blocked)?;
                if reason.trim().is_empty() {
                    return Err(RuntimeError::InvalidCommand("task block reason cannot be empty"));
                }
                Ok(vec![EventKind::TaskBlocked { task_id, reason }])
            }
            RuntimeCommand::UnblockTask { task_id } => {
                let task = task(projection, task_id)?;
                task.status.validate_transition(TaskStatus::Pending)?;
                Ok(vec![EventKind::TaskUnblocked { task_id }])
            }
            RuntimeCommand::SucceedTask { task_id, output } => {
                let task = task(projection, task_id)?;
                task.status.validate_transition(TaskStatus::Succeeded)?;
                Ok(vec![EventKind::TaskSucceeded { task_id, output }])
            }
            RuntimeCommand::FailTask { task_id, error, retryable, operation_id } => {
                let task = task(projection, task_id)?;
                task.status.validate_transition(TaskStatus::Failed)?;
                let failure = FailureRecord {
                    id: format!("task:{task_id}:attempt:{}", task.attempt),
                    approach: task.spec.title.clone(),
                    reason: error.clone(),
                    operation_id: operation_id.clone(),
                    retryable,
                };
                Ok(vec![
                    EventKind::TaskFailed { task_id, error, retryable, operation_id },
                    EventKind::FailureRemembered { failure },
                ])
            }
            RuntimeCommand::RetryTask { task_id } => {
                let task = task(projection, task_id)?;
                task.status.validate_transition(TaskStatus::Retrying)?;
                if task.attempt > task.spec.max_retries {
                    return Err(CommandError::RetryExhausted(task_id).into());
                }
                Ok(vec![EventKind::TaskRetried { task_id, attempt: task.attempt }])
            }
            RuntimeCommand::CancelTask { task_id, reason } => {
                let task = task(projection, task_id)?;
                task.status.validate_transition(TaskStatus::Cancelled)?;
                Ok(vec![EventKind::TaskCancelled { task_id, reason }])
            }
            RuntimeCommand::RememberFailure { failure } => {
                Ok(vec![EventKind::FailureRemembered { failure }])
            }
            RuntimeCommand::RecordDecision { decision } => {
                Ok(vec![EventKind::DecisionMade { decision }])
            }
            RuntimeCommand::RecordEvidence { evidence } => {
                Ok(vec![EventKind::EvidenceRecorded { evidence }])
            }
            RuntimeCommand::CreateMemory { memory } => {
                if memory.content.trim().is_empty() {
                    return Err(RuntimeError::InvalidCommand("memory content cannot be empty"));
                }
                if memory.content.chars().count() > MAX_SEMANTIC_MEMORY_CONTENT_CHARS {
                    return Err(RuntimeError::InvalidCommand(
                        "memory content exceeds the maximum length",
                    ));
                }
                if !memory.importance.is_finite()
                    || !(0.0..=1.0).contains(&memory.importance)
                    || !memory.confidence.is_finite()
                    || !(0.0..=1.0).contains(&memory.confidence)
                {
                    return Err(RuntimeError::InvalidCommand(
                        "memory importance and confidence must be finite values between 0 and 1",
                    ));
                }
                if projection.semantic_memories.contains_key(&memory.id) {
                    return Err(RuntimeError::InvalidCommand(
                        "memory id already exists in this run",
                    ));
                }
                Ok(vec![EventKind::MemoryCreated { memory }])
            }
            RuntimeCommand::RetrieveSemanticMemory { query, limit } => {
                if query.trim().is_empty() {
                    return Err(RuntimeError::InvalidCommand(
                        "semantic memory query cannot be empty",
                    ));
                }
                if query.chars().count() > MAX_SEMANTIC_MEMORY_QUERY_CHARS {
                    return Err(RuntimeError::InvalidCommand(
                        "semantic memory query exceeds the maximum length",
                    ));
                }
                if limit == 0 || limit > MAX_SEMANTIC_MEMORY_RETRIEVAL_HITS {
                    return Err(RuntimeError::InvalidCommand(
                        "semantic memory retrieval limit must be between 1 and 8",
                    ));
                }
                let memories = projection.semantic_memories.values().cloned().collect::<Vec<_>>();
                let ranked = SemanticMemoryRetrievalPolicy.retrieve(&query, &memories, limit);
                let hits = ranked
                    .into_iter()
                    .filter_map(|ranked_memory| {
                        projection.semantic_memories.get(&ranked_memory.memory_id).map(|memory| {
                            SemanticMemoryHit {
                                memory_id: memory.id,
                                kind: memory.kind,
                                content: memory.content.clone(),
                                score_milli: ranked_memory.score_milli,
                            }
                        })
                    })
                    .collect();
                Ok(vec![EventKind::SemanticMemoryRetrieved {
                    retrieval: SemanticMemoryRetrieval {
                        query,
                        algorithm: SEMANTIC_RETRIEVAL_ALGORITHM.to_owned(),
                        hits,
                    },
                }])
            }
            RuntimeCommand::RecordToolInvocation { operation_id, tool, input } => {
                if operation_id.trim().is_empty() || tool.trim().is_empty() {
                    return Err(RuntimeError::InvalidCommand(
                        "tool operation_id and tool name cannot be empty",
                    ));
                }
                Ok(vec![EventKind::ToolInvoked { operation_id, tool, input }])
            }
            RuntimeCommand::RecordToolSuccess { operation_id, output } => {
                if operation_id.trim().is_empty() {
                    return Err(RuntimeError::InvalidCommand("tool operation_id cannot be empty"));
                }
                Ok(vec![EventKind::ToolSucceeded { operation_id, output }])
            }
            RuntimeCommand::RecordToolFailure { operation_id, error, retryable } => {
                if operation_id.trim().is_empty() || error.trim().is_empty() {
                    return Err(RuntimeError::InvalidCommand(
                        "tool operation_id and error cannot be empty",
                    ));
                }
                Ok(vec![EventKind::ToolFailed { operation_id, error, retryable }])
            }
            RuntimeCommand::RecordToolResult { result } => {
                if result.operation_id.trim().is_empty() || result.tool.trim().is_empty() {
                    return Err(RuntimeError::InvalidCommand(
                        "tool result operation_id and tool name cannot be empty",
                    ));
                }
                if result.status != ToolResultStatus::Succeeded
                    && result.error.as_deref().is_none_or(str::is_empty)
                {
                    return Err(RuntimeError::InvalidCommand(
                        "a non-successful tool result must include an error",
                    ));
                }
                Ok(vec![EventKind::ToolResultRecorded { result }])
            }
            RuntimeCommand::UpdateBudget { budget } => {
                // Actual usage becomes known only after a model or process
                // call. Preserve budget overruns as durable telemetry rather
                // than rejecting the observation; a policy can suspend on the
                // next boundary while replay retains the fact.
                Ok(vec![EventKind::BudgetUpdated { budget }])
            }
            RuntimeCommand::UpdateEnvironment { environment } => {
                Ok(vec![EventKind::EnvironmentUpdated { environment }])
            }
            RuntimeCommand::ApplyIntervention { assessment } => {
                validate_intervention_assessment(projection, &assessment)?;
                let mut events =
                    vec![EventKind::StateDecayAssessed { assessment: assessment.clone() }];
                if assessment.action.injects_anchor() {
                    events.push(EventKind::StateAnchorInjected {
                        anchor: horizon_core::AnchorRecord {
                            risk_score_milli: assessment.risk_score_milli,
                            reason: assessment.reason,
                            content: projection.cognitive.render_anchor(),
                        },
                    });
                }
                Ok(events)
            }
            RuntimeCommand::Suspend { reason } => {
                projection.state.validate_transition(AgentState::Suspended)?;
                Ok(vec![
                    EventKind::StateTransitioned {
                        from: projection.state,
                        to: AgentState::Suspended,
                    },
                    EventKind::AgentSuspended { reason },
                ])
            }
        }
    }

    async fn append_apply(
        &self,
        projection: &mut RunProjection,
        event: EventKind,
    ) -> Result<EventRecord, RuntimeError> {
        let event = self.append_apply_without_auto_checkpoint(projection, event).await?;
        self.maybe_automatic_checkpoint(projection).await?;
        Ok(event)
    }

    async fn append_apply_without_auto_checkpoint(
        &self,
        projection: &mut RunProjection,
        event: EventKind,
    ) -> Result<EventRecord, RuntimeError> {
        let mut records =
            self.append_apply_many_without_auto_checkpoint(projection, vec![event]).await?;
        Ok(records.pop().expect("single event append returned one record"))
    }

    /// Persist a whole logical command in one SQLite transaction, then apply its
    /// committed events in order. A process can therefore observe either all or
    /// none of command-coupled records such as task failure and failure memory.
    async fn append_apply_many_without_auto_checkpoint(
        &self,
        projection: &mut RunProjection,
        events: Vec<EventKind>,
    ) -> Result<Vec<EventRecord>, RuntimeError> {
        let expected_sequence = projection.sequence;
        let new_events =
            events.into_iter().map(|event| NewEvent::new(projection.run_id, event)).collect();
        let records = self.store.append_many_if_sequence(new_events, expected_sequence).await?;
        for record in &records {
            projection.apply(record)?;
            horizon_trace::emit_persisted_event(record);
            debug!(run_id = %projection.run_id, sequence = record.sequence, event = record.event.name(), "runtime applied event");
        }
        Ok(records)
    }

    async fn maybe_automatic_checkpoint(
        &self,
        projection: &mut RunProjection,
    ) -> Result<(), RuntimeError> {
        let interval = self.config.checkpoint_every_events;
        if interval == 0
            || projection.sequence == 0
            || projection.sequence.saturating_sub(projection.last_checkpoint_sequence) < interval
        {
            return Ok(());
        }
        let sequence_before_checkpoint = projection.sequence;
        self.append_apply_without_auto_checkpoint(
            projection,
            EventKind::CheckpointCreated { sequence: sequence_before_checkpoint + 1 },
        )
        .await?;
        self.persist_snapshot(projection).await?;
        info!(run_id = %projection.run_id, sequence = projection.sequence, "automatic checkpoint created");
        Ok(())
    }

    async fn persist_snapshot(&self, projection: &RunProjection) -> Result<(), RuntimeError> {
        let snapshot =
            encode_snapshot(projection, self.config.snapshot_encoding).map_err(StoreError::from)?;
        self.store
            .save_checkpoint(CheckpointRecord::from_encoded(
                projection.run_id,
                projection.sequence,
                Utc::now(),
                snapshot,
            ))
            .await?;
        if let Some(retain_latest) = self.config.checkpoint_retention {
            self.store.compact_checkpoints(projection.run_id, retain_latest).await?;
        }
        Ok(())
    }

    async fn load_projection_with_origin(
        &self,
        run_id: RunId,
    ) -> Result<LoadedProjection, RuntimeError> {
        let checkpoint = self.store.load_latest_checkpoint(run_id).await?;
        let (mut projection, checkpoint_sequence) = match checkpoint {
            Some(checkpoint) => {
                let projection: RunProjection =
                    decode_snapshot(&checkpoint).map_err(StoreError::from)?;
                if projection.run_id != run_id || projection.sequence != checkpoint.sequence {
                    return Err(RuntimeError::InvalidCheckpoint {
                        run_id,
                        sequence: checkpoint.sequence,
                    });
                }
                (projection, Some(checkpoint.sequence))
            }
            None => (RunProjection::empty(run_id), None),
        };
        let events = self.store.load_events(run_id, projection.sequence).await?;
        for event in events {
            projection.apply(&event)?;
        }
        Ok(LoadedProjection { projection, checkpoint_sequence })
    }
}

fn task(projection: &RunProjection, task_id: TaskId) -> Result<&TaskRecord, RuntimeError> {
    projection.tasks.get(&task_id).ok_or_else(|| CommandError::UnknownTask(task_id).into())
}

fn dependencies_succeeded(projection: &RunProjection, task: &TaskRecord) -> bool {
    task.spec.dependencies.iter().all(|dependency| {
        projection
            .tasks
            .get(dependency)
            .is_some_and(|dependency_task| dependency_task.status == TaskStatus::Succeeded)
    })
}

fn validate_intervention_assessment(
    projection: &RunProjection,
    assessment: &InterventionAssessment,
) -> Result<(), RuntimeError> {
    if assessment.policy_id.trim().is_empty() {
        return Err(RuntimeError::InvalidCommand("intervention policy_id cannot be empty"));
    }
    if assessment.policy_version.as_deref().is_some_and(|version| version.trim().is_empty()) {
        return Err(RuntimeError::InvalidCommand("intervention policy_version cannot be empty"));
    }
    if assessment.risk_score_milli > 1_000 || assessment.threshold_milli > 1_000 {
        return Err(RuntimeError::InvalidCommand(
            "intervention risk and threshold must be between 0 and 1000",
        ));
    }
    if !assessment.signals.context_pressure.is_finite()
        || !(0.0..=1.0).contains(&assessment.signals.context_pressure)
    {
        return Err(RuntimeError::InvalidCommand(
            "intervention context_pressure must be a finite value between 0 and 1",
        ));
    }
    let expected_anchor_distance =
        projection.sequence.saturating_sub(projection.last_anchor_sequence);
    if assessment.signals.steps_since_anchor != expected_anchor_distance {
        return Err(RuntimeError::InvalidCommand(
            "intervention steps_since_anchor must match the durable projection",
        ));
    }
    if assessment.reason.trim().is_empty() {
        return Err(RuntimeError::InvalidCommand("intervention reason cannot be empty"));
    }
    let should_inject = assessment.risk_score_milli >= assessment.threshold_milli;
    if assessment.action.injects_anchor() != should_inject {
        return Err(RuntimeError::InvalidCommand(
            "intervention action must agree with whether risk is at or above its threshold",
        ));
    }
    Ok(())
}

/// Applies prospective event kinds to a cloned projection without writing them.
/// It is used only to choose recovery-time intervention content before the whole
/// recovery boundary is atomically persisted.
fn simulate_events(
    projection: &RunProjection,
    event_kinds: &[EventKind],
) -> Result<RunProjection, RuntimeError> {
    let mut simulated = projection.clone();
    for event in event_kinds {
        let record = EventRecord {
            id: EventId::new(),
            run_id: simulated.run_id,
            sequence: simulated.sequence + 1,
            timestamp: Utc::now(),
            schema_version: CURRENT_EVENT_SCHEMA_VERSION,
            event: event.clone(),
            metadata: serde_json::Value::Null,
        };
        simulated.apply(&record)?;
    }
    Ok(simulated)
}

struct LoadedProjection {
    projection: RunProjection,
    checkpoint_sequence: Option<u64>,
}

struct RuntimeTaskExecutor<S: EventStore + ?Sized> {
    runtime: Arc<HorizonRuntime<S>>,
    run_id: RunId,
}

#[async_trait::async_trait]
impl<S: EventStore + ?Sized> TaskExecutor for RuntimeTaskExecutor<S> {
    async fn execute(&self, task_spec: horizon_core::TaskSpec) -> TaskExecutionResult {
        let task_id = task_spec.id;
        let result = self.execute_inner(task_spec).await;
        TaskExecutionResult { task_id, output: result }
    }
}

impl<S: EventStore + ?Sized> RuntimeTaskExecutor<S> {
    async fn execute_inner(&self, task_spec: horizon_core::TaskSpec) -> Result<String, String> {
        let initial_projection =
            self.runtime.projection(self.run_id).await.map_err(|error| error.to_string())?;
        let Some(current_task) = initial_projection.tasks.get(&task_spec.id) else {
            return Err(format!("task {} disappeared before execution", task_spec.id));
        };
        if current_task.status != TaskStatus::Pending
            || !dependencies_succeeded(&initial_projection, current_task)
        {
            return Err(format!("task {} is no longer ready", task_spec.id));
        }
        let operation_id =
            task_spec.operation_id.clone().unwrap_or_else(|| format!("task:{}", task_spec.id));
        if initial_projection.is_operation_completed(&operation_id) {
            self.runtime
                .dispatch(
                    self.run_id,
                    RuntimeCommand::SucceedTask {
                        task_id: task_spec.id,
                        output: format!(
                            "[idempotent] operation {operation_id} was already completed"
                        ),
                    },
                )
                .await
                .map_err(|error| error.to_string())?;
            return Ok(format!("[idempotent] operation {operation_id} was already completed"));
        }
        if let Some(failure) =
            matching_failure(&initial_projection.cognitive.failed_attempts, &operation_id)
        {
            if !failure.retryable {
                return Err(format!(
                    "blocked by failure memory: {} ({})",
                    failure.approach, failure.reason
                ));
            }
        }
        self.runtime
            .dispatch(self.run_id, RuntimeCommand::StartTask { task_id: task_spec.id })
            .await
            .map_err(|error| error.to_string())?;
        let command = task_spec.command.as_ref().expect("filtered by execute_ready_tasks");
        let (program, args) = command
            .split_first()
            .ok_or_else(|| "task command must contain a program".to_owned())?;
        let invocation = serde_json::json!({
            "program": program,
            "args": args,
            "task_id": task_spec.id,
            "operation_id": operation_id,
            "executor": task_spec.executor,
            "resources": task_spec.resources,
        });
        if let Err(error) = self
            .runtime
            .append_raw(
                self.run_id,
                EventKind::ToolInvoked {
                    operation_id: operation_id.clone(),
                    tool: "process".into(),
                    input: invocation,
                },
            )
            .await
        {
            return Err(error.to_string());
        }
        let output = self
            .runtime
            .supervisor
            .run(ProcessRequest {
                operation_id: operation_id.clone(),
                program: program.clone(),
                args: args.to_vec(),
                working_dir: task_spec.working_dir.clone().map(Into::into),
                environment: BTreeMap::from([(
                    "HORIZON_OPERATION_ID".to_owned(),
                    operation_id.clone(),
                )]),
                timeout_ms: task_spec.timeout_ms.unwrap_or(300_000),
                resources: task_spec.resources.clone(),
                executor: task_spec.executor.clone(),
            })
            .await;
        self.persist_process_result(task_spec.id, operation_id, output).await
    }

    async fn persist_process_result(
        &self,
        task_id: TaskId,
        operation_id: String,
        output: Result<ProcessOutput, horizon_process::ProcessError>,
    ) -> Result<String, String> {
        match output {
            Ok(output) if output.succeeded() => {
                self.runtime
                    .append_raw_many(
                        self.run_id,
                        vec![
                            EventKind::ProcessCompleted {
                                operation_id: operation_id.clone(),
                                exit_code: output.exit_code,
                            },
                            EventKind::ToolResultRecorded {
                                result: process_tool_result(task_id, &operation_id, &output),
                            },
                        ],
                    )
                    .await
                    .map_err(|error| error.to_string())?;
                let summary = process_summary(&output);
                self.runtime
                    .dispatch(
                        self.run_id,
                        RuntimeCommand::SucceedTask { task_id, output: summary.clone() },
                    )
                    .await
                    .map_err(|error| error.to_string())?;
                Ok(summary)
            }
            Ok(output) => {
                let error = process_summary(&output);
                self.runtime
                    .append_raw_many(
                        self.run_id,
                        vec![
                            EventKind::ProcessCompleted {
                                operation_id: operation_id.clone(),
                                exit_code: output.exit_code,
                            },
                            EventKind::ToolResultRecorded {
                                result: process_tool_result(task_id, &operation_id, &output),
                            },
                        ],
                    )
                    .await
                    .map_err(|error| error.to_string())?;
                self.runtime
                    .dispatch(
                        self.run_id,
                        RuntimeCommand::FailTask {
                            task_id,
                            error: error.clone(),
                            retryable: !output.timed_out,
                            operation_id: Some(operation_id),
                        },
                    )
                    .await
                    .map_err(|dispatch_error| dispatch_error.to_string())?;
                Err(error)
            }
            Err(error) => {
                let message = format!("process supervisor error: {error}");
                self.runtime
                    .append_raw(
                        self.run_id,
                        EventKind::ToolResultRecorded {
                            result: ToolResult {
                                operation_id: operation_id.clone(),
                                tool: "process".into(),
                                status: ToolResultStatus::Failed,
                                output: serde_json::Value::Null,
                                error: Some(message.clone()),
                                duration_ms: None,
                                metadata: serde_json::json!({ "task_id": task_id }),
                            },
                        },
                    )
                    .await
                    .map_err(|dispatch_error| dispatch_error.to_string())?;
                self.runtime
                    .dispatch(
                        self.run_id,
                        RuntimeCommand::FailTask {
                            task_id,
                            error: message.clone(),
                            retryable: true,
                            operation_id: Some(operation_id),
                        },
                    )
                    .await
                    .map_err(|dispatch_error| dispatch_error.to_string())?;
                Err(message)
            }
        }
    }
}

impl<S: EventStore + ?Sized> HorizonRuntime<S> {
    /// Internal escape hatch used only for events produced as a consequence of a
    /// supervised side effect. It still does the same durable append-and-apply
    /// sequence, and is intentionally not exported as a public free function.
    async fn append_raw(
        &self,
        run_id: RunId,
        event: EventKind,
    ) -> Result<EventRecord, RuntimeError> {
        let mut records = self.append_raw_many(run_id, vec![event]).await?;
        Ok(records.pop().expect("single raw event append returned one record"))
    }

    async fn append_raw_many(
        &self,
        run_id: RunId,
        events: Vec<EventKind>,
    ) -> Result<Vec<EventRecord>, RuntimeError> {
        let _gate = self.mutation_gate.lock().await;
        let mut projection = self.load_projection_with_origin(run_id).await?.projection;
        if projection.sequence == 0 {
            return Err(RuntimeError::UnknownRun(run_id));
        }
        let records =
            self.append_apply_many_without_auto_checkpoint(&mut projection, events).await?;
        self.maybe_automatic_checkpoint(&mut projection).await?;
        Ok(records)
    }
}

fn process_summary(output: &ProcessOutput) -> String {
    let stdout = output.stdout.trim();
    let stderr = output.stderr.trim();
    let details = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else if output.timed_out {
        "process timed out"
    } else {
        "process produced no output"
    };
    let truncation = match (output.stdout_truncated, output.stderr_truncated) {
        (false, false) => "",
        (true, false) => " [stdout truncated]",
        (false, true) => " [stderr truncated]",
        (true, true) => " [stdout/stderr truncated]",
    };
    format!(
        "exit={:?}, timeout={}, duration={}ms: {}{}",
        output.exit_code, output.timed_out, output.duration_ms, details, truncation
    )
}

fn process_tool_result(task_id: TaskId, operation_id: &str, output: &ProcessOutput) -> ToolResult {
    let status = if output.succeeded() {
        ToolResultStatus::Succeeded
    } else if output.timed_out {
        ToolResultStatus::TimedOut
    } else {
        ToolResultStatus::Failed
    };
    ToolResult {
        operation_id: operation_id.to_owned(),
        tool: "process".into(),
        status,
        output: serde_json::json!({
            "exit_code": output.exit_code,
            "stdout": output.stdout,
            "stderr": output.stderr,
            "stdout_truncated": output.stdout_truncated,
            "stderr_truncated": output.stderr_truncated,
            "timed_out": output.timed_out,
            "executor": output.executor,
            "resource_limits": output.resource_limits,
        }),
        error: (!output.succeeded()).then(|| process_summary(output)),
        duration_ms: u64::try_from(output.duration_ms).ok(),
        metadata: serde_json::json!({ "task_id": task_id }),
    }
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("run {0} does not exist")]
    UnknownRun(RunId),
    #[error("checkpoint for run {run_id} at sequence {sequence} does not match its snapshot")]
    InvalidCheckpoint { run_id: RunId, sequence: u64 },
    #[error("invalid runtime command: {0}")]
    InvalidCommand(&'static str),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Projection(#[from] ProjectionError),
    #[error(transparent)]
    Command(#[from] CommandError),
    #[error(transparent)]
    State(#[from] horizon_core::StateTransitionError),
    #[error(transparent)]
    Task(#[from] horizon_core::TaskTransitionError),
    #[error(transparent)]
    TaskSpec(#[from] TaskSpecValidationError),
    #[error(transparent)]
    Scheduler(#[from] horizon_scheduler::SchedulerError),
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Utc;
    use horizon_core::{
        AgentState, Constraint, EventId, InterventionAction, InterventionAssessment, MemoryId,
        MemoryItem, MemoryKind, RuntimeCommand, StateDecaySignals, TaskSpec, TaskStatus,
        ToolResult, ToolResultStatus,
    };
    use horizon_store::SqliteEventStore;

    use super::*;

    async fn runtime() -> Arc<HorizonRuntime<SqliteEventStore>> {
        Arc::new(HorizonRuntime::with_config(
            Arc::new(SqliteEventStore::in_memory().await.unwrap()),
            RuntimeConfig { checkpoint_every_events: 3, ..Default::default() },
        ))
    }

    #[tokio::test]
    async fn command_event_projection_path_is_durable() {
        let runtime = runtime().await;
        let run = runtime.create_run("deliver a durable demo").await.unwrap();
        let run_id = run.projection.run_id;
        runtime
            .dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Planning })
            .await
            .unwrap();
        runtime
            .dispatch(
                run_id,
                RuntimeCommand::AddConstraint {
                    constraint: Constraint {
                        id: "c1".into(),
                        content: "Do not redo known failures".into(),
                        source: None,
                    },
                },
            )
            .await
            .unwrap();
        let replayed = runtime.projection(run_id).await.unwrap();
        assert_eq!(replayed.state, AgentState::Planning);
        assert_eq!(replayed.cognitive.constraints.len(), 1);
        assert!(runtime.store().load_latest_checkpoint(run_id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn checkpoints_are_compressed_and_can_be_compacted_without_losing_replay() {
        let store = Arc::new(SqliteEventStore::in_memory().await.unwrap());
        let runtime = HorizonRuntime::with_config(
            Arc::clone(&store),
            RuntimeConfig {
                checkpoint_every_events: 0,
                checkpoint_retention: None,
                ..Default::default()
            },
        );
        let run_id = runtime.create_run("retain durable state").await.unwrap().projection.run_id;
        runtime.checkpoint(run_id).await.unwrap();
        runtime
            .dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Planning })
            .await
            .unwrap();
        runtime.checkpoint(run_id).await.unwrap();
        runtime
            .dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Executing })
            .await
            .unwrap();
        runtime.checkpoint(run_id).await.unwrap();
        let latest = store.load_latest_checkpoint(run_id).await.unwrap().unwrap();
        assert_eq!(latest.encoding, SnapshotEncoding::ZstdJson);
        assert!(latest.checksum.is_some());
        let compacted = runtime.compact_checkpoints(run_id, 2).await.unwrap();
        assert_eq!(compacted.retained, 2);
        assert_eq!(compacted.removed, 1);
        let replayed = runtime.projection(run_id).await.unwrap();
        assert_eq!(replayed.state, AgentState::Executing);
        assert_eq!(replayed.cognitive.primary_goal.as_deref(), Some("retain durable state"));
    }

    #[tokio::test]
    async fn zero_checkpoint_retention_keeps_snapshots_for_direct_config_callers() {
        let runtime = HorizonRuntime::with_config(
            Arc::new(SqliteEventStore::in_memory().await.unwrap()),
            RuntimeConfig {
                checkpoint_every_events: 0,
                checkpoint_retention: Some(0),
                ..Default::default()
            },
        );
        let run_id = runtime.create_run("preserve every snapshot").await.unwrap().projection.run_id;
        runtime.checkpoint(run_id).await.unwrap();
        runtime.checkpoint(run_id).await.unwrap();
        assert_eq!(runtime.config().checkpoint_retention, None);
    }

    #[tokio::test]
    async fn recovery_replays_after_checkpoint_and_injects_anchor() {
        let runtime = runtime().await;
        let run = runtime.create_run("retain the objective").await.unwrap();
        let run_id = run.projection.run_id;
        runtime
            .dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Planning })
            .await
            .unwrap();
        runtime
            .dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Executing })
            .await
            .unwrap();
        let before = runtime.projection(run_id).await.unwrap();
        assert_eq!(before.state, AgentState::Executing);
        let recovered = runtime.recover(run_id).await.unwrap();
        assert_eq!(recovered.projection.state, AgentState::Recovering);
        assert!(recovered.projection.last_anchor_sequence > before.last_anchor_sequence);
        assert_eq!(
            recovered.projection.cognitive.primary_goal.as_deref(),
            Some("retain the objective")
        );
    }

    #[tokio::test]
    async fn task_failure_becomes_failure_memory() {
        let runtime = runtime().await;
        let run = runtime.create_run("test memory").await.unwrap();
        let run_id = run.projection.run_id;
        let task = TaskSpec::new("unstable approach");
        let task_id = task.id;
        runtime.dispatch(run_id, RuntimeCommand::CreateTask { task }).await.unwrap();
        runtime.dispatch(run_id, RuntimeCommand::StartTask { task_id }).await.unwrap();
        let outcome = runtime
            .dispatch(
                run_id,
                RuntimeCommand::FailTask {
                    task_id,
                    error: "schema mismatch".into(),
                    retryable: false,
                    operation_id: Some("bad-op".into()),
                },
            )
            .await
            .unwrap();
        assert_eq!(outcome.events.len(), 2);
        assert_eq!(outcome.events[0].sequence + 1, outcome.events[1].sequence);
        assert!(matches!(outcome.events[0].event, EventKind::TaskFailed { .. }));
        assert!(matches!(outcome.events[1].event, EventKind::FailureRemembered { .. }));
        let projection = runtime.projection(run_id).await.unwrap();
        assert_eq!(projection.tasks[&task_id].status, TaskStatus::Failed);
        assert!(runtime.known_failure(run_id, "bad-op").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn blocked_task_requires_durable_unblock_before_execution() {
        let runtime = runtime().await;
        let run_id = runtime.create_run("exercise blocked status").await.unwrap().projection.run_id;
        let task = TaskSpec::new("wait for external prerequisite");
        let task_id = task.id;
        runtime.dispatch(run_id, RuntimeCommand::CreateTask { task }).await.unwrap();
        runtime
            .dispatch(
                run_id,
                RuntimeCommand::BlockTask {
                    task_id,
                    reason: "external prerequisite is unavailable".into(),
                },
            )
            .await
            .unwrap();
        assert_eq!(
            runtime.projection(run_id).await.unwrap().tasks[&task_id].status,
            TaskStatus::Blocked
        );
        runtime.dispatch(run_id, RuntimeCommand::UnblockTask { task_id }).await.unwrap();
        assert_eq!(
            runtime.projection(run_id).await.unwrap().tasks[&task_id].status,
            TaskStatus::Pending
        );
    }

    #[tokio::test]
    async fn rejects_duplicate_operation_ids_within_a_run() {
        let runtime = runtime().await;
        let run_id =
            runtime.create_run("protect idempotency namespace").await.unwrap().projection.run_id;
        let mut first = TaskSpec::new("first");
        first.operation_id = Some("shared-operation".into());
        runtime.dispatch(run_id, RuntimeCommand::CreateTask { task: first }).await.unwrap();
        let mut second = TaskSpec::new("second");
        second.operation_id = Some("shared-operation".into());
        let error = runtime
            .dispatch(run_id, RuntimeCommand::CreateTask { task: second })
            .await
            .unwrap_err();
        assert!(matches!(error, RuntimeError::Command(CommandError::DuplicateOperationId(_))));
    }

    #[tokio::test]
    async fn external_tool_success_is_recorded_as_completed_operation() {
        let runtime = runtime().await;
        let run_id = runtime.create_run("audit external tool").await.unwrap().projection.run_id;
        runtime
            .dispatch(
                run_id,
                RuntimeCommand::RecordToolInvocation {
                    operation_id: "search-42".into(),
                    tool: "search".into(),
                    input: serde_json::json!({"query": "durable agents"}),
                },
            )
            .await
            .unwrap();
        runtime
            .dispatch(
                run_id,
                RuntimeCommand::RecordToolSuccess {
                    operation_id: "search-42".into(),
                    output: serde_json::json!({"hits": 3}),
                },
            )
            .await
            .unwrap();
        assert!(runtime.projection(run_id).await.unwrap().is_operation_completed("search-42"));
    }

    #[tokio::test]
    async fn structured_tool_result_is_replayed_and_marks_success() {
        let runtime = runtime().await;
        let run_id =
            runtime.create_run("audit structured tool result").await.unwrap().projection.run_id;
        runtime
            .dispatch(
                run_id,
                RuntimeCommand::RecordToolResult {
                    result: ToolResult {
                        operation_id: "adapter-42".into(),
                        tool: "external_adapter".into(),
                        status: ToolResultStatus::Succeeded,
                        output: serde_json::json!({"artifact": "ready"}),
                        error: None,
                        duration_ms: Some(12),
                        metadata: serde_json::json!({"provider": "fixture"}),
                    },
                },
            )
            .await
            .unwrap();
        let projection = runtime.projection(run_id).await.unwrap();
        assert!(projection.is_operation_completed("adapter-42"));
        assert_eq!(projection.tool_results.len(), 1);
        assert_eq!(projection.tool_results[0].duration_ms, Some(12));
    }

    #[tokio::test]
    async fn semantic_memory_retrieval_is_durable_and_never_uses_caller_supplied_hits() {
        let runtime = runtime().await;
        let run_id = runtime
            .create_run("restore a durable PostgreSQL migration")
            .await
            .unwrap()
            .projection
            .run_id;
        let relevant = MemoryItem {
            id: MemoryId::new(),
            kind: MemoryKind::Episodic,
            content: "Recover PostgreSQL from its checkpoint before retrying the migration.".into(),
            source_event: EventId::new(),
            importance: 0.9,
            confidence: 0.95,
            created_at: Utc::now(),
        };
        let irrelevant = MemoryItem {
            id: MemoryId::new(),
            kind: MemoryKind::Environment,
            content: "Water the botanical garden on Tuesday morning.".into(),
            source_event: EventId::new(),
            importance: 1.0,
            confidence: 1.0,
            created_at: Utc::now(),
        };
        runtime
            .dispatch(run_id, RuntimeCommand::CreateMemory { memory: relevant.clone() })
            .await
            .unwrap();
        runtime
            .dispatch(run_id, RuntimeCommand::CreateMemory { memory: irrelevant })
            .await
            .unwrap();

        let outcome = runtime
            .retrieve_semantic_memory(run_id, "recover postgres checkpoint", 4)
            .await
            .unwrap();
        assert_eq!(outcome.events.len(), 1);
        let EventKind::SemanticMemoryRetrieved { retrieval } = &outcome.events[0].event else {
            panic!("retrieve command must persist a semantic_memory_retrieved event");
        };
        assert_eq!(retrieval.query, "recover postgres checkpoint");
        assert_eq!(retrieval.algorithm, "hybrid_lexical_v1");
        assert_eq!(retrieval.hits[0].memory_id, relevant.id);
        assert!(retrieval.hits.iter().all(|hit| hit.score_milli <= 1_000));
        assert!(
            retrieval
                .hits
                .iter()
                .all(|hit| hit.content != "Water the botanical garden on Tuesday morning.")
        );

        let projection = runtime.projection(run_id).await.unwrap();
        assert_eq!(projection.semantic_memories.len(), 2);
        assert_eq!(projection.semantic_memory_retrievals, 1);
        assert_eq!(
            projection.last_semantic_memory_retrieval.as_ref().unwrap().hits[0].memory_id,
            relevant.id
        );
    }

    #[tokio::test]
    async fn semantic_memory_retrieval_rejects_invalid_query_and_limit() {
        let runtime = runtime().await;
        let run_id =
            runtime.create_run("validate semantic retrieval").await.unwrap().projection.run_id;
        let empty_query = runtime.retrieve_semantic_memory(run_id, "   ", 1).await.unwrap_err();
        assert!(matches!(empty_query, RuntimeError::InvalidCommand(_)));
        let zero_limit = runtime.retrieve_semantic_memory(run_id, "anything", 0).await.unwrap_err();
        assert!(matches!(zero_limit, RuntimeError::InvalidCommand(_)));
        let over_limit = runtime.retrieve_semantic_memory(run_id, "anything", 9).await.unwrap_err();
        assert!(matches!(over_limit, RuntimeError::InvalidCommand(_)));
    }

    #[tokio::test]
    async fn learned_intervention_is_a_durable_assessment_and_anchor_boundary() {
        let runtime = runtime().await;
        let run_id =
            runtime.create_run("apply a learned intervention").await.unwrap().projection.run_id;
        let outcome = runtime
            .apply_intervention(
                run_id,
                InterventionAssessment {
                    policy_id: "logistic_state_decay".into(),
                    policy_version: Some("fixture-v1".into()),
                    risk_score_milli: 810,
                    threshold_milli: 640,
                    signals: StateDecaySignals {
                        steps_since_anchor: 1,
                        context_pressure: 0.72,
                        subgoal_switches: 2,
                        recent_failures: 1,
                        recovered_session: false,
                    },
                    action: InterventionAction::InjectAnchor,
                    reason: "learned risk exceeded the adaptive budget threshold".into(),
                    metadata: serde_json::json!({
                        "estimated_anchor_tokens": 120,
                        "controller": "adaptive_context_budget_v1",
                    }),
                },
            )
            .await
            .unwrap();
        assert_eq!(outcome.events.len(), 2);
        assert!(matches!(outcome.events[0].event, EventKind::StateDecayAssessed { .. }));
        assert!(matches!(outcome.events[1].event, EventKind::StateAnchorInjected { .. }));
        let projection = runtime.projection(run_id).await.unwrap();
        assert_eq!(projection.intervention_assessments, 1);
        assert_eq!(projection.last_anchor_sequence, outcome.events[1].sequence);
        assert_eq!(
            projection.last_intervention_assessment.unwrap().policy_id,
            "logistic_state_decay"
        );
    }

    #[tokio::test]
    async fn rejects_an_anchor_request_below_its_declared_risk_threshold() {
        let runtime = runtime().await;
        let run_id =
            runtime.create_run("reject invalid intervention").await.unwrap().projection.run_id;
        let error = runtime
            .apply_intervention(
                run_id,
                InterventionAssessment {
                    policy_id: "logistic_state_decay".into(),
                    policy_version: None,
                    risk_score_milli: 499,
                    threshold_milli: 500,
                    signals: StateDecaySignals { steps_since_anchor: 1, ..Default::default() },
                    action: InterventionAction::InjectAnchor,
                    reason: "invalid test decision".into(),
                    metadata: serde_json::Value::Null,
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, RuntimeError::InvalidCommand(_)));
    }

    #[tokio::test]
    async fn rejects_a_continue_decision_above_its_declared_risk_threshold() {
        let runtime = runtime().await;
        let run_id = runtime
            .create_run("reject inconsistent continue intervention")
            .await
            .unwrap()
            .projection
            .run_id;
        let error = runtime
            .apply_intervention(
                run_id,
                InterventionAssessment {
                    policy_id: "logistic_state_decay".into(),
                    policy_version: None,
                    risk_score_milli: 800,
                    threshold_milli: 500,
                    signals: StateDecaySignals { steps_since_anchor: 1, ..Default::default() },
                    action: InterventionAction::Continue,
                    reason: "invalid test decision".into(),
                    metadata: serde_json::Value::Null,
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, RuntimeError::InvalidCommand(_)));
    }

    #[tokio::test]
    async fn rejects_an_intervention_with_a_forged_anchor_distance() {
        let runtime = runtime().await;
        let run_id =
            runtime.create_run("reject forged anchor distance").await.unwrap().projection.run_id;
        let error = runtime
            .apply_intervention(
                run_id,
                InterventionAssessment {
                    policy_id: "logistic_state_decay".into(),
                    policy_version: None,
                    risk_score_milli: 100,
                    threshold_milli: 500,
                    signals: StateDecaySignals { steps_since_anchor: 99, ..Default::default() },
                    action: InterventionAction::Continue,
                    reason: "invalid test decision".into(),
                    metadata: serde_json::Value::Null,
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, RuntimeError::InvalidCommand(_)));
    }

    #[tokio::test]
    async fn process_task_runs_and_persists_output() {
        let runtime = runtime().await;
        let run_id = runtime.create_run("execute one task").await.unwrap().projection.run_id;
        let mut task = TaskSpec::new("say hello");
        task.command = Some(vec!["sh".into(), "-c".into(), "printf hello".into()]);
        let task_id = task.id;
        runtime.dispatch(run_id, RuntimeCommand::CreateTask { task }).await.unwrap();
        let outcomes = runtime.execute_ready_tasks(run_id).await.unwrap();
        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].output.as_ref().unwrap().contains("hello"));
        let projection = runtime.projection(run_id).await.unwrap();
        assert_eq!(projection.tasks[&task_id].status, TaskStatus::Succeeded);
        let result = projection.tool_results.last().unwrap();
        assert_eq!(result.tool, "process");
        assert_eq!(result.status, ToolResultStatus::Succeeded);
        assert_eq!(result.output["stdout"], "hello");
    }

    #[tokio::test]
    async fn checkpoint_event_matches_its_snapshot_sequence() {
        let runtime = runtime().await;
        let run_id = runtime.create_run("checkpoint integrity").await.unwrap().projection.run_id;
        let outcome = runtime.checkpoint(run_id).await.unwrap();
        let event = outcome.events.first().unwrap();
        assert_eq!(event.sequence, outcome.projection.last_checkpoint_sequence);
        assert!(matches!(
            event.event,
            EventKind::CheckpointCreated { sequence } if sequence == event.sequence
        ));
        let checkpoint = runtime.store().load_latest_checkpoint(run_id).await.unwrap().unwrap();
        assert_eq!(checkpoint.sequence, event.sequence);
    }

    #[tokio::test]
    async fn recovery_requeues_an_interrupted_running_task_within_retry_budget() {
        let runtime = runtime().await;
        let run_id =
            runtime.create_run("recover interrupted task").await.unwrap().projection.run_id;
        let mut task = TaskSpec::new("long experiment");
        task.max_retries = 1;
        let task_id = task.id;
        runtime.dispatch(run_id, RuntimeCommand::CreateTask { task }).await.unwrap();
        runtime.dispatch(run_id, RuntimeCommand::StartTask { task_id }).await.unwrap();
        runtime.checkpoint(run_id).await.unwrap();

        let recovered = runtime.recover(run_id).await.unwrap();
        let record = &recovered.projection.tasks[&task_id];
        assert_eq!(record.status, TaskStatus::Pending);
        assert_eq!(record.attempt, 1);
        assert!(
            recovered
                .projection
                .cognitive
                .failed_attempts
                .iter()
                .any(|failure| failure.reason.contains("interrupted"))
        );

        runtime.dispatch(run_id, RuntimeCommand::StartTask { task_id }).await.unwrap();
        assert_eq!(runtime.projection(run_id).await.unwrap().tasks[&task_id].attempt, 2);
    }
}

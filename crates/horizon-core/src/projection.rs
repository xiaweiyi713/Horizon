use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    AgentState, CognitiveState, EventKind, EventRecord, RunId, TaskId, TaskRecord, TaskStatus,
    ToolResult,
};

/// Materialized, deterministic view of a run. It can always be rebuilt from
/// events and is optionally serialized into checkpoints for faster recovery.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunProjection {
    pub run_id: RunId,
    pub sequence: u64,
    pub state: AgentState,
    pub created_at: Option<DateTime<Utc>>,
    pub updated_at: Option<DateTime<Utc>>,
    pub cognitive: CognitiveState,
    #[serde(default)]
    pub tasks: BTreeMap<TaskId, TaskRecord>,
    #[serde(default)]
    pub completed_operations: BTreeSet<String>,
    /// Terminal tool outcomes are a materialized operational view; the event
    /// stream remains authoritative for a full audit.
    #[serde(default)]
    pub tool_results: Vec<ToolResult>,
    /// Sequence of the last checkpoint event applied to this projection. It is
    /// persisted in snapshots so checkpoint cadence is not distorted by the
    /// checkpoint event itself.
    #[serde(default)]
    pub last_checkpoint_sequence: u64,
    pub last_anchor_sequence: u64,
}

impl RunProjection {
    #[must_use]
    pub fn empty(run_id: RunId) -> Self {
        Self {
            run_id,
            sequence: 0,
            state: AgentState::Created,
            created_at: None,
            updated_at: None,
            cognitive: CognitiveState::default(),
            tasks: BTreeMap::new(),
            completed_operations: BTreeSet::new(),
            tool_results: Vec::new(),
            last_checkpoint_sequence: 0,
            last_anchor_sequence: 0,
        }
    }

    pub fn replay(
        run_id: RunId,
        events: impl IntoIterator<Item = EventRecord>,
    ) -> Result<Self, ProjectionError> {
        let mut projection = Self::empty(run_id);
        for event in events {
            projection.apply(&event)?;
        }
        Ok(projection)
    }

    pub fn apply(&mut self, record: &EventRecord) -> Result<(), ProjectionError> {
        if record.run_id != self.run_id {
            return Err(ProjectionError::WrongRun { expected: self.run_id, actual: record.run_id });
        }
        if record.sequence != self.sequence + 1 {
            return Err(ProjectionError::NonContiguousSequence {
                expected: self.sequence + 1,
                actual: record.sequence,
            });
        }

        match &record.event {
            EventKind::RunCreated { goal } | EventKind::GoalRegistered { goal } => {
                self.cognitive.primary_goal = Some(goal.clone());
                self.created_at.get_or_insert(record.timestamp);
            }
            EventKind::StateTransitioned { from, to } => {
                if *from != self.state {
                    return Err(ProjectionError::TransitionSourceMismatch {
                        expected: self.state,
                        actual: *from,
                    });
                }
                from.validate_transition(*to)?;
                self.state = *to;
            }
            EventKind::ConstraintAdded { constraint } => {
                self.cognitive.constraints.push(constraint.clone())
            }
            EventKind::PlanCreated { steps } => self.cognitive.current_plan = steps.clone(),
            EventKind::SubgoalOpened { subgoal } => {
                self.cognitive.open_subgoals.push(subgoal.clone())
            }
            EventKind::SubgoalCompleted { id } => {
                if let Some(index) =
                    self.cognitive.open_subgoals.iter().position(|item| item.id == *id)
                {
                    let mut subgoal = self.cognitive.open_subgoals.remove(index);
                    subgoal.completed = true;
                    self.cognitive.completed_subgoals.push(subgoal);
                }
            }
            EventKind::TaskCreated { task } => {
                self.tasks.insert(task.id, TaskRecord::from_spec(task.clone(), record.timestamp));
            }
            EventKind::TaskStarted { task_id, attempt } => {
                let task = self.task_mut(*task_id)?;
                task.status.validate_transition(TaskStatus::Running)?;
                task.status = TaskStatus::Running;
                task.attempt = *attempt;
                task.updated_at = record.timestamp;
            }
            EventKind::TaskBlocked { task_id, reason } => {
                let task = self.task_mut(*task_id)?;
                task.status.validate_transition(TaskStatus::Blocked)?;
                task.status = TaskStatus::Blocked;
                task.last_error = Some(reason.clone());
                task.updated_at = record.timestamp;
            }
            EventKind::TaskUnblocked { task_id } => {
                let task = self.task_mut(*task_id)?;
                task.status.validate_transition(TaskStatus::Pending)?;
                task.status = TaskStatus::Pending;
                task.last_error = None;
                task.updated_at = record.timestamp;
            }
            EventKind::TaskSucceeded { task_id, .. } => {
                let task = self.task_mut(*task_id)?;
                task.status.validate_transition(TaskStatus::Succeeded)?;
                task.status = TaskStatus::Succeeded;
                task.last_error = None;
                task.updated_at = record.timestamp;
            }
            EventKind::TaskFailed { task_id, error, .. } => {
                let task = self.task_mut(*task_id)?;
                task.status.validate_transition(TaskStatus::Failed)?;
                task.status = TaskStatus::Failed;
                task.last_error = Some(error.clone());
                task.updated_at = record.timestamp;
            }
            EventKind::TaskRetried { task_id, attempt } => {
                let task = self.task_mut(*task_id)?;
                task.status.validate_transition(TaskStatus::Retrying)?;
                task.status = TaskStatus::Retrying;
                if *attempt != task.attempt {
                    return Err(ProjectionError::RetryAttemptMismatch {
                        expected: task.attempt,
                        actual: *attempt,
                    });
                }
                task.updated_at = record.timestamp;
                task.status.validate_transition(TaskStatus::Pending)?;
                task.status = TaskStatus::Pending;
            }
            EventKind::TaskCancelled { task_id, reason } => {
                let task = self.task_mut(*task_id)?;
                task.status.validate_transition(TaskStatus::Cancelled)?;
                task.status = TaskStatus::Cancelled;
                task.last_error = Some(reason.clone());
                task.updated_at = record.timestamp;
            }
            EventKind::ToolSucceeded { operation_id, .. } => {
                self.completed_operations.insert(operation_id.clone());
            }
            EventKind::ToolResultRecorded { result } => {
                if result.status.is_success() {
                    self.completed_operations.insert(result.operation_id.clone());
                }
                self.tool_results.push(result.clone());
            }
            EventKind::ProcessCompleted { operation_id, exit_code: Some(0) } => {
                self.completed_operations.insert(operation_id.clone());
            }
            EventKind::ProcessCompleted { .. } => {}
            EventKind::MemoryCreated { memory } => match memory.kind {
                crate::MemoryKind::Goal => {
                    self.cognitive.primary_goal = Some(memory.content.clone())
                }
                crate::MemoryKind::Constraint => {
                    self.cognitive.constraints.push(crate::Constraint {
                        id: memory.id.to_string(),
                        content: memory.content.clone(),
                        source: Some("memory".to_owned()),
                    })
                }
                _ => {}
            },
            EventKind::FailureRemembered { failure } => {
                self.cognitive.failed_attempts.push(failure.clone())
            }
            EventKind::DecisionMade { decision } => self.cognitive.decisions.push(decision.clone()),
            EventKind::EvidenceRecorded { evidence } => {
                self.cognitive.evidence.push(evidence.clone())
            }
            EventKind::BudgetUpdated { budget } => self.cognitive.budget = budget.clone(),
            EventKind::EnvironmentUpdated { environment } => {
                self.cognitive.environment = environment.clone()
            }
            EventKind::StateAnchorInjected { .. } => self.last_anchor_sequence = record.sequence,
            EventKind::AgentSuspended { .. }
            | EventKind::AgentRecovered { .. }
            | EventKind::ToolInvoked { .. }
            | EventKind::ToolFailed { .. }
            | EventKind::Note { .. } => {}
            EventKind::CheckpointCreated { sequence } => {
                if *sequence != record.sequence {
                    return Err(ProjectionError::CheckpointSequenceMismatch {
                        payload: *sequence,
                        event: record.sequence,
                    });
                }
                self.last_checkpoint_sequence = record.sequence;
            }
        }
        self.sequence = record.sequence;
        self.updated_at = Some(record.timestamp);
        Ok(())
    }

    fn task_mut(&mut self, id: TaskId) -> Result<&mut TaskRecord, ProjectionError> {
        self.tasks.get_mut(&id).ok_or(ProjectionError::UnknownTask(id))
    }

    #[must_use]
    pub fn is_operation_completed(&self, operation_id: &str) -> bool {
        self.completed_operations.contains(operation_id)
    }
}

#[derive(Debug, Error)]
pub enum ProjectionError {
    #[error("event for run {actual} cannot be applied to run {expected}")]
    WrongRun { expected: RunId, actual: RunId },
    #[error("event sequence must be contiguous: expected {expected}, received {actual}")]
    NonContiguousSequence { expected: u64, actual: u64 },
    #[error("state transition source mismatch: projection is {expected:?}, event says {actual:?}")]
    TransitionSourceMismatch { expected: AgentState, actual: AgentState },
    #[error("unknown task {0}")]
    UnknownTask(TaskId),
    #[error("retry event attempt mismatch: task is on attempt {expected}, event says {actual}")]
    RetryAttemptMismatch { expected: u32, actual: u32 },
    #[error("checkpoint payload sequence {payload} does not match event sequence {event}")]
    CheckpointSequenceMismatch { payload: u64, event: u64 },
    #[error(transparent)]
    InvalidState(#[from] crate::StateTransitionError),
    #[error(transparent)]
    InvalidTask(#[from] crate::TaskTransitionError),
}

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use crate::{CURRENT_EVENT_SCHEMA_VERSION, EventId, EventKind, EventRecord};

    use super::*;

    #[test]
    fn projection_replays_happy_path() {
        let run_id = RunId::new();
        let events = vec![
            EventRecord {
                id: EventId::new(),
                run_id,
                sequence: 1,
                timestamp: Utc::now(),
                schema_version: CURRENT_EVENT_SCHEMA_VERSION,
                event: EventKind::RunCreated { goal: "ship".into() },
                metadata: serde_json::Value::Null,
            },
            EventRecord {
                id: EventId::new(),
                run_id,
                sequence: 2,
                timestamp: Utc::now(),
                schema_version: CURRENT_EVENT_SCHEMA_VERSION,
                event: EventKind::StateTransitioned {
                    from: AgentState::Created,
                    to: AgentState::Planning,
                },
                metadata: serde_json::Value::Null,
            },
        ];
        let projection = RunProjection::replay(run_id, events).unwrap();
        assert_eq!(projection.sequence, 2);
        assert_eq!(projection.state, AgentState::Planning);
        assert_eq!(projection.cognitive.primary_goal.as_deref(), Some("ship"));
    }
}

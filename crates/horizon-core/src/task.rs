use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::TaskId;

/// Scheduler-visible task status. It is separate from the agent lifecycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    #[default]
    Pending,
    Running,
    Blocked,
    Succeeded,
    Failed,
    Retrying,
    Cancelled,
}

impl TaskStatus {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }

    #[must_use]
    pub const fn can_transition_to(self, target: Self) -> bool {
        use TaskStatus::*;
        matches!(
            (self, target),
            (Pending, Running | Blocked | Cancelled)
                | (Blocked, Pending | Cancelled)
                | (Running, Succeeded | Failed | Retrying | Cancelled)
                | (Retrying, Pending | Running | Failed | Cancelled)
                | (Failed, Retrying)
                | (Succeeded, Succeeded)
                | (Failed, Failed)
                | (Cancelled, Cancelled)
        )
    }

    pub fn validate_transition(self, target: Self) -> Result<(), TaskTransitionError> {
        if self.can_transition_to(target) {
            Ok(())
        } else {
            Err(TaskTransitionError { from: self, to: target })
        }
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("invalid task transition: {from:?} -> {to:?}")]
pub struct TaskTransitionError {
    pub from: TaskStatus,
    pub to: TaskStatus,
}

/// A side-effect-free task declaration. Commands are argv vectors rather than
/// shell strings so the process layer can execute them without implicit shell
/// interpolation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskSpec {
    pub id: TaskId,
    pub title: String,
    /// Optional logical parent for plan/tree visualizations. Scheduling remains
    /// dependency-driven; this field preserves subtask provenance across replay.
    #[serde(default)]
    pub parent: Option<TaskId>,
    /// Stable logical operation identifier, reused across retries and supplied
    /// to child processes as `HORIZON_OPERATION_ID`. A task attempt is not an
    /// operation ID: attempts may be delivered at least once after recovery.
    #[serde(default)]
    pub operation_id: Option<String>,
    #[serde(default)]
    pub command: Option<Vec<String>>,
    #[serde(default)]
    pub working_dir: Option<String>,
    #[serde(default)]
    pub priority: u8,
    #[serde(default)]
    pub dependencies: Vec<TaskId>,
    #[serde(default = "TaskSpec::default_max_retries")]
    pub max_retries: u32,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

impl TaskSpec {
    const fn default_max_retries() -> u32 {
        1
    }

    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        let id = TaskId::new();
        Self {
            id,
            title: title.into(),
            parent: None,
            operation_id: Some(format!("task:{id}")),
            command: None,
            working_dir: None,
            priority: 0,
            dependencies: Vec::new(),
            max_retries: Self::default_max_retries(),
            timeout_ms: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskRecord {
    pub spec: TaskSpec,
    pub status: TaskStatus,
    pub attempt: u32,
    pub last_error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl TaskRecord {
    #[must_use]
    pub fn from_spec(spec: TaskSpec, now: DateTime<Utc>) -> Self {
        Self {
            spec,
            status: TaskStatus::Pending,
            attempt: 0,
            last_error: None,
            created_at: now,
            updated_at: now,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_flow_is_legal() {
        assert!(TaskStatus::Running.can_transition_to(TaskStatus::Failed));
        assert!(TaskStatus::Failed.can_transition_to(TaskStatus::Retrying));
        assert!(TaskStatus::Retrying.can_transition_to(TaskStatus::Pending));
        assert!(!TaskStatus::Succeeded.can_transition_to(TaskStatus::Running));
    }

    #[test]
    fn new_tasks_have_no_parent_until_the_plan_assigns_one() {
        assert_eq!(TaskSpec::new("root task").parent, None);
    }
}

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

/// Portable execution limits associated with one process-backed task.
///
/// `max_memory_bytes` and `max_cpu_time_ms` are enforced by a local Unix child
/// or translated to Docker flags. `max_output_bytes` is enforced separately
/// for each stdout/stderr stream by Horizon's readers on every supported
/// platform.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskResourceLimits {
    #[serde(default)]
    pub max_memory_bytes: Option<u64>,
    #[serde(default)]
    pub max_cpu_time_ms: Option<u64>,
    #[serde(default)]
    pub max_output_bytes: Option<u64>,
}

impl TaskResourceLimits {
    pub fn validate(&self) -> Result<(), TaskSpecValidationError> {
        for (field, value) in [
            ("max_memory_bytes", self.max_memory_bytes),
            ("max_cpu_time_ms", self.max_cpu_time_ms),
            ("max_output_bytes", self.max_output_bytes),
        ] {
            if value == Some(0) {
                return Err(TaskSpecValidationError::ZeroResourceLimit(field));
            }
        }
        Ok(())
    }
}

/// Where a task's argv program is executed. Docker is intentionally a small,
/// explicit execution target rather than a general sandbox orchestrator.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskExecutionBackend {
    #[default]
    Local,
    Docker {
        image: String,
        /// Docker networking is disabled unless this is explicitly true.
        #[serde(default)]
        allow_network: bool,
        /// Docker's writable root filesystem is disabled by default. A mounted
        /// task working directory remains writable when supplied.
        #[serde(default = "default_docker_read_only")]
        read_only: bool,
    },
}

const fn default_docker_read_only() -> bool {
    true
}

impl TaskExecutionBackend {
    pub fn validate(&self) -> Result<(), TaskSpecValidationError> {
        match self {
            Self::Local => Ok(()),
            Self::Docker { image, .. } if image.trim().is_empty() => {
                Err(TaskSpecValidationError::EmptyDockerImage)
            }
            Self::Docker { .. } => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum TaskSpecValidationError {
    #[error("task resource limit `{0}` must be greater than zero when configured")]
    ZeroResourceLimit(&'static str),
    #[error("Docker execution requires a non-empty image")]
    EmptyDockerImage,
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
    #[serde(default)]
    pub resources: TaskResourceLimits,
    #[serde(default)]
    pub executor: TaskExecutionBackend,
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
            resources: TaskResourceLimits::default(),
            executor: TaskExecutionBackend::Local,
        }
    }

    pub fn validate_execution(&self) -> Result<(), TaskSpecValidationError> {
        self.resources.validate()?;
        self.executor.validate()
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

    #[test]
    fn rejects_zero_resource_budget_and_empty_docker_image() {
        let mut task = TaskSpec::new("bounded work");
        task.resources.max_output_bytes = Some(0);
        assert!(matches!(
            task.validate_execution(),
            Err(TaskSpecValidationError::ZeroResourceLimit(_))
        ));
        task.resources.max_output_bytes = None;
        task.executor = TaskExecutionBackend::Docker {
            image: " ".into(),
            allow_network: false,
            read_only: true,
        };
        assert_eq!(task.validate_execution(), Err(TaskSpecValidationError::EmptyDockerImage));
    }
}

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::{
    AgentState, BudgetState, Constraint, DecisionRecord, EvidenceRecord, FailureRecord, MemoryItem,
    Subgoal, TaskSpec, ToolResult,
};

/// Commands are intent, never persisted truth. The runtime validates a command,
/// creates its event, persists it, then applies it to the projection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum RuntimeCommand {
    Transition {
        to: AgentState,
    },
    RegisterGoal {
        goal: String,
    },
    AddConstraint {
        constraint: Constraint,
    },
    SetPlan {
        steps: Vec<String>,
    },
    OpenSubgoal {
        subgoal: Subgoal,
    },
    CompleteSubgoal {
        id: String,
    },
    CreateTask {
        task: TaskSpec,
    },
    StartTask {
        task_id: crate::TaskId,
    },
    BlockTask {
        task_id: crate::TaskId,
        reason: String,
    },
    UnblockTask {
        task_id: crate::TaskId,
    },
    SucceedTask {
        task_id: crate::TaskId,
        output: String,
    },
    FailTask {
        task_id: crate::TaskId,
        error: String,
        retryable: bool,
        operation_id: Option<String>,
    },
    RetryTask {
        task_id: crate::TaskId,
    },
    CancelTask {
        task_id: crate::TaskId,
        reason: String,
    },
    RememberFailure {
        failure: FailureRecord,
    },
    RecordDecision {
        decision: DecisionRecord,
    },
    RecordEvidence {
        evidence: EvidenceRecord,
    },
    CreateMemory {
        memory: MemoryItem,
    },
    RecordToolInvocation {
        operation_id: String,
        tool: String,
        input: Value,
    },
    RecordToolSuccess {
        operation_id: String,
        output: Value,
    },
    RecordToolFailure {
        operation_id: String,
        error: String,
        retryable: bool,
    },
    RecordToolResult {
        result: ToolResult,
    },
    UpdateBudget {
        budget: BudgetState,
    },
    UpdateEnvironment {
        environment: Value,
    },
    Suspend {
        reason: String,
    },
}

#[derive(Debug, Error)]
pub enum CommandError {
    #[error("a run must have a non-empty goal")]
    EmptyGoal,
    #[error("a plan must contain at least one step")]
    EmptyPlan,
    #[error("a subgoal must have a non-empty id and content")]
    EmptySubgoal,
    #[error("subgoal `{0}` already exists")]
    SubgoalAlreadyExists(String),
    #[error("open subgoal `{0}` does not exist")]
    UnknownSubgoal(String),
    #[error("task {0} does not exist")]
    UnknownTask(crate::TaskId),
    #[error("task {0} already exists")]
    TaskAlreadyExists(crate::TaskId),
    #[error("operation id `{0}` is already assigned to another task in this run")]
    DuplicateOperationId(String),
    #[error("task {0} is not ready because a dependency has not succeeded")]
    TaskNotReady(crate::TaskId),
    #[error("task {0} has exhausted its retry budget")]
    RetryExhausted(crate::TaskId),
    #[error(transparent)]
    InvalidState(#[from] crate::StateTransitionError),
    #[error(transparent)]
    InvalidTask(#[from] crate::TaskTransitionError),
}

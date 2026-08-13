//! The deterministic domain model behind Horizon.
//!
//! `horizon-core` deliberately contains no database, network, process, or LLM
//! code. A [`RunProjection`] is reconstructed only by applying ordered events,
//! which makes it suitable for replay, auditing, and crash recovery.

mod command;
mod event;
mod ids;
mod projection;
mod state;
mod task;

pub use command::{CommandError, RuntimeCommand};
pub use event::{
    AnchorRecord, BudgetState, CURRENT_EVENT_SCHEMA_VERSION, CognitiveState, Constraint,
    DecisionRecord, EventKind, EventRecord, EvidenceRecord, FailureRecord, MemoryItem, MemoryKind,
    NewEvent, Subgoal, ToolResult, ToolResultStatus,
};
pub use ids::{EventId, MemoryId, RunId, TaskId};
pub use projection::{ProjectionError, RunProjection};
pub use state::{AgentState, StateTransitionError};
pub use task::{
    TaskExecutionBackend, TaskRecord, TaskResourceLimits, TaskSpec, TaskSpecValidationError,
    TaskStatus, TaskTransitionError,
};

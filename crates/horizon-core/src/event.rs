use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{AgentState, EventId, MemoryId, RunId, TaskId, TaskSpec};

/// Structured memory types deliberately favor execution state over generic
/// semantic retrieval. They are first-class event payloads and survive replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Goal,
    Constraint,
    Decision,
    Failure,
    Evidence,
    Progress,
    Environment,
    Episodic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: MemoryId,
    pub kind: MemoryKind,
    pub content: String,
    pub source_event: EventId,
    pub importance: f32,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraint {
    pub id: String,
    pub content: String,
    pub source: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Subgoal {
    pub id: String,
    pub content: String,
    pub completed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureRecord {
    pub id: String,
    pub approach: String,
    pub reason: String,
    pub operation_id: Option<String>,
    pub retryable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub id: String,
    pub decision: String,
    pub rationale: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub id: String,
    pub content: String,
    pub source: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BudgetState {
    pub token_budget: Option<u64>,
    pub tokens_used: u64,
    pub wall_time_budget_ms: Option<u64>,
    pub wall_time_used_ms: u64,
}

/// The information that must remain behaviorally available to a long-horizon
/// agent. This is intentionally not a transcript.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CognitiveState {
    pub primary_goal: Option<String>,
    #[serde(default)]
    pub constraints: Vec<Constraint>,
    #[serde(default)]
    pub current_plan: Vec<String>,
    #[serde(default)]
    pub open_subgoals: Vec<Subgoal>,
    #[serde(default)]
    pub completed_subgoals: Vec<Subgoal>,
    #[serde(default)]
    pub failed_attempts: Vec<FailureRecord>,
    #[serde(default)]
    pub decisions: Vec<DecisionRecord>,
    #[serde(default)]
    pub evidence: Vec<EvidenceRecord>,
    #[serde(default)]
    pub environment: Value,
    #[serde(default)]
    pub budget: BudgetState,
}

impl CognitiveState {
    /// Human-readable compact context intended for injection into an LLM policy.
    #[must_use]
    pub fn render_anchor(&self) -> String {
        fn section(lines: &mut Vec<String>, title: &str, values: impl Iterator<Item = String>) {
            let values: Vec<_> = values.collect();
            if !values.is_empty() {
                lines.push(format!("{title}:"));
                lines.extend(values.into_iter().map(|item| format!("- {item}")));
            }
        }

        let mut lines = vec!["STATE ANCHOR".to_owned()];
        if let Some(goal) = &self.primary_goal {
            lines.push(format!("Primary Goal: {goal}"));
        }
        section(
            &mut lines,
            "Constraints",
            self.constraints.iter().map(|item| item.content.clone()),
        );
        section(&mut lines, "Current Plan", self.current_plan.iter().cloned());
        section(
            &mut lines,
            "Open Subgoals",
            self.open_subgoals.iter().map(|item| item.content.clone()),
        );
        section(
            &mut lines,
            "Important Decisions",
            self.decisions
                .iter()
                .map(|item| format!("{} (because {})", item.decision, item.rationale)),
        );
        section(
            &mut lines,
            "Failed Approaches",
            self.failed_attempts.iter().map(|item| format!("{} — {}", item.approach, item.reason)),
        );
        section(&mut lines, "Evidence", self.evidence.iter().map(|item| item.content.clone()));
        lines.push(format!(
            "Budget: {} tokens used{}",
            self.budget.tokens_used,
            self.budget.token_budget.map(|total| format!(" / {total}")).unwrap_or_default()
        ));
        lines.join("\n")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorRecord {
    pub risk_score_milli: u32,
    pub reason: String,
    pub content: String,
}

/// Every state mutation travels through this immutable event vocabulary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum EventKind {
    RunCreated { goal: String },
    StateTransitioned { from: AgentState, to: AgentState },
    GoalRegistered { goal: String },
    ConstraintAdded { constraint: Constraint },
    PlanCreated { steps: Vec<String> },
    SubgoalOpened { subgoal: Subgoal },
    SubgoalCompleted { id: String },
    TaskCreated { task: TaskSpec },
    TaskStarted { task_id: TaskId, attempt: u32 },
    TaskBlocked { task_id: TaskId, reason: String },
    TaskUnblocked { task_id: TaskId },
    TaskSucceeded { task_id: TaskId, output: String },
    TaskFailed { task_id: TaskId, error: String, retryable: bool, operation_id: Option<String> },
    TaskRetried { task_id: TaskId, attempt: u32 },
    TaskCancelled { task_id: TaskId, reason: String },
    ToolInvoked { operation_id: String, tool: String, input: Value },
    ToolSucceeded { operation_id: String, output: Value },
    ToolFailed { operation_id: String, error: String, retryable: bool },
    ProcessCompleted { operation_id: String, exit_code: Option<i32> },
    MemoryCreated { memory: MemoryItem },
    FailureRemembered { failure: FailureRecord },
    DecisionMade { decision: DecisionRecord },
    EvidenceRecorded { evidence: EvidenceRecord },
    BudgetUpdated { budget: BudgetState },
    EnvironmentUpdated { environment: Value },
    StateAnchorInjected { anchor: AnchorRecord },
    CheckpointCreated { sequence: u64 },
    AgentSuspended { reason: String },
    AgentRecovered { from_sequence: u64 },
    Note { message: String },
}

impl EventKind {
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::RunCreated { .. } => "run_created",
            Self::StateTransitioned { .. } => "state_transitioned",
            Self::GoalRegistered { .. } => "goal_registered",
            Self::ConstraintAdded { .. } => "constraint_added",
            Self::PlanCreated { .. } => "plan_created",
            Self::SubgoalOpened { .. } => "subgoal_opened",
            Self::SubgoalCompleted { .. } => "subgoal_completed",
            Self::TaskCreated { .. } => "task_created",
            Self::TaskStarted { .. } => "task_started",
            Self::TaskBlocked { .. } => "task_blocked",
            Self::TaskUnblocked { .. } => "task_unblocked",
            Self::TaskSucceeded { .. } => "task_succeeded",
            Self::TaskFailed { .. } => "task_failed",
            Self::TaskRetried { .. } => "task_retried",
            Self::TaskCancelled { .. } => "task_cancelled",
            Self::ToolInvoked { .. } => "tool_invoked",
            Self::ToolSucceeded { .. } => "tool_succeeded",
            Self::ToolFailed { .. } => "tool_failed",
            Self::ProcessCompleted { .. } => "process_completed",
            Self::MemoryCreated { .. } => "memory_created",
            Self::FailureRemembered { .. } => "failure_remembered",
            Self::DecisionMade { .. } => "decision_made",
            Self::EvidenceRecorded { .. } => "evidence_recorded",
            Self::BudgetUpdated { .. } => "budget_updated",
            Self::EnvironmentUpdated { .. } => "environment_updated",
            Self::StateAnchorInjected { .. } => "state_anchor_injected",
            Self::CheckpointCreated { .. } => "checkpoint_created",
            Self::AgentSuspended { .. } => "agent_suspended",
            Self::AgentRecovered { .. } => "agent_recovered",
            Self::Note { .. } => "note",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NewEvent {
    pub run_id: RunId,
    pub event: EventKind,
    #[serde(default)]
    pub metadata: Value,
}

impl NewEvent {
    #[must_use]
    pub fn new(run_id: RunId, event: EventKind) -> Self {
        Self { run_id, event, metadata: Value::Null }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventRecord {
    pub id: EventId,
    pub run_id: RunId,
    pub sequence: u64,
    pub timestamp: DateTime<Utc>,
    #[serde(flatten)]
    pub event: EventKind,
    #[serde(default)]
    pub metadata: Value,
}

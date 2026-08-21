use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{AgentState, EventId, MemoryId, RunId, TaskId, TaskSpec};

/// Version of the serialized [`EventKind`] payload understood by this build.
///
/// The store persists this beside every event instead of inferring it from a
/// database migration. That lets future releases upcast older immutable event
/// payloads without rewriting an audit log in place.
///
/// Version 3 adds durable semantic-memory retrieval records. The event-schema
/// decoder keeps version-1 and version-2 histories readable rather than
/// rewriting them.
pub const CURRENT_EVENT_SCHEMA_VERSION: u32 = 3;

const fn default_event_schema_version() -> u32 {
    CURRENT_EVENT_SCHEMA_VERSION
}

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

/// One selected memory together with the deterministic score that selected it.
///
/// Content and kind are copied into the immutable retrieval event so a later
/// reader can audit exactly what was injected, even if a future version adds a
/// separate memory lifecycle or index implementation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticMemoryHit {
    pub memory_id: MemoryId,
    pub kind: MemoryKind,
    pub content: String,
    /// Integer score in `[0, 1000]`, avoiding float-formatting drift in audit
    /// and cross-language clients.
    pub score_milli: u32,
}

/// A replayable semantic-memory query boundary.
///
/// The runtime owns the selected hits; policy code supplies only a query and
/// result limit. Recording both empty and non-empty lookups makes retrieval
/// behavior reproducible and supports retrieval-specific ablations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticMemoryRetrieval {
    pub query: String,
    /// Stable retrieval implementation identifier, for example
    /// `hybrid_lexical_v1`.
    pub algorithm: String,
    #[serde(default)]
    pub hits: Vec<SemanticMemoryHit>,
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

/// Normalized terminal result for a tool or external executor. This avoids
/// forcing consumers to infer success from ad-hoc strings or process exit codes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultStatus {
    #[default]
    Succeeded,
    Failed,
    TimedOut,
    Cancelled,
}

impl ToolResultStatus {
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Succeeded)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub operation_id: String,
    pub tool: String,
    pub status: ToolResultStatus,
    #[serde(default)]
    pub output: Value,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub metadata: Value,
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
        section(&mut lines, "Approved plan", self.current_plan.iter().cloned());
        // Keep the same bounded progress facts available in full Anchors and
        // compact Python context. A policy can therefore evaluate a plan step
        // such as "if evidence=0" without inferring counts from repeated prose.
        lines.push(format!(
            "Durable records: decisions={}, evidence={}, remembered failures={}",
            self.decisions.len(),
            self.evidence.len(),
            self.failed_attempts.len()
        ));
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

/// Observable inputs to a state-decay policy at one execution boundary.
///
/// This belongs to the durable domain model rather than a particular policy
/// implementation. Python may train and run a predictor, while Rust persists
/// the exact inputs that led to an intervention decision for replay and later
/// calibration analysis.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StateDecaySignals {
    /// Number of durable event boundaries since the last State Anchor.
    #[serde(default)]
    pub steps_since_anchor: u64,
    /// Current context pressure normalized to `[0, 1]`.
    #[serde(default)]
    pub context_pressure: f32,
    /// Recent switches between active subgoals.
    #[serde(default)]
    pub subgoal_switches: u32,
    /// Recent process, tool, or policy failures.
    #[serde(default)]
    pub recent_failures: u32,
    /// Whether the boundary follows recovery from another process/session.
    #[serde(default)]
    pub recovered_session: bool,
}

/// The policy-selected action after assessing state-decay risk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionAction {
    /// Continue with compact context at this boundary.
    #[default]
    Continue,
    /// Inject a Rust-rendered State Anchor from the durable projection.
    InjectAnchor,
}

impl InterventionAction {
    #[must_use]
    pub const fn injects_anchor(self) -> bool {
        matches!(self, Self::InjectAnchor)
    }
}

/// A replayable policy assessment, including decisions that intentionally do
/// not inject an anchor. `risk_score_milli` and `threshold_milli` use integers
/// in `[0, 1000]` so audit records do not depend on floating-point formatting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterventionAssessment {
    /// Stable identifier for the policy or model family, for example
    /// `logistic_state_decay`.
    pub policy_id: String,
    /// Optional immutable model/calibration revision.
    #[serde(default)]
    pub policy_version: Option<String>,
    pub risk_score_milli: u32,
    pub threshold_milli: u32,
    pub signals: StateDecaySignals,
    pub action: InterventionAction,
    /// Human-readable explanation generated by the policy/controller.
    pub reason: String,
    /// Optional structured diagnostics such as estimated anchor token cost.
    #[serde(default)]
    pub metadata: Value,
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
    ToolResultRecorded { result: ToolResult },
    ProcessCompleted { operation_id: String, exit_code: Option<i32> },
    MemoryCreated { memory: MemoryItem },
    SemanticMemoryRetrieved { retrieval: SemanticMemoryRetrieval },
    FailureRemembered { failure: FailureRecord },
    DecisionMade { decision: DecisionRecord },
    EvidenceRecorded { evidence: EvidenceRecord },
    BudgetUpdated { budget: BudgetState },
    EnvironmentUpdated { environment: Value },
    StateDecayAssessed { assessment: InterventionAssessment },
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
            Self::ToolResultRecorded { .. } => "tool_result_recorded",
            Self::ProcessCompleted { .. } => "process_completed",
            Self::MemoryCreated { .. } => "memory_created",
            Self::SemanticMemoryRetrieved { .. } => "semantic_memory_retrieved",
            Self::FailureRemembered { .. } => "failure_remembered",
            Self::DecisionMade { .. } => "decision_made",
            Self::EvidenceRecorded { .. } => "evidence_recorded",
            Self::BudgetUpdated { .. } => "budget_updated",
            Self::EnvironmentUpdated { .. } => "environment_updated",
            Self::StateDecayAssessed { .. } => "state_decay_assessed",
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
    /// Schema version of `event` at the time it was produced.
    #[serde(default = "default_event_schema_version")]
    pub schema_version: u32,
    pub event: EventKind,
    #[serde(default)]
    pub metadata: Value,
}

impl NewEvent {
    #[must_use]
    pub fn new(run_id: RunId, event: EventKind) -> Self {
        Self { run_id, schema_version: CURRENT_EVENT_SCHEMA_VERSION, event, metadata: Value::Null }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventRecord {
    pub id: EventId,
    pub run_id: RunId,
    pub sequence: u64,
    pub timestamp: DateTime<Utc>,
    /// The schema version used to decode this persisted event payload.
    #[serde(default = "default_event_schema_version")]
    pub schema_version: u32,
    #[serde(flatten)]
    pub event: EventKind,
    #[serde(default)]
    pub metadata: Value,
}

#[cfg(test)]
mod tests {
    use super::{CognitiveState, DecisionRecord, EvidenceRecord, FailureRecord};

    #[test]
    fn rendered_anchor_exposes_durable_record_counts_before_detail_sections() {
        let state = CognitiveState {
            current_plan: vec!["Persist each required durable record once.".to_owned()],
            decisions: vec![DecisionRecord {
                id: "decision-1".to_owned(),
                decision: "Use the checkpointed rollback path.".to_owned(),
                rationale: "It preserves the frozen artifact.".to_owned(),
            }],
            evidence: vec![EvidenceRecord {
                id: "evidence-1".to_owned(),
                content: "The checksum matches.".to_owned(),
                source: None,
            }],
            failed_attempts: vec![FailureRecord {
                id: "failure-1".to_owned(),
                approach: "restart from event zero".to_owned(),
                reason: "It discards durable progress.".to_owned(),
                operation_id: None,
                retryable: false,
            }],
            ..CognitiveState::default()
        };

        let anchor = state.render_anchor();
        let counts = "Durable records: decisions=1, evidence=1, remembered failures=1";
        assert!(anchor.contains("Approved plan:\n- Persist each required durable record once."));
        assert!(anchor.contains(counts));
        let counts_index = anchor.find(counts).expect("Anchor should include record counts");
        let decisions_index = anchor
            .find("Important Decisions:")
            .expect("Anchor should include the decision detail section");
        assert!(counts_index < decisions_index);
    }
}

//! Proactive cognitive-state intervention.
//!
//! Horizon treats memory as execution-state management. This crate contains a
//! transparent heuristic policy for when an agent needs an explicit state anchor
//! rather than injecting all memory on every turn.

use horizon_core::{AnchorRecord, CognitiveState, FailureRecord};
use serde::{Deserialize, Serialize};

/// Observable factors that predict behavioral state decay at an execution
/// boundary. Inputs are normalized where possible to keep configuration stable.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DecaySignals {
    /// Number of event boundaries since the last state anchor.
    pub steps_since_anchor: u64,
    /// Context use as a fraction in `[0, 1]`; values are clamped by the policy.
    pub context_pressure: f32,
    /// Number of recent switches between subgoals.
    pub subgoal_switches: u32,
    /// Number of recent task/tool failures.
    pub recent_failures: u32,
    /// True after a cross-process or cross-session restore.
    pub recovered_session: bool,
}

/// Explainable implementation of `R_t = w1L + w2C + w3S + w4F + w5X`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateAnchorPolicy {
    /// Risk threshold in `[0, 1]`; an anchor is injected only above it.
    pub threshold: f32,
    pub distance_weight: f32,
    pub context_weight: f32,
    pub switch_weight: f32,
    pub failure_weight: f32,
    pub recovery_weight: f32,
    /// Number of events considered a maximum-distance signal.
    pub distance_scale: u64,
    /// Number of switches that saturates the switch signal.
    pub switch_scale: u32,
    /// Number of failures that saturates the failure signal.
    pub failure_scale: u32,
}

impl Default for StateAnchorPolicy {
    fn default() -> Self {
        Self {
            threshold: 0.55,
            distance_weight: 0.25,
            context_weight: 0.25,
            switch_weight: 0.15,
            failure_weight: 0.20,
            recovery_weight: 0.60,
            distance_scale: 12,
            switch_scale: 3,
            failure_scale: 2,
        }
    }
}

impl StateAnchorPolicy {
    #[must_use]
    pub fn risk(&self, signals: &DecaySignals) -> f32 {
        let distance = normalized_u64(signals.steps_since_anchor, self.distance_scale);
        let context = signals.context_pressure.clamp(0.0, 1.0);
        let switches = normalized_u32(signals.subgoal_switches, self.switch_scale);
        let failures = normalized_u32(signals.recent_failures, self.failure_scale);
        let recovery = f32::from(signals.recovered_session);
        (self.distance_weight * distance
            + self.context_weight * context
            + self.switch_weight * switches
            + self.failure_weight * failures
            + self.recovery_weight * recovery)
            .clamp(0.0, 1.0)
    }

    #[must_use]
    pub fn should_inject(&self, signals: &DecaySignals) -> bool {
        self.risk(signals) >= self.threshold.clamp(0.0, 1.0)
    }

    /// Builds an immutable audit record only when the intervention is justified.
    #[must_use]
    pub fn decide(&self, state: &CognitiveState, signals: &DecaySignals) -> Option<AnchorRecord> {
        let risk = self.risk(signals);
        self.should_inject(signals).then(|| AnchorRecord {
            risk_score_milli: (risk * 1_000.0).round() as u32,
            reason: self.explain(signals),
            content: state.render_anchor(),
        })
    }

    #[must_use]
    pub fn explain(&self, signals: &DecaySignals) -> String {
        let mut causes = Vec::new();
        if normalized_u64(signals.steps_since_anchor, self.distance_scale) >= 0.5 {
            causes.push("anchor distance");
        }
        if signals.context_pressure.clamp(0.0, 1.0) >= 0.5 {
            causes.push("context pressure");
        }
        if normalized_u32(signals.subgoal_switches, self.switch_scale) >= 0.5 {
            causes.push("subgoal switching");
        }
        if normalized_u32(signals.recent_failures, self.failure_scale) >= 0.5 {
            causes.push("recent failures");
        }
        if signals.recovered_session {
            causes.push("session recovery");
        }
        if causes.is_empty() { "policy threshold".to_owned() } else { causes.join(", ") }
    }
}

fn normalized_u64(value: u64, scale: u64) -> f32 {
    if scale == 0 { 1.0 } else { (value as f32 / scale as f32).clamp(0.0, 1.0) }
}

fn normalized_u32(value: u32, scale: u32) -> f32 {
    if scale == 0 { 1.0 } else { (value as f32 / scale as f32).clamp(0.0, 1.0) }
}

/// Query helper used by agent policies before attempting an operation. Exact
/// `operation_id` matches are strongest; normalized approach matches cover
/// recurrent plans that receive a fresh operation ID on each attempt.
#[must_use]
pub fn matching_failure<'a>(
    failures: &'a [FailureRecord],
    operation_or_approach: &str,
) -> Option<&'a FailureRecord> {
    let needle = normalize(operation_or_approach);
    failures.iter().rev().find(|failure| {
        failure.operation_id.as_deref().is_some_and(|id| normalize(id) == needle)
            || normalize(&failure.approach) == needle
    })
}

fn normalize(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

#[cfg(test)]
mod tests {
    use horizon_core::{CognitiveState, FailureRecord};

    use super::*;

    #[test]
    fn recovery_triggers_anchor_with_default_policy() {
        let state = CognitiveState {
            primary_goal: Some("Do not forget constraints".into()),
            ..Default::default()
        };
        let decision = StateAnchorPolicy::default().decide(
            &state,
            &DecaySignals { recovered_session: true, context_pressure: 0.9, ..Default::default() },
        );
        let anchor = decision.expect("risk should be high");
        assert!(anchor.content.contains("Primary Goal"));
        assert!(anchor.reason.contains("session recovery"));
    }

    #[test]
    fn finds_previous_failure_by_operation() {
        let failure = FailureRecord {
            id: "f1".into(),
            approach: "try model A".into(),
            reason: "incompatible schema".into(),
            operation_id: Some("operation-42".into()),
            retryable: false,
        };
        assert_eq!(matching_failure(&[failure], "operation-42").unwrap().id, "f1");
    }
}

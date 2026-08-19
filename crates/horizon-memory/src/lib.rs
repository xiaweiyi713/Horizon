//! Proactive cognitive-state intervention and deterministic memory ranking.
//!
//! Horizon treats memory as execution-state management. This crate contains a
//! transparent heuristic policy for when an agent needs an explicit state anchor
//! rather than injecting all memory on every turn, and a dependency-free lexical
//! retriever that ranks durable memory items for compact injection.

use std::collections::BTreeSet;

use horizon_core::{AnchorRecord, CognitiveState, FailureRecord, MemoryId, MemoryItem};
use serde::{Deserialize, Serialize};

/// Backwards-compatible policy-facing name for the durable domain signals.
/// The type lives in `horizon-core` because learned policy assessments persist
/// it in the event log; this crate remains responsible only for heuristic
/// scoring and explanation.
pub use horizon_core::StateDecaySignals as DecaySignals;

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

/// Stable identifier for this retriever, recorded by runtime audit events.
pub const SEMANTIC_RETRIEVAL_ALGORITHM: &str = "hybrid_lexical_v1";

/// Token-set Jaccard is weighted more heavily than character-trigram Jaccard.
const TOKEN_OVERLAP_WEIGHT: u32 = 3;
const TRIGRAM_SIMILARITY_WEIGHT: u32 = 2;
/// Discard residual / coincidental overlap before quality weighting.
const MIN_LEXICAL_RELEVANCE_MILLI: u32 = 50;
const QUALITY_BAND_DENOMINATOR: u64 = 1_100;

/// One ranked memory together with its integer retrieval score.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievedMemory {
    pub memory_id: MemoryId,
    /// Integer score in `0..=1000`.
    pub score_milli: u32,
}

/// Deterministic hybrid lexical ranker (`hybrid_lexical_v1`).
///
/// Scoring is integer-only after tokenization so the same catalog and query
/// always produce the same ranking:
///
/// 1. Unicode-lowercase the text and split on non-alphanumeric characters
///    (`char::is_alphanumeric`). Tokens are the remaining alphanumeric runs.
/// 2. **Token overlap** is the Jaccard index of the query and memory token
///    sets, scaled to milli: `round(1000 * |Q ∩ D| / |Q ∪ D|)`.
/// 3. **Character-trigram similarity** is the same Jaccard over overlapping
///    Unicode-scalar 3-grams of those tokens joined by a single space.
/// 4. **Lexical relevance** is a 3:2 blend of the token and trigram milli
///    scores. A memory with no token or trigram overlap, or with blended
///    relevance below 50, is discarded. Importance and confidence cannot
///    create a hit from zero lexical relevance.
/// 5. Importance and confidence (each clamped to `[0, 1]`, non-finite → 0)
///    form a quality term in `[0, 1000]`. Quality may scale a *positive*
///    lexical score within a 10% band:
///    `score = round(lexical * (1000 + quality / 10) / 1100)`.
///    High quality never exceeds the lexical score; low quality withholds
///    up to 10%.
/// 6. Results are unique by `MemoryId` (first occurrence wins), sorted by
///    descending `score_milli`, then ascending `MemoryId`, then truncated
///    to `limit`. Empty/whitespace queries, punctuation-only queries, zero
///    `limit`, and blank memory content produce no corresponding hits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticMemoryRetrievalPolicy;

impl SemanticMemoryRetrievalPolicy {
    #[must_use]
    pub fn retrieve(
        &self,
        query: &str,
        memories: &[MemoryItem],
        limit: usize,
    ) -> Vec<RetrievedMemory> {
        if limit == 0 || query.trim().is_empty() {
            return Vec::new();
        }
        let query_features = TextFeatures::analyze(query);
        if query_features.is_empty() {
            return Vec::new();
        }

        let mut seen = BTreeSet::new();
        let mut ranked = Vec::new();
        for memory in memories {
            if !seen.insert(memory.id) {
                continue;
            }
            if let Some(hit) = score_memory(&query_features, memory) {
                ranked.push(hit);
            }
        }
        ranked.sort_by(|left, right| {
            right
                .score_milli
                .cmp(&left.score_milli)
                .then_with(|| left.memory_id.cmp(&right.memory_id))
        });
        ranked.truncate(limit);
        ranked
    }
}

struct TextFeatures {
    tokens: BTreeSet<String>,
    trigrams: BTreeSet<(char, char, char)>,
}

impl TextFeatures {
    fn analyze(text: &str) -> Self {
        let tokens = alphanumeric_tokens(text);
        let trigrams = char_trigrams(&tokens.join(" "));
        Self { tokens: tokens.into_iter().collect(), trigrams }
    }

    fn is_empty(&self) -> bool {
        self.tokens.is_empty() && self.trigrams.is_empty()
    }
}

fn alphanumeric_tokens(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in input.chars() {
        for lower in ch.to_lowercase() {
            if lower.is_alphanumeric() {
                current.push(lower);
            } else if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn char_trigrams(text: &str) -> BTreeSet<(char, char, char)> {
    let chars: Vec<char> = text.chars().collect();
    chars.windows(3).map(|window| (window[0], window[1], window[2])).collect()
}

fn jaccard_milli<T: Ord>(left: &BTreeSet<T>, right: &BTreeSet<T>) -> u32 {
    if left.is_empty() || right.is_empty() {
        return 0;
    }
    let intersection = left.intersection(right).count() as u64;
    let union = left.len() as u64 + right.len() as u64 - intersection;
    let numerator = intersection * 1_000 + union / 2;
    numerator.checked_div(union).unwrap_or_default() as u32
}

fn score_memory(query: &TextFeatures, memory: &MemoryItem) -> Option<RetrievedMemory> {
    if memory.content.trim().is_empty() {
        return None;
    }
    let features = TextFeatures::analyze(&memory.content);
    if features.is_empty() {
        return None;
    }
    let token_milli = jaccard_milli(&query.tokens, &features.tokens);
    let trigram_milli = jaccard_milli(&query.trigrams, &features.trigrams);
    if token_milli == 0 && trigram_milli == 0 {
        return None;
    }
    let lexical_milli =
        weighted_milli(token_milli, TOKEN_OVERLAP_WEIGHT, trigram_milli, TRIGRAM_SIMILARITY_WEIGHT);
    if lexical_milli < MIN_LEXICAL_RELEVANCE_MILLI {
        return None;
    }
    let score_milli =
        apply_quality(lexical_milli, quality_milli(memory.importance, memory.confidence));
    (score_milli > 0).then_some(RetrievedMemory { memory_id: memory.id, score_milli })
}

fn weighted_milli(left: u32, left_weight: u32, right: u32, right_weight: u32) -> u32 {
    let numerator =
        u64::from(left) * u64::from(left_weight) + u64::from(right) * u64::from(right_weight);
    let denominator = u64::from(left_weight) + u64::from(right_weight);
    ((numerator + denominator / 2) / denominator) as u32
}

fn quality_milli(importance: f32, confidence: f32) -> u32 {
    (unit_interval_milli(importance) + unit_interval_milli(confidence)) / 2
}

fn unit_interval_milli(value: f32) -> u32 {
    if !value.is_finite() { 0 } else { (value.clamp(0.0, 1.0) * 1_000.0).round() as u32 }
}

fn apply_quality(lexical_milli: u32, quality_milli: u32) -> u32 {
    let factor = 1_000 + u64::from(quality_milli) / 10;
    let rounded = (u64::from(lexical_milli) * factor + QUALITY_BAND_DENOMINATOR / 2)
        / QUALITY_BAND_DENOMINATOR;
    rounded.min(1_000) as u32
}

#[cfg(test)]
mod tests {
    use horizon_core::{CognitiveState, EventId, FailureRecord, MemoryItem, MemoryKind};

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

    const ID_LOW: &str = "00000000-0000-0000-0000-000000000001";
    const ID_HIGH: &str = "00000000-0000-0000-0000-000000000002";
    const ID_THIRD: &str = "00000000-0000-0000-0000-000000000003";

    fn sample_memory(id: &str, content: &str, importance: f32, confidence: f32) -> MemoryItem {
        MemoryItem {
            id: id.parse().expect("memory id"),
            kind: MemoryKind::Evidence,
            content: content.to_owned(),
            source_event: EventId::default(),
            importance,
            confidence,
            created_at: Default::default(),
        }
    }

    #[test]
    fn ranks_relevant_memories_above_irrelevant_ones() {
        let relevant =
            sample_memory(ID_LOW, "sqlite checkpoint compaction preserves the event log", 0.4, 0.4);
        let irrelevant = sample_memory(ID_HIGH, "banana pancake recipe with maple syrup", 0.4, 0.4);
        let hits = SemanticMemoryRetrievalPolicy.retrieve(
            "sqlite checkpoint compaction",
            &[irrelevant.clone(), relevant.clone()],
            8,
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, relevant.id);
        assert!((1..=1_000).contains(&hits[0].score_milli));
    }

    #[test]
    fn normalizes_case_and_punctuation() {
        let memory = sample_memory(ID_LOW, "Postgres, Recovery!  Ready.", 0.5, 0.5);
        let hits = SemanticMemoryRetrievalPolicy.retrieve(
            "POSTGRES recovery ready???",
            std::slice::from_ref(&memory),
            4,
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, memory.id);
        assert!((1..=1_000).contains(&hits[0].score_milli));
    }

    #[test]
    fn quality_does_not_rescue_irrelevant_text() {
        let relevant = sample_memory(ID_LOW, "postgres connection pool timeout", 0.05, 0.05);
        let shiny = sample_memory(ID_HIGH, "xylophone umbrellas juggle quartz", 1.0, 1.0);
        let hits = SemanticMemoryRetrievalPolicy.retrieve(
            "postgres connection pool",
            &[shiny.clone(), relevant.clone()],
            8,
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, relevant.id);
        assert!((1..=1_000).contains(&hits[0].score_milli));
    }

    #[test]
    fn breaks_score_ties_by_ascending_memory_id() {
        let later = sample_memory(ID_HIGH, "durable event sourced runtime", 0.5, 0.5);
        let earlier = sample_memory(ID_LOW, "durable event sourced runtime", 0.5, 0.5);
        let policy = SemanticMemoryRetrievalPolicy;
        let hits =
            policy.retrieve("durable event sourced runtime", &[later.clone(), earlier.clone()], 8);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].score_milli, hits[1].score_milli);
        assert_eq!(hits[0].memory_id, earlier.id);
        assert_eq!(hits[1].memory_id, later.id);
        assert_eq!(
            policy.retrieve("durable event sourced runtime", &[earlier.clone(), later.clone()], 8,),
            hits
        );
    }

    #[test]
    fn empty_query_zero_limit_and_blank_content_yield_no_hits() {
        let filled = sample_memory(ID_LOW, "postgres checkpoint", 1.0, 1.0);
        let blank = sample_memory(ID_HIGH, "   ", 1.0, 1.0);
        let empty = sample_memory(ID_THIRD, "", 1.0, 1.0);
        let memories = [filled.clone(), blank, empty];
        let policy = SemanticMemoryRetrievalPolicy;
        assert!(policy.retrieve("   ", &memories, 8).is_empty());
        assert!(policy.retrieve("\t\n", &memories, 8).is_empty());
        assert!(policy.retrieve("postgres checkpoint", &memories, 0).is_empty());
        let hits = policy.retrieve("postgres checkpoint", &memories, 8);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, filled.id);
        assert!((1..=1_000).contains(&hits[0].score_milli));
    }
}

//! Event payload upcasting boundary.
//!
//! Historical events are immutable. When an event payload changes, a future
//! release adds an upcaster here and reads old records through it rather than
//! mutating the audit log in place.

use horizon_core::{CURRENT_EVENT_SCHEMA_VERSION, EventKind};
use serde_json::Value;

use crate::StoreError;

/// Decode an event payload stored with `schema_version` into today's domain
/// event type. The match is explicit so each new payload version requires a
/// deliberate upcaster rather than a silent deserialization assumption.
pub fn decode_event_payload(schema_version: u32, payload: &str) -> Result<EventKind, StoreError> {
    if schema_version == 0 || schema_version > CURRENT_EVENT_SCHEMA_VERSION {
        return Err(StoreError::UnsupportedEventSchema {
            found: schema_version,
            supported: CURRENT_EVENT_SCHEMA_VERSION,
        });
    }
    let value: Value = serde_json::from_str(payload)?;
    let current = upcast_payload(schema_version, value)?;
    Ok(serde_json::from_value(current)?)
}

fn upcast_payload(schema_version: u32, value: Value) -> Result<Value, StoreError> {
    match schema_version {
        // v1 had the original event vocabulary. Version 2 adds a new event
        // variant for state-decay assessments but does not alter any existing
        // payload, so its structural upcast is intentionally identity.
        1 => upcast_v1_to_v2(value),
        2 => Ok(value),
        _ => Err(StoreError::UnsupportedEventSchema {
            found: schema_version,
            supported: CURRENT_EVENT_SCHEMA_VERSION,
        }),
    }
}

fn upcast_v1_to_v2(value: Value) -> Result<Value, StoreError> {
    // Keep this explicit identity boundary: future schema changes chain from
    // here, while immutable v1 events remain byte-for-byte unmodified at rest.
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_a_v1_payload_after_the_v2_event_vocabulary_upgrade() {
        let event =
            decode_event_payload(1, r#"{"type":"note","data":{"message":"legacy"}}"#).unwrap();
        assert!(matches!(event, EventKind::Note { message } if message == "legacy"));
    }

    #[test]
    fn decodes_a_current_versioned_payload() {
        let event = decode_event_payload(
            CURRENT_EVENT_SCHEMA_VERSION,
            r#"{"type":"state_decay_assessed","data":{"assessment":{"policy_id":"test","policy_version":"v1","risk_score_milli":800,"threshold_milli":700,"signals":{"steps_since_anchor":4,"context_pressure":0.6,"subgoal_switches":1,"recent_failures":0,"recovered_session":false},"action":"inject_anchor","reason":"calibrated risk","metadata":null}}}"#,
        )
        .unwrap();
        assert!(matches!(event, EventKind::StateDecayAssessed { .. }));
    }

    #[test]
    fn rejects_a_future_unknown_event_schema() {
        let error = decode_event_payload(CURRENT_EVENT_SCHEMA_VERSION + 1, "{}").unwrap_err();
        assert!(matches!(error, StoreError::UnsupportedEventSchema { .. }));
    }
}

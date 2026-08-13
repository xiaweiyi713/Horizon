//! Event payload upcasting boundary.
//!
//! Historical events are immutable. When an event payload changes, a future
//! release adds an upcaster here and reads old records through it rather than
//! mutating the audit log in place.

use horizon_core::{CURRENT_EVENT_SCHEMA_VERSION, EventKind};
use serde_json::Value;

use crate::StoreError;

/// Decode an event payload stored with `schema_version` into today's domain
/// event type. Version 1 is the initial payload schema; the match is explicit
/// so adding version 2 requires a deliberate migration rather than a silent
/// deserialization assumption.
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
        // v1 is already compatible with the current event vocabulary. Future
        // versions should retain v1's branch and chain each next upcaster.
        1 => Ok(value),
        _ => Err(StoreError::UnsupportedEventSchema {
            found: schema_version,
            supported: CURRENT_EVENT_SCHEMA_VERSION,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_initial_versioned_payload() {
        let event =
            decode_event_payload(1, r#"{"type":"note","data":{"message":"legacy"}}"#).unwrap();
        assert!(matches!(event, EventKind::Note { message } if message == "legacy"));
    }

    #[test]
    fn rejects_a_future_unknown_event_schema() {
        let error = decode_event_payload(CURRENT_EVENT_SCHEMA_VERSION + 1, "{}").unwrap_err();
        assert!(matches!(error, StoreError::UnsupportedEventSchema { .. }));
    }
}

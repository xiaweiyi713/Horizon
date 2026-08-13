//! Horizon's event-to-tracing bridge.
//!
//! Event persistence remains the authoritative audit log. This crate emits
//! lightweight structured `tracing` records for operational visibility. The
//! optional `otel` feature additionally streams a span for each *already
//! committed* durable event through an OTLP/HTTP collector.

#[cfg(feature = "otel")]
mod otel;

use chrono::{DateTime, Utc};
use horizon_core::{EventRecord, RunId};
use serde::Serialize;

#[cfg(feature = "otel")]
pub use otel::{OtelError, OtelGuard, init_otlp_http};

/// Stable telemetry fields common to every durable runtime event.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RuntimeTraceEvent {
    pub run_id: RunId,
    pub sequence: u64,
    pub timestamp: DateTime<Utc>,
    pub event_type: &'static str,
}

impl From<&EventRecord> for RuntimeTraceEvent {
    fn from(record: &EventRecord) -> Self {
        Self {
            run_id: record.run_id,
            sequence: record.sequence,
            timestamp: record.timestamp,
            event_type: record.event.name(),
        }
    }
}

/// Emit a structured event after the durable store has successfully committed
/// it. This ordering prevents telemetry from claiming mutations that were never
/// persisted.
pub fn emit_persisted_event(record: &EventRecord) {
    let event = RuntimeTraceEvent::from(record);
    tracing::info!(
        target: "horizon.runtime",
        run_id = %event.run_id,
        sequence = event.sequence,
        event_type = event.event_type,
        timestamp = %event.timestamp,
        "durable runtime event"
    );
    #[cfg(feature = "otel")]
    otel::emit_durable_event(record);
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use horizon_core::{CURRENT_EVENT_SCHEMA_VERSION, EventId, EventKind};

    use super::*;

    #[test]
    fn maps_stable_event_fields() {
        let run_id = RunId::new();
        let record = EventRecord {
            id: EventId::new(),
            run_id,
            sequence: 7,
            timestamp: Utc::now(),
            schema_version: CURRENT_EVENT_SCHEMA_VERSION,
            event: EventKind::Note { message: "trace".into() },
            metadata: serde_json::Value::Null,
        };
        let trace = RuntimeTraceEvent::from(&record);
        assert_eq!(trace.run_id, run_id);
        assert_eq!(trace.sequence, 7);
        assert_eq!(trace.event_type, "note");
    }
}

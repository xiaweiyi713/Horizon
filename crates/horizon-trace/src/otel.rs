//! Optional OTLP/HTTP streaming for durable Horizon events.
//!
//! This module deliberately exports one short span per committed event instead
//! of making tracing output authoritative. If a collector is unavailable, the
//! event remains safely present in the EventStore and can be replayed later.

use horizon_core::EventRecord;
use opentelemetry::{
    KeyValue, global,
    trace::{Span, Tracer},
};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig};
use opentelemetry_sdk::{Resource, trace::SdkTracerProvider};
use thiserror::Error;

/// Failure while configuring or closing the optional OTLP exporter.
#[derive(Debug, Error)]
pub enum OtelError {
    #[error("OTLP endpoint cannot be empty")]
    EmptyEndpoint,
    #[error("OpenTelemetry service name cannot be empty")]
    EmptyServiceName,
    #[error("could not initialize OTLP HTTP trace exporter: {0}")]
    Exporter(String),
    #[error("could not shut down OTLP trace exporter: {0}")]
    Shutdown(String),
}

/// Owns Horizon's OTLP tracer provider until the application exits.
///
/// Call [`OtelGuard::shutdown`] at a controlled process boundary so the batch
/// processor flushes committed event spans. Dropping the guard also performs a
/// best-effort shutdown.
#[derive(Debug)]
pub struct OtelGuard {
    provider: SdkTracerProvider,
    shut_down: bool,
}

impl OtelGuard {
    /// Flush and stop this provider. Calling this after a successful shutdown
    /// is harmless, which makes cleanup paths straightforward.
    pub fn shutdown(&mut self) -> Result<(), OtelError> {
        if self.shut_down {
            return Ok(());
        }
        self.provider.shutdown().map_err(|error| OtelError::Shutdown(error.to_string()))?;
        self.shut_down = true;
        Ok(())
    }
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// Configure a batch OTLP/HTTP protobuf exporter and install it as the global
/// OpenTelemetry tracer provider used by [`super::emit_persisted_event`].
///
/// `endpoint` should be the collector base URL (for example
/// `http://127.0.0.1:4318`); the OTLP exporter appends `/v1/traces` for HTTP
/// protobuf transport.
pub fn init_otlp_http(
    endpoint: impl Into<String>,
    service_name: impl Into<String>,
) -> Result<OtelGuard, OtelError> {
    let endpoint = endpoint.into();
    let service_name = service_name.into();
    if endpoint.trim().is_empty() {
        return Err(OtelError::EmptyEndpoint);
    }
    if service_name.trim().is_empty() {
        return Err(OtelError::EmptyServiceName);
    }

    let exporter = SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(endpoint)
        .build()
        .map_err(|error| OtelError::Exporter(error.to_string()))?;
    let provider = SdkTracerProvider::builder()
        .with_resource(Resource::builder().with_service_name(service_name).build())
        .with_batch_exporter(exporter)
        .build();
    global::set_tracer_provider(provider.clone());
    Ok(OtelGuard { provider, shut_down: false })
}

/// Export one root span only after an event is durable.
pub(super) fn emit_durable_event(record: &EventRecord) {
    let tracer = global::tracer("horizon.runtime");
    let mut span = tracer.start("horizon.durable_event");
    span.set_attributes([
        KeyValue::new("horizon.run_id", record.run_id.to_string()),
        KeyValue::new("horizon.event.sequence", record.sequence.to_string()),
        KeyValue::new("horizon.event.schema_version", record.schema_version.to_string()),
        KeyValue::new("horizon.event.type", record.event.name()),
        KeyValue::new("horizon.event.timestamp", record.timestamp.to_rfc3339()),
    ]);
    span.end();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_blank_configuration_without_installing_a_global_provider() {
        assert!(matches!(init_otlp_http(" ", "horizon"), Err(OtelError::EmptyEndpoint)));
        assert!(matches!(
            init_otlp_http("http://127.0.0.1:4318", " "),
            Err(OtelError::EmptyServiceName)
        ));
    }
}

//! Durable, append-only storage for Horizon events and checkpoint snapshots.
//!
//! SQLite is intentionally the MVP backend: local-first, inspectable, and
//! service-free. The [`EventStore`] trait keeps a future PostgreSQL backend out
//! of the runtime's domain logic.

mod event_schema;
mod postgres;
mod snapshot;
mod sqlite;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use horizon_core::{EventRecord, NewEvent, RunId};
use serde::Serialize;
use thiserror::Error;

pub use postgres::PostgresEventStore;
pub use snapshot::{
    CURRENT_SNAPSHOT_SCHEMA_VERSION, EncodedSnapshot, SnapshotEncoding, SnapshotError,
    decode_snapshot, encode_snapshot,
};
pub use sqlite::SqliteEventStore;

#[derive(Clone, Debug)]
pub struct CheckpointRecord {
    pub run_id: RunId,
    pub sequence: u64,
    pub created_at: DateTime<Utc>,
    pub state_snapshot: Vec<u8>,
    /// Encoding and schema are stored per snapshot so old checkpoints remain
    /// readable after runtime upgrades.
    pub encoding: SnapshotEncoding,
    pub schema_version: u32,
    /// Optional fields are `None` for snapshots written by v0.1 before these
    /// integrity metadata columns existed.
    pub uncompressed_size: Option<u64>,
    pub checksum: Option<String>,
}

impl CheckpointRecord {
    #[must_use]
    pub fn from_encoded(
        run_id: RunId,
        sequence: u64,
        created_at: DateTime<Utc>,
        snapshot: EncodedSnapshot,
    ) -> Self {
        Self {
            run_id,
            sequence,
            created_at,
            state_snapshot: snapshot.bytes,
            encoding: snapshot.encoding,
            schema_version: snapshot.schema_version,
            uncompressed_size: Some(snapshot.uncompressed_size),
            checksum: Some(snapshot.checksum),
        }
    }

    #[must_use]
    pub fn legacy_json(
        run_id: RunId,
        sequence: u64,
        created_at: DateTime<Utc>,
        state_snapshot: Vec<u8>,
    ) -> Self {
        Self {
            run_id,
            sequence,
            created_at,
            state_snapshot,
            encoding: SnapshotEncoding::Json,
            schema_version: CURRENT_SNAPSHOT_SCHEMA_VERSION,
            uncompressed_size: None,
            checksum: None,
        }
    }
}

/// Result of safely discarding old checkpoint *snapshots*. Events are never
/// pruned: they remain Horizon's immutable audit log and replay source.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CheckpointCompaction {
    pub run_id: RunId,
    pub retained: u64,
    pub removed: u64,
    pub reclaimed_bytes: u64,
}

#[async_trait]
pub trait EventStore: Send + Sync + 'static {
    /// Atomically assigns the next per-run sequence and persists an event.
    async fn append(&self, event: NewEvent) -> Result<EventRecord, StoreError>;
    /// Atomically persists a causally ordered group of events. The runtime uses
    /// this for one logical command (for example `TaskFailed` together with its
    /// `FailureRemembered` record), so a crash cannot expose half a command.
    async fn append_many(&self, events: Vec<NewEvent>) -> Result<Vec<EventRecord>, StoreError>;
    /// Same as [`EventStore::append_many`], but commits only when the durable
    /// run sequence still equals `expected_sequence`. Runtime commands use this
    /// optimistic concurrency guard to reject stale cross-process decisions.
    async fn append_many_if_sequence(
        &self,
        events: Vec<NewEvent>,
        expected_sequence: u64,
    ) -> Result<Vec<EventRecord>, StoreError>;
    async fn load_events(
        &self,
        run_id: RunId,
        after_sequence: u64,
    ) -> Result<Vec<EventRecord>, StoreError>;
    async fn save_checkpoint(&self, checkpoint: CheckpointRecord) -> Result<(), StoreError>;
    async fn load_latest_checkpoint(
        &self,
        run_id: RunId,
    ) -> Result<Option<CheckpointRecord>, StoreError>;
    async fn list_run_ids(&self) -> Result<Vec<RunId>, StoreError>;
    async fn last_sequence(&self, run_id: RunId) -> Result<u64, StoreError>;
    /// Retain the newest `retain_latest` checkpoint snapshots for a run. This
    /// bounds snapshot storage without rewriting or deleting domain events.
    async fn compact_checkpoints(
        &self,
        run_id: RunId,
        retain_latest: usize,
    ) -> Result<CheckpointCompaction, StoreError>;
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("event batch cannot be empty")]
    EmptyBatch,
    #[error("conditional event batch must target one run")]
    MixedRunBatch,
    #[error(
        "concurrent modification of run {run_id}: expected sequence {expected}, found {actual}"
    )]
    ConcurrentModification { run_id: RunId, expected: u64, actual: u64 },
    #[error(
        "event schema version {found} is unsupported; this runtime supports through {supported}"
    )]
    UnsupportedEventSchema { found: u32, supported: u32 },
    #[error("checkpoint retention must be at least one")]
    InvalidCheckpointRetention,
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("could not serialize event: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
    #[error("invalid persisted {field}: {value}")]
    InvalidPersistedValue { field: &'static str, value: String },
    #[error("integer conversion failed for {field}: {value}")]
    IntegerConversion { field: &'static str, value: i64 },
}

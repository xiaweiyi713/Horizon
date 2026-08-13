//! Durable, append-only storage for Horizon events and checkpoint snapshots.
//!
//! SQLite is intentionally the MVP backend: local-first, inspectable, and
//! service-free. The [`EventStore`] trait keeps a future PostgreSQL backend out
//! of the runtime's domain logic.

mod sqlite;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use horizon_core::{EventRecord, NewEvent, RunId};
use thiserror::Error;

pub use sqlite::SqliteEventStore;

#[derive(Clone, Debug)]
pub struct CheckpointRecord {
    pub run_id: RunId,
    pub sequence: u64,
    pub created_at: DateTime<Utc>,
    pub state_snapshot: Vec<u8>,
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
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("could not serialize event: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("invalid persisted {field}: {value}")]
    InvalidPersistedValue { field: &'static str, value: String },
    #[error("integer conversion failed for {field}: {value}")]
    IntegerConversion { field: &'static str, value: i64 },
}

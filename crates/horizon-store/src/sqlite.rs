use std::{path::Path, str::FromStr, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use horizon_core::{CURRENT_EVENT_SCHEMA_VERSION, EventId, EventRecord, NewEvent, RunId};
use sqlx::{
    Row, SqlitePool,
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous},
};
use tokio::sync::Mutex;
use tracing::debug;

use crate::{
    CheckpointCompaction, CheckpointRecord, EventStore, SnapshotEncoding, StoreError,
    event_schema::decode_event_payload,
};

/// SQLite event store. The local append mutex prevents two runtime instances in
/// this process from racing sequence allocation; SQLite's transaction and unique
/// index still protect integrity for independent processes.
#[derive(Debug)]
pub struct SqliteEventStore {
    pool: SqlitePool,
    append_lock: Mutex<()>,
}

impl SqliteEventStore {
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        // SQLite's `:memory:` database is private to one connection. Horizon's
        // normal file pool has several connections, so route this common test/
        // embedding spelling through the one-connection constructor instead.
        if path.as_ref() == Path::new(":memory:") {
            return Self::in_memory().await;
        }
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .foreign_keys(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5))
            .synchronous(SqliteSynchronous::Normal);
        Self::connect_with(options).await
    }

    /// Connect using a SQLx SQLite URL, useful for `sqlite::memory:` in tests.
    pub async fn connect(database_url: &str) -> Result<Self, StoreError> {
        if database_url.starts_with("sqlite::memory:") {
            return Self::in_memory().await;
        }
        let options = SqliteConnectOptions::from_str(database_url)
            .map_err(|error| StoreError::InvalidPersistedValue {
                field: "database_url",
                value: error.to_string(),
            })?
            .create_if_missing(true)
            .foreign_keys(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5))
            .synchronous(SqliteSynchronous::Normal);
        Self::connect_with(options).await
    }

    pub async fn in_memory() -> Result<Self, StoreError> {
        let options = SqliteConnectOptions::from_str("sqlite::memory:")
            .expect("SQLite in-memory URL is valid")
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new().max_connections(1).connect_with(options).await?;
        let store = Self { pool, append_lock: Mutex::new(()) };
        store.migrate().await?;
        Ok(store)
    }

    async fn connect_with(options: SqliteConnectOptions) -> Result<Self, StoreError> {
        let pool = SqlitePoolOptions::new().max_connections(8).connect_with(options).await?;
        let store = Self { pool, append_lock: Mutex::new(()) };
        store.migrate().await?;
        Ok(store)
    }

    async fn migrate(&self) -> Result<(), StoreError> {
        // The schema is deliberately simple and inspectable using sqlite3.
        // Additive migrations preserve v0.1 database files in place.
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS horizon_events (
                id TEXT PRIMARY KEY NOT NULL,
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                metadata TEXT NOT NULL,
                schema_version INTEGER NOT NULL DEFAULT 1,
                UNIQUE(run_id, sequence)
            );
            "#,
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_horizon_events_run_sequence
            ON horizon_events(run_id, sequence);
            "#,
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS horizon_run_sequences (
                run_id TEXT PRIMARY KEY NOT NULL,
                next_sequence INTEGER NOT NULL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS horizon_checkpoints (
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                state_snapshot BLOB NOT NULL,
                snapshot_encoding TEXT NOT NULL DEFAULT 'json',
                snapshot_schema_version INTEGER NOT NULL DEFAULT 1,
                uncompressed_size INTEGER,
                checksum TEXT,
                PRIMARY KEY(run_id, sequence)
            );
            "#,
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_horizon_checkpoints_latest
            ON horizon_checkpoints(run_id, sequence DESC);
            "#,
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS horizon_schema_migrations (
                version INTEGER PRIMARY KEY NOT NULL,
                applied_at TEXT NOT NULL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;
        self.ensure_column("horizon_events", "schema_version", "INTEGER NOT NULL DEFAULT 1")
            .await?;
        self.ensure_column(
            "horizon_checkpoints",
            "snapshot_encoding",
            "TEXT NOT NULL DEFAULT 'json'",
        )
        .await?;
        self.ensure_column(
            "horizon_checkpoints",
            "snapshot_schema_version",
            "INTEGER NOT NULL DEFAULT 1",
        )
        .await?;
        self.ensure_column("horizon_checkpoints", "uncompressed_size", "INTEGER").await?;
        self.ensure_column("horizon_checkpoints", "checksum", "TEXT").await?;
        // Older databases already normally have sequence rows, but rebuild or
        // repair can leave them absent/stale. Reconcile from immutable events
        // before the first v0.2 append so we never reuse a sequence number.
        sqlx::query(
            r#"
            INSERT INTO horizon_run_sequences(run_id, next_sequence)
            SELECT run_id, MAX(sequence) + 1
            FROM horizon_events
            GROUP BY run_id
            ON CONFLICT(run_id) DO UPDATE SET
                next_sequence = MAX(horizon_run_sequences.next_sequence, excluded.next_sequence)
            "#,
        )
        .execute(&self.pool)
        .await?;
        let applied_at = Utc::now().to_rfc3339();
        for version in [1_i64, 2] {
            sqlx::query(
                "INSERT OR IGNORE INTO horizon_schema_migrations(version, applied_at) VALUES (?, ?)",
            )
            .bind(version)
            .bind(&applied_at)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    async fn ensure_column(
        &self,
        table: &'static str,
        column: &'static str,
        definition: &'static str,
    ) -> Result<(), StoreError> {
        let rows =
            sqlx::query(&format!("PRAGMA table_info({table})")).fetch_all(&self.pool).await?;
        let exists = rows
            .iter()
            .any(|row| row.try_get::<String, _>("name").is_ok_and(|name| name == column));
        if !exists {
            sqlx::query(&format!("ALTER TABLE {table} ADD COLUMN {column} {definition}"))
                .execute(&self.pool)
                .await?;
        }
        Ok(())
    }

    /// Database schema migration version applied by this SQLite store.
    pub async fn schema_version(&self) -> Result<u32, StoreError> {
        let version: i64 =
            sqlx::query_scalar("SELECT COALESCE(MAX(version), 0) FROM horizon_schema_migrations")
                .fetch_one(&self.pool)
                .await?;
        u32::try_from(version)
            .map_err(|_| StoreError::IntegerConversion { field: "schema.version", value: version })
    }

    async fn append_many_inner(
        &self,
        new_events: Vec<NewEvent>,
        expected_sequence: Option<u64>,
    ) -> Result<Vec<EventRecord>, StoreError> {
        if new_events.is_empty() {
            return Err(StoreError::EmptyBatch);
        }
        let conditional_run = if expected_sequence.is_some() {
            let run_id = new_events[0].run_id;
            if new_events.iter().any(|event| event.run_id != run_id) {
                return Err(StoreError::MixedRunBatch);
            }
            Some(run_id)
        } else {
            None
        };
        let _guard = self.append_lock.lock().await;
        let mut connection = self.pool.acquire().await?;
        // SQLite's BEGIN IMMEDIATE obtains the writer reservation before the
        // sequence check, making check-and-append serializable across separate
        // runtime processes sharing this database file.
        sqlx::query("BEGIN IMMEDIATE").execute(&mut *connection).await?;
        let operation = async {
            if let (Some(run_id), Some(expected_sequence)) = (conditional_run, expected_sequence) {
                let actual: i64 = sqlx::query_scalar(
                    "SELECT COALESCE(MAX(sequence), 0) FROM horizon_events WHERE run_id = ?",
                )
                .bind(run_id.to_string())
                .fetch_one(&mut *connection)
                .await?;
                let actual = u64::try_from(actual).map_err(|_| StoreError::IntegerConversion {
                    field: "event.sequence",
                    value: actual,
                })?;
                if actual != expected_sequence {
                    return Err(StoreError::ConcurrentModification {
                        run_id,
                        expected: expected_sequence,
                        actual,
                    });
                }
            }

            let mut records = Vec::with_capacity(new_events.len());
            for new_event in new_events {
                if new_event.schema_version != CURRENT_EVENT_SCHEMA_VERSION {
                    return Err(StoreError::UnsupportedEventSchema {
                        found: new_event.schema_version,
                        supported: CURRENT_EVENT_SCHEMA_VERSION,
                    });
                }
                let sequence: i64 = sqlx::query_scalar(
                    r#"
                    INSERT INTO horizon_run_sequences(run_id, next_sequence)
                    VALUES (?, 2)
                    ON CONFLICT(run_id) DO UPDATE SET next_sequence = next_sequence + 1
                    RETURNING next_sequence - 1
                    "#,
                )
                .bind(new_event.run_id.to_string())
                .fetch_one(&mut *connection)
                .await?;
                let sequence_u64 = u64::try_from(sequence).map_err(|_| {
                    StoreError::IntegerConversion { field: "event.sequence", value: sequence }
                })?;
                let record = EventRecord {
                    id: EventId::new(),
                    run_id: new_event.run_id,
                    sequence: sequence_u64,
                    timestamp: Utc::now(),
                    schema_version: new_event.schema_version,
                    event: new_event.event,
                    metadata: new_event.metadata,
                };
                let payload = serde_json::to_string(&record.event)?;
                let metadata = serde_json::to_string(&record.metadata)?;
                sqlx::query(
                    r#"
                    INSERT INTO horizon_events
                        (id, run_id, sequence, timestamp, event_type, payload, metadata, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    "#,
                )
                .bind(record.id.to_string())
                .bind(record.run_id.to_string())
                .bind(sequence)
                .bind(record.timestamp.to_rfc3339())
                .bind(record.event.name())
                .bind(payload)
                .bind(metadata)
                .bind(i64::from(record.schema_version))
                .execute(&mut *connection)
                .await?;
                records.push(record);
            }
            Ok(records)
        }
        .await;

        match operation {
            Ok(records) => {
                sqlx::query("COMMIT").execute(&mut *connection).await?;
                for record in &records {
                    debug!(run_id = %record.run_id, sequence = record.sequence, event = record.event.name(), "event appended");
                }
                Ok(records)
            }
            Err(error) => {
                // Rollback failure is secondary; preserve the primary error.
                let _ = sqlx::query("ROLLBACK").execute(&mut *connection).await;
                Err(error)
            }
        }
    }

    fn parse_event(row: sqlx::sqlite::SqliteRow) -> Result<EventRecord, StoreError> {
        let id: String = row.try_get("id")?;
        let run_id: String = row.try_get("run_id")?;
        let sequence: i64 = row.try_get("sequence")?;
        let timestamp: String = row.try_get("timestamp")?;
        let schema_version: i64 = row.try_get("schema_version")?;
        let payload: String = row.try_get("payload")?;
        let metadata: String = row.try_get("metadata")?;
        let id = id
            .parse()
            .map_err(|_| StoreError::InvalidPersistedValue { field: "event.id", value: id })?;
        let run_id = run_id.parse().map_err(|_| StoreError::InvalidPersistedValue {
            field: "event.run_id",
            value: run_id,
        })?;
        let sequence = u64::try_from(sequence).map_err(|_| StoreError::IntegerConversion {
            field: "event.sequence",
            value: sequence,
        })?;
        let timestamp = DateTime::parse_from_rfc3339(&timestamp)
            .map_err(|_| StoreError::InvalidPersistedValue {
                field: "event.timestamp",
                value: timestamp,
            })?
            .with_timezone(&Utc);
        let schema_version = u32::try_from(schema_version).map_err(|_| {
            StoreError::IntegerConversion { field: "event.schema_version", value: schema_version }
        })?;
        let event = decode_event_payload(schema_version, &payload)?;
        let metadata = serde_json::from_str(&metadata)?;
        Ok(EventRecord { id, run_id, sequence, timestamp, schema_version, event, metadata })
    }
}

#[async_trait]
impl EventStore for SqliteEventStore {
    async fn append(&self, new_event: NewEvent) -> Result<EventRecord, StoreError> {
        let mut records = self.append_many(vec![new_event]).await?;
        // A one-event vector always yields exactly one record.
        Ok(records.pop().expect("append_many returned one record"))
    }

    async fn append_many(&self, new_events: Vec<NewEvent>) -> Result<Vec<EventRecord>, StoreError> {
        self.append_many_inner(new_events, None).await
    }

    async fn append_many_if_sequence(
        &self,
        new_events: Vec<NewEvent>,
        expected_sequence: u64,
    ) -> Result<Vec<EventRecord>, StoreError> {
        self.append_many_inner(new_events, Some(expected_sequence)).await
    }

    async fn load_events(
        &self,
        run_id: RunId,
        after_sequence: u64,
    ) -> Result<Vec<EventRecord>, StoreError> {
        let after_sequence = i64::try_from(after_sequence).map_err(|_| {
            StoreError::IntegerConversion { field: "after_sequence", value: i64::MAX }
        })?;
        let rows = sqlx::query(
            r#"
            SELECT id, run_id, sequence, timestamp, schema_version, payload, metadata
            FROM horizon_events
            WHERE run_id = ? AND sequence > ?
            ORDER BY sequence ASC
            "#,
        )
        .bind(run_id.to_string())
        .bind(after_sequence)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(Self::parse_event).collect()
    }

    async fn save_checkpoint(&self, checkpoint: CheckpointRecord) -> Result<(), StoreError> {
        let sequence = i64::try_from(checkpoint.sequence).map_err(|_| {
            StoreError::IntegerConversion { field: "checkpoint.sequence", value: i64::MAX }
        })?;
        sqlx::query(
            r#"
            INSERT INTO horizon_checkpoints(
                run_id, sequence, created_at, state_snapshot, snapshot_encoding,
                snapshot_schema_version, uncompressed_size, checksum
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, sequence) DO UPDATE SET
                created_at = excluded.created_at,
                state_snapshot = excluded.state_snapshot,
                snapshot_encoding = excluded.snapshot_encoding,
                snapshot_schema_version = excluded.snapshot_schema_version,
                uncompressed_size = excluded.uncompressed_size,
                checksum = excluded.checksum
            "#,
        )
        .bind(checkpoint.run_id.to_string())
        .bind(sequence)
        .bind(checkpoint.created_at.to_rfc3339())
        .bind(checkpoint.state_snapshot)
        .bind(checkpoint.encoding.as_str())
        .bind(i64::from(checkpoint.schema_version))
        .bind(checkpoint.uncompressed_size.map(i64::try_from).transpose().map_err(|_| {
            StoreError::IntegerConversion { field: "checkpoint.uncompressed_size", value: i64::MAX }
        })?)
        .bind(checkpoint.checksum)
        .execute(&self.pool)
        .await?;
        debug!(run_id = %checkpoint.run_id, sequence = checkpoint.sequence, "checkpoint persisted");
        Ok(())
    }

    async fn load_latest_checkpoint(
        &self,
        run_id: RunId,
    ) -> Result<Option<CheckpointRecord>, StoreError> {
        let row = sqlx::query(
            r#"
            SELECT run_id, sequence, created_at, state_snapshot, snapshot_encoding,
                snapshot_schema_version, uncompressed_size, checksum
            FROM horizon_checkpoints
            WHERE run_id = ?
            ORDER BY sequence DESC
            LIMIT 1
            "#,
        )
        .bind(run_id.to_string())
        .fetch_optional(&self.pool)
        .await?;
        row.map(|row| {
            let run_id: String = row.try_get("run_id")?;
            let sequence: i64 = row.try_get("sequence")?;
            let created_at: String = row.try_get("created_at")?;
            let state_snapshot: Vec<u8> = row.try_get("state_snapshot")?;
            let encoding: String = row.try_get("snapshot_encoding")?;
            let schema_version: i64 = row.try_get("snapshot_schema_version")?;
            let uncompressed_size: Option<i64> = row.try_get("uncompressed_size")?;
            let checksum: Option<String> = row.try_get("checksum")?;
            let run_id = run_id.parse().map_err(|_| StoreError::InvalidPersistedValue {
                field: "checkpoint.run_id",
                value: run_id,
            })?;
            let sequence = u64::try_from(sequence).map_err(|_| StoreError::IntegerConversion {
                field: "checkpoint.sequence",
                value: sequence,
            })?;
            let created_at = DateTime::parse_from_rfc3339(&created_at)
                .map_err(|_| StoreError::InvalidPersistedValue {
                    field: "checkpoint.created_at",
                    value: created_at,
                })?
                .with_timezone(&Utc);
            let encoding = SnapshotEncoding::parse_persisted(&encoding)?;
            let schema_version =
                u32::try_from(schema_version).map_err(|_| StoreError::IntegerConversion {
                    field: "checkpoint.schema_version",
                    value: schema_version,
                })?;
            let uncompressed_size = uncompressed_size
                .map(|size| {
                    u64::try_from(size).map_err(|_| StoreError::IntegerConversion {
                        field: "checkpoint.uncompressed_size",
                        value: size,
                    })
                })
                .transpose()?;
            Ok(CheckpointRecord {
                run_id,
                sequence,
                created_at,
                state_snapshot,
                encoding,
                schema_version,
                uncompressed_size,
                checksum,
            })
        })
        .transpose()
    }

    async fn list_run_ids(&self) -> Result<Vec<RunId>, StoreError> {
        let rows = sqlx::query(
            "SELECT run_id FROM horizon_events GROUP BY run_id ORDER BY MIN(timestamp)",
        )
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                let value: String = row.try_get("run_id")?;
                value
                    .parse()
                    .map_err(|_| StoreError::InvalidPersistedValue { field: "event.run_id", value })
            })
            .collect()
    }

    async fn last_sequence(&self, run_id: RunId) -> Result<u64, StoreError> {
        let sequence: i64 = sqlx::query_scalar(
            "SELECT COALESCE(MAX(sequence), 0) FROM horizon_events WHERE run_id = ?",
        )
        .bind(run_id.to_string())
        .fetch_one(&self.pool)
        .await?;
        u64::try_from(sequence)
            .map_err(|_| StoreError::IntegerConversion { field: "event.sequence", value: sequence })
    }

    async fn compact_checkpoints(
        &self,
        run_id: RunId,
        retain_latest: usize,
    ) -> Result<CheckpointCompaction, StoreError> {
        if retain_latest == 0 {
            return Err(StoreError::InvalidCheckpointRetention);
        }
        let _guard = self.append_lock.lock().await;
        let mut transaction = self.pool.begin().await?;
        let rows = sqlx::query(
            r#"
            SELECT sequence, length(state_snapshot) AS byte_len
            FROM horizon_checkpoints
            WHERE run_id = ?
            ORDER BY sequence DESC
            "#,
        )
        .bind(run_id.to_string())
        .fetch_all(&mut *transaction)
        .await?;
        let mut removed = 0_u64;
        let mut reclaimed_bytes = 0_u64;
        for row in rows.iter().skip(retain_latest) {
            let sequence: i64 = row.try_get("sequence")?;
            let byte_len: i64 = row.try_get("byte_len")?;
            sqlx::query("DELETE FROM horizon_checkpoints WHERE run_id = ? AND sequence = ?")
                .bind(run_id.to_string())
                .bind(sequence)
                .execute(&mut *transaction)
                .await?;
            removed += 1;
            reclaimed_bytes += u64::try_from(byte_len).map_err(|_| {
                StoreError::IntegerConversion { field: "checkpoint.byte_len", value: byte_len }
            })?;
        }
        transaction.commit().await?;
        let retained = u64::try_from(rows.len().min(retain_latest)).map_err(|_| {
            StoreError::IntegerConversion { field: "checkpoint.retained", value: i64::MAX }
        })?;
        Ok(CheckpointCompaction { run_id, retained, removed, reclaimed_bytes })
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use horizon_core::{EventKind, NewEvent, RunId};
    use sqlx::{
        SqlitePool,
        sqlite::{SqliteConnectOptions, SqlitePoolOptions},
    };
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn appends_ordered_events_and_loads_them() {
        let store = SqliteEventStore::in_memory().await.unwrap();
        let run_id = RunId::new();
        let first = store
            .append(NewEvent::new(run_id, EventKind::RunCreated { goal: "test".into() }))
            .await
            .unwrap();
        let second = store
            .append(NewEvent::new(run_id, EventKind::Note { message: "two".into() }))
            .await
            .unwrap();
        assert_eq!((first.sequence, second.sequence), (1, 2));
        assert_eq!(store.load_events(run_id, 1).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn memory_path_and_url_keep_schema_on_one_connection() {
        for store in [
            SqliteEventStore::open(":memory:").await.unwrap(),
            SqliteEventStore::connect("sqlite::memory:").await.unwrap(),
        ] {
            let run_id = RunId::new();
            store
                .append(NewEvent::new(run_id, EventKind::RunCreated { goal: "memory".into() }))
                .await
                .unwrap();
            assert_eq!(store.load_events(run_id, 0).await.unwrap().len(), 1);
        }
    }

    #[tokio::test]
    async fn appends_a_logical_batch_with_contiguous_sequences() {
        let store = SqliteEventStore::in_memory().await.unwrap();
        let run_id = RunId::new();
        let records = store
            .append_many(vec![
                NewEvent::new(run_id, EventKind::RunCreated { goal: "batch".into() }),
                NewEvent::new(run_id, EventKind::Note { message: "failure memory follows".into() }),
            ])
            .await
            .unwrap();
        assert_eq!(records.iter().map(|record| record.sequence).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(store.load_events(run_id, 0).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn rejects_a_stale_conditional_batch_without_appending_it() {
        let store = SqliteEventStore::in_memory().await.unwrap();
        let run_id = RunId::new();
        store
            .append(NewEvent::new(run_id, EventKind::RunCreated { goal: "first".into() }))
            .await
            .unwrap();
        let error = store
            .append_many_if_sequence(
                vec![NewEvent::new(run_id, EventKind::Note { message: "stale".into() })],
                0,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, StoreError::ConcurrentModification { expected: 0, actual: 1, .. }));
        assert_eq!(store.load_events(run_id, 0).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn compacts_old_snapshots_without_deleting_events() {
        let store = SqliteEventStore::in_memory().await.unwrap();
        let run_id = RunId::new();
        store
            .append(NewEvent::new(run_id, EventKind::RunCreated { goal: "retain events".into() }))
            .await
            .unwrap();
        for sequence in 1..=3 {
            store
                .save_checkpoint(CheckpointRecord::legacy_json(
                    run_id,
                    sequence,
                    Utc::now(),
                    format!("{{\"sequence\":{sequence}}}").into_bytes(),
                ))
                .await
                .unwrap();
        }
        let result = store.compact_checkpoints(run_id, 2).await.unwrap();
        assert_eq!(result.retained, 2);
        assert_eq!(result.removed, 1);
        assert!(result.reclaimed_bytes > 0);
        assert_eq!(store.load_latest_checkpoint(run_id).await.unwrap().unwrap().sequence, 3);
        assert_eq!(store.load_events(run_id, 0).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn migrates_a_v0_1_sqlite_file_additively() {
        let directory = tempdir().unwrap();
        let database = directory.path().join("legacy.db");
        let options = SqliteConnectOptions::new().filename(&database).create_if_missing(true);
        let pool: SqlitePool =
            SqlitePoolOptions::new().max_connections(1).connect_with(options).await.unwrap();
        sqlx::query(
            r#"
            CREATE TABLE horizon_events (
                id TEXT PRIMARY KEY NOT NULL,
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                metadata TEXT NOT NULL,
                UNIQUE(run_id, sequence)
            )
            "#,
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE TABLE horizon_run_sequences (run_id TEXT PRIMARY KEY NOT NULL, next_sequence INTEGER NOT NULL)",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            r#"
            CREATE TABLE horizon_checkpoints (
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                state_snapshot BLOB NOT NULL,
                PRIMARY KEY(run_id, sequence)
            )
            "#,
        )
        .execute(&pool)
        .await
        .unwrap();
        let run_id = RunId::new();
        let event = EventKind::RunCreated { goal: "legacy event".into() };
        sqlx::query(
            "INSERT INTO horizon_events(id, run_id, sequence, timestamp, event_type, payload, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(horizon_core::EventId::new().to_string())
        .bind(run_id.to_string())
        .bind(1_i64)
        .bind(Utc::now().to_rfc3339())
        .bind(event.name())
        .bind(serde_json::to_string(&event).unwrap())
        .bind("null")
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO horizon_checkpoints(run_id, sequence, created_at, state_snapshot) VALUES (?, ?, ?, ?)",
        )
        .bind(run_id.to_string())
        .bind(1_i64)
        .bind(Utc::now().to_rfc3339())
        .bind(b"{}".to_vec())
        .execute(&pool)
        .await
        .unwrap();
        pool.close().await;

        let store = SqliteEventStore::open(&database).await.unwrap();
        assert_eq!(store.schema_version().await.unwrap(), 2);
        let events = store.load_events(run_id, 0).await.unwrap();
        // The migration preserves the original payload version; loading passes
        // it through the explicit event-schema upcast boundary.
        assert_eq!(events[0].schema_version, 1);
        assert!(matches!(events[0].event, EventKind::RunCreated { .. }));
        let checkpoint = store.load_latest_checkpoint(run_id).await.unwrap().unwrap();
        assert_eq!(checkpoint.encoding, SnapshotEncoding::Json);
        assert_eq!(checkpoint.uncompressed_size, None);
        assert_eq!(checkpoint.checksum, None);
        let appended = store
            .append(NewEvent::new(run_id, EventKind::Note { message: "after migration".into() }))
            .await
            .unwrap();
        assert_eq!(appended.sequence, 2);
    }
}

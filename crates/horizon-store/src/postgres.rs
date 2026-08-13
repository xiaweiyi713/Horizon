//! PostgreSQL implementation of Horizon's durable event store.
//!
//! PostgreSQL is deliberately a drop-in [`EventStore`](crate::EventStore), not
//! a second runtime. Per-run sequence rows are locked inside a transaction so
//! the same optimistic command semantics used by SQLite hold across processes
//! and hosts.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use horizon_core::{CURRENT_EVENT_SCHEMA_VERSION, EventId, EventRecord, NewEvent, RunId};
use sqlx::{
    PgPool, Row,
    postgres::{PgPoolOptions, PgRow},
};
use tracing::debug;

use crate::{
    CheckpointCompaction, CheckpointRecord, EventStore, SnapshotEncoding, StoreError,
    event_schema::decode_event_payload,
};

/// PostgreSQL-backed event store for multi-process or multi-host deployments.
#[derive(Debug)]
pub struct PostgresEventStore {
    pool: PgPool,
}

impl PostgresEventStore {
    /// Connect and apply Horizon's additive storage migrations.
    pub async fn connect(database_url: &str) -> Result<Self, StoreError> {
        let pool = PgPoolOptions::new().max_connections(16).connect(database_url).await?;
        let store = Self { pool };
        store.migrate().await?;
        Ok(store)
    }

    /// Exposes the connection pool for health checks or deployment-owned
    /// observability. Domain code should use [`EventStore`] instead.
    #[must_use]
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    async fn migrate(&self) -> Result<(), StoreError> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS horizon_events (
                id TEXT PRIMARY KEY NOT NULL,
                run_id TEXT NOT NULL,
                sequence BIGINT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                event_type TEXT NOT NULL,
                payload JSONB NOT NULL,
                metadata JSONB NOT NULL,
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
                next_sequence BIGINT NOT NULL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS horizon_checkpoints (
                run_id TEXT NOT NULL,
                sequence BIGINT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                state_snapshot BYTEA NOT NULL,
                snapshot_encoding TEXT NOT NULL DEFAULT 'json',
                snapshot_schema_version INTEGER NOT NULL DEFAULT 1,
                uncompressed_size BIGINT,
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
                applied_at TIMESTAMPTZ NOT NULL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;
        // A database created by an earlier Horizon version only needs additive
        // columns; event payloads stay immutable and are upcast while loading.
        for statement in [
            "ALTER TABLE horizon_events ADD COLUMN IF NOT EXISTS schema_version INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE horizon_checkpoints ADD COLUMN IF NOT EXISTS snapshot_encoding TEXT NOT NULL DEFAULT 'json'",
            "ALTER TABLE horizon_checkpoints ADD COLUMN IF NOT EXISTS snapshot_schema_version INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE horizon_checkpoints ADD COLUMN IF NOT EXISTS uncompressed_size BIGINT",
            "ALTER TABLE horizon_checkpoints ADD COLUMN IF NOT EXISTS checksum TEXT",
        ] {
            sqlx::query(statement).execute(&self.pool).await?;
        }
        // PostgreSQL is new in v0.2, but retain the same repair-safe behavior
        // as SQLite if events were imported before their sequence rows.
        sqlx::query(
            r#"
            INSERT INTO horizon_run_sequences(run_id, next_sequence)
            SELECT run_id, MAX(sequence) + 1
            FROM horizon_events
            GROUP BY run_id
            ON CONFLICT(run_id) DO UPDATE SET
                next_sequence = GREATEST(
                    horizon_run_sequences.next_sequence,
                    EXCLUDED.next_sequence
                )
            "#,
        )
        .execute(&self.pool)
        .await?;
        for version in [1_i32, 2] {
            sqlx::query(
                "INSERT INTO horizon_schema_migrations(version, applied_at) VALUES ($1, $2) ON CONFLICT (version) DO NOTHING",
            )
            .bind(version)
            .bind(Utc::now())
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    /// Database schema migration version applied by this PostgreSQL store.
    pub async fn schema_version(&self) -> Result<u32, StoreError> {
        let version: i32 =
            sqlx::query_scalar("SELECT COALESCE(MAX(version), 0) FROM horizon_schema_migrations")
                .fetch_one(&self.pool)
                .await?;
        u32::try_from(version).map_err(|_| StoreError::IntegerConversion {
            field: "schema.version",
            value: i64::from(version),
        })
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
        let mut transaction = self.pool.begin().await?;
        if let (Some(run_id), Some(expected)) = (conditional_run, expected_sequence) {
            let actual = Self::lock_and_read_sequence(&mut transaction, run_id).await?;
            if actual != expected {
                return Err(StoreError::ConcurrentModification { run_id, expected, actual });
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
            let sequence = Self::allocate_sequence(&mut transaction, new_event.run_id).await?;
            let record = EventRecord {
                id: EventId::new(),
                run_id: new_event.run_id,
                sequence,
                timestamp: Utc::now(),
                schema_version: new_event.schema_version,
                event: new_event.event,
                metadata: new_event.metadata,
            };
            let payload = serde_json::to_string(&record.event)?;
            let metadata = serde_json::to_string(&record.metadata)?;
            let sequence_i64 = i64::try_from(sequence).map_err(|_| {
                StoreError::IntegerConversion { field: "event.sequence", value: i64::MAX }
            })?;
            sqlx::query(
                r#"
                INSERT INTO horizon_events(
                    id, run_id, sequence, timestamp, event_type, payload, metadata, schema_version
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8)
                "#,
            )
            .bind(record.id.to_string())
            .bind(record.run_id.to_string())
            .bind(sequence_i64)
            .bind(record.timestamp)
            .bind(record.event.name())
            .bind(payload)
            .bind(metadata)
            .bind(i32::try_from(record.schema_version).map_err(|_| {
                StoreError::IntegerConversion {
                    field: "event.schema_version",
                    value: i64::from(record.schema_version),
                }
            })?)
            .execute(&mut *transaction)
            .await?;
            records.push(record);
        }
        transaction.commit().await?;
        for record in &records {
            debug!(run_id = %record.run_id, sequence = record.sequence, event = record.event.name(), "event appended");
        }
        Ok(records)
    }

    async fn lock_and_read_sequence(
        transaction: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        run_id: RunId,
    ) -> Result<u64, StoreError> {
        sqlx::query(
            "INSERT INTO horizon_run_sequences(run_id, next_sequence) VALUES ($1, 1) ON CONFLICT (run_id) DO NOTHING",
        )
        .bind(run_id.to_string())
        .execute(&mut **transaction)
        .await?;
        let next_sequence: i64 = sqlx::query_scalar(
            "SELECT next_sequence FROM horizon_run_sequences WHERE run_id = $1 FOR UPDATE",
        )
        .bind(run_id.to_string())
        .fetch_one(&mut **transaction)
        .await?;
        let next_sequence = u64::try_from(next_sequence).map_err(|_| {
            StoreError::IntegerConversion { field: "event.next_sequence", value: next_sequence }
        })?;
        Ok(next_sequence.saturating_sub(1))
    }

    async fn allocate_sequence(
        transaction: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        run_id: RunId,
    ) -> Result<u64, StoreError> {
        sqlx::query(
            "INSERT INTO horizon_run_sequences(run_id, next_sequence) VALUES ($1, 1) ON CONFLICT (run_id) DO NOTHING",
        )
        .bind(run_id.to_string())
        .execute(&mut **transaction)
        .await?;
        let sequence: i64 = sqlx::query_scalar(
            "UPDATE horizon_run_sequences SET next_sequence = next_sequence + 1 WHERE run_id = $1 RETURNING next_sequence - 1",
        )
        .bind(run_id.to_string())
        .fetch_one(&mut **transaction)
        .await?;
        u64::try_from(sequence)
            .map_err(|_| StoreError::IntegerConversion { field: "event.sequence", value: sequence })
    }

    fn parse_event(row: PgRow) -> Result<EventRecord, StoreError> {
        let id: String = row.try_get("id")?;
        let run_id: String = row.try_get("run_id")?;
        let sequence: i64 = row.try_get("sequence")?;
        let timestamp: DateTime<Utc> = row.try_get("timestamp")?;
        let schema_version: i32 = row.try_get("schema_version")?;
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
        let schema_version =
            u32::try_from(schema_version).map_err(|_| StoreError::IntegerConversion {
                field: "event.schema_version",
                value: i64::from(schema_version),
            })?;
        let event = decode_event_payload(schema_version, &payload)?;
        let metadata = serde_json::from_str(&metadata)?;
        Ok(EventRecord { id, run_id, sequence, timestamp, schema_version, event, metadata })
    }

    fn parse_checkpoint(row: PgRow) -> Result<CheckpointRecord, StoreError> {
        let run_id: String = row.try_get("run_id")?;
        let sequence: i64 = row.try_get("sequence")?;
        let created_at: DateTime<Utc> = row.try_get("created_at")?;
        let state_snapshot: Vec<u8> = row.try_get("state_snapshot")?;
        let encoding: String = row.try_get("snapshot_encoding")?;
        let schema_version: i32 = row.try_get("snapshot_schema_version")?;
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
        let encoding = SnapshotEncoding::parse_persisted(&encoding)?;
        let schema_version =
            u32::try_from(schema_version).map_err(|_| StoreError::IntegerConversion {
                field: "checkpoint.schema_version",
                value: i64::from(schema_version),
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
    }
}

#[async_trait]
impl EventStore for PostgresEventStore {
    async fn append(&self, event: NewEvent) -> Result<EventRecord, StoreError> {
        let mut records = self.append_many(vec![event]).await?;
        Ok(records.pop().expect("append_many returned one record"))
    }

    async fn append_many(&self, events: Vec<NewEvent>) -> Result<Vec<EventRecord>, StoreError> {
        self.append_many_inner(events, None).await
    }

    async fn append_many_if_sequence(
        &self,
        events: Vec<NewEvent>,
        expected_sequence: u64,
    ) -> Result<Vec<EventRecord>, StoreError> {
        self.append_many_inner(events, Some(expected_sequence)).await
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
            SELECT id, run_id, sequence, timestamp, schema_version,
                payload::text AS payload, metadata::text AS metadata
            FROM horizon_events
            WHERE run_id = $1 AND sequence > $2
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
        let schema_version = i32::try_from(checkpoint.schema_version).map_err(|_| {
            StoreError::IntegerConversion {
                field: "checkpoint.schema_version",
                value: i64::from(checkpoint.schema_version),
            }
        })?;
        let uncompressed_size = checkpoint
            .uncompressed_size
            .map(|size| {
                i64::try_from(size).map_err(|_| StoreError::IntegerConversion {
                    field: "checkpoint.uncompressed_size",
                    value: i64::MAX,
                })
            })
            .transpose()?;
        sqlx::query(
            r#"
            INSERT INTO horizon_checkpoints(
                run_id, sequence, created_at, state_snapshot, snapshot_encoding,
                snapshot_schema_version, uncompressed_size, checksum
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (run_id, sequence) DO UPDATE SET
                created_at = EXCLUDED.created_at,
                state_snapshot = EXCLUDED.state_snapshot,
                snapshot_encoding = EXCLUDED.snapshot_encoding,
                snapshot_schema_version = EXCLUDED.snapshot_schema_version,
                uncompressed_size = EXCLUDED.uncompressed_size,
                checksum = EXCLUDED.checksum
            "#,
        )
        .bind(checkpoint.run_id.to_string())
        .bind(sequence)
        .bind(checkpoint.created_at)
        .bind(checkpoint.state_snapshot)
        .bind(checkpoint.encoding.as_str())
        .bind(schema_version)
        .bind(uncompressed_size)
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
            WHERE run_id = $1
            ORDER BY sequence DESC
            LIMIT 1
            "#,
        )
        .bind(run_id.to_string())
        .fetch_optional(&self.pool)
        .await?;
        row.map(Self::parse_checkpoint).transpose()
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
            "SELECT COALESCE(MAX(sequence), 0) FROM horizon_events WHERE run_id = $1",
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
        let mut transaction = self.pool.begin().await?;
        let rows = sqlx::query(
            r#"
            SELECT sequence, octet_length(state_snapshot) AS byte_len
            FROM horizon_checkpoints
            WHERE run_id = $1
            ORDER BY sequence DESC
            FOR UPDATE
            "#,
        )
        .bind(run_id.to_string())
        .fetch_all(&mut *transaction)
        .await?;
        let mut removed = 0_u64;
        let mut reclaimed_bytes = 0_u64;
        for row in rows.iter().skip(retain_latest) {
            let sequence: i64 = row.try_get("sequence")?;
            let byte_len: i32 = row.try_get("byte_len")?;
            sqlx::query("DELETE FROM horizon_checkpoints WHERE run_id = $1 AND sequence = $2")
                .bind(run_id.to_string())
                .bind(sequence)
                .execute(&mut *transaction)
                .await?;
            removed += 1;
            reclaimed_bytes +=
                u64::try_from(byte_len).map_err(|_| StoreError::IntegerConversion {
                    field: "checkpoint.byte_len",
                    value: i64::from(byte_len),
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
    use std::env;

    use horizon_core::{
        EventKind, InterventionAction, InterventionAssessment, NewEvent, RunId, StateDecaySignals,
    };

    use super::*;

    /// This test is intentionally opt-in so a normal local `cargo test` has no
    /// service dependency. CI and contributors can set HORIZON_POSTGRES_URL to
    /// validate the exact same EventStore contract against a real database.
    #[tokio::test]
    async fn round_trips_against_configured_postgres() {
        let Ok(url) = env::var("HORIZON_POSTGRES_URL") else {
            return;
        };
        let store = PostgresEventStore::connect(&url).await.unwrap();
        let run_id = RunId::new();
        let appended = store
            .append_many_if_sequence(
                vec![
                    NewEvent::new(run_id, EventKind::RunCreated { goal: "postgres".into() }),
                    NewEvent::new(
                        run_id,
                        EventKind::StateDecayAssessed {
                            assessment: InterventionAssessment {
                                policy_id: "postgres-contract".into(),
                                policy_version: Some("v1".into()),
                                risk_score_milli: 700,
                                threshold_milli: 600,
                                signals: StateDecaySignals {
                                    context_pressure: 0.7,
                                    ..Default::default()
                                },
                                action: InterventionAction::InjectAnchor,
                                reason: "exercise JSONB event round trip".into(),
                                metadata: serde_json::json!({"source": "integration-test"}),
                            },
                        },
                    ),
                ],
                0,
            )
            .await
            .unwrap();
        assert_eq!(appended.iter().map(|event| event.sequence).collect::<Vec<_>>(), [1, 2]);
        let events = store.load_events(run_id, 0).await.unwrap();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[1].event, EventKind::StateDecayAssessed { .. }));
        assert!(store.schema_version().await.unwrap() >= 2);
    }
}

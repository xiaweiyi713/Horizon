//! Versioned, integrity-checked checkpoint payloads.
//!
//! Checkpoints are an optimization over the immutable event log, so corruption
//! must be detected before a projection is trusted. The codec is deliberately
//! stored with each checkpoint: upgrading a runtime never requires rewriting an
//! existing snapshot just to change its compression setting.

use std::io::Cursor;

use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::CheckpointRecord;

/// Current schema for a serialized projection snapshot.
pub const CURRENT_SNAPSHOT_SCHEMA_VERSION: u32 = 1;

/// Encoding stored alongside a checkpoint payload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotEncoding {
    /// Uncompressed JSON retained for legacy compatibility and debugging.
    #[default]
    Json,
    /// JSON compressed with Zstandard. This is the v0.2 default.
    ZstdJson,
}

impl SnapshotEncoding {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::ZstdJson => "zstd_json",
        }
    }

    pub fn parse_persisted(value: &str) -> Result<Self, SnapshotError> {
        match value {
            "json" => Ok(Self::Json),
            "zstd_json" => Ok(Self::ZstdJson),
            _ => Err(SnapshotError::UnknownEncoding(value.to_owned())),
        }
    }
}

/// Encoded checkpoint material ready for durable storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedSnapshot {
    pub bytes: Vec<u8>,
    pub encoding: SnapshotEncoding,
    pub schema_version: u32,
    pub uncompressed_size: u64,
    pub checksum: String,
}

/// Serialize and optionally compress a projection snapshot.
pub fn encode_snapshot<T: Serialize>(
    value: &T,
    encoding: SnapshotEncoding,
) -> Result<EncodedSnapshot, SnapshotError> {
    let uncompressed = serde_json::to_vec(value)?;
    let uncompressed_size = u64::try_from(uncompressed.len())
        .map_err(|_| SnapshotError::PayloadTooLarge(uncompressed.len()))?;
    let checksum = blake3::hash(&uncompressed).to_hex().to_string();
    let bytes = match encoding {
        SnapshotEncoding::Json => uncompressed,
        SnapshotEncoding::ZstdJson => zstd::stream::encode_all(Cursor::new(uncompressed), 3)?,
    };
    Ok(EncodedSnapshot {
        bytes,
        encoding,
        schema_version: CURRENT_SNAPSHOT_SCHEMA_VERSION,
        uncompressed_size,
        checksum,
    })
}

/// Decode and validate a persisted checkpoint.
pub fn decode_snapshot<T: DeserializeOwned>(
    checkpoint: &CheckpointRecord,
) -> Result<T, SnapshotError> {
    if checkpoint.schema_version == 0 || checkpoint.schema_version > CURRENT_SNAPSHOT_SCHEMA_VERSION
    {
        return Err(SnapshotError::UnsupportedSchema {
            found: checkpoint.schema_version,
            supported: CURRENT_SNAPSHOT_SCHEMA_VERSION,
        });
    }
    let uncompressed = match checkpoint.encoding {
        SnapshotEncoding::Json => checkpoint.state_snapshot.clone(),
        SnapshotEncoding::ZstdJson => {
            zstd::stream::decode_all(Cursor::new(&checkpoint.state_snapshot))?
        }
    };
    if let Some(expected) = checkpoint.uncompressed_size {
        let actual = u64::try_from(uncompressed.len())
            .map_err(|_| SnapshotError::PayloadTooLarge(uncompressed.len()))?;
        if expected != actual {
            return Err(SnapshotError::SizeMismatch { expected, actual });
        }
    }
    if let Some(expected) = &checkpoint.checksum {
        let actual = blake3::hash(&uncompressed).to_hex().to_string();
        if expected != &actual {
            return Err(SnapshotError::ChecksumMismatch { expected: expected.clone(), actual });
        }
    }
    Ok(serde_json::from_slice(&uncompressed)?)
}

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("unsupported snapshot encoding `{0}`")]
    UnknownEncoding(String),
    #[error(
        "unsupported snapshot schema version {found}; this runtime supports through {supported}"
    )]
    UnsupportedSchema { found: u32, supported: u32 },
    #[error("snapshot decompressed size mismatch: expected {expected} bytes, got {actual}")]
    SizeMismatch { expected: u64, actual: u64 },
    #[error("snapshot checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    #[error("snapshot payload is too large to represent ({0} bytes)")]
    PayloadTooLarge(usize),
    #[error("snapshot JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("snapshot compression error: {0}")]
    Compression(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use horizon_core::RunId;

    use super::*;

    #[test]
    fn zstd_round_trip_validates_integrity() {
        let encoded =
            encode_snapshot(&vec!["goal", "constraint", "failure"], SnapshotEncoding::ZstdJson)
                .unwrap();
        let checkpoint = CheckpointRecord {
            run_id: RunId::new(),
            sequence: 7,
            created_at: Utc::now(),
            state_snapshot: encoded.bytes,
            encoding: encoded.encoding,
            schema_version: encoded.schema_version,
            uncompressed_size: Some(encoded.uncompressed_size),
            checksum: Some(encoded.checksum),
        };
        let decoded: Vec<String> = decode_snapshot(&checkpoint).unwrap();
        assert_eq!(decoded, vec!["goal", "constraint", "failure"]);
    }

    #[test]
    fn checksum_detects_a_tampered_json_snapshot() {
        let encoded = encode_snapshot(&"stable", SnapshotEncoding::Json).unwrap();
        let checkpoint = CheckpointRecord {
            run_id: RunId::new(),
            sequence: 1,
            created_at: Utc::now(),
            state_snapshot: b"\"tampered\"".to_vec(),
            encoding: encoded.encoding,
            schema_version: encoded.schema_version,
            uncompressed_size: Some(10),
            checksum: Some(encoded.checksum),
        };
        assert!(matches!(
            decode_snapshot::<String>(&checkpoint),
            Err(SnapshotError::ChecksumMismatch { .. })
        ));
    }
}

//! Optional native Python bindings for Horizon.
//!
//! The Python research layer can use HTTP by default. This module provides a
//! synchronous, local SQLite alternative for embedding the already-stable
//! runtime contract in Python without recreating state mutation logic there.

use std::{future::Future, path::PathBuf, sync::Arc};

use horizon_core::{RunId, RuntimeCommand};
use horizon_memory::DecaySignals;
use horizon_runtime::{HorizonRuntime, RuntimeConfig, RuntimeError};
use horizon_store::{SnapshotEncoding, SqliteEventStore};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyModule,
};
use serde::Serialize;

/// Native, synchronous façade over a local durable Horizon runtime.
///
/// Python methods return JSON strings deliberately: the public shape matches
/// Horizon's HTTP responses exactly, avoiding a second Python-specific state
/// model and keeping command validation in Rust.
#[pyclass(name = "Runtime")]
pub struct NativeRuntime {
    runtime: Arc<HorizonRuntime<SqliteEventStore>>,
    tokio: tokio::runtime::Runtime,
}

#[pymethods]
impl NativeRuntime {
    #[new]
    #[pyo3(signature = (database_path, *, checkpoint_every=12, retain_checkpoints=8, snapshot_encoding="zstd_json"))]
    fn new(
        database_path: PathBuf,
        checkpoint_every: u64,
        retain_checkpoints: usize,
        snapshot_encoding: &str,
    ) -> PyResult<Self> {
        let snapshot_encoding = parse_snapshot_encoding(snapshot_encoding)?;
        let tokio = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(runtime_error)?;
        let store =
            tokio.block_on(SqliteEventStore::open(&database_path)).map_err(runtime_error)?;
        let runtime = Arc::new(HorizonRuntime::with_config(
            Arc::new(store),
            RuntimeConfig {
                checkpoint_every_events: checkpoint_every,
                checkpoint_retention: (retain_checkpoints > 0).then_some(retain_checkpoints),
                snapshot_encoding,
                ..RuntimeConfig::default()
            },
        ));
        Ok(Self { runtime, tokio })
    }

    /// Create a run and return the normal `CommandOutcome` JSON object.
    fn create_run(&self, py: Python<'_>, goal: String) -> PyResult<String> {
        self.run_json(py, self.runtime.create_run(goal))
    }

    /// Submit a serialized `RuntimeCommand` and return `CommandOutcome` JSON.
    fn dispatch(&self, py: Python<'_>, run_id: &str, command_json: &str) -> PyResult<String> {
        let run_id = parse_run_id(run_id)?;
        let command = serde_json::from_str::<RuntimeCommand>(command_json).map_err(|error| {
            PyValueError::new_err(format!("invalid RuntimeCommand JSON: {error}"))
        })?;
        self.run_json(py, self.runtime.dispatch(run_id, command))
    }

    /// Return the current materialized projection as JSON without mutation.
    fn projection(&self, py: Python<'_>, run_id: &str) -> PyResult<String> {
        self.run_json(py, self.runtime.projection(parse_run_id(run_id)?))
    }

    /// Return the immutable ordered event stream as JSON.
    fn events(&self, py: Python<'_>, run_id: &str) -> PyResult<String> {
        self.run_json(py, self.runtime.events(parse_run_id(run_id)?))
    }

    /// Render the compact State Anchor in the same shape as the HTTP API.
    fn state_anchor(&self, py: Python<'_>, run_id: &str) -> PyResult<String> {
        let run_id = parse_run_id(run_id)?;
        let runtime = Arc::clone(&self.runtime);
        self.run_json(py, async move {
            let projection = runtime.projection(run_id).await?;
            if projection.sequence == 0 {
                return Err(RuntimeError::UnknownRun(run_id));
            }
            Ok(serde_json::json!({
                "run_id": run_id,
                "sequence": projection.sequence,
                "last_anchor_sequence": projection.last_anchor_sequence,
                "content": projection.cognitive.render_anchor(),
            }))
        })
    }

    /// Return all local run projections as JSON.
    fn list_runs(&self, py: Python<'_>) -> PyResult<String> {
        self.run_json(py, self.runtime.list_runs())
    }

    /// Persist an explicit checkpoint and return `CommandOutcome` JSON.
    fn checkpoint(&self, py: Python<'_>, run_id: &str) -> PyResult<String> {
        self.run_json(py, self.runtime.checkpoint(parse_run_id(run_id)?))
    }

    /// Replay from the latest checkpoint and record the recovery boundary.
    fn recover(&self, py: Python<'_>, run_id: &str) -> PyResult<String> {
        self.run_json(py, self.runtime.recover(parse_run_id(run_id)?))
    }

    /// Evaluate an intervention request encoded as `DecaySignals` JSON.
    fn intervene(&self, py: Python<'_>, run_id: &str, signals_json: &str) -> PyResult<String> {
        let signals = serde_json::from_str::<DecaySignals>(signals_json).map_err(|error| {
            PyValueError::new_err(format!("invalid DecaySignals JSON: {error}"))
        })?;
        self.run_json(py, self.runtime.intervene_if_needed(parse_run_id(run_id)?, signals))
    }

    /// Execute currently dependency-ready process tasks and return their JSON outcomes.
    fn execute_ready_tasks(&self, py: Python<'_>, run_id: &str) -> PyResult<String> {
        self.run_json(py, Arc::clone(&self.runtime).execute_ready_tasks(parse_run_id(run_id)?))
    }

    /// Retain recent snapshots while preserving the immutable event log.
    fn compact_checkpoints(
        &self,
        py: Python<'_>,
        run_id: &str,
        retain_latest: usize,
    ) -> PyResult<String> {
        self.run_json(py, self.runtime.compact_checkpoints(parse_run_id(run_id)?, retain_latest))
    }
}

impl NativeRuntime {
    fn run_json<T>(
        &self,
        py: Python<'_>,
        operation: impl Future<Output = Result<T, RuntimeError>> + Send,
    ) -> PyResult<String>
    where
        T: Serialize + Send,
    {
        // Rust can await/process independently while the Python interpreter is
        // available to another thread. Horizon's own mutation gate preserves
        // the durable command ordering.
        let result = py.detach(|| self.tokio.block_on(operation));
        let value = result.map_err(runtime_error)?;
        serde_json::to_string(&value).map_err(runtime_error)
    }
}

fn parse_run_id(value: &str) -> PyResult<RunId> {
    value
        .parse()
        .map_err(|error| PyValueError::new_err(format!("invalid run id `{value}`: {error}")))
}

fn parse_snapshot_encoding(value: &str) -> PyResult<SnapshotEncoding> {
    match value {
        "json" => Ok(SnapshotEncoding::Json),
        "zstd_json" | "zstd-json" => Ok(SnapshotEncoding::ZstdJson),
        _ => Err(PyValueError::new_err("snapshot_encoding must be `json` or `zstd_json`")),
    }
}

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A Python extension module implemented in Rust.
#[pymodule]
fn horizon_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeRuntime>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use pyo3::Python;

    use super::*;

    #[test]
    fn native_runtime_uses_the_same_durable_json_contract() {
        let runtime = NativeRuntime::new(PathBuf::from(":memory:"), 0, 0, "json").unwrap();
        Python::attach(|py| {
            let created = runtime.create_run(py, "native durability".into()).unwrap();
            let run_id = serde_json::from_str::<serde_json::Value>(&created).unwrap()["projection"]
                ["run_id"]
                .as_str()
                .unwrap()
                .to_owned();
            let projection = runtime.projection(py, &run_id).unwrap();
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&projection).unwrap()["cognitive"]["primary_goal"],
                "native durability"
            );
        });
    }

    #[test]
    fn snapshot_encoding_parser_is_strict() {
        assert!(parse_snapshot_encoding("zstd_json").is_ok());
        assert!(parse_snapshot_encoding("gzip").is_err());
    }
}

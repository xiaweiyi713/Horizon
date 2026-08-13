//! Small JSON/HTTP façade for the Python research layer.
//!
//! The HTTP API intentionally mirrors durable runtime primitives instead of
//! embedding an LLM policy in Rust. Python remains free to provide prompts,
//! providers, and experiment policies while Rust owns correctness boundaries.

use std::{net::SocketAddr, sync::Arc};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use horizon_core::{RunId, RuntimeCommand};
use horizon_memory::DecaySignals;
use horizon_store::SqliteEventStore;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::{CommandOutcome, HorizonRuntime, ReadyTaskOutcome, RecoveryOutcome, RuntimeError};

type Runtime = HorizonRuntime<SqliteEventStore>;

#[derive(Clone)]
struct AppState {
    runtime: Arc<Runtime>,
}

#[derive(Clone, Debug, Deserialize)]
struct CreateRunRequest {
    goal: String,
}

#[derive(Clone, Debug, Serialize)]
struct ApiError {
    error: String,
}

#[derive(Clone, Debug, Serialize)]
struct AnchorResponse {
    run_id: RunId,
    sequence: u64,
    last_anchor_sequence: u64,
    content: String,
}

impl IntoResponse for HttpApiError {
    fn into_response(self) -> Response {
        let status = match self.0 {
            RuntimeError::UnknownRun(_) => StatusCode::NOT_FOUND,
            RuntimeError::Store(horizon_store::StoreError::ConcurrentModification { .. }) => {
                StatusCode::CONFLICT
            }
            RuntimeError::Command(_)
            | RuntimeError::State(_)
            | RuntimeError::Scheduler(_)
            | RuntimeError::InvalidCommand(_) => StatusCode::UNPROCESSABLE_ENTITY,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(ApiError { error: self.0.to_string() })).into_response()
    }
}

struct HttpApiError(RuntimeError);

impl From<RuntimeError> for HttpApiError {
    fn from(error: RuntimeError) -> Self {
        Self(error)
    }
}

/// Starts the self-contained local API. No authentication is included because
/// the intended MVP binding is localhost; place a proxy in front for remote use.
pub async fn serve(address: SocketAddr, runtime: Arc<Runtime>) -> Result<(), HttpServerError> {
    let state = AppState { runtime };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/runs", post(create_run).get(list_runs))
        .route("/v1/runs/{run_id}", get(get_run))
        .route("/v1/runs/{run_id}/events", get(events))
        .route("/v1/runs/{run_id}/anchor", get(anchor))
        .route("/v1/runs/{run_id}/commands", post(command))
        .route("/v1/runs/{run_id}/checkpoint", post(checkpoint))
        .route("/v1/runs/{run_id}/recover", post(recover))
        .route("/v1/runs/{run_id}/interventions", post(intervene))
        .route("/v1/runs/{run_id}/tasks/execute", post(execute_tasks))
        .with_state(state)
        .layer(TraceLayer::new_for_http());
    let listener = TcpListener::bind(address).await?;
    info!(address = %address, "Horizon HTTP runtime listening");
    axum::serve(listener, app).await.map_err(HttpServerError::Serve)?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

async fn create_run(
    State(state): State<AppState>,
    Json(request): Json<CreateRunRequest>,
) -> Result<Json<CommandOutcome>, HttpApiError> {
    Ok(Json(state.runtime.create_run(request.goal).await?))
}

async fn list_runs(
    State(state): State<AppState>,
) -> Result<Json<Vec<horizon_core::RunProjection>>, HttpApiError> {
    Ok(Json(state.runtime.list_runs().await?))
}

async fn get_run(
    State(state): State<AppState>,
    Path(run_id): Path<RunId>,
) -> Result<Json<horizon_core::RunProjection>, HttpApiError> {
    let projection = state.runtime.projection(run_id).await?;
    if projection.sequence == 0 {
        return Err(HttpApiError(RuntimeError::UnknownRun(run_id)));
    }
    Ok(Json(projection))
}

async fn events(
    State(state): State<AppState>,
    Path(run_id): Path<RunId>,
) -> Result<Json<Vec<horizon_core::EventRecord>>, HttpApiError> {
    let records = state.runtime.events(run_id).await?;
    if records.is_empty() {
        return Err(HttpApiError(RuntimeError::UnknownRun(run_id)));
    }
    Ok(Json(records))
}

async fn anchor(
    State(state): State<AppState>,
    Path(run_id): Path<RunId>,
) -> Result<Json<AnchorResponse>, HttpApiError> {
    let projection = state.runtime.projection(run_id).await?;
    if projection.sequence == 0 {
        return Err(HttpApiError(RuntimeError::UnknownRun(run_id)));
    }
    Ok(Json(AnchorResponse {
        run_id,
        sequence: projection.sequence,
        last_anchor_sequence: projection.last_anchor_sequence,
        content: projection.cognitive.render_anchor(),
    }))
}

async fn command(
    State(state): State<AppState>,
    Path(run_id): Path<RunId>,
    Json(command): Json<RuntimeCommand>,
) -> Result<Json<CommandOutcome>, HttpApiError> {
    Ok(Json(state.runtime.dispatch(run_id, command).await?))
}

async fn checkpoint(
    State(state): State<AppState>,
    Path(run_id): Path<RunId>,
) -> Result<Json<CommandOutcome>, HttpApiError> {
    Ok(Json(state.runtime.checkpoint(run_id).await?))
}

async fn recover(
    State(state): State<AppState>,
    Path(run_id): Path<RunId>,
) -> Result<Json<RecoveryOutcome>, HttpApiError> {
    Ok(Json(state.runtime.recover(run_id).await?))
}

async fn intervene(
    State(state): State<AppState>,
    Path(run_id): Path<RunId>,
    Json(signals): Json<DecaySignals>,
) -> Result<Json<Option<CommandOutcome>>, HttpApiError> {
    Ok(Json(state.runtime.intervene_if_needed(run_id, signals).await?))
}

async fn execute_tasks(
    State(state): State<AppState>,
    Path(run_id): Path<RunId>,
) -> Result<Json<Vec<ReadyTaskOutcome>>, HttpApiError> {
    Ok(Json(state.runtime.execute_ready_tasks(run_id).await?))
}

#[derive(Debug, Error)]
pub enum HttpServerError {
    #[error("failed to bind Horizon HTTP server: {0}")]
    Bind(#[from] std::io::Error),
    #[error("Horizon HTTP server failed: {0}")]
    Serve(std::io::Error),
}

//! Safe-ish process supervision for task execution.
//!
//! This crate intentionally accepts an argv vector, not a shell string. Horizon
//! records an operation id so callers can use at-least-once execution together
//! with idempotency checks in the durable event stream.

use std::{collections::BTreeMap, path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{io::AsyncReadExt, process::Command, time::timeout};
use tracing::{debug, warn};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessRequest {
    pub operation_id: String,
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub working_dir: Option<PathBuf>,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    #[serde(default = "ProcessRequest::default_timeout_ms")]
    pub timeout_ms: u64,
}

impl ProcessRequest {
    const fn default_timeout_ms() -> u64 {
        300_000
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessOutput {
    pub operation_id: String,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub duration_ms: u128,
}

impl ProcessOutput {
    #[must_use]
    pub fn succeeded(&self) -> bool {
        !self.timed_out && self.exit_code == Some(0)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProcessSupervisor;

impl ProcessSupervisor {
    pub async fn run(&self, request: ProcessRequest) -> Result<ProcessOutput, ProcessError> {
        if request.operation_id.trim().is_empty() {
            return Err(ProcessError::EmptyOperationId);
        }
        if request.program.trim().is_empty() {
            return Err(ProcessError::EmptyProgram);
        }
        let started = std::time::Instant::now();
        debug!(operation_id = %request.operation_id, program = %request.program, "spawning process");
        let mut command = Command::new(&request.program);
        command.args(&request.args).kill_on_drop(true);
        if let Some(working_dir) = &request.working_dir {
            command.current_dir(working_dir);
        }
        command.envs(&request.environment);
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        let mut child = command.spawn().map_err(ProcessError::Spawn)?;
        let mut stdout = child.stdout.take().expect("stdout configured as piped");
        let mut stderr = child.stderr.take().expect("stderr configured as piped");
        let stdout_task = tokio::spawn(async move {
            let mut bytes = Vec::new();
            stdout.read_to_end(&mut bytes).await.map(|_| bytes)
        });
        let stderr_task = tokio::spawn(async move {
            let mut bytes = Vec::new();
            stderr.read_to_end(&mut bytes).await.map(|_| bytes)
        });

        let duration = Duration::from_millis(request.timeout_ms.max(1));
        let (status, timed_out) = match timeout(duration, child.wait()).await {
            Ok(status) => (status.map_err(ProcessError::Wait)?, false),
            Err(_) => {
                warn!(operation_id = %request.operation_id, timeout_ms = request.timeout_ms, "process timed out; killing child");
                child.kill().await.map_err(ProcessError::Kill)?;
                let status = child.wait().await.map_err(ProcessError::Wait)?;
                (status, true)
            }
        };
        let stdout = stdout_task.await.map_err(ProcessError::Join)?.map_err(ProcessError::Read)?;
        let stderr = stderr_task.await.map_err(ProcessError::Join)?.map_err(ProcessError::Read)?;
        Ok(ProcessOutput {
            operation_id: request.operation_id,
            exit_code: status.code(),
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
            timed_out,
            duration_ms: started.elapsed().as_millis(),
        })
    }
}

#[derive(Debug, Error)]
pub enum ProcessError {
    #[error("process operation id cannot be empty")]
    EmptyOperationId,
    #[error("process program cannot be empty")]
    EmptyProgram,
    #[error("failed to spawn process: {0}")]
    Spawn(std::io::Error),
    #[error("failed while waiting for process: {0}")]
    Wait(std::io::Error),
    #[error("failed to kill timed-out process: {0}")]
    Kill(std::io::Error),
    #[error("failed to join output reader: {0}")]
    Join(tokio::task::JoinError),
    #[error("failed to read process output: {0}")]
    Read(std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn captures_successful_process_output() {
        let output = ProcessSupervisor
            .run(ProcessRequest {
                operation_id: "test-success".into(),
                program: "sh".into(),
                args: vec!["-c".into(), "printf horizon".into()],
                working_dir: None,
                environment: BTreeMap::new(),
                timeout_ms: 1_000,
            })
            .await
            .unwrap();
        assert!(output.succeeded());
        assert_eq!(output.stdout, "horizon");
    }
}

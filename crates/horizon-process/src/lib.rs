//! Timeout-aware, resource-bounded process supervision for Horizon tasks.
//!
//! Horizon accepts argv vectors rather than shell strings. A task can execute
//! locally with Unix `setrlimit` safeguards or through an explicit Docker
//! backend; neither mode claims to be a complete security sandbox.

use std::{
    collections::BTreeMap,
    io,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use horizon_core::{TaskExecutionBackend, TaskResourceLimits, TaskSpecValidationError};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    process::Command,
    time::timeout,
};
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
    #[serde(default)]
    pub resources: TaskResourceLimits,
    #[serde(default)]
    pub executor: TaskExecutionBackend,
}

impl ProcessRequest {
    const fn default_timeout_ms() -> u64 {
        300_000
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessOutput {
    pub operation_id: String,
    pub executor: TaskExecutionBackend,
    pub resource_limits: TaskResourceLimits,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
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
        request.resources.validate()?;
        request.executor.validate()?;
        let started = std::time::Instant::now();
        debug!(
            operation_id = %request.operation_id,
            program = %request.program,
            executor = ?request.executor,
            resources = ?request.resources,
            "spawning process"
        );
        let mut spawned = build_command(&request)?;
        let mut child = spawned.command.spawn().map_err(ProcessError::Spawn)?;
        let stdout = child.stdout.take().expect("stdout configured as piped");
        let stderr = child.stderr.take().expect("stderr configured as piped");
        let output_limit = request.resources.max_output_bytes;
        let stdout_task = tokio::spawn(read_capped(stdout, output_limit));
        let stderr_task = tokio::spawn(read_capped(stderr, output_limit));

        let duration = Duration::from_millis(request.timeout_ms.max(1));
        let (status, timed_out) = match timeout(duration, child.wait()).await {
            Ok(status) => (status.map_err(ProcessError::Wait)?, false),
            Err(_) => {
                warn!(operation_id = %request.operation_id, timeout_ms = request.timeout_ms, "process timed out; killing child");
                child.kill().await.map_err(ProcessError::Kill)?;
                let status = child.wait().await.map_err(ProcessError::Wait)?;
                if let Some(container_name) = spawned.container_name.take() {
                    cleanup_container(&container_name).await;
                }
                (status, true)
            }
        };
        let stdout = stdout_task.await.map_err(ProcessError::Join)?.map_err(ProcessError::Read)?;
        let stderr = stderr_task.await.map_err(ProcessError::Join)?.map_err(ProcessError::Read)?;
        Ok(ProcessOutput {
            operation_id: request.operation_id,
            executor: request.executor,
            resource_limits: request.resources,
            exit_code: status.code(),
            stdout: String::from_utf8_lossy(&stdout.bytes).into_owned(),
            stderr: String::from_utf8_lossy(&stderr.bytes).into_owned(),
            stdout_truncated: stdout.truncated,
            stderr_truncated: stderr.truncated,
            timed_out,
            duration_ms: started.elapsed().as_millis(),
        })
    }
}

struct SpawnedCommand {
    command: Command,
    /// A Docker name is retained only so a timeout can forcibly clean up a
    /// detached-or-slowly-shutting-down container after the client is killed.
    container_name: Option<String>,
}

fn build_command(request: &ProcessRequest) -> Result<SpawnedCommand, ProcessError> {
    match &request.executor {
        TaskExecutionBackend::Local => {
            let mut command = Command::new(&request.program);
            command.args(&request.args).kill_on_drop(true);
            if let Some(working_dir) = &request.working_dir {
                command.current_dir(working_dir);
            }
            command.envs(&request.environment);
            configure_local_resource_limits(&mut command, &request.resources)?;
            configure_pipes(&mut command);
            Ok(SpawnedCommand { command, container_name: None })
        }
        TaskExecutionBackend::Docker { image, allow_network, read_only } => {
            let container_name = docker_container_name(&request.operation_id);
            let mut command = Command::new("docker");
            command.arg("run").arg("--rm").arg("--init").arg("--name").arg(&container_name);
            if !allow_network {
                command.args(["--network", "none"]);
            }
            if *read_only {
                command.arg("--read-only");
            }
            if let Some(memory) = request.resources.max_memory_bytes {
                command.arg("--memory").arg(memory.to_string());
            }
            if let Some(cpu_time_ms) = request.resources.max_cpu_time_ms {
                command.arg("--ulimit").arg(format!("cpu={}", cpu_seconds(cpu_time_ms)));
            }
            if let Some(working_dir) = &request.working_dir {
                let host_path = std::fs::canonicalize(working_dir).map_err(|source| {
                    ProcessError::WorkingDirectory { path: working_dir.clone(), source }
                })?;
                command
                    .arg("--mount")
                    .arg(format!("type=bind,src={},dst=/workspace", host_path.display()))
                    .args(["--workdir", "/workspace"]);
            }
            for (key, value) in &request.environment {
                command.arg("--env").arg(format!("{key}={value}"));
            }
            command
                .arg("--entrypoint")
                .arg(&request.program)
                .arg(image)
                .args(&request.args)
                .kill_on_drop(true);
            configure_pipes(&mut command);
            Ok(SpawnedCommand { command, container_name: Some(container_name) })
        }
    }
}

fn configure_pipes(command: &mut Command) {
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
}

#[cfg(unix)]
fn configure_local_resource_limits(
    command: &mut Command,
    limits: &TaskResourceLimits,
) -> Result<(), ProcessError> {
    use std::os::unix::process::CommandExt;

    // macOS's sandboxed process model rejects memory rlimit changes from the
    // pre-exec child path. Rejecting the request is safer than pretending to
    // enforce it; Docker remains the portable memory-isolation option there.
    #[cfg(target_os = "macos")]
    if limits.max_memory_bytes.is_some() {
        return Err(ProcessError::UnsupportedLocalMemoryLimit);
    }

    let memory = limits
        .max_memory_bytes
        .map(|value| {
            libc::rlim_t::try_from(value).map_err(|_| ProcessError::ResourceLimitTooLarge {
                field: "max_memory_bytes",
                value,
            })
        })
        .transpose()?;
    let cpu = limits
        .max_cpu_time_ms
        .map(cpu_seconds)
        .map(|value| {
            libc::rlim_t::try_from(value).map_err(|_| ProcessError::ResourceLimitTooLarge {
                field: "max_cpu_time_ms",
                value,
            })
        })
        .transpose()?;
    if memory.is_none() && cpu.is_none() {
        return Ok(());
    }
    // `pre_exec` runs in the child after fork and before exec. The closure only
    // uses setrlimit and avoids logging/allocation on the normal path.
    unsafe {
        command.as_std_mut().pre_exec(move || {
            if let Some(memory) = memory {
                let mut limit = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
                let get_result = libc::getrlimit(libc::RLIMIT_AS, &mut limit);
                if get_result != 0 {
                    return Err(io::Error::last_os_error());
                }
                // Preserve the inherited hard cap.
                limit.rlim_cur = memory.min(limit.rlim_max);
                let result = libc::setrlimit(libc::RLIMIT_AS, &limit);
                if result != 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            if let Some(cpu) = cpu {
                let mut limit = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
                if libc::getrlimit(libc::RLIMIT_CPU, &mut limit) != 0 {
                    return Err(io::Error::last_os_error());
                }
                limit.rlim_cur = cpu.min(limit.rlim_max);
                if libc::setrlimit(libc::RLIMIT_CPU, &limit) != 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    Ok(())
}

#[cfg(not(unix))]
fn configure_local_resource_limits(
    _command: &mut Command,
    limits: &TaskResourceLimits,
) -> Result<(), ProcessError> {
    if limits.max_memory_bytes.is_some() || limits.max_cpu_time_ms.is_some() {
        return Err(ProcessError::UnsupportedLocalResourceLimits);
    }
    Ok(())
}

const fn cpu_seconds(milliseconds: u64) -> u64 {
    milliseconds.saturating_add(999) / 1_000
}

fn docker_container_name(operation_id: &str) -> String {
    static NEXT_CONTAINER_ID: AtomicU64 = AtomicU64::new(1);
    let normalized: String = operation_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '-'
            }
        })
        .take(45)
        .collect();
    let instance = NEXT_CONTAINER_ID.fetch_add(1, Ordering::Relaxed);
    format!("horizon-{}-{}-{instance}", normalized.trim_matches('-'), std::process::id())
}

async fn cleanup_container(name: &str) {
    match Command::new("docker").args(["rm", "-f", name]).output().await {
        Ok(output) if output.status.success() => {
            debug!(container = name, "cleaned up timed-out Docker container")
        }
        Ok(output) => warn!(
            container = name,
            status = ?output.status.code(),
            stderr = %String::from_utf8_lossy(&output.stderr),
            "could not confirm Docker container cleanup"
        ),
        Err(error) => warn!(container = name, %error, "could not invoke Docker cleanup"),
    }
}

struct CapturedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

async fn read_capped<R: AsyncRead + Unpin>(
    mut reader: R,
    max_output_bytes: Option<u64>,
) -> io::Result<CapturedOutput> {
    let max = max_output_bytes
        .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
        .unwrap_or(usize::MAX);
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 8_192];
    let mut truncated = false;
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        let remaining = max.saturating_sub(bytes.len());
        let retained = read.min(remaining);
        bytes.extend_from_slice(&buffer[..retained]);
        truncated |= retained < read;
    }
    Ok(CapturedOutput { bytes, truncated })
}

#[derive(Debug, Error)]
pub enum ProcessError {
    #[error("process operation id cannot be empty")]
    EmptyOperationId,
    #[error("process program cannot be empty")]
    EmptyProgram,
    #[error(transparent)]
    InvalidTaskSpecification(#[from] TaskSpecValidationError),
    #[error("resource limit `{field}` cannot represent {value} on this platform")]
    ResourceLimitTooLarge { field: &'static str, value: u64 },
    #[error(
        "memory and CPU limits for local processes require a Unix host; use the Docker backend"
    )]
    UnsupportedLocalResourceLimits,
    #[error("local memory limits are unavailable on macOS; use the Docker backend")]
    UnsupportedLocalMemoryLimit,
    #[error("failed to resolve task working directory {path}: {source}")]
    WorkingDirectory { path: PathBuf, source: io::Error },
    #[error("failed to spawn process: {0}")]
    Spawn(io::Error),
    #[error("failed while waiting for process: {0}")]
    Wait(io::Error),
    #[error("failed to kill timed-out process: {0}")]
    Kill(io::Error),
    #[error("failed to join output reader: {0}")]
    Join(tokio::task::JoinError),
    #[error("failed to read process output: {0}")]
    Read(io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_request(script: &str) -> ProcessRequest {
        ProcessRequest {
            operation_id: "test-success".into(),
            program: "sh".into(),
            args: vec!["-c".into(), script.into()],
            working_dir: None,
            environment: BTreeMap::new(),
            timeout_ms: 1_000,
            resources: TaskResourceLimits::default(),
            executor: TaskExecutionBackend::Local,
        }
    }

    #[tokio::test]
    async fn captures_successful_process_output() {
        let output = ProcessSupervisor.run(local_request("printf horizon")).await.unwrap();
        assert!(output.succeeded());
        assert_eq!(output.stdout, "horizon");
        assert!(!output.stdout_truncated);
    }

    #[tokio::test]
    async fn caps_output_without_blocking_the_child() {
        let mut request = local_request("printf 123456789; printf abcdefghi >&2");
        request.resources.max_output_bytes = Some(4);
        let output = ProcessSupervisor.run(request).await.unwrap();
        assert!(output.succeeded());
        assert_eq!(output.stdout, "1234");
        assert_eq!(output.stderr, "abcd");
        assert!(output.stdout_truncated);
        assert!(output.stderr_truncated);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn applies_a_local_cpu_limit_to_short_lived_work() {
        let mut request = local_request("printf bounded");
        request.resources.max_cpu_time_ms = Some(1_000);
        let output = ProcessSupervisor.run(request).await.unwrap();
        assert!(output.succeeded());
        assert_eq!(output.stdout, "bounded");
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn rejects_unenforceable_local_memory_limits() {
        let mut request = local_request("printf bounded");
        request.resources.max_memory_bytes = Some(64 * 1024 * 1024);
        assert!(matches!(
            ProcessSupervisor.run(request).await,
            Err(ProcessError::UnsupportedLocalMemoryLimit)
        ));
    }

    #[test]
    fn docker_backend_maps_isolation_and_resource_flags_without_needing_a_daemon() {
        let mut request = local_request("printf ignored");
        request.operation_id = "container operation/42".into();
        request.resources.max_memory_bytes = Some(1_048_576);
        request.resources.max_cpu_time_ms = Some(1_500);
        request.executor = TaskExecutionBackend::Docker {
            image: "alpine:3.21".into(),
            allow_network: false,
            read_only: true,
        };
        let command = build_command(&request).unwrap();
        let rendered = format!("{:?}", command.command);
        assert!(command.container_name.unwrap().starts_with("horizon-container-operation-42"));
        assert!(rendered.contains("--network") && rendered.contains("none"));
        assert!(rendered.contains("--read-only"));
        assert!(rendered.contains("--memory") && rendered.contains("1048576"));
        assert!(rendered.contains("--ulimit") && rendered.contains("cpu=2"));
    }

    #[test]
    fn docker_container_names_are_unique_for_concurrent_operations() {
        assert_ne!(
            docker_container_name("same-operation"),
            docker_container_name("same-operation")
        );
    }
}

use std::{
    collections::HashMap,
    net::SocketAddr,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use horizon_core::{
    AgentState, Constraint, EventRecord, RunId, RuntimeCommand, TaskId, TaskSpec, TaskStatus,
};
use horizon_memory::DecaySignals;
use horizon_runtime::{HorizonRuntime, RuntimeConfig, serve};
use horizon_store::SqliteEventStore;
use serde::Deserialize;
use tokio::fs;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "horizon",
    version,
    about = "A durable cognitive runtime for long-horizon AI agents"
)]
struct Cli {
    /// SQLite event-store file. Horizon is local-first; this is all the service
    /// infrastructure required for the MVP.
    #[arg(long, global = true, env = "HORIZON_DB", default_value = "horizon.db")]
    db: PathBuf,
    /// Number of domain events between automatic snapshots; 0 disables automatic snapshots.
    #[arg(long, global = true, default_value_t = 12)]
    checkpoint_every: u64,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Create a durable run from a task YAML file and execute ready local tasks.
    Run {
        file: PathBuf,
        #[arg(long)]
        no_execute: bool,
    },
    /// List all local durable runs, or show one run by ID.
    Status { run_id: Option<String> },
    /// Move a nonterminal run into Suspended state.
    Pause {
        run_id: String,
        #[arg(long, default_value = "paused by operator")]
        reason: String,
    },
    /// Recover a run from checkpoint + event replay and return it to Planning.
    Resume {
        run_id: String,
        #[arg(long)]
        execute: bool,
    },
    /// Print the materialized execution and cognitive state.
    Inspect {
        run_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Print ordered immutable events for a run.
    Events {
        run_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Persist a checkpoint at the current boundary.
    Checkpoint { run_id: String },
    /// Rebuild a run from checkpoint plus its event suffix without mutation.
    Replay {
        run_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Print an LLM-ready State Anchor generated from durable cognitive state.
    Memory { run_id: String },
    /// Print remembered failed approaches and their reasons.
    Failures { run_id: String },
    /// Evaluate the state-anchor intervention heuristic at a boundary.
    Intervene(InterveneArgs),
    /// Execute currently dependency-ready process-backed tasks.
    Execute { run_id: String },
    /// Start a local JSON API for Python agent policies and experiments.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8787")]
        address: String,
    },
    /// Run a self-contained durable recovery demonstration with no LLM/API key.
    Demo,
}

#[derive(Args, Debug)]
struct InterveneArgs {
    run_id: String,
    #[arg(long, default_value_t = 0.0)]
    context_pressure: f32,
    #[arg(long, default_value_t = 0)]
    subgoal_switches: u32,
    #[arg(long, default_value_t = 0)]
    recent_failures: u32,
    #[arg(long)]
    recovered_session: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunFile {
    goal: String,
    #[serde(default)]
    constraints: Vec<ConstraintFile>,
    #[serde(default)]
    plan: Vec<String>,
    #[serde(default)]
    tasks: Vec<TaskFile>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ConstraintFile {
    Text(String),
    Detailed { id: Option<String>, content: String, source: Option<String> },
}

impl ConstraintFile {
    fn into_constraint(self, index: usize) -> Constraint {
        match self {
            Self::Text(content) => {
                Constraint { id: format!("constraint-{}", index + 1), content, source: None }
            }
            Self::Detailed { id, content, source } => Constraint {
                id: id.unwrap_or_else(|| format!("constraint-{}", index + 1)),
                content,
                source,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskFile {
    name: String,
    /// Optional hierarchy-only parent. Unlike `depends_on`, this does not
    /// affect scheduling readiness.
    #[serde(default)]
    parent: Option<String>,
    /// Stable idempotency key reused after a crash. Omit to derive one from the
    /// task UUID; provide one for an external system with its own key scheme.
    #[serde(default)]
    operation_id: Option<String>,
    #[serde(default)]
    command: Option<Vec<String>>,
    #[serde(default)]
    depends_on: Vec<String>,
    #[serde(default)]
    priority: u8,
    #[serde(default = "default_retries")]
    max_retries: u32,
    #[serde(default)]
    timeout_ms: Option<u64>,
    #[serde(default)]
    working_dir: Option<String>,
}

const fn default_retries() -> u32 {
    1
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| "horizon=info".into()),
        )
        .with_target(false)
        .init();
    let runtime = Arc::new(HorizonRuntime::with_config(
        Arc::new(SqliteEventStore::open(&cli.db).await.context("opening SQLite event store")?),
        RuntimeConfig { checkpoint_every_events: cli.checkpoint_every, ..Default::default() },
    ));
    match cli.command {
        Command::Run { file, no_execute } => command_run(&runtime, &file, no_execute).await,
        Command::Status { run_id } => command_status(&runtime, run_id.as_deref()).await,
        Command::Pause { run_id, reason } => {
            let run_id = parse_run_id(&run_id)?;
            let outcome = runtime.dispatch(run_id, RuntimeCommand::Suspend { reason }).await?;
            print_projection(&outcome.projection, false)
        }
        Command::Resume { run_id, execute } => {
            command_resume(&runtime, parse_run_id(&run_id)?, execute).await
        }
        Command::Inspect { run_id, json } => {
            print_projection(&runtime.projection(parse_run_id(&run_id)?).await?, json)
        }
        Command::Events { run_id, json } => {
            print_events(&runtime.events(parse_run_id(&run_id)?).await?, json)
        }
        Command::Checkpoint { run_id } => {
            let outcome = runtime.checkpoint(parse_run_id(&run_id)?).await?;
            println!("Checkpoint stored at sequence {}", outcome.projection.sequence);
            Ok(())
        }
        Command::Replay { run_id, json } => {
            print_projection(&runtime.projection(parse_run_id(&run_id)?).await?, json)
        }
        Command::Memory { run_id } => {
            let projection = runtime.projection(parse_run_id(&run_id)?).await?;
            println!("{}", projection.cognitive.render_anchor());
            Ok(())
        }
        Command::Failures { run_id } => {
            let projection = runtime.projection(parse_run_id(&run_id)?).await?;
            if projection.cognitive.failed_attempts.is_empty() {
                println!("No remembered failures.");
            } else {
                for failure in &projection.cognitive.failed_attempts {
                    println!(
                        "{}\n  approach: {}\n  reason: {}\n  retryable: {}",
                        failure.id, failure.approach, failure.reason, failure.retryable
                    );
                }
            }
            Ok(())
        }
        Command::Intervene(args) => command_intervene(&runtime, args).await,
        Command::Execute { run_id } => {
            let outcomes = runtime.execute_ready_tasks(parse_run_id(&run_id)?).await?;
            print_task_outcomes(&outcomes);
            Ok(())
        }
        Command::Serve { address } => {
            let address =
                SocketAddr::from_str(&address).context("address must look like 127.0.0.1:8787")?;
            serve(address, runtime).await.map_err(anyhow::Error::from)
        }
        Command::Demo => command_demo(&runtime).await,
    }
}

async fn command_run(
    runtime: &Arc<HorizonRuntime<SqliteEventStore>>,
    file: &Path,
    no_execute: bool,
) -> Result<()> {
    let content = fs::read_to_string(file)
        .await
        .with_context(|| format!("reading task file {}", file.display()))?;
    let specification: RunFile = serde_yaml::from_str(&content).context("parsing task YAML")?;
    if specification.goal.trim().is_empty() {
        bail!("task YAML `goal` cannot be empty");
    }
    let run_id = runtime.create_run(specification.goal).await?.projection.run_id;
    runtime.dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Planning }).await?;
    for (index, constraint) in specification.constraints.into_iter().enumerate() {
        runtime
            .dispatch(
                run_id,
                RuntimeCommand::AddConstraint { constraint: constraint.into_constraint(index) },
            )
            .await?;
    }
    if !specification.plan.is_empty() {
        runtime.dispatch(run_id, RuntimeCommand::SetPlan { steps: specification.plan }).await?;
    }
    let tasks = resolve_tasks(specification.tasks)?;
    for task in tasks {
        runtime.dispatch(run_id, RuntimeCommand::CreateTask { task }).await?;
    }
    runtime.dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Executing }).await?;
    println!("Run created: {run_id}");
    if !no_execute {
        execute_until_idle(runtime, run_id).await?;
    }
    maybe_complete(runtime, run_id).await?;
    print_projection(&runtime.projection(run_id).await?, false)
}

fn resolve_tasks(raw_tasks: Vec<TaskFile>) -> Result<Vec<TaskSpec>> {
    let mut ids = HashMap::<String, TaskId>::new();
    for task in &raw_tasks {
        if task.name.trim().is_empty() {
            bail!("task name cannot be empty");
        }
        if ids.insert(task.name.clone(), TaskId::new()).is_some() {
            bail!("duplicate task name: {}", task.name);
        }
        if task.command.as_ref().is_some_and(Vec::is_empty) {
            bail!("task `{}` has an empty command", task.name);
        }
    }
    raw_tasks
        .into_iter()
        .map(|task| {
            let parent = task
                .parent
                .as_deref()
                .map(|name| {
                    ids.get(name).copied().with_context(|| {
                        format!("task `{}` has unknown parent `{name}`", task.name)
                    })
                })
                .transpose()?;
            let dependencies = task
                .depends_on
                .iter()
                .map(|name| {
                    ids.get(name).copied().with_context(|| {
                        format!("task `{}` depends on unknown task `{name}`", task.name)
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(TaskSpec {
                id: ids[&task.name],
                title: task.name,
                parent,
                operation_id: task.operation_id,
                command: task.command,
                working_dir: task.working_dir,
                priority: task.priority,
                dependencies,
                max_retries: task.max_retries,
                timeout_ms: task.timeout_ms,
            })
        })
        .collect()
}

async fn execute_until_idle(
    runtime: &Arc<HorizonRuntime<SqliteEventStore>>,
    run_id: RunId,
) -> Result<()> {
    loop {
        let outcomes = runtime.execute_ready_tasks(run_id).await?;
        if outcomes.is_empty() {
            break;
        }
        print_task_outcomes(&outcomes);
    }
    Ok(())
}

async fn maybe_complete(
    runtime: &Arc<HorizonRuntime<SqliteEventStore>>,
    run_id: RunId,
) -> Result<()> {
    let projection = runtime.projection(run_id).await?;
    if projection.state != AgentState::Executing {
        return Ok(());
    }
    let all_succeeded = !projection.tasks.is_empty()
        && projection.tasks.values().all(|task| task.status == TaskStatus::Succeeded);
    if all_succeeded {
        runtime.dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Evaluating }).await?;
        runtime.dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Completed }).await?;
    }
    Ok(())
}

async fn command_status(
    runtime: &Arc<HorizonRuntime<SqliteEventStore>>,
    run_id: Option<&str>,
) -> Result<()> {
    if let Some(run_id) = run_id {
        return print_projection(&runtime.projection(parse_run_id(run_id)?).await?, false);
    }
    let runs = runtime.list_runs().await?;
    if runs.is_empty() {
        println!("No runs in this store.");
        return Ok(());
    }
    println!("RUN ID                                  STATE       SEQUENCE  GOAL");
    for run in runs {
        println!(
            "{:<39} {:<11?} {:<9} {}",
            run.run_id,
            run.state,
            run.sequence,
            run.cognitive.primary_goal.unwrap_or_default()
        );
    }
    Ok(())
}

async fn command_resume(
    runtime: &Arc<HorizonRuntime<SqliteEventStore>>,
    run_id: RunId,
    execute: bool,
) -> Result<()> {
    let outcome = runtime.recover(run_id).await?;
    if outcome.projection.state == AgentState::Recovering {
        runtime.dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Planning }).await?;
        runtime.dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Executing }).await?;
    }
    if execute {
        execute_until_idle(runtime, run_id).await?;
        maybe_complete(runtime, run_id).await?;
    }
    println!(
        "Recovered from checkpoint: {}",
        outcome
            .restored_from_checkpoint
            .map_or_else(|| "none (full replay)".into(), |sequence| sequence.to_string())
    );
    print_projection(&runtime.projection(run_id).await?, false)
}

async fn command_intervene(
    runtime: &Arc<HorizonRuntime<SqliteEventStore>>,
    args: InterveneArgs,
) -> Result<()> {
    let outcome = runtime
        .intervene_if_needed(
            parse_run_id(&args.run_id)?,
            DecaySignals {
                context_pressure: args.context_pressure,
                subgoal_switches: args.subgoal_switches,
                recent_failures: args.recent_failures,
                recovered_session: args.recovered_session,
                ..Default::default()
            },
        )
        .await?;
    match outcome {
        Some(outcome) => {
            println!("State Anchor injected at sequence {}", outcome.projection.sequence)
        }
        None => println!("No State Anchor needed at this boundary."),
    }
    Ok(())
}

async fn command_demo(runtime: &Arc<HorizonRuntime<SqliteEventStore>>) -> Result<()> {
    println!("Horizon durable recovery demo\n");
    let run_id = runtime
        .create_run("Produce a durable result without repeating a failed method")
        .await?
        .projection
        .run_id;
    runtime.dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Planning }).await?;
    runtime
        .dispatch(
            run_id,
            RuntimeCommand::AddConstraint {
                constraint: Constraint {
                    id: "avoid-repeat".into(),
                    content: "Do not retry the full-sample Sharpe approach after it fails.".into(),
                    source: Some("demo".into()),
                },
            },
        )
        .await?;
    runtime
        .dispatch(
            run_id,
            RuntimeCommand::SetPlan {
                steps: vec![
                    "Evaluate performance by market regime".into(),
                    "Compare out-of-sample results".into(),
                ],
            },
        )
        .await?;
    runtime.dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Executing }).await?;
    runtime
        .dispatch(
            run_id,
            RuntimeCommand::RememberFailure {
                failure: horizon_core::FailureRecord {
                    id: "demo-failure".into(),
                    approach: "full-sample Sharpe".into(),
                    reason: "cannot distinguish regime-specific degradation".into(),
                    operation_id: Some("full-sample-sharpe".into()),
                    retryable: false,
                },
            },
        )
        .await?;
    runtime.checkpoint(run_id).await?;
    println!("Run {run_id} checkpointed. Simulating a process crash…");
    drop(runtime.projection(run_id).await?);
    let recovered = runtime.recover(run_id).await?;
    println!(
        "Restored from checkpoint {:?}; current state: {:?}\n",
        recovered.restored_from_checkpoint, recovered.projection.state
    );
    println!("{}\n", recovered.projection.cognitive.render_anchor());
    let remembered = runtime.known_failure(run_id, "full-sample-sharpe").await?;
    println!(
        "Failure guard: {}",
        remembered.map_or_else(
            || "MISSING (unexpected)".into(),
            |failure| format!("blocked `{}` because {}", failure.approach, failure.reason)
        )
    );
    Ok(())
}

fn parse_run_id(value: &str) -> Result<RunId> {
    value.parse().with_context(|| format!("invalid run ID `{value}`"))
}

fn print_projection(projection: &horizon_core::RunProjection, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(projection)?);
        return Ok(());
    }
    println!(
        "Run: {}\nState: {:?}\nSequence: {}\nGoal: {}",
        projection.run_id,
        projection.state,
        projection.sequence,
        projection.cognitive.primary_goal.as_deref().unwrap_or("<unset>")
    );
    println!(
        "Tasks: {} ({} succeeded, {} failed)",
        projection.tasks.len(),
        projection.tasks.values().filter(|task| task.status == TaskStatus::Succeeded).count(),
        projection.tasks.values().filter(|task| task.status == TaskStatus::Failed).count()
    );
    println!(
        "Constraints: {} | Decisions: {} | Failure memories: {} | Last anchor: {}",
        projection.cognitive.constraints.len(),
        projection.cognitive.decisions.len(),
        projection.cognitive.failed_attempts.len(),
        projection.last_anchor_sequence
    );
    Ok(())
}

fn print_events(events: &[EventRecord], json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(events)?);
        return Ok(());
    }
    for event in events {
        println!("#{:<4} {} {}", event.sequence, event.timestamp.to_rfc3339(), event.event.name());
    }
    Ok(())
}

fn print_task_outcomes(outcomes: &[horizon_runtime::ReadyTaskOutcome]) {
    for outcome in outcomes {
        match &outcome.output {
            Ok(output) => println!("task {} succeeded: {}", outcome.task_id, output),
            Err(error) => println!("task {} failed: {}", outcome.task_id, error),
        }
    }
}

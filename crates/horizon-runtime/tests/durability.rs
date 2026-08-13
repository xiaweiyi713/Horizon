use std::sync::Arc;

use horizon_core::{AgentState, RuntimeCommand, TaskSpec, TaskStatus};
use horizon_runtime::{HorizonRuntime, RuntimeConfig};
use horizon_store::{EventStore, SqliteEventStore};
use tempfile::tempdir;

#[tokio::test]
async fn independent_runtime_instance_restores_checkpoint_and_event_suffix() {
    let directory = tempdir().unwrap();
    let database = directory.path().join("durable.db");
    let config = RuntimeConfig { checkpoint_every_events: 0, ..Default::default() };

    let store_before_crash = Arc::new(SqliteEventStore::open(&database).await.unwrap());
    let runtime_before_crash =
        HorizonRuntime::with_config(Arc::clone(&store_before_crash), config.clone());
    let run_id = runtime_before_crash
        .create_run("resume from a separate process")
        .await
        .unwrap()
        .projection
        .run_id;
    runtime_before_crash
        .dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Planning })
        .await
        .unwrap();
    runtime_before_crash
        .dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Executing })
        .await
        .unwrap();
    let task = TaskSpec::new("interrupted task");
    let task_id = task.id;
    runtime_before_crash.dispatch(run_id, RuntimeCommand::CreateTask { task }).await.unwrap();
    runtime_before_crash.dispatch(run_id, RuntimeCommand::StartTask { task_id }).await.unwrap();
    runtime_before_crash.checkpoint(run_id).await.unwrap();
    let last_before_crash = store_before_crash.last_sequence(run_id).await.unwrap();
    drop(runtime_before_crash);
    drop(store_before_crash);

    let store_after_crash = Arc::new(SqliteEventStore::open(&database).await.unwrap());
    let runtime_after_crash = HorizonRuntime::with_config(Arc::clone(&store_after_crash), config);
    let recovered = runtime_after_crash.recover(run_id).await.unwrap();
    assert!(recovered.restored_from_checkpoint.is_some());
    assert_eq!(recovered.projection.state, AgentState::Recovering);
    assert_eq!(recovered.projection.tasks[&task_id].status, TaskStatus::Pending);
    assert!(
        recovered
            .projection
            .cognitive
            .failed_attempts
            .iter()
            .any(|failure| failure.reason.contains("interrupted"))
    );
    assert!(store_after_crash.last_sequence(run_id).await.unwrap() > last_before_crash);

    runtime_after_crash
        .dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Planning })
        .await
        .unwrap();
    runtime_after_crash
        .dispatch(run_id, RuntimeCommand::Transition { to: AgentState::Executing })
        .await
        .unwrap();
    runtime_after_crash.dispatch(run_id, RuntimeCommand::StartTask { task_id }).await.unwrap();
    runtime_after_crash
        .dispatch(
            run_id,
            RuntimeCommand::SucceedTask { task_id, output: "continued after restart".into() },
        )
        .await
        .unwrap();
    assert_eq!(
        runtime_after_crash.projection(run_id).await.unwrap().tasks[&task_id].status,
        TaskStatus::Succeeded
    );
}

#[tokio::test]
async fn process_receives_stable_operation_id() {
    let runtime = Arc::new(HorizonRuntime::with_config(
        Arc::new(SqliteEventStore::in_memory().await.unwrap()),
        RuntimeConfig { checkpoint_every_events: 0, ..Default::default() },
    ));
    let run_id = runtime.create_run("test idempotency key").await.unwrap().projection.run_id;
    let mut task = TaskSpec::new("read stable operation id");
    task.operation_id = Some("external-operation-42".into());
    task.command =
        Some(vec!["sh".into(), "-c".into(), "printf '%s' \"$HORIZON_OPERATION_ID\"".into()]);
    let task_id = task.id;
    runtime.dispatch(run_id, RuntimeCommand::CreateTask { task }).await.unwrap();
    let result = runtime.execute_ready_tasks(run_id).await.unwrap();
    assert!(result[0].output.as_ref().unwrap().contains("external-operation-42"));
    let projection = runtime.projection(run_id).await.unwrap();
    assert!(projection.is_operation_completed("external-operation-42"));
    assert_eq!(projection.tasks[&task_id].status, TaskStatus::Succeeded);
}

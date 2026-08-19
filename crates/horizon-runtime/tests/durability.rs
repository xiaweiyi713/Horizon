use std::sync::Arc;

use chrono::Utc;
use horizon_core::{
    AgentState, EventId, InterventionAction, InterventionAssessment, MemoryId, MemoryItem,
    MemoryKind, RuntimeCommand, StateDecaySignals, TaskSpec, TaskStatus,
};
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

#[tokio::test]
async fn learned_intervention_assessments_survive_checkpoint_and_event_suffix_replay() {
    let directory = tempdir().unwrap();
    let database = directory.path().join("learned-intervention.db");
    let config = RuntimeConfig { checkpoint_every_events: 0, ..Default::default() };
    let store_before_restart = Arc::new(SqliteEventStore::open(&database).await.unwrap());
    let runtime_before_restart =
        HorizonRuntime::with_config(Arc::clone(&store_before_restart), config.clone());
    let run_id = runtime_before_restart
        .create_run("replay learned intervention audit state")
        .await
        .unwrap()
        .projection
        .run_id;
    runtime_before_restart
        .apply_intervention(
            run_id,
            InterventionAssessment {
                policy_id: "logistic_state_decay".into(),
                policy_version: Some("durability-fixture-v1".into()),
                risk_score_milli: 900,
                threshold_milli: 600,
                signals: StateDecaySignals {
                    steps_since_anchor: 1,
                    context_pressure: 0.8,
                    subgoal_switches: 2,
                    recent_failures: 1,
                    recovered_session: false,
                },
                action: InterventionAction::InjectAnchor,
                reason: "fixture risk requires anchor".into(),
                metadata: serde_json::json!({"estimated_anchor_tokens": 99}),
            },
        )
        .await
        .unwrap();
    runtime_before_restart.checkpoint(run_id).await.unwrap();
    runtime_before_restart
        .apply_intervention(
            run_id,
            InterventionAssessment {
                policy_id: "logistic_state_decay".into(),
                policy_version: Some("durability-fixture-v1".into()),
                risk_score_milli: 150,
                threshold_milli: 600,
                signals: StateDecaySignals { steps_since_anchor: 1, ..Default::default() },
                action: InterventionAction::Continue,
                reason: "fixture risk stays below threshold".into(),
                metadata: serde_json::json!({"estimated_compact_tokens": 24}),
            },
        )
        .await
        .unwrap();
    drop(runtime_before_restart);
    drop(store_before_restart);

    let store_after_restart = Arc::new(SqliteEventStore::open(&database).await.unwrap());
    let runtime_after_restart = HorizonRuntime::with_config(store_after_restart, config);
    let projection = runtime_after_restart.projection(run_id).await.unwrap();
    assert_eq!(projection.intervention_assessments, 2);
    assert_eq!(
        projection.last_intervention_assessment.as_ref().unwrap().action,
        InterventionAction::Continue
    );
    assert!(projection.last_anchor_sequence > 0);
    let events = runtime_after_restart.events(run_id).await.unwrap();
    assert_eq!(
        events.iter().filter(|event| event.event.name() == "state_decay_assessed").count(),
        2
    );
}

#[tokio::test]
async fn semantic_memory_catalog_and_retrieval_survive_checkpoint_suffix_replay() {
    let directory = tempdir().unwrap();
    let database = directory.path().join("semantic-memory.db");
    let config = RuntimeConfig { checkpoint_every_events: 0, ..Default::default() };
    let store_before_restart = Arc::new(SqliteEventStore::open(&database).await.unwrap());
    let runtime_before_restart =
        HorizonRuntime::with_config(Arc::clone(&store_before_restart), config.clone());
    let run_id = runtime_before_restart
        .create_run("replay semantic-memory retrieval")
        .await
        .unwrap()
        .projection
        .run_id;
    let memory = MemoryItem {
        id: MemoryId::new(),
        kind: MemoryKind::Episodic,
        content: "Restore PostgreSQL from checkpoint before retrying the migration.".into(),
        source_event: EventId::new(),
        importance: 0.9,
        confidence: 0.95,
        created_at: Utc::now(),
    };
    runtime_before_restart
        .dispatch(run_id, RuntimeCommand::CreateMemory { memory: memory.clone() })
        .await
        .unwrap();
    runtime_before_restart.checkpoint(run_id).await.unwrap();
    runtime_before_restart
        .retrieve_semantic_memory(run_id, "restore postgres checkpoint", 4)
        .await
        .unwrap();
    drop(runtime_before_restart);
    drop(store_before_restart);

    let store_after_restart = Arc::new(SqliteEventStore::open(&database).await.unwrap());
    let runtime_after_restart = HorizonRuntime::with_config(store_after_restart, config);
    let projection = runtime_after_restart.projection(run_id).await.unwrap();
    assert_eq!(projection.semantic_memories.get(&memory.id), Some(&memory));
    assert_eq!(projection.semantic_memory_retrievals, 1);
    let retrieval = projection.last_semantic_memory_retrieval.as_ref().unwrap();
    assert_eq!(retrieval.algorithm, "hybrid_lexical_v1");
    assert_eq!(retrieval.hits[0].memory_id, memory.id);
    let events = runtime_after_restart.events(run_id).await.unwrap();
    assert_eq!(
        events.iter().filter(|event| event.event.name() == "semantic_memory_retrieved").count(),
        1
    );
}

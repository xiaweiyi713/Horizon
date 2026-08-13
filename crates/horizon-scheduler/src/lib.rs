//! Dependency-aware task selection and bounded Tokio execution.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use async_trait::async_trait;
use futures::{StreamExt, stream::FuturesUnordered};
use horizon_core::{TaskId, TaskRecord, TaskSpec, TaskStatus};
use thiserror::Error;
use tokio::sync::Semaphore;

#[derive(Clone, Debug, Default)]
pub struct TaskDag {
    tasks: BTreeMap<TaskId, TaskRecord>,
}

impl TaskDag {
    pub fn from_records(
        records: impl IntoIterator<Item = TaskRecord>,
    ) -> Result<Self, SchedulerError> {
        let tasks = records.into_iter().map(|record| (record.spec.id, record)).collect();
        let dag = Self { tasks };
        dag.validate()?;
        Ok(dag)
    }

    pub fn validate(&self) -> Result<(), SchedulerError> {
        for record in self.tasks.values() {
            for dependency in &record.spec.dependencies {
                if !self.tasks.contains_key(dependency) {
                    return Err(SchedulerError::UnknownDependency {
                        task_id: record.spec.id,
                        dependency: *dependency,
                    });
                }
            }
        }
        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        for id in self.tasks.keys().copied() {
            self.visit(id, &mut visiting, &mut visited)?;
        }
        Ok(())
    }

    fn visit(
        &self,
        id: TaskId,
        visiting: &mut BTreeSet<TaskId>,
        visited: &mut BTreeSet<TaskId>,
    ) -> Result<(), SchedulerError> {
        if visited.contains(&id) {
            return Ok(());
        }
        if !visiting.insert(id) {
            return Err(SchedulerError::Cycle(id));
        }
        let record = self.tasks.get(&id).expect("id originated from task map");
        for dependency in &record.spec.dependencies {
            self.visit(*dependency, visiting, visited)?;
        }
        visiting.remove(&id);
        visited.insert(id);
        Ok(())
    }

    /// Pending tasks whose dependencies have all succeeded, ordered by highest
    /// priority then stable task id for deterministic replay and testing.
    #[must_use]
    pub fn ready(&self) -> Vec<TaskSpec> {
        let mut ready: Vec<_> = self
            .tasks
            .values()
            .filter(|task| task.status == TaskStatus::Pending)
            .filter(|task| {
                task.spec.dependencies.iter().all(|dependency| {
                    self.tasks.get(dependency).is_some_and(|dependency_task| {
                        dependency_task.status == TaskStatus::Succeeded
                    })
                })
            })
            .map(|task| task.spec.clone())
            .collect();
        ready.sort_by(|left, right| {
            right.priority.cmp(&left.priority).then_with(|| left.id.cmp(&right.id))
        });
        ready
    }

    #[must_use]
    pub fn blocked(&self) -> Vec<TaskId> {
        self.tasks
            .values()
            .filter(|task| task.status == TaskStatus::Pending)
            .filter(|task| !self.ready().iter().any(|ready| ready.id == task.spec.id))
            .map(|task| task.spec.id)
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskExecutionResult {
    pub task_id: TaskId,
    pub output: Result<String, String>,
}

#[async_trait]
pub trait TaskExecutor: Send + Sync {
    async fn execute(&self, task: TaskSpec) -> TaskExecutionResult;
}

/// Executes independent tasks with a Tokio semaphore. It does not mutate the
/// event stream; the runtime remains the sole writer of task lifecycle events.
pub struct AsyncTaskScheduler {
    concurrency: usize,
}

impl AsyncTaskScheduler {
    #[must_use]
    pub const fn new(concurrency: usize) -> Self {
        Self { concurrency }
    }

    pub async fn execute<E: TaskExecutor + 'static>(
        &self,
        tasks: Vec<TaskSpec>,
        executor: Arc<E>,
    ) -> Vec<TaskExecutionResult> {
        let semaphore = Arc::new(Semaphore::new(self.concurrency.max(1)));
        let mut pending = FuturesUnordered::new();
        for task in tasks {
            let semaphore = Arc::clone(&semaphore);
            let executor = Arc::clone(&executor);
            pending.push(async move {
                let _permit = semaphore.acquire_owned().await.expect("semaphore stays open");
                executor.execute(task).await
            });
        }
        let mut results = Vec::new();
        while let Some(result) = pending.next().await {
            results.push(result);
        }
        results.sort_by_key(|result| result.task_id);
        results
    }
}

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("task {task_id} depends on unknown task {dependency}")]
    UnknownDependency { task_id: TaskId, dependency: TaskId },
    #[error("task DAG contains a dependency cycle at {0}")]
    Cycle(TaskId),
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use horizon_core::{TaskRecord, TaskSpec, TaskStatus};

    use super::*;

    #[test]
    fn dependent_task_becomes_ready_after_predecessor() {
        let first = TaskSpec::new("first");
        let mut second = TaskSpec::new("second");
        second.dependencies.push(first.id);
        let mut first_record = TaskRecord::from_spec(first, Utc::now());
        first_record.status = TaskStatus::Succeeded;
        let dag = TaskDag::from_records([
            first_record,
            TaskRecord::from_spec(second.clone(), Utc::now()),
        ])
        .unwrap();
        assert_eq!(dag.ready(), vec![second]);
    }

    #[test]
    fn rejects_cycle() {
        let mut first = TaskSpec::new("first");
        let mut second = TaskSpec::new("second");
        first.dependencies.push(second.id);
        second.dependencies.push(first.id);
        let error = TaskDag::from_records([
            TaskRecord::from_spec(first, Utc::now()),
            TaskRecord::from_spec(second, Utc::now()),
        ])
        .unwrap_err();
        assert!(matches!(error, SchedulerError::Cycle(_)));
    }
}

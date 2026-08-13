use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Lifecycle states owned by the runtime, not by an LLM policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentState {
    #[default]
    Created,
    Planning,
    Executing,
    Waiting,
    Evaluating,
    Recovering,
    Suspended,
    Completed,
    Failed,
}

impl AgentState {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }

    /// Returns whether a transition is legal. Recovery is intentionally allowed
    /// from every non-terminal state: a process may disappear at any boundary.
    #[must_use]
    pub const fn can_transition_to(self, target: Self) -> bool {
        use AgentState::*;

        matches!(
            (self, target),
            (Created, Planning | Suspended | Recovering | Failed)
                | (Planning, Executing | Waiting | Suspended | Recovering | Failed)
                | (Executing, Waiting | Evaluating | Planning | Recovering | Suspended | Failed)
                | (Waiting, Executing | Planning | Recovering | Suspended | Failed)
                | (Evaluating, Completed | Planning | Recovering | Suspended | Failed)
                | (Recovering, Planning | Executing | Suspended | Failed)
                | (Suspended, Recovering | Failed)
                | (Completed, Completed)
                | (Failed, Failed)
        )
    }

    pub fn validate_transition(self, target: Self) -> Result<(), StateTransitionError> {
        if self.can_transition_to(target) {
            Ok(())
        } else {
            Err(StateTransitionError { from: self, to: target })
        }
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("invalid agent state transition: {from:?} -> {to:?}")]
pub struct StateTransitionError {
    pub from: AgentState,
    pub to: AgentState,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_is_legal() {
        let path = [
            AgentState::Created,
            AgentState::Planning,
            AgentState::Executing,
            AgentState::Evaluating,
            AgentState::Completed,
        ];

        for states in path.windows(2) {
            assert!(states[0].can_transition_to(states[1]));
        }
    }

    #[test]
    fn created_cannot_complete_directly() {
        assert_eq!(
            AgentState::Created.validate_transition(AgentState::Completed),
            Err(StateTransitionError { from: AgentState::Created, to: AgentState::Completed })
        );
    }
}

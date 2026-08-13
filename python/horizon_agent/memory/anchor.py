"""Local bookkeeping for risk signals sent to the Rust State Anchor policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class InterventionTracker:
    """Tracks observable Python-side signals without duplicating Rust scoring."""

    context_budget_tokens: Optional[int] = None
    tokens_used: int = 0
    subgoal_switches: int = 0
    recent_failures: int = 0
    recovered_session: bool = False

    def context_pressure(self) -> float:
        if not self.context_budget_tokens:
            return 0.0
        return max(0.0, min(1.0, self.tokens_used / self.context_budget_tokens))

    def run_budget_pressure(self) -> float:
        """Fraction of the configured total agent token budget consumed.

        ``context_pressure`` remains the backwards-compatible signal name sent
        to the Rust heuristic. Learned controllers can use this explicit name
        alongside their estimate of the immediate prompt's context pressure.
        """
        return self.context_pressure()

    def note_response(self, input_tokens: int, output_tokens: int) -> None:
        self.tokens_used += max(0, input_tokens) + max(0, output_tokens)

    def note_subgoal_switch(self) -> None:
        self.subgoal_switches += 1

    def note_failure(self) -> None:
        self.recent_failures += 1

    def consume_boundary(self, *, steps_since_anchor: int = 0) -> dict[str, object]:
        """Return signals for one boundary and decay short-lived indicators."""
        result = {
            "steps_since_anchor": max(0, steps_since_anchor),
            "context_pressure": self.context_pressure(),
            "subgoal_switches": self.subgoal_switches,
            "recent_failures": self.recent_failures,
            "recovered_session": self.recovered_session,
        }
        self.subgoal_switches = 0
        self.recent_failures = 0
        self.recovered_session = False
        return result

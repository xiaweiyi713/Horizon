"""Provider protocol kept deliberately small for model-agnostic evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ProviderResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Optional[Any] = None


class LlmProvider(Protocol):
    """Minimal chat-completion interface consumed by :class:`DurableAgent`."""

    def complete(self, messages: Sequence[ChatMessage], *, json_mode: bool = True) -> ProviderResponse:
        """Return one model response, preferably a JSON object when requested."""

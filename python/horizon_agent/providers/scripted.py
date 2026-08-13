"""Deterministic provider for demos, regression tests, and benchmark fixtures."""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Iterable, Sequence, Union

from .base import ChatMessage, ProviderResponse


ScriptedItem = Union[str, dict[str, Any], ProviderResponse]


class ScriptedProvider:
    """Return a finite, deterministic sequence of responses.

    This lets the Python layer be tested without a network call or an API key.
    ``history`` retains prompts for assertions about conditional State Anchors.
    """

    def __init__(self, responses: Iterable[ScriptedItem]) -> None:
        self._responses = deque(responses)
        self.history: list[tuple[ChatMessage, ...]] = []

    def complete(self, messages: Sequence[ChatMessage], *, json_mode: bool = True) -> ProviderResponse:
        del json_mode
        self.history.append(tuple(messages))
        if not self._responses:
            raise RuntimeError("ScriptedProvider exhausted its response sequence")
        response = self._responses.popleft()
        if isinstance(response, ProviderResponse):
            return response
        if isinstance(response, dict):
            return ProviderResponse(content=json.dumps(response))
        return ProviderResponse(content=response)

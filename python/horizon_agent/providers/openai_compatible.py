"""OpenAI-compatible provider implemented with Python's standard library."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import ChatMessage, ProviderResponse


class OpenAICompatibleProvider:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint.

    It is intentionally generic: use ``base_url`` and ``model`` to point at a
    compatible local or hosted provider. No SDK dependency is required.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def complete(self, messages: Sequence[ChatMessage], *, json_mode: bool = True) -> ProviderResponse:
        if not self.api_key:
            raise RuntimeError(
                "No API key supplied. Pass api_key=... or set OPENAI_API_KEY before using "
                "OpenAICompatibleProvider."
            )
        payload: Mapping[str, Any] = {
            "model": self.model,
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "temperature": self.temperature,
        }
        if json_mode:
            payload = dict(payload)
            payload["response_format"] = {"type": "json_object"}
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError("LLM endpoint returned HTTP {}: {}".format(error.code, detail)) from error
        except URLError as error:
            raise RuntimeError("could not reach LLM endpoint: {}".format(error.reason)) from error
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("LLM endpoint returned an unexpected response: {}".format(body)) from error
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in content
            )
        usage = body.get("usage") or {}
        return ProviderResponse(
            content=str(content),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            raw=body,
        )

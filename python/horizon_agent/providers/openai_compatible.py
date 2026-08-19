"""OpenAI-compatible provider implemented with Python's standard library."""

from __future__ import annotations

import json
import math
import os
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import ChatMessage, ProviderResponse

_ALLOWED_DECODING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "stop",
        "presence_penalty",
        "frequency_penalty",
    }
)
_RESERVED_REQUEST_KEYS = ("model", "messages", "response_format")
_NUMERIC_DECODING_KEYS = frozenset(
    {"temperature", "top_p", "presence_penalty", "frequency_penalty"}
)
_INTEGER_DECODING_KEYS = frozenset({"seed", "max_tokens", "max_completion_tokens"})
_POSITIVE_INT_DECODING_KEYS = frozenset({"max_tokens", "max_completion_tokens"})


def _validate_decoding(decoding: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a detached copy of allowed decoding controls."""

    if decoding is None:
        return {}
    if not isinstance(decoding, Mapping):
        raise ValueError("decoding must be a mapping of request controls")

    reserved = [key for key in _RESERVED_REQUEST_KEYS if key in decoding]
    if reserved:
        raise ValueError(
            "decoding must not override model, messages, or response_format (got {})".format(
                ", ".join(reserved)
            )
        )

    unknown = sorted(str(key) for key in decoding if key not in _ALLOWED_DECODING_KEYS)
    if unknown:
        raise ValueError("decoding contains unrecognized fields: {}".format(", ".join(unknown)))

    validated: dict[str, Any] = {}
    for key, value in decoding.items():
        validated[key] = _validate_decoding_value(key, value)
    return validated


def _validate_decoding_value(key: str, value: Any) -> Any:
    if key in _NUMERIC_DECODING_KEYS:
        return _require_finite_number(value, key)
    if key in _INTEGER_DECODING_KEYS:
        number = _require_int(value, key)
        if key in _POSITIVE_INT_DECODING_KEYS and number <= 0:
            raise ValueError("decoding {} must be a positive integer".format(key))
        return number
    if key == "stop":
        return _validate_stop(value)
    raise ValueError("decoding contains unrecognized fields: {}".format(key))


def _require_finite_number(value: Any, field: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("decoding {} must be a finite number, not a boolean or non-numeric value".format(field))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("decoding {} must be a finite number".format(field))
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("decoding {} must be an integer, not a boolean or non-integer value".format(field))
    return value


def _validate_stop(value: Any) -> Any:
    if isinstance(value, str):
        if not value:
            raise ValueError(
                "decoding stop must be a non-empty string or a non-empty sequence of non-empty strings"
            )
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = [item for item in value]
        if not items or any(not isinstance(item, str) or not item for item in items):
            raise ValueError(
                "decoding stop must be a non-empty string or a non-empty sequence of non-empty strings"
            )
        return tuple(items)
    raise ValueError(
        "decoding stop must be a non-empty string or a non-empty sequence of non-empty strings"
    )


def _detach_decoding(decoding: Mapping[str, Any]) -> dict[str, Any]:
    detached = dict(decoding)
    stop = detached.get("stop")
    if isinstance(stop, (list, tuple)):
        detached["stop"] = list(stop)
    return detached


class OpenAICompatibleProvider:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint.

    It is intentionally generic: use ``base_url`` and ``model`` to point at a
    compatible local or hosted provider. No SDK dependency is required.
    Optional ``decoding`` forwards a detached, validated subset of sampling
    controls; it cannot override ``model``, ``messages``, or ``response_format``.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        temperature: float = 0.0,
        decoding: Mapping[str, Any] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.decoding = None if decoding is None else MappingProxyType(_validate_decoding(decoding))

    def complete(self, messages: Sequence[ChatMessage], *, json_mode: bool = True) -> ProviderResponse:
        if not self.api_key:
            raise RuntimeError(
                "No API key supplied. Pass api_key=... or set OPENAI_API_KEY before using "
                "OpenAICompatibleProvider."
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "temperature": self.temperature,
        }
        if self.decoding:
            payload.update(_detach_decoding(_validate_decoding(self.decoding)))
        if json_mode:
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

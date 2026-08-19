"""Unit tests for OpenAI-compatible frozen decoding control transmission."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Mapping, Optional
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from horizon_agent.providers.base import ChatMessage  # noqa: E402
from horizon_agent.providers.openai_compatible import OpenAICompatibleProvider  # noqa: E402


_COMPLETION = {
    "choices": [{"message": {"content": '{"ok": true}'}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
}


class _FakeResponse:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._encoded = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._encoded

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class OpenAICompatibleDecodingTests(unittest.TestCase):
    def test_constructor_keeps_existing_positional_and_keyword_forms(self) -> None:
        positional = OpenAICompatibleProvider("demo-model")
        self.assertEqual(positional.model, "demo-model")
        self.assertEqual(positional.temperature, 0.0)
        self.assertIsNone(positional.decoding)

        keyword = OpenAICompatibleProvider(
            model="demo-model",
            api_key="sk-test",
            base_url="http://127.0.0.1:9/v1/",
            timeout=1.5,
            temperature=0.25,
        )
        self.assertEqual(keyword.api_key, "sk-test")
        self.assertEqual(keyword.base_url, "http://127.0.0.1:9/v1")
        self.assertEqual(keyword.timeout, 1.5)
        self.assertEqual(keyword.temperature, 0.25)
        self.assertIsNone(keyword.decoding)

    def test_default_request_remains_backward_compatible(self) -> None:
        payload = self._complete_payload(OpenAICompatibleProvider("demo-model", api_key="sk-test"))
        self.assertEqual(
            payload,
            {
                "model": "demo-model",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
        )

    def test_constructor_temperature_is_used_when_decoding_omits_it(self) -> None:
        provider = OpenAICompatibleProvider(
            "demo-model",
            api_key="sk-test",
            temperature=0.3,
            decoding={"max_tokens": 64, "seed": 11},
        )
        self.assertEqual(
            self._complete_payload(provider),
            {
                "model": "demo-model",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.3,
                "max_tokens": 64,
                "seed": 11,
                "response_format": {"type": "json_object"},
            },
        )

    def test_decoding_temperature_controls_payload_and_forwards_allowed_keys(self) -> None:
        provider = OpenAICompatibleProvider(
            "demo-model",
            api_key="sk-test",
            temperature=0.1,
            decoding={
                "temperature": 0.4,
                "top_p": 0.9,
                "max_tokens": 256,
                "max_completion_tokens": 128,
                "seed": 7,
                "stop": ["END", "STOP"],
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
            },
        )
        self.assertEqual(
            self._complete_payload(provider),
            {
                "model": "demo-model",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.4,
                "top_p": 0.9,
                "max_tokens": 256,
                "max_completion_tokens": 128,
                "seed": 7,
                "stop": ["END", "STOP"],
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
                "response_format": {"type": "json_object"},
            },
        )

    def test_json_mode_keeps_response_format_and_decoding_cannot_override_it(self) -> None:
        provider = OpenAICompatibleProvider(
            "demo-model",
            api_key="sk-test",
            decoding={"stop": "END"},
        )
        self.assertEqual(
            self._complete_payload(provider)["response_format"],
            {"type": "json_object"},
        )
        self.assertNotIn(
            "response_format",
            self._complete_payload(provider, json_mode=False),
        )
        with patch("horizon_agent.providers.openai_compatible.urlopen") as mocked:
            with self.assertRaisesRegex(ValueError, "response_format"):
                OpenAICompatibleProvider(
                    "demo-model",
                    api_key="sk-test",
                    decoding={"response_format": {"type": "text"}},
                )
            mocked.assert_not_called()

    def test_request_uses_detached_validated_mapping(self) -> None:
        controls: dict[str, Any] = {"temperature": 0.4, "stop": ["END"]}
        provider = OpenAICompatibleProvider("demo-model", api_key="sk-test", decoding=controls)
        controls["temperature"] = 0.9
        controls["stop"].append("OTHER")
        controls["max_tokens"] = 8
        self.assertEqual(self._complete_payload(provider)["temperature"], 0.4)
        self.assertEqual(self._complete_payload(provider)["stop"], ["END"])
        self.assertNotIn("max_tokens", self._complete_payload(provider))

        with self.assertRaises(TypeError):
            provider.decoding["temperature"] = 1.0  # type: ignore[index]

    def test_invalid_controls_fail_before_network(self) -> None:
        cases: list[tuple[Any, str]] = [
            (["temperature", 0.2], "mapping"),
            ({"n": 1}, "unrecognized"),
            ({"logit_bias": {}}, "unrecognized"),
            ({"model": "other-model"}, "model, messages, or response_format"),
            ({"messages": [{"role": "user", "content": "nope"}]}, "model, messages, or response_format"),
            ({"response_format": {"type": "text"}}, "model, messages, or response_format"),
            ({"temperature": True}, "boolean"),
            ({"top_p": False}, "boolean"),
            ({"presence_penalty": float("nan")}, "finite"),
            ({"frequency_penalty": float("inf")}, "finite"),
            ({"seed": True}, "boolean"),
            ({"max_tokens": 0}, "positive"),
            ({"max_tokens": -3}, "positive"),
            ({"max_completion_tokens": 0}, "positive"),
            ({"max_tokens": 1.5}, "integer"),
            ({"stop": ""}, "stop"),
            ({"stop": []}, "stop"),
            ({"stop": ["ok", ""]}, "stop"),
            ({"stop": ["ok", 1]}, "stop"),
            ({"stop": b"END"}, "stop"),
        ]
        for decoding, pattern in cases:
            with self.subTest(decoding=decoding):
                with patch("horizon_agent.providers.openai_compatible.urlopen") as mocked:
                    with self.assertRaisesRegex(ValueError, pattern):
                        OpenAICompatibleProvider("demo-model", api_key="sk-test", decoding=decoding)
                    mocked.assert_not_called()

    def _complete_payload(
        self,
        provider: OpenAICompatibleProvider,
        *,
        json_mode: bool = True,
        messages: Optional[list[ChatMessage]] = None,
    ) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def fake_urlopen(request: Any, timeout: Any = None) -> _FakeResponse:
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _FakeResponse(_COMPLETION)

        if messages is None:
            messages = [ChatMessage(role="user", content="hello")]
        with patch(
            "horizon_agent.providers.openai_compatible.urlopen",
            side_effect=fake_urlopen,
        ) as mocked:
            provider.complete(messages, json_mode=json_mode)
            mocked.assert_called_once()
        return captured["payload"]


if __name__ == "__main__":
    unittest.main()

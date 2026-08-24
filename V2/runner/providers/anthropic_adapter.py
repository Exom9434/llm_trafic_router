"""Anthropic Messages API 어댑터.

logprob을 주지 않으므로 정확도·일관성 지표로만 참여한다(설계서 4.2절).
system 메시지를 messages 배열이 아니라 top-level `system`으로 보내야 한다.
"""

from __future__ import annotations

from .base import BaseAdapter, RawResult

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"


class AnthropicAdapter(BaseAdapter):
    def _endpoint(self) -> str:
        base = (self.spec.base_url or DEFAULT_BASE_URL).rstrip("/")
        return f"{base}/messages"

    def _headers(self) -> dict:
        return {
            "x-api-key": self.spec.env_key() or "",
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _payload(self, messages, temperature, max_tokens, want_logprobs, top_logprobs) -> dict:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat = [m for m in messages if m["role"] != "system"]
        payload = {
            "model": self.spec.model,
            "messages": chat,
            "max_tokens": max_tokens,
        }
        # Sonnet 5는 temperature를 폐기해 보내면 400을 낸다.
        if self.spec.supports_temperature:
            payload["temperature"] = temperature
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        # Sonnet 5는 adaptive thinking이 기본 ON이라 disabled를 명시해야 한다.
        payload.update(self.spec.extra_body)
        return payload

    def _parse(self, data: dict) -> RawResult:
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage") or {}
        return RawResult(
            text=text or None,
            returned_model=data.get("model"),
            system_fingerprint=None,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )

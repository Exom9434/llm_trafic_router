"""Naver HyperCLOVA X (CLOVA Studio) 어댑터.

OpenAI 스키마와 다른 점이 세 가지다.
  1. 엔드포인트에 모델명이 경로로 들어간다: /v3/chat-completions/{model}
  2. 파라미터 이름이 camelCase다: maxTokens, topP, repeatPenalty
  3. 성공해도 HTTP 200에 status.code로 실패를 알린다.

2026-08 재조사로 확인한 것: v3 경로와 Bearer 인증은 현행이 맞고, temperature는
0.00~1.00으로 0을 허용한다(3월에 걱정하던 제약은 없었다). 반면 반복 억제
파라미터 이름은 repeatPenalty가 아니라 repetitionPenalty다.

남은 미확인: KRW 단가. 요금 페이지가 JS 렌더링이라 값을 못 읽었으므로
콘솔에서 직접 채워 넣어야 한다. 그 전까지 비용 추정에서 이 모델은 빠진다.
"""

from __future__ import annotations

import uuid

from .base import BaseAdapter, RawResult

DEFAULT_BASE_URL = "https://clovastudio.stream.ntruss.com"


class HyperClovaAdapter(BaseAdapter):
    def _endpoint(self) -> str:
        base = (self.spec.base_url or DEFAULT_BASE_URL).rstrip("/")
        return f"{base}/v3/chat-completions/{self.spec.model}"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.spec.env_key()}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": uuid.uuid4().hex,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _payload(self, messages, temperature, max_tokens, want_logprobs, top_logprobs) -> dict:
        payload = {
            "messages": messages,
            "maxTokens": max_tokens,
            "topP": 0.8,
            "topK": 0,
            "repetitionPenalty": 1.1,
            "includeAiFilters": False,
        }
        if self.spec.supports_temperature:
            payload["temperature"] = temperature
        return payload

    def _parse(self, data: dict) -> RawResult:
        status = (data.get("status") or {})
        code = status.get("code")
        if code and str(code) != "20000":
            return RawResult(error=f"clova status {code}: {status.get('message')}")

        result = data.get("result") or {}
        message = result.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):   # 멀티모달 응답 형태 대비
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))

        usage = result.get("usage") or {}
        return RawResult(
            text=content or None,
            returned_model=self.spec.model,
            input_tokens=usage.get("promptTokens") or result.get("inputLength"),
            output_tokens=usage.get("completionTokens") or result.get("outputLength"),
        )

    # ── 스트리밍 (지연 프로브 전용) ──
    #
    # CLOVA Studio는 Accept 헤더로 스트리밍을 켠다. 페이로드에 stream 필드를
    # 넣지 않는다는 점이 OpenAI 계열과 다르다.
    #
    # 주의할 것은 마지막 result 이벤트다. 거기 실린 message.content는 증분이
    # 아니라 전문이다. token 이벤트에서 이미 모은 텍스트에 그것을 더하면
    # 응답이 두 번 들어간다. 그래서 이벤트 이름으로 갈라야 하고, 이름을
    # 못 읽었을 때는 더하지 않는 쪽으로 둔다.

    def _stream_headers(self) -> dict:
        headers = self._headers()
        headers["Accept"] = "text/event-stream"
        return headers

    def _stream_payload(self, payload: dict) -> dict:
        return payload

    def _stream_event(self, event: str, obj: dict, acc: dict) -> str:
        acc.setdefault("returned_model", self.spec.model)

        if event == "result":
            result = obj.get("result") or obj
            usage = result.get("usage") or {}
            acc["input_tokens"] = usage.get("promptTokens") or result.get("inputLength")
            acc["output_tokens"] = usage.get("completionTokens") or result.get("outputLength")
            return ""

        if event != "token":
            return ""

        message = obj.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        return content or ""

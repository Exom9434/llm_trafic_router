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
        return {
            "messages": messages,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": 0.8,
            "topK": 0,
            "repetitionPenalty": 1.1,
            "includeAiFilters": False,
        }

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

"""OpenAI Chat Completions 호환 엔드포인트 어댑터.

OpenAI·Google(호환 엔드포인트)·DeepSeek·Qwen·Upstage가 모두 이 스키마를 쓴다.
차이는 base_url과 인증 헤더뿐이므로 한 클래스로 덮는다.
"""

from __future__ import annotations

from .base import BaseAdapter, RawResult

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatAdapter(BaseAdapter):
    def _endpoint(self) -> str:
        base = (self.spec.base_url or DEFAULT_BASE_URL).rstrip("/")
        return f"{base}/chat/completions"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.spec.env_key()}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages, temperature, max_tokens, want_logprobs, top_logprobs) -> dict:
        payload = {
            "model": self.spec.model,
            "messages": messages,
            # reasoning 모델은 max_tokens를 안 받고 max_completion_tokens를 받는다.
            self.spec.max_tokens_param: max_tokens,
        }
        # temperature를 폐기한 모델에 보내면 400이 난다.
        if self.spec.supports_temperature:
            payload["temperature"] = temperature
        # thinking/reasoning 차단 같은 프로바이더별 파라미터. 안 넣으면 추론 토큰이
        # 출력 상한을 먹어 빈 응답이 온다.
        payload.update(self.spec.extra_body)
        # 미지원 프로바이더에 logprobs를 보내면 400이 난다.
        # "unknown"일 때는 일단 보내 보고, smoke_test가 결과를 확정한다.
        if want_logprobs and self.spec.supports_logprobs in ("yes", "unknown"):
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs
        return payload

    def _parse(self, data: dict) -> RawResult:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}

        first_lp = None
        top = None
        lp = choice.get("logprobs") or {}
        content_lp = lp.get("content") or []
        if content_lp:
            # 직답 프로브에서는 첫 토큰이 곧 답 글자다.
            head = content_lp[0]
            first_lp = head.get("logprob")
            raw_top = head.get("top_logprobs") or []
            top = [
                {"token": t.get("token"), "logprob": t.get("logprob")}
                for t in raw_top
                if t.get("logprob") is not None
            ] or None

        details = usage.get("completion_tokens_details") or {}
        return RawResult(
            text=message.get("content"),
            returned_model=data.get("model"),
            system_fingerprint=data.get("system_fingerprint"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            first_token_logprob=first_lp,
            top_logprobs=top,
        )

    # ── 스트리밍 (지연 프로브 전용) ──

    # usage를 스트림 끝에 붙여 달라는 옵션. OpenAI가 정의했고 호환
    # 엔드포인트 대부분이 따라왔으나 전부는 아니다. 거부하는 프로바이더가
    # 있으면 러너가 시작 점검에서 이 값을 False로 내리고 다시 시도한다.
    # 그 경우 지연 프로브의 토큰 수가 비는데, 지연 지표는 TTFT와 총 소요시간이라
    # 지장이 없다.
    stream_usage = True

    def _stream_payload(self, payload: dict) -> dict:
        payload["stream"] = True
        if self.stream_usage:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _stream_event(self, event: str, obj: dict, acc: dict) -> str:
        if obj.get("model"):
            acc["returned_model"] = obj["model"]
        if obj.get("system_fingerprint"):
            acc["system_fingerprint"] = obj["system_fingerprint"]

        usage = obj.get("usage")
        if usage:
            acc["input_tokens"] = usage.get("prompt_tokens")
            acc["output_tokens"] = usage.get("completion_tokens")
            details = usage.get("completion_tokens_details") or {}
            acc["reasoning_tokens"] = details.get("reasoning_tokens")

        choices = obj.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        return delta.get("content") or ""

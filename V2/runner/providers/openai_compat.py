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
            "temperature": temperature,
            # reasoning 모델은 max_tokens를 안 받고 max_completion_tokens를 받는다.
            self.spec.max_tokens_param: max_tokens,
        }
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

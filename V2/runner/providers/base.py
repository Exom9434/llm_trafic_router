"""어댑터 공통 인터페이스.

프로바이더마다 API 스키마가 다르지만 실험이 필요로 하는 것은 같다.
어댑터는 콜 하나를 쏘고 `RawResult` 하나를 돌려준다. 성공·실패 모두 결과다.
예외를 밖으로 던지지 않고 error 필드에 담는 이유는, 한 콜의 실패가
슬롯 전체를 죽이면 안 되기 때문이다.

SDK 대신 requests로 직접 친다. 상태코드·레이트리밋 헤더·request-id를
로깅 스키마에 그대로 남기려면 원시 응답이 필요하다.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MAX_RETRIES, RETRY_BASE_SLEEP, REQUEST_TIMEOUT  # noqa: E402


@dataclass
class RawResult:
    text: str | None = None
    returned_model: str | None = None
    system_fingerprint: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    # 추론 모델이 보이지 않게 태운 토큰. logprob을 대부분 잃은 뒤
    # 이 값이 '부하 때 추론 예산을 깎는가'를 재는 주력 지표가 됐다.
    reasoning_tokens: int | None = None
    first_token_logprob: float | None = None
    top_logprobs: list[dict] | None = None      # [{"token": "A", "logprob": -0.1}, ...]
    http_status: int | None = None
    retries: int = 0
    error: str | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    endpoint_host: str | None = None
    request_id: str | None = None
    rate_limit: dict | None = None


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
RATE_LIMIT_HEADER_PREFIXES = ("x-ratelimit", "ratelimit", "retry-after", "anthropic-ratelimit")


def collect_rate_limit(headers) -> dict:
    out = {}
    for k, v in headers.items():
        lk = k.lower()
        if any(lk.startswith(p) for p in RATE_LIMIT_HEADER_PREFIXES):
            out[lk] = v
    return out


def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


class BaseAdapter:
    """프로바이더 어댑터의 뼈대. 재시도와 오류 포장을 여기서 처리한다."""

    def __init__(self, spec):
        self.spec = spec
        self.session = requests.Session()

    # 하위 클래스가 구현한다 ────────────────────────────────
    def _endpoint(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict:
        raise NotImplementedError

    def _payload(self, messages, temperature, max_tokens, want_logprobs, top_logprobs) -> dict:
        raise NotImplementedError

    def _parse(self, data: dict) -> RawResult:
        raise NotImplementedError

    # 공통 실행부 ─────────────────────────────────────────
    def chat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        want_logprobs: bool = False,
        top_logprobs: int = 10,
    ) -> RawResult:
        url = self._endpoint()
        payload = self._payload(messages, temperature, max_tokens, want_logprobs, top_logprobs)
        last_error = None
        status = None

        for attempt in range(MAX_RETRIES):
            t0 = time.perf_counter()
            try:
                resp = self.session.post(
                    url, headers=self._headers(), json=payload, timeout=REQUEST_TIMEOUT
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                status = resp.status_code
                rl = collect_rate_limit(resp.headers)
                rid = resp.headers.get("x-request-id") or resp.headers.get("request-id")

                if resp.status_code == 200:
                    result = self._parse(resp.json())
                    result.http_status = 200
                    result.retries = attempt
                    result.total_ms = elapsed_ms
                    result.endpoint_host = host_of(url)
                    result.request_id = rid
                    result.rate_limit = rl or None
                    return result

                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in RETRYABLE_STATUS:
                    break
            except requests.RequestException as e:
                last_error = f"{type(e).__name__}: {e}"

            if attempt < MAX_RETRIES - 1:
                sleep = RETRY_BASE_SLEEP * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep)

        return RawResult(
            error=last_error or "unknown error",
            http_status=status,
            retries=MAX_RETRIES - 1,
            endpoint_host=host_of(url),
        )

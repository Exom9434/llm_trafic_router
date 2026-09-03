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
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MAX_RETRIES, RETRY_BASE_SLEEP, CONNECT_TIMEOUT, READ_TIMEOUT  # noqa: E402


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
        # (연결, 읽기) 초. 읽기 쪽은 총 소요시간이 아니라 바이트가 도착하지
        # 않고 흐른 시간이다. 모델별 근거는 config.py의 read_timeout 주석에 있다.
        self.timeout = (
            CONNECT_TIMEOUT,
            getattr(spec, "read_timeout", None) or READ_TIMEOUT,
        )
        # 모델 하나당 어댑터 하나를 그 모델의 스레드 풀이 공유하므로,
        # 여기서 간격을 지키면 그 모델의 전체 발사 속도가 잡힌다.
        self._rate_lock = threading.Lock()
        self._next_send_at = 0.0

    def _wait_for_slot(self) -> None:
        """분당 요청 상한을 지킨다. 상한이 없으면 그냥 통과한다.

        재시도도 요청이므로 매 시도마다 부른다. 429를 맞고 백오프로
        회수하는 것보다 애초에 안 맞는 편이 낫다. 로그에 오류가 쌓이지
        않고, 실패한 콜이 재개 대상으로 남지도 않는다.
        """
        rpm = getattr(self.spec, "max_rpm", None)
        if not rpm:
            return
        gap = 60.0 / rpm
        while True:
            with self._rate_lock:
                now = time.monotonic()
                if now >= self._next_send_at:
                    self._next_send_at = now + gap
                    return
                sleep = self._next_send_at - now
            time.sleep(sleep)

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
            self._wait_for_slot()
            t0 = time.perf_counter()
            try:
                resp = self.session.post(
                    url, headers=self._headers(), json=payload, timeout=self.timeout
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
            except requests.Timeout as e:
                # 타임아웃 결측은 다른 실패와 성격이 다르다. 오래 걸린 콜부터
                # 사라지므로 결측이 지표와 상관된다(2026-09-03 Qwen 건).
                # 진단이 따로 셀 수 있게 설정값을 오류 문자열에 남긴다.
                last_error = (
                    f"{type(e).__name__}: timeout={self.timeout[1]:.0f}s {e}"
                )
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

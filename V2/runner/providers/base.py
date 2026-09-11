"""어댑터 공통 인터페이스.

프로바이더마다 API 스키마가 다르지만 실험이 필요로 하는 것은 같다.
어댑터는 콜 하나를 쏘고 `RawResult` 하나를 돌려준다. 성공·실패 모두 결과다.
예외를 밖으로 던지지 않고 error 필드에 담는 이유는, 한 콜의 실패가
슬롯 전체를 죽이면 안 되기 때문이다.

SDK 대신 requests로 직접 친다. 상태코드·레이트리밋 헤더·request-id를
로깅 스키마에 그대로 남기려면 원시 응답이 필요하다.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    CONNECT_TIMEOUT,
    MAX_RETRIES,
    READ_TIMEOUT,
    RETRY_BASE_SLEEP,
    STREAM_CHUNK_TIMEOUT,
    STREAM_TOTAL_CAP,
)


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

    # 스트리밍 실행부 ─────────────────────────────────────
    #
    # 지연 프로브 전용이다(설계서 4.1절). 비스트리밍 응답에서는 첫 토큰이
    # 언제 왔는지 알 수 없어 TTFT가 비고, 남는 total_ms에는 생성 시간 전체가
    # 섞인다. 부하의 증거로 쓰려면 대기와 생성을 갈라야 한다.
    #
    # 타임아웃을 둘로 나눈 이유는 2026-09-03에 확인한 것 때문이다. requests의
    # read timeout은 총 소요시간이 아니라 바이트가 도착하지 않고 흐른 시간이라,
    # 청크를 꾸준히 흘리는 프로바이더에서는 콜 길이를 전혀 제한하지 않는다.
    # 21일 무인 실행에서 매달린 연결 하나가 워커를 붙잡으면 그 슬롯이 빈다.
    # 그래서 청크 간 간격(STREAM_CHUNK_TIMEOUT)과 총 소요시간(STREAM_TOTAL_CAP)에
    # 각각 상한을 건다.

    def _stream_payload(self, payload: dict) -> dict:
        """스트리밍용으로 페이로드를 고친다."""
        payload["stream"] = True
        return payload

    def _stream_headers(self) -> dict:
        return self._headers()

    def _stream_event(self, event: str, obj: dict, acc: dict) -> str:
        """SSE 이벤트 하나를 읽어 텍스트 증분을 돌려주고 부수 정보를 acc에 담는다.

        acc에 모으는 것: returned_model, system_fingerprint, input_tokens,
        output_tokens, reasoning_tokens. 하위 클래스가 채운다.
        """
        raise NotImplementedError

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> RawResult:
        url = self._endpoint()
        payload = self._stream_payload(
            self._payload(messages, temperature, max_tokens, False, 0))
        last_error = None
        status = None

        for attempt in range(MAX_RETRIES):
            self._wait_for_slot()
            acc: dict = {}
            chunks: list[str] = []
            ttft_ms = None
            t0 = time.perf_counter()
            resp = None
            try:
                resp = self.session.post(
                    url, headers=self._stream_headers(), json=payload, stream=True,
                    timeout=(CONNECT_TIMEOUT, STREAM_CHUNK_TIMEOUT),
                )
                status = resp.status_code
                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    if resp.status_code not in RETRYABLE_STATUS:
                        break
                    raise _StreamRetry()

                rl = collect_rate_limit(resp.headers)
                rid = resp.headers.get("x-request-id") or resp.headers.get("request-id")
                event = ""
                data_lines: list[str] = []

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if time.perf_counter() - t0 > STREAM_TOTAL_CAP:
                        raise _StreamOverrun()

                    if raw_line is None:
                        continue
                    line = raw_line.rstrip("\r")

                    # 빈 줄이 이벤트의 끝이다. 모아 둔 data 줄을 여기서 넘긴다.
                    if line == "":
                        if data_lines:
                            body = "\n".join(data_lines)
                            data_lines = []
                            name, event = event, ""
                            if body.strip() == "[DONE]":
                                break
                            try:
                                obj = json.loads(body)
                            except json.JSONDecodeError:
                                continue
                            delta = self._stream_event(name, obj, acc)
                            if delta:
                                if ttft_ms is None:
                                    ttft_ms = (time.perf_counter() - t0) * 1000
                                chunks.append(delta)
                                # 작업량을 맞추는 것은 서버 상한의 몫이고, 이건
                                # 그 상한이 안 먹을 때를 위한 폭주 가드다. 콜이
                                # 길어지면 읽기 타임아웃에 걸리고, 그 결측은
                                # 오래 걸린 콜부터 사라져 지표와 상관된다(8.1절).
                                # 청크 하나가 토큰 하나가 아니므로 넉넉히 4배를
                                # 두어 정상 운용에는 걸리지 않게 한다.
                                # ponytail: 끊긴 것을 레코드에 표시하지 않는다.
                                # 서버 상한이 도는 한 발동하지 않아서다. 실제로
                                # 걸리기 시작하면 RawResult에 플래그를 단다.
                                if len(chunks) >= max_tokens * 4:
                                    break
                        continue

                    if line.startswith(":"):          # 주석(하트비트)
                        continue
                    field, _, value = line.partition(":")
                    value = value[1:] if value.startswith(" ") else value
                    if field == "event":
                        event = value
                    elif field == "data":
                        data_lines.append(value)

                total_ms = (time.perf_counter() - t0) * 1000
                text = "".join(chunks)
                return RawResult(
                    text=text or None,
                    returned_model=acc.get("returned_model"),
                    system_fingerprint=acc.get("system_fingerprint"),
                    input_tokens=acc.get("input_tokens"),
                    output_tokens=acc.get("output_tokens"),
                    reasoning_tokens=acc.get("reasoning_tokens"),
                    http_status=200,
                    retries=attempt,
                    ttft_ms=ttft_ms,
                    total_ms=total_ms,
                    endpoint_host=host_of(url),
                    request_id=rid,
                    rate_limit=rl or None,
                )
            except _StreamOverrun:
                last_error = f"StreamOverrun: total>{STREAM_TOTAL_CAP:.0f}s"
            except _StreamRetry:
                pass
            except requests.Timeout as e:
                last_error = f"{type(e).__name__}: chunk_gap={STREAM_CHUNK_TIMEOUT:.0f}s {e}"
            except requests.RequestException as e:
                last_error = f"{type(e).__name__}: {e}"
            finally:
                if resp is not None:
                    resp.close()

            if attempt < MAX_RETRIES - 1:
                sleep = RETRY_BASE_SLEEP * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep)

        return RawResult(
            error=last_error or "unknown stream error",
            http_status=status,
            retries=MAX_RETRIES - 1,
            endpoint_host=host_of(url),
        )


class _StreamRetry(Exception):
    """재시도 가능한 상태코드를 공통 경로로 보내기 위한 내부 신호."""


class _StreamOverrun(Exception):
    """청크는 오는데 끝나지 않는 연결. 총 소요시간 상한에 걸렸다."""

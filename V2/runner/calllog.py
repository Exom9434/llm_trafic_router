"""콜 단위 로깅 스키마와 JSONL 기록기.

설계서 4.3절의 로깅 스키마를 그대로 옮겼다. 한 줄이 한 번의 API 콜이다.
재개(resume)는 이 파일을 되읽어 이미 끝난 콜 키를 모으는 방식으로 한다.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CallRecord:
    # ── 식별 ──
    call_key: str                    # 재개용 고유 키
    run_id: str
    phase: str                       # "calibration" | "main" | "smoke"
    probe: str                       # "quality" | "latency"
    mode: str                        # "direct" | "cot"
    # ── 시각 ──
    ts_utc: str = ""
    ts_local: str = ""
    slot: str = ""                   # 본실험 슬롯 라벨 (보정 패스에서는 빈 값)
    vantage: str = ""                # 측정 지점 라벨
    # ── 대상 ──
    model_key: str = ""
    provider: str = ""
    requested_model: str = ""
    returned_model: str | None = None
    system_fingerprint: str | None = None
    endpoint_host: str | None = None
    # ── 요청 ──
    item_id: str | None = None
    subject: str | None = None
    rep: int = 0
    temperature: float = 0.0
    max_tokens: int = 0
    nonce: str = ""
    # ── 응답 ──
    http_status: int | None = None
    retries: int = 0
    error: str | None = None
    raw_text: str | None = None
    parsed_letter: str | None = None
    gold_letter: str | None = None
    correct: int | None = None
    answer_logprob: float | None = None
    letter_probs: dict | None = None      # 선택지 위 정규화 확률분포
    p_gold: float | None = None           # 정답 글자의 확률
    margin: float | None = None           # 1위-2위 확률 차
    # ── 성능 ──
    ttft_ms: float | None = None
    total_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    tps: float | None = None
    # ── 레이트리밋 헤더 ──
    rate_limit: dict | None = None
    request_id: str | None = None

    def stamp(self, tz_local=None) -> "CallRecord":
        now = datetime.now(timezone.utc)
        self.ts_utc = now.isoformat()
        self.ts_local = now.astimezone(tz_local).isoformat() if tz_local else now.astimezone().isoformat()
        return self


class JsonlLogger:
    """스레드 안전 append 기록기. 한 줄씩 flush 해 중단에도 데이터가 남게 한다."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: CallRecord | dict) -> None:
        payload = asdict(record) if isinstance(record, CallRecord) else record
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def load_done_keys(path: str | Path) -> set[str]:
    """이미 성공적으로 끝난 콜의 call_key 집합. 실패 레코드는 다시 돌린다."""
    p = Path(path)
    if not p.exists():
        return set()
    done: set[str] = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") is None and rec.get("call_key"):
                done.add(rec["call_key"])
    return done


def read_records(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out

"""네트워크 기준선 (사전등록 4.2절, 5.1절 부하 조작 확인과 공통 시각 요인 검정).

서울 서버에서 잰 TTFT에는 세 몫이 섞인다. 서울에서 제공사 서버까지 오가는
망 시간, 제공사 쪽 대기열, 첫 토큰 생성이다. 부하를 반영하는 것은 둘째뿐인데
첫째에도 하루 주기가 있을 수 있다. 미국 팔의 피크 슬롯(12·15 UTC)은 한국시간
21·24시로 국내 인터넷 사용이 가장 많은 저녁이라, 그 시간에 해외 구간이
붐비면 미국 제공사의 TTFT가 늘고 그것이 부하 신호처럼 보인다. 그래서 망
몫만 따로 잰다.

재는 것은 슬롯마다 각 API 호스트에 대한 연결 수립 시간 셋이다.

  dns_ms  이름 풀이. 대부분 캐시에서 나오지만 캐시가 비는 순간을 가르려고 둔다.
  tcp_ms  TCP 연결. 왕복 1회라 ping에 해당하는 값이다.
  tls_ms  TLS 핸드셰이크. 왕복에 서버 쪽 암호 연산이 더해진다.

ICMP ping을 쓰지 않는 이유는 둘이다. 클라우드와 CDN 앞단이 ICMP를 막거나
낮은 우선순위로 처리하는 경우가 많고, 막히지 않더라도 API 요청이 실제로 타는
경로(TCP 443)와 다를 수 있다. TCP 연결 시간은 요청과 같은 경로의 왕복이다.

API 키도 요청 본문도 보내지 않는다. 비용이 없고 제공사 쪽 사용량에 잡히지
않는다. 표준 라이브러리만 쓴다.
"""

from __future__ import annotations

import json
import socket
import ssl
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from config import NET_REPS_PER_SLOT, NET_TIMEOUT, ModelSpec, condition_label

# 어댑터별 기본 주소. ModelSpec.base_url이 비어 있으면 어댑터가 이 값을 쓴다.
# 어댑터 모듈에서 가져오지 않는 이유는 그 모듈들이 requests를 불러오기 때문이다.
_DEFAULT_BASE = {
    "openai_compat": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "hyperclova": "https://clovastudio.stream.ntruss.com",
}


def endpoint_host(model: ModelSpec) -> str:
    base = model.base_url or _DEFAULT_BASE.get(model.adapter) or ""
    host = urlparse(base).hostname
    if not host:
        raise ValueError(f"{model.key}: 호스트를 정할 수 없다 (base_url={model.base_url!r})")
    return host


def hosts_for(models) -> dict[str, list[ModelSpec]]:
    """호스트별로 모델을 묶는다. 같은 호스트를 쓰는 모델은 한 번만 잰다."""
    out: dict[str, list[ModelSpec]] = {}
    for m in models:
        out.setdefault(endpoint_host(m), []).append(m)
    return out


def measure_once(host: str, port: int = 443, timeout: float = NET_TIMEOUT) -> dict:
    """연결 한 번을 열고 닫으며 세 구간의 시간을 잰다. 예외는 기록으로 돌린다."""
    row = {"ip": None, "dns_ms": None, "tcp_ms": None, "tls_ms": None,
           "tls_version": None, "error": None}
    sock = None
    try:
        t0 = time.perf_counter()
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        row["dns_ms"] = (time.perf_counter() - t0) * 1000
        # IPv4를 먼저 쓴다. 서버의 IPv6 경로가 따로 놀면 값이 섞인다.
        infos.sort(key=lambda i: 0 if i[0] == socket.AF_INET else 1)
        family, stype, proto, _, addr = infos[0]
        row["ip"] = addr[0]

        sock = socket.socket(family, stype, proto)
        sock.settimeout(timeout)
        t1 = time.perf_counter()
        sock.connect(addr)
        row["tcp_ms"] = (time.perf_counter() - t1) * 1000

        ctx = ssl.create_default_context()
        t2 = time.perf_counter()
        tls = ctx.wrap_socket(sock, server_hostname=host)
        row["tls_ms"] = (time.perf_counter() - t2) * 1000
        row["tls_version"] = tls.version()
        sock = tls
    except Exception as e:  # noqa: BLE001  망 오류는 전부 기록 대상이다
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
    return row


def probe_slot(models, slot: str, slot_index: int, start: datetime, run_id: str,
               vantage: str, reps: int = NET_REPS_PER_SLOT,
               measure=measure_once) -> list[dict]:
    """한 슬롯의 기준선. 반복을 바깥 고리에 두어 호스트마다 시각을 고르게 섞는다."""
    groups = hosts_for(models)
    rows = []
    for rep in range(reps):
        for host, ms in groups.items():
            r = measure(host)
            rows.append({
                "kind": "net_baseline",
                "run_id": run_id,
                "vantage": vantage,
                "slot": slot,
                "slot_index": slot_index,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "host": host,
                "rep": rep,
                "models": [m.key for m in ms],
                # 조건 라벨도 실행 시각에 박는다(설계서 3.2절). 호스트 하나를
                # 여러 모델이 쓰면 모델마다 라벨이 따로 붙는다.
                "conditions": {m.key: condition_label(m.region, start.hour, start.date())
                               for m in ms},
                **r,
            })
    return rows


def net_log_path(call_log: Path) -> Path:
    """콜 로그 옆에 둔다. 리허설이 --log를 바꾸면 기준선도 따라 갈라진다."""
    return call_log.with_name(call_log.stem + ".net.jsonl")


def logged_slots(path: Path) -> set[str]:
    """이미 기준선을 적은 슬롯. 재시작해서 같은 슬롯을 다시 재지 않는다."""
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.add(json.loads(line)["slot"])
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


if __name__ == "__main__":
    # 서버에서 배선만 확인한다. 로그에는 쓰지 않는다.
    #   python netbase.py
    from config import ALL_MODELS, RETIRED

    live = [m for m in ALL_MODELS if m.key not in {r.key for r in RETIRED}]
    for host, ms in hosts_for(live).items():
        rows = [measure_once(host) for _ in range(NET_REPS_PER_SLOT)]
        ok = [r for r in rows if not r["error"]]
        if not ok:
            print(f"{host:40s} 실패 — {rows[0]['error']}")
            continue
        med = lambda k: sorted(r[k] for r in ok)[len(ok) // 2]
        print(f"{host:40s} TCP {med('tcp_ms'):6.1f}ms  TLS {med('tls_ms'):6.1f}ms  "
              f"DNS {med('dns_ms'):5.1f}ms  {ok[0]['ip']}  ({len(ok)}/{len(rows)})")

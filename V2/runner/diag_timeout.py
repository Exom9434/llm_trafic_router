"""모델별 읽기 타임아웃이 실제로 무는지 확인한다.

2026-09-03에 만들었다. 보정 패스에서 Qwen이 같은 5문항을 14~18번 두들기고
전부 ReadTimeout으로 죽었다. 원인은 REQUEST_TIMEOUT=60초였고, 그 60초가
총 소요시간이 아니라 "바이트가 도착하지 않고 흐른 시간"이라는 점이 문제를
프로바이더마다 다르게 만들었다. DeepSeek은 헤더를 먼저 흘려 203초짜리 콜도
통과시켰고, Qwen은 흘리지 않아 60초에서 잘렸다.

타임아웃이 무는 결측은 다른 결측과 성격이 다르다. 오래 추론한 콜부터
사라지므로 결측이 주력 지표(추론 토큰)와 상관된다. 부하 때 추론이 길어지면
그 콜들이 빠지고, 남은 관측만 보면 추론 토큰이 짧아진 것처럼 보인다.
가설이 예측하는 바로 그 방향이라 가짜 양성이 된다.

이 스크립트는 죽었던 문항을 새 타임아웃으로 다시 쳐서
(1) 회수되는가, (2) 옛 60초였다면 몇 건이 죽었을 것인가를 본다.

    python diag_timeout.py --model qwen_flash
    python diag_timeout.py --model qwen_flash --items physics:9554,math:8181 --reps 5

인자를 안 주면 보정 로그에서 그 모델의 타임아웃 미회수 문항을 스스로 찾는다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
load_dotenv(HERE.parent.parent / ".env")

from config import LINEUP, ANCHORS, CONNECT_TIMEOUT, READ_TIMEOUT  # noqa: E402
from itembank import load_pool  # noqa: E402
from prompts import build_messages, letters_for, parse_letter  # noqa: E402
from providers import build_adapter  # noqa: E402

SPECS = {m.key: m for m in list(LINEUP) + list(ANCHORS)}
DEFAULT_LOG = HERE / "outputs" / "calibration_calls.jsonl"


def find_timeout_items(model_key: str, log: Path) -> list[str]:
    """로그에서 타임아웃을 맞고 끝내 회수되지 않은 문항을 찾는다."""
    hit: set[str] = set()
    ok: set[str] = set()
    with log.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("model_key") != model_key:
                continue
            ck = r.get("call_key")
            if not ck:
                continue
            if r.get("error"):
                if "Timeout" in str(r["error"]):
                    hit.add(ck)
            else:
                ok.add(ck)
    lost = hit - ok
    items = sorted({k.split("|")[3] for k in lost if len(k.split("|")) > 3})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="모델별 타임아웃 확인")
    ap.add_argument("--model", required=True, help="예: qwen_flash")
    ap.add_argument("--items", default=None, help="쉼표로 나눈 item_id. 없으면 로그에서 찾는다")
    ap.add_argument("--reps", type=int, default=3, help="문항당 반복")
    ap.add_argument("--log", default=None)
    ap.add_argument("--old-timeout", type=float, default=60.0,
                    help="이 값이었다면 몇 건이 죽었을지 함께 보고한다")
    args = ap.parse_args()

    spec = SPECS.get(args.model)
    if spec is None:
        raise SystemExit(f"모르는 모델: {args.model}")
    if not spec.env_key():
        raise SystemExit(f"{spec.api_key_env}가 비어 있다")

    if args.items:
        item_ids = [s.strip() for s in args.items.split(",") if s.strip()]
    else:
        log = Path(args.log) if args.log else DEFAULT_LOG
        if not log.exists():
            raise SystemExit(f"로그가 없다: {log}. --items로 직접 지정할 것")
        item_ids = find_timeout_items(args.model, log)
        if not item_ids:
            print(f"{args.model}: 로그에 타임아웃 미회수 문항이 없다. 확인할 것이 없다.")
            return

    pool = {it["item_id"]: it for it in load_pool()}
    missing = [i for i in item_ids if i not in pool]
    if missing:
        raise SystemExit(f"후보 풀에 없는 문항: {missing}")

    adapter = build_adapter(spec)
    read_to = adapter.timeout[1]
    print(f"모델      {spec.key} ({spec.model})")
    print(f"타임아웃  연결 {CONNECT_TIMEOUT:.0f}s · 읽기 {read_to:.0f}s "
          f"(기본 {READ_TIMEOUT:.0f}s)")
    print(f"문항      {len(item_ids)}개 × {args.reps}회 = {len(item_ids) * args.reps}콜")
    print(f"출력 상한 {spec.direct_max_tokens:,}토큰\n")

    print(f"{'item_id':22s} {'rep':>3s} {'초':>7s} {'출력':>7s} {'추론':>7s} "
          f"{'답':>3s} {'정답':>4s} {'옛60s':>6s}")
    print("-" * 70)

    lat: list[float] = []
    fails = 0
    would_die = 0
    for item_id in item_ids:
        item = pool[item_id]
        for rep in range(args.reps):
            msgs = build_messages(item, "direct", nonce=f"diagto-{int(time.time())}-{rep}")
            t0 = time.perf_counter()
            res = adapter.chat(
                msgs,
                temperature=0.0 if rep == 0 else 1.0,
                max_tokens=spec.direct_max_tokens,
            )
            el = time.perf_counter() - t0
            if res.error:
                fails += 1
                print(f"{item_id:22s} {rep:3d} {el:7.1f} {'-':>7s} {'-':>7s} "
                      f"{'-':>3s} {'-':>4s} {'-':>6s}  {res.error[:60]}")
                continue
            lat.append(el)
            cap = spec.direct_max_tokens
            truncated = bool(res.output_tokens and res.output_tokens >= cap - 2)
            letter = parse_letter(res.text, letters_for(item), truncated)
            gold = item.get("answer")
            dead = el > args.old_timeout
            would_die += 1 if dead else 0
            print(f"{item_id:22s} {rep:3d} {el:7.1f} "
                  f"{(res.output_tokens or 0):7,} {(res.reasoning_tokens or 0):7,} "
                  f"{str(letter or '-'):>3s} {str(gold or '-'):>4s} "
                  f"{'죽음' if dead else '':>6s}")

    n = len(lat)
    print("-" * 70)
    print(f"성공 {n}건 · 실패 {fails}건")
    if n:
        lat.sort()
        print(f"지연  p50 {statistics.median(lat):.1f}s · "
              f"p90 {lat[min(n - 1, int(0.9 * (n - 1)))]:.1f}s · 최대 {lat[-1]:.1f}s")
        print(f"옛 {args.old_timeout:.0f}초였다면 {would_die}/{n}건이 죽었다.")
        head = read_to / lat[-1] if lat[-1] else float("inf")
        print(f"새 상한까지 여유 {head:.1f}배.")
    if fails:
        print("\n실패가 남았다. 타임아웃이 아직 무는지 오류 문자열을 볼 것.")
    else:
        print("\n전부 회수됐다. 이 문항들은 은행 재선별 대상이다.")


if __name__ == "__main__":
    main()

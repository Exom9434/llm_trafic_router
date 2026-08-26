"""보정 패스 로그의 오류·파싱실패를 모델별로 분류한다.

무엇이 실패했는지보다 **왜** 실패했는지가 중요하다. 실패가 무작위면
다시 돌려 채우면 그만이지만, 특정 성질의 문항에 몰려 있으면 그 문항들이
빠진 채로 은행이 만들어진다. 특히 추론이 출력 상한을 다 먹어 답이 잘린
경우라면, 잘린 문항이야말로 추론 예산에 가장 민감한 문항이므로 우리가
재려는 신호가 제일 센 것들을 골라 버리게 된다.

실행:
    python diag_errors.py
    python diag_errors.py --log outputs/calibration_calls.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from config import ANCHORS, LINEUP, OUTPUT_DIR

DEFAULT_LOG = OUTPUT_DIR / "calibration_calls.jsonl"
SPECS = {m.key: m for m in list(LINEUP) + list(ANCHORS)}


def normalize(err: str) -> str:
    """비슷한 오류를 한 덩어리로 묶는다. 숫자·id는 지운다."""
    e = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", str(err))
    e = re.sub(r"\d+", "N", e)
    return e[:110]


def main() -> None:
    ap = argparse.ArgumentParser(description="보정 패스 오류 분류")
    ap.add_argument("--log", default=None)
    ap.add_argument("--samples", type=int, default=2, help="오류 유형당 원문 예시 수")
    args = ap.parse_args()

    path = Path(args.log) if args.log else DEFAULT_LOG
    if not path.exists():
        raise SystemExit(f"로그가 없다: {path}")

    total = 0
    ok = collections.Counter()
    errs = collections.Counter()
    err_kinds = collections.defaultdict(collections.Counter)
    err_samples = collections.defaultdict(list)
    parse_fail = collections.Counter()
    truncated = collections.Counter()
    out_max = collections.defaultdict(int)
    http_codes = collections.defaultdict(collections.Counter)
    fail_items = collections.defaultdict(collections.Counter)

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            key = r.get("model_key") or "?"

            if r.get("error"):
                errs[key] += 1
                kind = normalize(r["error"])
                err_kinds[key][kind] += 1
                if len(err_samples[kind]) < args.samples:
                    err_samples[kind].append(str(r["error"])[:200])
                if r.get("http_status"):
                    http_codes[key][r["http_status"]] += 1
                if r.get("item_id"):
                    fail_items[key][r["item_id"]] += 1
                continue

            ok[key] += 1
            out = r.get("output_tokens")
            if out is not None:
                out_max[key] = max(out_max[key], out)
            if r.get("parsed_letter") is None:
                parse_fail[key] += 1
                if r.get("item_id"):
                    fail_items[key][r["item_id"]] += 1
                # 출력 상한을 다 쓰고 답이 안 나왔으면 추론이 자리를 먹은 것이다
                cap = r.get("max_tokens") or 0
                if out is not None and cap and out >= cap - 2:
                    truncated[key] += 1

    print(f"로그: {path}  ({total:,}행)\n")

    hdr = (f"{'모델':28s} {'성공':>8s} {'오류':>7s} {'파싱실패':>9s} "
           f"{'상한소진':>9s} {'출력상한':>9s} {'최대출력':>9s}")
    print(hdr)
    print("-" * 92)
    for key in sorted(set(ok) | set(errs)):
        spec = SPECS.get(key)
        cap = spec.direct_max_tokens if spec else 0
        print(f"{key:28s} {ok[key]:8,} {errs[key]:7,} {parse_fail[key]:9,} "
              f"{truncated[key]:9,} {cap:9,} {out_max[key]:9,}")

    print(f"\n합계 — 성공 {sum(ok.values()):,} · 오류 {sum(errs.values()):,} · "
          f"파싱실패 {sum(parse_fail.values()):,}")

    if errs:
        print("\n" + "=" * 72)
        print("오류 유형별 (재개 시 다시 시도되는 콜들이다)")
        print("=" * 72)
        for key in sorted(err_kinds):
            print(f"\n[{key}]  총 {errs[key]:,}건")
            if http_codes[key]:
                codes = ", ".join(f"{c}: {n:,}" for c, n in http_codes[key].most_common())
                print(f"  HTTP  {codes}")
            for kind, n in err_kinds[key].most_common(5):
                print(f"  {n:6,}건  {kind}")
                for s in err_samples[kind][:1]:
                    print(f"          예: {s}")

    # 실패가 특정 문항에 몰렸는가 — 무작위면 대부분 1회씩 흩어진다
    print("\n" + "=" * 72)
    print("실패가 특정 문항에 몰렸는가")
    print("=" * 72)
    for key in sorted(fail_items):
        c = fail_items[key]
        repeats = sum(1 for n in c.values() if n > 1)
        top = c.most_common(3)
        print(f"{key:28s} 문항 {len(c):4,}개에서 실패, 2회 이상 실패한 문항 {repeats:4,}개")
        if repeats:
            print(f"{'':28s}  최다: " + ", ".join(f"{i}({n}회)" for i, n in top))
    print("\n2회 이상 실패한 문항이 많으면 무작위 장애가 아니라 그 문항의 성질 문제다.")
    print("'상한소진'이 0이 아니면 direct_max_tokens를 올리고 그 모델만 다시 돌릴 것.")


if __name__ == "__main__":
    main()

"""출력 상한을 실제 문항으로 재본다.

보정 패스 1차에서 파싱 실패 5,697건이 났고 거의 전부가 출력 상한 소진이었다.
상한을 2026-08-24 스모크 테스트("파리는 어디의 수도인가")로 정했는데 그 문항은
답이 3~4토큰이면 끝난다. MMLU-Pro의 40~85% 밴드 문항은 그렇지 않다.

넉넉한 상한으로 소수 문항만 돌려 실제 출력 길이 분포를 잰다. 그 분포에서
상한을 정한 뒤 보정 패스를 다시 돌린다. 짐작으로 올리면 같은 실수를 반복한다.

실행:
    python probe_caps.py --items 24
    python probe_caps.py --report-only
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from calllog import JsonlLogger, read_records
from config import DIRECT_TEMPERATURE, OUTPUT_DIR, apply_capabilities, get_models
from core import CallSpec, run_batch
from itembank import load_pool

PROBE_LOG = OUTPUT_DIR / "probe_caps_calls.jsonl"

# 넉넉한 상한. 추론을 못 끄는 모델은 크게 잡는다. 1차에서 Qwen이 상한 768을
# 무시하고 10,730토큰까지 쓴 것을 봤으므로 추론 모델의 꼬리는 매우 길다.
GENEROUS = {"deepseek_v4_flash": 16384, "qwen_flash": 16384}
GENEROUS_DEFAULT = 512      # 추론을 끈 모델도 8~16으로는 확실히 모자랐다


def pct(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round((len(xs) - 1) * q)))]


def main() -> None:
    ap = argparse.ArgumentParser(description="모델별 출력 상한 실측")
    ap.add_argument("--items", type=int, default=24, help="문항 수 (모델당 이만큼 콜)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--report-only", action="store_true", help="콜 없이 기존 로그만 집계")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else PROBE_LOG
    models = apply_capabilities(get_models(args.models, include_anchors=True))
    if not models:
        sys.exit("돌릴 모델이 없다.")

    if not args.report_only:
        pool = load_pool(Path(args.pool) if args.pool else None)
        # 실패가 math에 몰렸으므로 과목을 고루 섞는다.
        by_subject = {}
        for it in pool:
            by_subject.setdefault(it.get("subject") or "?", []).append(it)
        subjects = sorted(by_subject)
        picked, i = [], 0
        while len(picked) < args.items and any(by_subject.values()):
            s = subjects[i % len(subjects)]
            if by_subject[s]:
                picked.append(by_subject[s].pop(0))
            i += 1

        specs = []
        for m in models:
            cap = GENEROUS.get(m.key, GENEROUS_DEFAULT)
            for it in picked:
                specs.append(CallSpec(model=m, item=it, mode="direct",
                                      temperature=DIRECT_TEMPERATURE,
                                      max_tokens=cap, rep=0, phase="probe"))
        print(f"문항 {len(picked)}개 x 모델 {len(models)}개 = {len(specs)}콜")
        logger = JsonlLogger(log_path)
        seen = {"n": 0}

        def on_done(rec):
            seen["n"] += 1
            if seen["n"] % 10 == 0:
                print(f"  {seen['n']}/{len(specs)}", flush=True)

        try:
            run_batch(specs, uuid.uuid4().hex[:12], logger, vantage="probe", on_done=on_done)
        finally:
            logger.close()

    records = [r for r in read_records(log_path) if r.get("phase") == "probe"]
    if not records:
        sys.exit(f"로그가 비었다: {log_path}")

    print(f"\n로그: {log_path} ({len(records)}행)\n")
    hdr = (f"{'모델':28s} {'성공':>5s} {'무응답':>6s} {'p50':>7s} {'p90':>7s} "
           f"{'p99':>7s} {'최대':>7s} {'현재상한':>8s} {'권장':>7s}")
    print(hdr)
    print("-" * 96)

    recommend = {}
    for m in models:
        rs = [r for r in records if r.get("model_key") == m.key and not r.get("error")]
        outs = [r["output_tokens"] for r in rs if r.get("output_tokens") is not None]
        blank = sum(1 for r in rs if r.get("parsed_letter") is None)
        if not outs:
            print(f"{m.key:28s} {len(rs):5d} {blank:6d}  (출력 토큰 미보고)")
            continue
        p99 = pct(outs, 0.99)
        want = max(32, int(p99 * 1.5))
        rec = 1
        while rec < want:
            rec *= 2
        recommend[m.key] = rec
        print(f"{m.key:28s} {len(rs):5d} {blank:6d} {pct(outs,.5):7,} {pct(outs,.9):7,} "
              f"{p99:7,} {max(outs):7,} {m.direct_max_tokens:8,} {rec:7,}")

    print("\n권장값은 p99 x 1.5를 2의 거듭제곱으로 올린 것이다. 꼬리가 길어 여유를 둔다.")
    print("무응답이 0이 아니면 이 상한으로도 모자란 것이니 --items를 늘려 다시 잴 것.")
    if recommend:
        out = OUTPUT_DIR / "probe_caps.json"
        out.write_text(json.dumps(recommend, indent=2), encoding="utf-8")
        print(f"\nconfig.py의 direct_max_tokens에 반영할 값 -> {out}")
        for k, v in recommend.items():
            print(f"  {k:28s} direct_max_tokens={v}")


if __name__ == "__main__":
    main()

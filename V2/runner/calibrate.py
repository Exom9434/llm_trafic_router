"""보정 패스 러너 (설계서 6절).

저부하 시간에 라인업 전체를 후보 풀에 통과시킨다. 두 가지를 한 번에 얻는다.
  (1) 문항별 난이도 → 40~85% 선별의 재료
  (2) 모델별 노이즈 바닥선 → 리뷰어 공격 5.1의 방어 재료

한 문항당 콜은 (temp=0 직답 1회) + (temp>0 직답 k회)다.
중단해도 같은 명령으로 다시 실행하면 남은 콜부터 이어간다.

실행:
    python calibrate.py --pool data/candidate_pool.json --k 5
    python calibrate.py --models openai_gpt4o_mini anthropic_haiku --limit 20
    python calibrate.py --dry-run          # 콜 수·예상 비용만 계산
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))

from calllog import JsonlLogger, load_done_keys, read_records
from config import (
    CALIBRATION_WINDOWS_KST,
    CONSISTENCY_K,
    CONSISTENCY_TEMPERATURE,
    DIRECT_MAX_TOKENS,
    DIRECT_TEMPERATURE,
    OUTPUT_DIR,
    REGION_LABELS,
    apply_capabilities,
    get_models,
    in_window,
)
from core import CallSpec, estimate_cost, run_batch
from itembank import load_pool

DEFAULT_LOG = OUTPUT_DIR / "calibration_calls.jsonl"

# 직답 프로브의 평균 입력 길이 추정치(문항 하나가 대략 이 정도 토큰이다).
# --dry-run 비용 추정에만 쓴다.
EST_INPUT_TOKENS = 320


def check_window(models, force: bool) -> None:
    """지금이 각 모델의 저부하 시간인지 확인한다.

    노이즈 바닥선을 그 모델의 피크 시간에 재면 바닥선 자체가 부하를 먹은
    값이 된다. 본실험에서 대비가 줄어 검정력을 깎으므로, 시간대를 벗어났으면
    막는다. 9개 모델이 동시에 한가한 시각은 없으므로 지역별로 나눠 돌린다.
    """
    hour = datetime.now(KST).hour
    off = {}
    for m in models:
        window = CALIBRATION_WINDOWS_KST.get(m.region)
        if window and not in_window(hour, window):
            off.setdefault(m.region, (window, []))[1].append(m.key)

    if not off:
        return

    print(f"\n현재 시각 KST {hour:02d}시 — 저부하 시간대를 벗어난 모델이 있다.\n")
    for region, (window, keys) in sorted(off.items()):
        label = REGION_LABELS.get(region, region)
        print(f"  {label}({region}) 시간대 KST {window[0]:02d}~{window[1]:02d}시 — {', '.join(keys)}")
    print("\n지역별로 나눠 돌리는 편이 낫다. 예시:")
    for region, window in CALIBRATION_WINDOWS_KST.items():
        label = REGION_LABELS.get(region, region)
        print(f"  KST {window[0]:02d}~{window[1]:02d}시  python calibrate.py --region {region}"
              f"   # {label}")
    if not force:
        print("\n그래도 지금 돌리려면 --force를 붙인다. 노이즈 바닥선이 부하를 먹는다.")
        sys.exit(1)
    print()


def k_for(model, k: int, anchor_k: int) -> int:
    """앵커는 반복을 줄인다.

    반복의 목적은 자기일관성 추정인데, 앵커는 축소 일정에서 자기일관성을
    쓰지 않기로 했다(설계서 7.4절). 그래서 temp0 한 번이면 난이도 기여와
    천장 확인이라는 앵커의 역할이 충족된다. 비용이 1/6로 준다.
    """
    return anchor_k if model.tier == "flagship" else k


def build_call_specs(models, pool, k: int, done: set[str], anchor_k: int = 1) -> list[CallSpec]:
    specs: list[CallSpec] = []
    for model in models:
        model_k = k_for(model, k, anchor_k)
        for item in pool:
            candidates = [
                CallSpec(
                    model=model, item=item, mode="direct",
                    temperature=DIRECT_TEMPERATURE, max_tokens=model.direct_max_tokens,
                    rep=0, phase="calibration",
                )
            ]
            for r in range(1, model_k + 1):
                candidates.append(
                    CallSpec(
                        model=model, item=item, mode="direct",
                        temperature=CONSISTENCY_TEMPERATURE, max_tokens=model.direct_max_tokens,
                        rep=r, phase="calibration",
                    )
                )
            specs.extend(cs for cs in candidates if cs.call_key() not in done)
    return specs


def dry_run_report(models, pool, k: int, specs, anchor_k: int) -> None:
    print(f"문항 {len(pool)}개 | 라인업 콜 {1 + k}회/문항 | 앵커 콜 {1 + anchor_k}회/문항")
    print(f"남은 콜: {len(specs):,}회\n")

    total = 0.0
    estimated = []
    for m in models:
        n = sum(1 for cs in specs if cs.model.key == m.key)
        # 상한이 아니라 실측 출력 토큰을 쓴다. 상한을 쓰면 크게 과대평가된다.
        out = m.measured_output_tokens
        if out is None:
            out = m.direct_max_tokens
            estimated.append(m.key)
        cost = n * (EST_INPUT_TOKENS / 1e6 * m.price_in + out / 1e6 * m.price_out)
        total += cost
        tier = "앵커" if m.tier == "flagship" else "    "
        print(f"  {m.key:28s} {tier} {n:6,}콜  출력 {out:4d}토큰  ~${cost:6.3f}")
    print(f"  {'합계':28s}      {len(specs):6,}콜{'':14s}~${total:6.3f}")

    print(f"\n입력은 문항당 {EST_INPUT_TOKENS}토큰으로 가정했다.")
    print("출력은 2026-08-24 실측 평균이다(diag_reasoning).")
    if estimated:
        print(f"실측값이 없어 상한으로 대체한 모델: {', '.join(estimated)} — 과대평가된다.")
    unpriced = [m.key for m in models if m.price_in == 0 and m.price_out == 0]
    if unpriced:
        print(f"단가 미확인이라 빠진 모델: {', '.join(unpriced)} — 과소평가된다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="보정 패스 실행")
    ap.add_argument("--pool", default=None, help="후보 풀 JSON (기본: data/candidate_pool.json)")
    ap.add_argument("--models", nargs="*", default=None, help="돌릴 모델 키 (기본: 키가 있는 전체)")
    ap.add_argument("--region", choices=["us", "cn", "kr"], default=None,
                    help="지역별로 나눠 돌린다. 보정 패스는 지역마다 저부하 시각이 다르다")
    ap.add_argument("--force", action="store_true",
                    help="저부하 시간대를 벗어나도 강행한다")
    ap.add_argument("--no-anchors", action="store_true", help="flagship 앵커 제외")
    ap.add_argument("--k", type=int, default=CONSISTENCY_K, help="일관성 샘플 반복 수")
    ap.add_argument("--anchor-k", type=int, default=1,
                    help="앵커의 반복 수. 앵커는 자기일관성을 쓰지 않으므로 기본 1이다")
    ap.add_argument("--limit", type=int, default=None, help="후보 풀에서 앞 N개만")
    ap.add_argument("--log", default=None, help="JSONL 로그 경로")
    ap.add_argument("--vantage", default="local", help="측정 지점 라벨")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = apply_capabilities(
        get_models(args.models, include_anchors=not args.no_anchors, region=args.region))
    if not models:
        sys.exit("돌릴 모델이 없다. .env에 API 키가 있는지, --region이 맞는지 확인할 것.")

    missing = [m.key for m in models if not m.env_key()]
    if missing:
        sys.exit(f"API 키 없음: {missing} — .env를 채우거나 --models로 제외할 것.")

    pool = load_pool(Path(args.pool) if args.pool else None)
    if args.limit:
        pool = pool[: args.limit]

    log_path = Path(args.log) if args.log else DEFAULT_LOG
    done = load_done_keys(log_path)
    specs = build_call_specs(models, pool, args.k, done, args.anchor_k)

    print(f"모델: {', '.join(m.key for m in models)}")
    print(f"문항: {len(pool)}개 | 이미 끝난 콜: {len(done)}회")

    if args.dry_run:
        print("\n(dry-run이므로 콜을 쏘지 않는다. 아래 시간대 안내는 참고용이다.)")
        check_window(models, force=True)
        dry_run_report(models, pool, args.k, specs, args.anchor_k)
        return

    if not specs:
        print("남은 콜이 없다. 보정 패스가 이미 끝났다.")
        return

    check_window(models, args.force)

    run_id = uuid.uuid4().hex[:12]
    logger = JsonlLogger(log_path)
    progress = {"n": 0, "err": 0}

    def on_done(rec):
        progress["n"] += 1
        if rec.error:
            progress["err"] += 1
        if progress["n"] % 25 == 0 or progress["n"] == len(specs):
            print(f"  {progress['n']}/{len(specs)} (오류 {progress['err']})", flush=True)

    try:
        run_batch(specs, run_id, logger, vantage=args.vantage, on_done=on_done)
    finally:
        logger.close()

    records = read_records(log_path)
    cost = estimate_cost(records, models)
    print(f"\n로그: {log_path}  ({len(records)}행)")
    print(f"누적 추정 비용: ${cost['_total']:.3f}")
    for key, val in sorted(cost.items()):
        if key != "_total":
            print(f"  {key:28s} ${val:.3f}")
    print("\n다음: python select_bank.py")


if __name__ == "__main__":
    main()

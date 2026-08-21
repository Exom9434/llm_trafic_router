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
from pathlib import Path

from calllog import JsonlLogger, load_done_keys, read_records
from config import (
    CONSISTENCY_K,
    CONSISTENCY_TEMPERATURE,
    DIRECT_MAX_TOKENS,
    DIRECT_TEMPERATURE,
    OUTPUT_DIR,
    apply_capabilities,
    get_models,
)
from core import CallSpec, estimate_cost, run_batch
from itembank import load_pool

DEFAULT_LOG = OUTPUT_DIR / "calibration_calls.jsonl"

# 직답 프로브의 평균 입력 길이 추정치(문항 하나가 대략 이 정도 토큰이다).
# --dry-run 비용 추정에만 쓴다.
EST_INPUT_TOKENS = 320


def build_call_specs(models, pool, k: int, done: set[str]) -> list[CallSpec]:
    specs: list[CallSpec] = []
    for model in models:
        for item in pool:
            candidates = [
                CallSpec(
                    model=model, item=item, mode="direct",
                    temperature=DIRECT_TEMPERATURE, max_tokens=model.direct_max_tokens,
                    rep=0, phase="calibration",
                )
            ]
            for r in range(1, k + 1):
                candidates.append(
                    CallSpec(
                        model=model, item=item, mode="direct",
                        temperature=CONSISTENCY_TEMPERATURE, max_tokens=model.direct_max_tokens,
                        rep=r, phase="calibration",
                    )
                )
            specs.extend(cs for cs in candidates if cs.call_key() not in done)
    return specs


def dry_run_report(models, pool, k: int, specs) -> None:
    per_item_calls = 1 + k
    print(f"모델 {len(models)}개 × 문항 {len(pool)}개 × 콜 {per_item_calls}회")
    print(f"남은 콜: {len(specs)}회")
    total = 0.0
    for m in models:
        n = sum(1 for cs in specs if cs.model.key == m.key)
        cost = n * (EST_INPUT_TOKENS / 1e6 * m.price_in + m.direct_max_tokens / 1e6 * m.price_out)
        total += cost
        print(f"  {m.key:28s} {n:6d}콜  ~${cost:6.3f}")
    print(f"  {'합계':28s} {len(specs):6d}콜  ~${total:6.3f}")
    print("\n(입력 토큰은 문항당 %d개로 가정한 근사치다.)" % EST_INPUT_TOKENS)
    print("주의: 출력은 상한까지 다 쓴다고 가정했다. thinking을 못 끄는 모델은")
    print("      추론 토큰이 더 붙어 실제 비용이 이보다 커진다. HyperCLOVA는 단가")
    print("      미확인이라 아예 빠져 있다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="보정 패스 실행")
    ap.add_argument("--pool", default=None, help="후보 풀 JSON (기본: data/candidate_pool.json)")
    ap.add_argument("--models", nargs="*", default=None, help="돌릴 모델 키 (기본: 키가 있는 전체)")
    ap.add_argument("--no-anchors", action="store_true", help="flagship 앵커 제외")
    ap.add_argument("--k", type=int, default=CONSISTENCY_K, help="일관성 샘플 반복 수")
    ap.add_argument("--limit", type=int, default=None, help="후보 풀에서 앞 N개만")
    ap.add_argument("--log", default=None, help="JSONL 로그 경로")
    ap.add_argument("--vantage", default="local", help="측정 지점 라벨")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = apply_capabilities(get_models(args.models, include_anchors=not args.no_anchors))
    if not models:
        sys.exit("돌릴 모델이 없다. .env에 API 키가 있는지 확인할 것.")

    missing = [m.key for m in models if not m.env_key()]
    if missing:
        sys.exit(f"API 키 없음: {missing} — .env를 채우거나 --models로 제외할 것.")

    pool = load_pool(Path(args.pool) if args.pool else None)
    if args.limit:
        pool = pool[: args.limit]

    log_path = Path(args.log) if args.log else DEFAULT_LOG
    done = load_done_keys(log_path)
    specs = build_call_specs(models, pool, args.k, done)

    print(f"모델: {', '.join(m.key for m in models)}")
    print(f"문항: {len(pool)}개 | 이미 끝난 콜: {len(done)}회")

    if args.dry_run:
        dry_run_report(models, pool, args.k, specs)
        return

    if not specs:
        print("남은 콜이 없다. 보정 패스가 이미 끝났다.")
        return

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

"""추론 조절 파라미터가 실제로 먹는지 재는 진단.

부록 B는 DeepSeek v4가 `reasoning_effort`로 low/high/max를 받는다고 적었고,
부록 C는 "추론을 끄는 파라미터가 없다"고 적었다. 두 기록이 어긋난다. 08-24
실측이 확인한 것은 끌 수 있는가(none이 먹는가)였지 낮출 수 있는가가 아니었을
수 있다. 설계서 4.2절의 차단 가능 여부 표기가 이 답에 달려 있다.

`diag_reasoning.py`와 다른 점은 묻는 질문이다. 그쪽은 차단이 되는가를 묻고,
이쪽은 어떤 값이 받아들여지고 그 값이 추론량을 실제로 바꾸는가를 묻는다.

세 가지를 갈라 본다.

  거절     — HTTP 400. 그 값은 못 쓴다.
  무시     — 200이 오지만 추론량이 기준선과 구별되지 않는다.
             Qwen의 max_tokens가 이 꼴이다. 받아들이고 안 지킨다.
  먹음     — 200이 오고 추론량이 기준선에서 벗어난다.

**반복이 필수다.** 딥시크는 같은 문항 같은 설정에서도 추론 토큰의 문항 내
변동계수가 0.32다. 값마다 한 콜만 쏘면 설정 효과와 콜 간 잡음을 구별할 수
없다. 그래서 값마다 여러 번 쏘고 분포로 비교한다.

문항도 여러 개를 쓴다. 추론량은 문항 난이도에 크게 좌우되므로(딥시크 보정
로그에서 p50 314, p90 4,013) 문항 하나로 재면 그 문항의 성질을 설정 효과로
읽을 수 있다. 같은 문항 집합을 모든 값에 통과시켜 문항 효과를 상쇄한다.

실행:
    uv run diag_effort.py --models deepseek_v4_flash
    uv run diag_effort.py --models deepseek_v4_flash --values low,medium,high,max
    uv run diag_effort.py --models openai_gpt56_luna --values none,minimal,low
    uv run diag_effort.py --models qwen_flash --param enable_thinking --values true,false
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import prompts
from calllog import JsonlLogger
from config import OUTPUT_DIR, get_models
from itembank import load_pool
from providers import build_adapter

DIAG_LOG = OUTPUT_DIR / "diag_effort_calls.jsonl"

# 기준선. 파라미터를 아예 안 보내는 조건이며, 지금 본실험이 도는 상태다.
BASELINE = "__baseline__"


def parse_value(raw: str):
    """CLI 문자열을 API가 받을 값으로 바꾼다."""
    if raw == BASELINE:
        return BASELINE
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if raw.strip().startswith("{"):
        return json.loads(raw)
    return raw.strip()


def pick_items(pool: list[dict], n: int) -> list[dict]:
    """과목을 고루 섞어 n개. 난이도가 과목마다 치우쳐 있어 한 과목만 쓰면 안 된다."""
    by_subject = defaultdict(list)
    for it in pool:
        by_subject[it.get("subject") or "?"].append(it)
    subjects = sorted(by_subject)
    picked, i = [], 0
    while len(picked) < n and any(by_subject.values()):
        s = subjects[i % len(subjects)]
        if by_subject[s]:
            picked.append(by_subject[s].pop(0))
        i += 1
    return picked


def one_call(spec, extra: dict, item: dict, max_tokens: int):
    adapter = build_adapter(spec)
    original = spec.extra_body
    spec.extra_body = extra
    try:
        raw = adapter.chat(
            prompts.build_messages(item, "direct", prompts.make_nonce()),
            temperature=0.0,
            max_tokens=max_tokens,
            want_logprobs=False,
        )
    finally:
        spec.extra_body = original
    letter = None
    if not raw.error:
        cut = raw.output_tokens is not None and raw.output_tokens >= max_tokens - 2
        letter = prompts.parse_letter(raw.text, prompts.letters_for(item), truncated=cut)
    return raw, letter


def summarize(values: list[float]):
    if not values:
        return None
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def verdict(base: list[int], test: list[int]) -> str:
    """기준선과 구별되는가.

    표본이 작으므로 검정을 하지 않는다. 대신 범위가 겹치는지를 본다.
    겹치지 않으면 먹은 것이고, 겹치면 이 표본으로는 구별할 수 없다는 뜻이다.
    후자를 '무시'라고 단정하지 않고 '구별 안 됨'이라고 적는 이유다.
    """
    if not base or not test:
        return "판정 불가"
    if max(test) < min(base) or min(test) > max(base):
        return "먹음 (범위가 안 겹침)"
    bm, tm = statistics.median(base), statistics.median(test)
    if bm > 0 and abs(tm - bm) / bm >= 0.5:
        return "먹는 듯 (중앙값 50% 이상 차이, 범위는 겹침)"
    return "구별 안 됨"


def main() -> None:
    ap = argparse.ArgumentParser(description="추론 조절 파라미터 실측")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--param", default="reasoning_effort",
                    help="시험할 파라미터 이름 (기본 reasoning_effort)")
    ap.add_argument("--values", default="none,low,medium,high,max",
                    help="쉼표로 구분한 값 목록. 기준선(파라미터 미전송)은 항상 함께 잰다.")
    ap.add_argument("--items", type=int, default=2, help="문항 수")
    ap.add_argument("--reps", type=int, default=3, help="문항·값마다 반복 횟수")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="출력 상한. 기본값은 모델의 direct_max_tokens다.")
    ap.add_argument("--pool", default=None)
    args = ap.parse_args()

    models = get_models(args.models)
    if not models:
        sys.exit("돌릴 모델이 없다. --models와 .env의 키를 확인할 것.")

    values = [BASELINE] + [parse_value(v) for v in args.values.split(",") if v.strip()]
    items = pick_items(load_pool(Path(args.pool) if args.pool else None), args.items)
    if not items:
        sys.exit("문항 풀이 비었다.")

    logger = JsonlLogger(DIAG_LOG)
    run_id = uuid.uuid4().hex[:12]

    try:
        for spec in models:
            cap = args.max_tokens or spec.direct_max_tokens
            n_calls = len(values) * len(items) * args.reps
            worst = n_calls * cap / 1e6 * spec.price_out
            print(f"\n{'=' * 78}")
            print(f"{spec.key}  ({spec.model})")
            print(f"  파라미터 {args.param} · 값 {len(values)}개 · 문항 {len(items)}개 · 반복 {args.reps}")
            print(f"  {n_calls}콜, 출력 상한 {cap:,}, 최악의 경우 ${worst:.2f}")
            print(f"{'=' * 78}")

            per_value: dict[str, list[int]] = {}
            per_value_out: dict[str, list[int]] = {}
            rejected: dict[str, str] = {}

            for value in values:
                label = "기준선(미전송)" if value is BASELINE else f"{args.param}={value}"
                extra = dict(spec.extra_body)
                if value is not BASELINE:
                    extra[args.param] = value
                reasoning, outputs, letters = [], [], []
                for item in items:
                    for rep in range(args.reps):
                        raw, letter = one_call(spec, extra, item, cap)
                        logger.write({
                            "run_id": run_id, "phase": "diag_effort",
                            "model_key": spec.key, "param": args.param,
                            "value": None if value is BASELINE else value,
                            "item_id": item["item_id"], "rep": rep,
                            "http_status": raw.http_status, "error": raw.error,
                            "raw_text": raw.text, "parsed_letter": letter,
                            "gold_letter": item.get("answer"),
                            "input_tokens": raw.input_tokens,
                            "output_tokens": raw.output_tokens,
                            "reasoning_tokens": raw.reasoning_tokens,
                            "max_tokens": cap, "total_ms": raw.total_ms,
                        })
                        if raw.error:
                            rejected.setdefault(label, raw.error[:160])
                            break
                        if raw.reasoning_tokens is not None:
                            reasoning.append(raw.reasoning_tokens)
                        if raw.output_tokens is not None:
                            outputs.append(raw.output_tokens)
                        letters.append(letter)
                    if label in rejected:
                        break
                if label in rejected:
                    print(f"  {label:<30} 거절: {rejected[label]}")
                    continue
                per_value[label] = reasoning
                per_value_out[label] = outputs
                s = summarize(reasoning) or summarize(outputs)
                field = "추론" if reasoning else "출력"
                ok = sum(1 for x in letters if x)
                print(f"  {label:<30} 200 · {field} 중앙 {s['median']:>7,.0f} "
                      f"(min {s['min']:,} max {s['max']:,}) · 답 읽힘 {ok}/{len(letters)}")

            base_label = "기준선(미전송)"
            base = per_value.get(base_label) or per_value_out.get(base_label, [])
            if base:
                print(f"\n  기준선 대비 판정")
                for label in per_value:
                    if label == base_label:
                        continue
                    test = per_value[label] or per_value_out[label]
                    print(f"    {label:<30} {verdict(base, test)}")
            if rejected:
                print(f"\n  거절된 값: {', '.join(rejected)}")
    finally:
        logger.close()

    print(f"\n원시 로그: {DIAG_LOG}")
    print("\n읽는 법")
    print("  거절        그 값은 못 쓴다. 설계서 표에 그대로 적는다.")
    print("  먹음        추론량이 기준선에서 벗어났다. 우리가 조절할 수 있다는 뜻이다.")
    print("  구별 안 됨  받아들이지만 이 표본으로는 효과를 못 봤다.")
    print("              --reps를 올려 다시 재거나, 효과가 없다고 보고 넘어간다.")
    print("\n조절이 가능하더라도 본실험에서 값을 박을 것인지는 별개 판단이다.")
    print("제공사가 부하 때 추론 예산을 깎는지가 가설인데, 우리가 먼저 깎으면")
    print("관측 대상이 사라진다(설계서 3.4절).")


if __name__ == "__main__":
    main()

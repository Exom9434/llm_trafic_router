"""추론 토큰 실측 진단.

smoke test에서 두 가지가 드러났다. 첫째, DeepSeek과 Upstage는 추론이 출력
상한을 통째로 먹어 답 토큰이 한 개도 안 나왔다(출력 8 = 추론 8, 출력 16 =
추론 16). 둘째, 추론 토큰을 usage에 보고하는 모델과 아예 보고하지 않는
모델이 갈린다.

여기서 설계의 자기모순이 드러난다. 주력 지표를 추론 토큰으로 옮겨 놓고,
비용을 아끼려고 config에서 추론을 끄고 있었다. 추론을 끄면 주력 지표가
항상 0이다. 둘 중 하나를 골라야 하고, 고르려면 숫자가 있어야 한다.

이 스크립트가 그 숫자를 만든다. 모델마다 두 조건으로 한 콜씩 쏜다.

  A. 차단   — 지금 config의 extra_body 그대로
  B. 허용   — 추론 차단 파라미터를 뺀 상태

둘 다 출력 상한을 넉넉히 열어(기본 512) 추론이 잘리지 않게 한다. 그래야
실제 추론 길이를 알 수 있다. 모델당 2콜이므로 비용은 몇 센트다.

실행:
    uv run diag_reasoning.py
    uv run diag_reasoning.py --max-tokens 1024
"""

from __future__ import annotations

import argparse
import sys

import prompts
from config import get_models
from providers import build_adapter

PROBE = {
    "item_id": "diag:0",
    "subject": "diag",
    "question": (
        "A train travels 60 km at 30 km/h, then 60 km at 60 km/h. "
        "What is the average speed for the whole trip?"
    ),
    "options": ["36 km/h", "40 km/h", "45 km/h", "50 km/h"],
    "answer": "B",
    "answer_index": 1,
}

# 추론 차단을 푸는 방법이 프로바이더마다 다르고, 허용값도 문서와 다를 수 있다.
# 2026-08-24 실측에서 OpenAI가 reasoning_effort="low"를 "Unsupported value"로
# 거절했다. 그래서 후보를 여러 개 두고 200이 뜰 때까지 순서대로 시도한다.
ALLOW_REASONING = {
    "reasoning_effort": ["minimal", "low", "medium"],
    "enable_thinking": [True],
    "thinking": [
        {"type": "enabled", "budget_tokens": 1024},
        {"type": "adaptive"},
    ],
}


def allow_variants(extra: dict) -> list[dict]:
    """차단 파라미터를 허용 쪽으로 뒤집은 후보들을 만든다.

    차단 파라미터가 없으면(DeepSeek처럼 끌 수 없는 경우) 빈 목록을 준다.
    """
    keys = [k for k in extra if k in ALLOW_REASONING]
    if not keys:
        return []
    out = []
    # 키가 하나뿐인 경우가 대부분이므로 첫 키의 후보만 훑는다.
    key = keys[0]
    for value in ALLOW_REASONING[key]:
        variant = dict(extra)
        variant[key] = value
        out.append(variant)
    return out


def run(spec, extra: dict, max_tokens: int):
    adapter = build_adapter(spec)
    original = spec.extra_body
    spec.extra_body = extra
    try:
        raw = adapter.chat(
            prompts.build_messages(PROBE, "direct", prompts.make_nonce()),
            temperature=0.0,
            max_tokens=max_tokens,
            want_logprobs=False,
        )
    finally:
        spec.extra_body = original
    letter = prompts.parse_letter(raw.text, prompts.letters_for(PROBE)) if not raw.error else None
    return raw, letter


def cost(spec, raw) -> float:
    return ((raw.input_tokens or 0) / 1e6 * spec.price_in
            + (raw.output_tokens or 0) / 1e6 * spec.price_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="추론 토큰 실측")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="추론이 잘리지 않게 넉넉히 연다")
    args = ap.parse_args()

    models = get_models(args.models)
    if not models:
        sys.exit("돌릴 모델이 없다.")

    print(f"출력 상한 {args.max_tokens}토큰. 정답은 B(40 km/h)다.\n")
    header = f"{'모델':<26} {'조건':<6} {'답':<6} {'출력':>6} {'추론':>6} {'지연ms':>8} {'콜당$':>10}"
    print(header)
    print("-" * len(header))

    def show(key, label, raw, letter, note=""):
        rt = raw.reasoning_tokens
        print(f"{key:<26} {label:<6} {(letter or '실패'):<6} "
              f"{(raw.output_tokens if raw.output_tokens is not None else -1):>6} "
              f"{(rt if rt is not None else -1):>6} "
              f"{(raw.total_ms or 0):>8.0f} {cost_of[key](raw):>10.6f}  {note}")

    cost_of = {m.key: (lambda r, s=m: cost(s, r)) for m in models}

    for spec in models:
        raw, letter = run(spec, spec.extra_body, args.max_tokens)
        if raw.error:
            print(f"{spec.key:<26} {'차단':<6} 실패: {raw.error[:70]}")
        else:
            show(spec.key, "차단", raw, letter)

        variants = allow_variants(spec.extra_body)
        if not variants:
            print(f"{'':<26} (차단 파라미터가 없다. 추론을 끌 수 없는 모델이다)")
            continue

        for variant in variants:
            key_shown = [f"{k}={v}" for k, v in variant.items() if k in ALLOW_REASONING]
            raw, letter = run(spec, variant, args.max_tokens)
            if raw.error:
                print(f"{spec.key:<26} {'허용':<6} 거절: {', '.join(key_shown)} "
                      f"→ {raw.error[:50]}")
                continue
            show(spec.key, "허용", raw, letter, f"({', '.join(key_shown)})")
            break   # 통과한 첫 후보에서 멈춘다

    print("\n판정 기준")
    print("  추론 = -1 : usage에 추론 토큰 필드가 없다. 이 모델로는 추론 토큰 지표를 못 쓴다.")
    print("  추론 = 0  : 차단이 먹었다.")
    print("  추론 > 0  : 측정 가능하다. 이 값이 본실험의 주력 지표가 된다.")
    print("  답 = 실패 : 추론이 출력 상한을 다 먹었거나 형식이 어긋났다.")
    print("\n'허용' 조건의 출력 토큰이 본실험 비용을 좌우한다.")
    print("직답 4토큰 대비 몇 배인지가 그대로 비용 배수다.")


if __name__ == "__main__":
    main()

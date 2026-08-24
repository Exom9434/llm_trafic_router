"""logprobs 실지원 진단 — 원시 응답을 그대로 본다.

smoke_test는 "logprobs가 왔는가"만 알려 준다. 안 왔을 때 이유가 세 가지라
구분이 안 된다. 프로바이더가 미지원인가, 파라미터 이름이 다른가, 우리
파서가 못 읽는가. 이 스크립트는 원시 JSON을 찍어 셋을 가른다.

문항은 "프랑스 수도" 한 개뿐이라 비용은 사실상 0이다.

실행:
    uv run diag_logprobs.py
    uv run diag_logprobs.py --models deepseek_v4_flash qwen_flash
    uv run diag_logprobs.py --full        # 응답 전문을 찍는다
"""

from __future__ import annotations

import argparse
import json
import sys

import requests

import prompts
from config import TOP_LOGPROBS, get_models
from providers.openai_compat import DEFAULT_BASE_URL

PROBE = {
    "item_id": "diag:0",
    "subject": "diag",
    "question": "What is the capital city of France?",
    "options": ["Berlin", "Madrid", "Paris", "Rome"],
    "answer": "C",
    "answer_index": 2,
}


def probe_raw(spec, want_logprobs: bool) -> tuple[int, dict | str]:
    base = (spec.base_url or DEFAULT_BASE_URL).rstrip("/")
    url = f"{base}/chat/completions"
    payload = {
        "model": spec.model,
        "messages": prompts.build_messages(PROBE, "direct", prompts.make_nonce()),
        spec.max_tokens_param: spec.direct_max_tokens,
    }
    if spec.supports_temperature:
        payload["temperature"] = 0.0
    payload.update(spec.extra_body)
    if want_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = TOP_LOGPROBS

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {spec.env_key()}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, resp.text[:800]


def main() -> None:
    ap = argparse.ArgumentParser(description="logprobs 실지원 진단")
    ap.add_argument("--models", nargs="*",
                    default=["deepseek_v4_flash", "qwen_flash"],
                    help="기본값은 문서상 logprobs를 지원한다는 두 곳이다")
    ap.add_argument("--full", action="store_true", help="응답 전문을 찍는다")
    args = ap.parse_args()

    models = [m for m in get_models(args.models) if m.adapter == "openai_compat"]
    if not models:
        sys.exit("OpenAI 호환 어댑터를 쓰는 모델이 없다.")

    for spec in models:
        print("=" * 72)
        print(f"{spec.key}  ({spec.model})  @ {spec.base_url or DEFAULT_BASE_URL}")
        print("=" * 72)

        status, body = probe_raw(spec, want_logprobs=True)
        print(f"[logprobs 요청] HTTP {status}")

        if status != 200:
            print("  거절됨. 응답 본문:")
            print("  " + json.dumps(body, ensure_ascii=False)[:600]
                  if isinstance(body, dict) else f"  {body}")
            print("\n  → logprobs 없이 재시도")
            status2, body2 = probe_raw(spec, want_logprobs=False)
            print(f"  HTTP {status2}"
                  + ("  (본체는 정상. logprobs 파라미터만 문제)" if status2 == 200 else ""))
            print()
            continue

        choice = (body.get("choices") or [{}])[0]
        lp = choice.get("logprobs")
        print(f"  반환 모델: {body.get('model')}")
        print(f"  fingerprint: {body.get('system_fingerprint')}")
        print(f"  응답 텍스트: {(choice.get('message') or {}).get('content')!r}")
        print(f"  usage: {body.get('usage')}")
        print(f"  choices[0].logprobs = {json.dumps(lp, ensure_ascii=False)[:500] if lp else lp}")

        if not lp:
            print("  → logprobs 필드 자체가 없거나 null. 프로바이더 미지원으로 판단.")
        elif not (lp.get("content")):
            print("  → logprobs는 있으나 content가 비었다. 스키마가 다를 수 있다. 아래 키 확인:")
            print(f"     {list(lp.keys())}")
        else:
            head = lp["content"][0]
            print(f"  → 지원됨. 첫 토큰 {head.get('token')!r} logprob={head.get('logprob')}")
            print(f"     top_logprobs {len(head.get('top_logprobs') or [])}개")

        if args.full:
            print("\n  [응답 전문]")
            print(json.dumps(body, ensure_ascii=False, indent=2)[:4000])
        print()


if __name__ == "__main__":
    main()

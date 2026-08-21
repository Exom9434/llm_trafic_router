"""프로바이더 실지원 확인 (설계서 9절 미확정 항목 1·2).

문서만 보고는 알 수 없는 것들을 콜 한 방씩으로 확정한다.
  · logprobs를 실제로 돌려주는가 (Solar Pro 3·Qwen·Gemini가 관건)
  · system_fingerprint가 오는가
  · 반환 모델 문자열이 요청한 것과 같은가
  · 직답 프롬프트에 글자 하나로 답하는가

결과는 capabilities.json에 남고, config.apply_capabilities()가 이 값을 읽어
이후 실행에서 logprobs 요청 여부를 결정한다. 비용은 모델당 두 콜이라 무시할
수준이다.

실행:
    python smoke_test.py
    python smoke_test.py --models upstage_solar_pro3 qwen_flash
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import prompts
from config import (
    CAPABILITIES_FILE,
    DIRECT_MAX_TOKENS,
    OUTPUT_DIR,
    TOP_LOGPROBS,
    get_models,
)
from providers import build_adapter

PROBE_ITEM = {
    "item_id": "smoke:0",
    "subject": "smoke",
    "question": "What is the capital city of France?",
    "options": ["Berlin", "Madrid", "Paris", "Rome"],
    "answer": "C",
    "answer_index": 2,
}


def probe(model, adapter, run_id: str, want_logprobs: bool):
    """logprobs를 요청하거나 요청하지 않고 한 번 쏜다."""
    valid = prompts.letters_for(PROBE_ITEM)
    nonce = prompts.make_nonce()
    messages = prompts.build_messages(PROBE_ITEM, "direct", nonce)
    raw = adapter.chat(
        messages,
        temperature=0.0,
        max_tokens=model.direct_max_tokens,
        want_logprobs=want_logprobs,
        top_logprobs=TOP_LOGPROBS,
    )
    letter = prompts.parse_letter(raw.text, valid) if raw.error is None else None
    return raw, letter


def main() -> None:
    ap = argparse.ArgumentParser(description="프로바이더 실지원 확인")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--no-anchors", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models = get_models(args.models, include_anchors=not args.no_anchors)
    models = [m for m in models if m.env_key()] if args.models is None else models
    if not models:
        sys.exit("돌릴 모델이 없다. .env에 API 키가 있는지 확인할 것.")

    run_id = uuid.uuid4().hex[:12]
    caps: dict[str, dict] = {}
    if CAPABILITIES_FILE.exists():
        caps = json.loads(CAPABILITIES_FILE.read_text(encoding="utf-8"))

    rows = []
    for model in models:
        if not model.env_key():
            rows.append({"model_key": model.key, "status": f"키 없음 ({model.api_key_env})"})
            caps[model.key] = {"reachable": False, "reason": "missing_api_key"}
            continue

        adapter = build_adapter(model)

        # 1차: logprobs 요청
        raw_lp, letter_lp = probe(model, adapter, run_id, want_logprobs=True)
        used_fallback = False
        raw, letter = raw_lp, letter_lp

        # logprobs 때문에 거절당했을 수 있으므로 한 번 더 시도한다.
        if raw_lp.error is not None:
            raw_plain, letter_plain = probe(model, adapter, run_id, want_logprobs=False)
            if raw_plain.error is None:
                used_fallback = True
                raw, letter = raw_plain, letter_plain

        has_lp = bool(raw_lp.error is None and raw_lp.top_logprobs)
        supports = "yes" if has_lp else ("no" if raw.error is None else "unknown")

        caps[model.key] = {
            "reachable": raw.error is None,
            "supports_logprobs": supports,
            "logprobs_request_rejected": used_fallback,
            "returned_model": raw.returned_model,
            "system_fingerprint": raw.system_fingerprint,
            "answered_single_letter": letter is not None,
            "reasoning_tokens": raw.reasoning_tokens,
            "thinking_off": raw.reasoning_tokens in (None, 0),
            "output_tokens": raw.output_tokens,
            "endpoint_host": raw.endpoint_host,
            "error": raw.error,
        }

        rows.append({
            "model_key": model.key,
            "status": "OK" if raw.error is None else f"실패: {(raw.error or '')[:120]}",
            "returned_model": raw.returned_model or "—",
            "logprob": "O" if has_lp else ("거절" if used_fallback else "X"),
            "fingerprint": raw.system_fingerprint or "—",
            "answer": letter or "파싱실패",
            "latency_ms": f"{raw.total_ms:.0f}" if raw.total_ms else "—",
            "out_tokens": raw.output_tokens if raw.output_tokens is not None else "—",
            "reasoning": raw.reasoning_tokens if raw.reasoning_tokens is not None else "—",
        })

        print(f"{model.key:28s} {rows[-1]['status']:10s} logprob={rows[-1]['logprob']}", flush=True)

    out = Path(args.out) if args.out else CAPABILITIES_FILE
    out.write_text(json.dumps(caps, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 프로바이더 실지원 확인 결과",
        "",
        "| 모델 | 상태 | 반환 모델 | logprob | fingerprint | 답 | 지연(ms) | 출력토큰 | 추론토큰 |",
        "|---|---|---|:--:|---|:--:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_key']} | {r.get('status','')} | {r.get('returned_model','—')} | "
            f"{r.get('logprob','—')} | {r.get('fingerprint','—')} | {r.get('answer','—')} | "
            f"{r.get('latency_ms','—')} | {r.get('out_tokens','—')} | {r.get('reasoning','—')} |"
        )
    lines += [
        "",
        "- `logprob = 거절`은 logprobs 파라미터를 넣으면 에러가 나고 빼면 되는 경우다.",
        "- 정답은 C(Paris)다. `답` 칸이 C가 아니면 프롬프트나 파서를 손봐야 한다.",
        "- **추론토큰이 0이 아니면 thinking 차단이 안 먹은 것이다.** `config.py`의",
        "  `extra_body`를 고치거나 `direct_max_tokens`를 올려야 한다. 그냥 두면",
        "  본실험 비용이 추정치를 크게 넘고, 심하면 빈 응답이 온다.",
        "- `답` 칸이 비어 있는데 추론토큰이 상한에 가깝다면 추론이 출력 상한을 다 먹은 것이다.",
        "",
    ]
    report = OUTPUT_DIR / "smoke_test.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"\ncapabilities: {out}")
    print(f"리포트: {report}")


if __name__ == "__main__":
    main()

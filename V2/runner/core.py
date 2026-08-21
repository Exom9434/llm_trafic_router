"""콜 하나를 실행해 CallRecord 하나로 만드는 공통 실행부.

보정 패스와 본실험이 같은 함수를 쓴다. 두 단계의 차이는 무엇을 언제 쏘느냐지
콜 자체가 아니기 때문이다. 여기서 프롬프트 조립·파싱·logprob 접기·채점을
한 번에 끝내고, 로깅 스키마를 채워 돌려준다.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import prompts
from calllog import CallRecord
from config import TOP_LOGPROBS, ModelSpec
from providers import build_adapter


@dataclass
class CallSpec:
    """한 콜이 무엇을 물어보는지."""
    model: ModelSpec
    item: dict
    mode: str = "direct"        # "direct" | "cot"
    temperature: float = 0.0
    max_tokens: int = 8
    rep: int = 0
    phase: str = "calibration"
    probe: str = "quality"
    slot: str = ""

    def call_key(self) -> str:
        return (
            f"{self.phase}|{self.slot}|{self.model.key}|{self.item['item_id']}"
            f"|{self.mode}|t{self.temperature}|r{self.rep}"
        )


def execute_call(cs: CallSpec, adapter, run_id: str, vantage: str = "") -> CallRecord:
    valid = prompts.letters_for(cs.item)
    nonce = prompts.make_nonce()
    messages = prompts.build_messages(cs.item, cs.mode, nonce)

    want_lp = cs.mode == "direct" and cs.model.supports_logprobs in ("yes", "unknown")
    raw = adapter.chat(
        messages,
        temperature=cs.temperature,
        max_tokens=cs.max_tokens,
        want_logprobs=want_lp,
        top_logprobs=TOP_LOGPROBS,
    )

    rec = CallRecord(
        call_key=cs.call_key(),
        run_id=run_id,
        phase=cs.phase,
        probe=cs.probe,
        mode=cs.mode,
        slot=cs.slot,
        vantage=vantage,
        model_key=cs.model.key,
        provider=cs.model.provider,
        requested_model=cs.model.model,
        returned_model=raw.returned_model,
        system_fingerprint=raw.system_fingerprint,
        endpoint_host=raw.endpoint_host,
        item_id=cs.item["item_id"],
        subject=cs.item.get("subject"),
        rep=cs.rep,
        temperature=cs.temperature,
        max_tokens=cs.max_tokens,
        nonce=nonce,
        http_status=raw.http_status,
        retries=raw.retries,
        error=raw.error,
        raw_text=raw.text,
        gold_letter=cs.item["answer"],
        ttft_ms=raw.ttft_ms,
        total_ms=raw.total_ms,
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        reasoning_tokens=raw.reasoning_tokens,
        rate_limit=raw.rate_limit,
        request_id=raw.request_id,
    ).stamp()

    if raw.error is None:
        letter = prompts.parse_letter(raw.text, valid)
        rec.parsed_letter = letter
        rec.correct = None if letter is None else int(letter == cs.item["answer"])
        rec.answer_logprob = raw.first_token_logprob

        dist = prompts.letter_distribution(raw.top_logprobs, valid)
        if dist:
            rec.letter_probs = dist
            rec.p_gold = dist.get(cs.item["answer"], 0.0)
            ordered = sorted(dist.values(), reverse=True)
            rec.margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)

        if raw.output_tokens and raw.total_ms:
            rec.tps = raw.output_tokens / (raw.total_ms / 1000.0)

    return rec


def run_batch(
    call_specs: list[CallSpec],
    run_id: str,
    logger,
    vantage: str = "",
    on_done=None,
) -> int:
    """모델별로 동시성 한도를 지키며 콜을 실행한다.

    한 프로바이더에 몰아치면 레이트리밋에 걸리므로 모델마다 별도 풀을 쓴다.
    모델 간에는 병렬로 돈다.
    """
    by_model: dict[str, list[CallSpec]] = {}
    for cs in call_specs:
        by_model.setdefault(cs.model.key, []).append(cs)

    adapters = {key: build_adapter(specs[0].model) for key, specs in by_model.items()}
    written = 0

    def run_one_model(key: str) -> int:
        specs = by_model[key]
        adapter = adapters[key]
        workers = max(1, specs[0].model.max_concurrency)
        count = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for rec in pool.map(lambda cs: execute_call(cs, adapter, run_id, vantage), specs):
                logger.write(rec)
                count += 1
                if on_done:
                    on_done(rec)
        return count

    with ThreadPoolExecutor(max_workers=max(1, len(by_model))) as outer:
        for n in outer.map(run_one_model, list(by_model.keys())):
            written += n

    return written


def estimate_cost(records: list[dict], models: list[ModelSpec]) -> dict:
    """누적 토큰으로 프로바이더별 비용을 추정한다(USD)."""
    price = {m.key: (m.price_in, m.price_out) for m in models}
    out: dict[str, float] = {}
    for rec in records:
        key = rec.get("model_key")
        if key not in price:
            continue
        pin, pout = price[key]
        cost = (rec.get("input_tokens") or 0) / 1e6 * pin + (rec.get("output_tokens") or 0) / 1e6 * pout
        out[key] = out.get(key, 0.0) + cost
    out["_total"] = sum(out.values())
    return out

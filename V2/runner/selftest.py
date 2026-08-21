"""API 키 없이 파이프라인 전체를 도는 자체 점검.

가짜 어댑터로 보정 패스를 흉내 내고, 그 로그를 select_bank의 선별 로직에
그대로 통과시킨다. 실제 API를 쓰기 전에 배선이 맞는지, 재개가 되는지,
선별이 의도대로 걸러지는지를 확인하는 용도다.

실행:
    python selftest.py
"""

from __future__ import annotations

import math
import random
import sys
import tempfile
import uuid
from pathlib import Path

import prompts
from calllog import JsonlLogger, load_done_keys, read_records
from config import ModelSpec
from core import CallSpec, run_batch
from itembank import SUBJECTS
from providers.base import RawResult
from select_bank import group_records, item_stats, noise_floor, select_bank

# ── 가짜 세계 ────────────────────────────────────────────────
# 문항마다 '진짜 난이도'를, 모델마다 '실력'과 '흔들림'을 준다.
# 선별 로직이 천장(전부 맞힘)·바닥(전부 틀림) 문항을 실제로 버리는지 본다.

RNG = random.Random(20260722)

MOCK_MODELS = [
    ModelSpec(key="mock_strong", provider="mock", model="mock-strong-v1",
              adapter="openai_compat", api_key_env="MOCK", supports_logprobs="yes"),
    ModelSpec(key="mock_mid", provider="mock", model="mock-mid-v1",
              adapter="openai_compat", api_key_env="MOCK", supports_logprobs="yes"),
    ModelSpec(key="mock_weak", provider="mock", model="mock-weak-v1",
              adapter="openai_compat", api_key_env="MOCK", supports_logprobs="no"),
]

SKILL = {"mock_strong": 0.25, "mock_mid": 0.0, "mock_weak": -0.25}


def make_pool(per_subject: int = 8) -> list[dict]:
    pool = []
    for subject in SUBJECTS:
        for i in range(per_subject):
            # 0.05~0.95를 고르게 훑어 밴드 바깥 문항도 섞이게 한다.
            base = 0.05 + 0.9 * (i / max(1, per_subject - 1))
            item = {
                "item_id": f"{subject}:{i}",
                "subject": subject,
                "question": f"[{subject}] synthetic question {i}",
                "options": [f"option {c}" for c in "ABCDEFGHIJ"],
                "answer": "C",
                "answer_index": 2,
                "_p": base,
            }
            pool.append(item)
    return pool


POOL = make_pool()
P_BY_ID = {i["item_id"]: i["_p"] for i in POOL}


class MockAdapter:
    def __init__(self, spec):
        self.spec = spec

    def chat(self, messages, temperature, max_tokens, want_logprobs=False, top_logprobs=10):
        # system 메시지에 nonce가 실렸는지 확인한다 (캐시 방지 배선 점검).
        system = messages[0]["content"]
        assert "[session:" in system, "nonce가 프롬프트에 안 실렸다"

        item_id = _current_item_id[0]
        p = min(0.98, max(0.02, P_BY_ID[item_id] + SKILL[self.spec.key]))
        # temp>0에서는 흔들림을 키운다.
        if temperature > 0:
            p = 0.5 * p + 0.5 * RNG.random()

        correct = RNG.random() < p
        letter = "C" if correct else RNG.choice([c for c in "ABDEFGHIJ"])

        top = None
        first_lp = None
        if want_logprobs and self.spec.supports_logprobs == "yes":
            first_lp = math.log(max(1e-6, p if correct else 1 - p))
            others = [c for c in "ABDEFGHIJ"][:4]
            top = [{"token": letter, "logprob": first_lp}] + [
                {"token": c, "logprob": first_lp - 2.0 - RNG.random()} for c in others
            ]

        # thinking을 못 끄는 모델을 흉내 낸다. 이 값이 노이즈 바닥선까지
        # 흘러가는지가 새 주력 지표의 배선 점검이다.
        reasoning = {"mock_strong": 12, "mock_mid": 0, "mock_weak": None}[self.spec.key]

        return RawResult(
            text=letter,
            returned_model=self.spec.model,
            system_fingerprint="fp_mock" if self.spec.key == "mock_strong" else None,
            input_tokens=320,
            output_tokens=1,
            reasoning_tokens=reasoning,
            first_token_logprob=first_lp,
            top_logprobs=top,
            http_status=200,
            total_ms=120.0,
            endpoint_host="mock.local",
        )


_current_item_id = [""]


def patched_execute(cs: CallSpec, adapter, run_id: str, vantage: str = ""):
    _current_item_id[0] = cs.item["item_id"]
    return _real_execute(cs, adapter, run_id, vantage)


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        _failures.append(label)


_failures: list[str] = []

if __name__ == "__main__":
    import core

    _real_execute = core.execute_call
    core.execute_call = patched_execute
    core.build_adapter = MockAdapter

    print("V2 runner 자체 점검\n")

    # 1. 프롬프트·파서 ────────────────────────────────────
    print("프롬프트와 파서")
    item = POOL[0]
    msgs = prompts.build_messages(item, "direct", prompts.make_nonce())
    check("system/user 두 메시지", len(msgs) == 2 and msgs[0]["role"] == "system")
    check("선택지가 프롬프트에 들어감", "J. option J" in msgs[1]["content"])
    valid = prompts.letters_for(item)
    cases = [
        ("C", "C"), (" C ", "C"), ("C.", "C"), ("C)", "C"), ("(C)", "C"),
        ("**C**", "C"), ("<Answer>C</Answer>", "C"),
        ("The answer is C", "C"), ("Answer: C", "C"), ("Correct option — C", "C"),
        # 아래는 반드시 파싱 실패여야 한다. 추측하면 정확도가 조용히 오염된다.
        ("I cannot answer", None), ("I'm sorry, I can't help with that.", None),
        ("As an AI language model", None), ("", None), ("   ", None),
    ]
    bad = [(r, prompts.parse_letter(r, valid)) for r, w in cases
           if prompts.parse_letter(r, valid) != w]
    check(f"답 파싱 {len(cases)}종", not bad, f"틀린 케이스 {bad}" if bad else "")
    dist = prompts.letter_distribution(
        [{"token": " C", "logprob": math.log(0.7)}, {"token": "A", "logprob": math.log(0.2)}], valid
    )
    check("logprob → 확률분포 정규화", abs(sum(dist.values()) - 1.0) < 1e-9 and dist["C"] > dist["A"], str(dist))

    # 2. 보정 패스 실행 ───────────────────────────────────
    print("\n보정 패스 (가짜 어댑터)")
    tmp = Path(tempfile.mkdtemp(prefix="v2selftest_"))
    log_path = tmp / "calibration_calls.jsonl"
    K = 5

    specs = []
    for m in MOCK_MODELS:
        for it in POOL:
            specs.append(CallSpec(model=m, item=it, mode="direct", temperature=0.0,
                                  max_tokens=8, rep=0, phase="calibration"))
            for r in range(1, K + 1):
                specs.append(CallSpec(model=m, item=it, mode="direct", temperature=1.0,
                                      max_tokens=8, rep=r, phase="calibration"))

    logger = JsonlLogger(log_path)
    n = run_batch(specs, uuid.uuid4().hex[:12], logger)
    logger.close()
    expected = len(MOCK_MODELS) * len(POOL) * (1 + K)
    check(f"콜 {expected}회 실행", n == expected, f"실제 {n}")

    records = read_records(log_path)
    check("로그 행 수 일치", len(records) == expected, f"{len(records)}행")
    check("call_key 중복 없음", len({r["call_key"] for r in records}) == expected)
    check("오류 0건", sum(1 for r in records if r["error"]) == 0)
    check("파싱 실패 0건", sum(1 for r in records if r["parsed_letter"] is None) == 0)
    lp_records = [r for r in records if r["model_key"] == "mock_strong" and r["rep"] == 0]
    check("logprob 모델에 p_gold 기록됨", all(r["p_gold"] is not None for r in lp_records))
    nolp = [r for r in records if r["model_key"] == "mock_weak"]
    check("비logprob 모델은 p_gold 없음", all(r["p_gold"] is None for r in nolp))

    # 3. 재개 ────────────────────────────────────────────
    print("\n재개")
    done = load_done_keys(log_path)
    check("done 키 수집", len(done) == expected, f"{len(done)}개")
    remaining = [cs for cs in specs if cs.call_key() not in done]
    check("남은 콜 0개", len(remaining) == 0, f"{len(remaining)}개")

    # 4. 선별과 노이즈 바닥선 ───────────────────────────
    print("\n선별과 노이즈 바닥선")
    grouped = group_records(records)
    stats = item_stats(grouped, {i["item_id"]: i for i in POOL})
    floors = noise_floor(grouped)
    kept, selected = select_bank(stats, 0.40, 0.85, per_subject=4,
                                 max_error=0.10, max_parse_fail=0.10)

    check("문항 통계 생성", len(stats) == len(POOL), f"{len(stats)}개")
    in_band = all(0.40 <= s["difficulty"] <= 0.85 for s in kept)
    check("밴드 밖 문항 제외됨", in_band and 0 < len(kept) < len(POOL), f"{len(kept)}/{len(POOL)} 통과")
    check("과목당 상한 지켜짐", all(
        sum(1 for s in selected if s["subject"] == subj) <= 4 for subj in SUBJECTS))
    check("노이즈 바닥선 3모델", len(floors) == len(MOCK_MODELS))
    strong = next(f for f in floors if f["model_key"] == "mock_strong")
    weak = next(f for f in floors if f["model_key"] == "mock_weak")
    check("강한 모델이 더 정확", (strong["temp0_accuracy"] or 0) > (weak["temp0_accuracy"] or 0),
          f"{strong['temp0_accuracy']} vs {weak['temp0_accuracy']}")
    check("자기일관성 [0,1] 범위", all(
        f["self_consistency"] is None or 0 <= f["self_consistency"] <= 1 for f in floors))
    check("logprob 유무 구분", strong["has_logprob"] and not weak["has_logprob"])
    check("추론 토큰이 바닥선까지 전달됨", strong["mean_reasoning_tokens"] == 12.0,
          f"strong={strong['mean_reasoning_tokens']}, weak={weak['mean_reasoning_tokens']}")
    check("추론 토큰 미보고 모델은 None", weak["mean_reasoning_tokens"] is None)
    rec_with_reasoning = [r for r in records if r["model_key"] == "mock_strong"]
    check("로그에 reasoning_tokens 기록됨",
          all(r.get("reasoning_tokens") == 12 for r in rec_with_reasoning))

    print()
    if _failures:
        print(f"실패 {len(_failures)}건: {_failures}")
        sys.exit(1)
    print("전부 통과.")

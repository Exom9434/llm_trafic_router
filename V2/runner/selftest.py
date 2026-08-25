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
from datetime import datetime
import tempfile
import uuid
from pathlib import Path

import prompts
from budget import DayLedger, SpendGuard, measure_token_profiles, project
from calllog import JsonlLogger, load_done_keys, read_records
from config import (
    KST,
    ModelSpec,
    in_window,
    seconds_left_in_window,
    seconds_until_window,
    wait_until_window,
)
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
              adapter="openai_compat", api_key_env="MOCK", supports_logprobs="yes",
              price_in=1.00, price_out=5.00),
    ModelSpec(key="mock_mid", provider="mock", model="mock-mid-v1",
              adapter="openai_compat", api_key_env="MOCK", supports_logprobs="yes",
              price_in=0.30, price_out=1.20),
    ModelSpec(key="mock_weak", provider="mock", model="mock-weak-v1",
              adapter="openai_compat", api_key_env="MOCK", supports_logprobs="no",
              price_in=0.05, price_out=0.20),
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

    # 5. 지출 가드 ────────────────────────────────────────
    print("\n저부하 시간대 대기")

    US, NIGHT = (10, 17), (23, 7)

    def kst(h, m=0, day=25):
        return datetime(2026, 8, day, h, m, tzinfo=KST)

    # 자정을 넘어가는 구간
    check("야간 시간대: 23시 안", in_window(23, NIGHT))
    check("야간 시간대: 03시 안", in_window(3, NIGHT))
    check("야간 시간대: 07시 밖", not in_window(7, NIGHT))
    check("야간 시간대: 16시 밖", not in_window(16, NIGHT))

    # 열릴 때까지
    check("열려 있으면 0초", seconds_until_window(kst(2), NIGHT) == 0)
    check("16:16 → 23시까지 6시간 44분",
          seconds_until_window(kst(16, 16), NIGHT) == 6 * 3600 + 44 * 60,
          str(seconds_until_window(kst(16, 16), NIGHT)))
    check("18시 → 미국 시간대는 내일 10시",
          seconds_until_window(kst(18), US) == 16 * 3600,
          str(seconds_until_window(kst(18), US)))

    # 닫히기까지 — 오늘 미국 배치를 접은 근거
    check("닫혀 있으면 0초", seconds_left_in_window(kst(18), US) == 0)
    check("16:16 → 미국 시간대 44분 남음",
          seconds_left_in_window(kst(16, 16), US) == 44 * 60,
          str(seconds_left_in_window(kst(16, 16), US)))
    check("23:30 → 야간 시간대 7시간 30분 남음",
          seconds_left_in_window(kst(23, 30), NIGHT) == 7 * 3600 + 30 * 60,
          str(seconds_left_in_window(kst(23, 30), NIGHT)))
    check("02:00 → 야간 시간대 5시간 남음",
          seconds_left_in_window(kst(2), NIGHT) == 5 * 3600)

    # wait_until_window: 가짜 시계로 대기 경로를 돈다
    ticks = {"n": 0}
    slept = []

    def fake_now(seq):
        it = iter(seq)
        return lambda: next(it)

    import config as _cfg
    real_sleep, real_dt = _cfg.time.sleep, _cfg.datetime

    class FakeDatetime:
        seq = []
        @classmethod
        def now(cls, tz=None):
            ticks["n"] += 1
            return cls.seq[min(ticks["n"] - 1, len(cls.seq) - 1)]

    _cfg.time.sleep = lambda s: slept.append(s)
    _cfg.datetime = FakeDatetime
    try:
        # 이미 열려 있으면 자지 않고 즉시 반환한다
        ticks["n"] = 0; slept.clear()
        FakeDatetime.seq = [kst(2)]
        wait_until_window(NIGHT, log=lambda *a: None)
        check("이미 열려 있으면 즉시 반환", slept == [], str(slept))

        # 닫혀 있으면 열릴 때까지 잔다
        ticks["n"] = 0; slept.clear()
        FakeDatetime.seq = [kst(22, 58), kst(22, 59), kst(23, 0)]
        wait_until_window(NIGHT, log=lambda *a: None)
        check("닫혀 있으면 열릴 때까지 대기", len(slept) == 2, str(slept))
        check("poll_sec을 넘겨 자지 않음", all(x <= 60 for x in slept), str(slept))

        # 열려 있어도 남은 시간이 모자라면 다음 회차를 노린다
        ticks["n"] = 0; slept.clear()
        FakeDatetime.seq = [kst(16, 16), kst(16, 17)] + [kst(10, 0, day=26)]
        wait_until_window(US, min_remaining_sec=2 * 3600, log=lambda *a: None)
        check("남은 시간 부족하면 다음 회차까지 대기", len(slept) == 2, str(slept))
    finally:
        _cfg.time.sleep = real_sleep
        _cfg.datetime = real_dt

    print("\n지출 가드")
    profiles = measure_token_profiles(log_path)
    check("실측 토큰 프로파일 3모델", len(profiles) == 3, str(sorted(profiles)))
    check("입력 토큰 실측값 반영",
          all(abs(p.mean_input - 320) < 1e-6 for p in profiles.values()))
    check("추론 토큰 실측값 반영",
          profiles["mock_strong"].mean_reasoning == 12.0
          and profiles["mock_mid"].mean_reasoning == 0.0)

    design = {"bank": 300, "items_per_slot": 100, "slots_per_day": 8, "days": 14, "k": 5}
    plan = project(MOCK_MODELS, profiles, lambda m: design)
    check("모델 3개 모두 투영됨", len(plan["models"]) == 3)
    e = plan["models"]["mock_strong"]
    check("상한 = 투영치 × 배수",
          abs(e["spend_cap"] - round(e["projected_total"] * 3.0, 2)) < 0.02,
          f"투영 ${e['projected_total']}, 상한 ${e['spend_cap']}")
    check("하루 예약 < 총 투영", e["day_reserve"] < e["projected_total"])

    guard = SpendGuard(plan)
    ok, _ = guard.can_start_day("mock_strong")
    check("예산 충분하면 하루 시작 허용", ok)

    # 상한 직전까지 쓴 상황을 만든다. 하루치 예약이 안 되면 시작을 막아야 한다.
    guard.spent["mock_strong"] = e["spend_cap"] - e["day_reserve"] * 0.5
    ok, reason = guard.can_start_day("mock_strong")
    check("하루치 예약 불가하면 시작 거부", not ok, reason)
    check("거부는 날 경계에서만 — 중간 점검은 아직 통과",
          guard.check_mid_day("mock_mid")[0])

    ok_other, _ = guard.can_start_day("mock_mid")
    check("모델별 독립 정지 (다른 모델은 계속)", ok_other)
    check("정지 사유가 기록됨", "mock_strong" in guard.stopped)

    # 재개: 로그에서 누적 비용을 복원한다
    guard2 = SpendGuard(plan)
    guard2.load_spent(log_path, MOCK_MODELS)
    expected = sum(1 for r in records if r["model_key"] == "mock_strong") * (
        320 / 1e6 * 1.00 + 1 / 1e6 * 5.00)
    check("재개 시 기존 지출 복원",
          abs(guard2.spent["mock_strong"] - expected) < 1e-9,
          f"{guard2.spent['mock_strong']:.6f} vs {expected:.6f}")

    # 날짜 완주 기록
    ledger = DayLedger(tmp / "day_status.jsonl")
    ledger.write("mock_strong", "2026-09-01", 8, 8, "complete")
    ledger.write("mock_strong", "2026-09-02", 8, 3, "aborted", "상한 초과")
    ledger.write("mock_mid", "2026-09-01", 8, 8, "complete")
    done = ledger.complete_days()
    check("완주한 날만 분석 대상", len(done) == 2 and all(d["status"] == "complete" for d in done))
    check("모델별 필터", len(ledger.complete_days("mock_strong")) == 1)

    print()
    if _failures:
        print(f"실패 {len(_failures)}건: {_failures}")
        sys.exit(1)
    print("전부 통과.")

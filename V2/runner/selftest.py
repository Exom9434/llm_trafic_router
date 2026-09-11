"""API 키 없이 파이프라인 전체를 도는 자체 점검.

가짜 어댑터로 보정 패스를 흉내 내고, 그 로그를 select_bank의 선별 로직에
그대로 통과시킨다. 실제 API를 쓰기 전에 배선이 맞는지, 재개가 되는지,
선별이 의도대로 걸러지는지를 확인하는 용도다.

실행:
    python selftest.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from datetime import datetime, timezone
import tempfile
import uuid
from pathlib import Path

import prompts
from budget import DayLedger, SpendGuard, measure_token_profiles, project
from calllog import JsonlLogger, load_done_keys, read_records
from config import (
    ALL_MODELS,
    ANCHOR_DESIGN,
    KST,
    MAIN_DESIGN,
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
    print("\n끝에 글자만 남긴 응답 파싱")

    V = list("ABCDEFGHIJ")
    trail_cases = [
        # (출력, 기대, 잘림)
        ("F = q(E + vxB) = 0, so E = -vxB ... E = 2ix - 2iy + iz  D", "D", False),
        ("The integral evaluates to pi*123 = 386.4158898 B", "B", False),
        ("...therefore the field is 3.0e3 N/C A", "A", False),
        ("(D)", "D", False),
        # 마크다운 강조로 감싼 마지막 글자 (2026-09-02)
        ("For small oscillations the condition gives **A**", "A", False),
        ("...so the total is 3 lb 15 oz **J**", "J", False),
        ("...checking the correlations again *B*", "B", False),
        ("...therefore [C]", "C", False),
        ("Answer: G", "G", False),
        ("<Answer>E</Answer>", "E", False),
        # 끊긴 응답의 마지막 글자는 답이 아니다
        ("I need to find the diameter of a steel rod under combined loads. Given: P", None, True),
        # 본문 한가운데 대문자는 줍지 않는다
        ("blah blah C blah blah", None, False),
        ("I cannot answer", None, False),
        ("Let me compute. The result is 42", None, False),
    ]
    bad = [(t[:30], want, prompts.parse_letter(t, V, truncated=cut))
           for t, want, cut in trail_cases
           if prompts.parse_letter(t, V, truncated=cut) != want]
    check(f"끝글자 규칙 {len(trail_cases)}종", not bad, f"틀린 케이스 {bad}" if bad else "")
    check("잘린 응답에는 끝글자 규칙을 쓰지 않는다",
          prompts.parse_letter("... the answer should be D", V, truncated=True) is None)
    check("안 잘렸으면 같은 문자열에서 답을 읽는다",
          prompts.parse_letter("... the answer should be D", V, truncated=False) == "D")

    # 답을 명시한 꼴이 여러 번 나오는 경우 (2026-09-01)
    print("\n답 명시 꼴이 여러 번 나올 때")
    phrase_cases = [
        # 검토 과정의 언급이 앞에 깔리고 결론이 뒤에 온다. 첫 매치를 쓰면 안 된다.
        ("Option A says the field is zero. Option C is closer. The answer is E.", "E", False),
        # answer가 없으면 마지막 매치를 쓴다.
        ("Let me check option B, then option F.", "F", False),
        # answer 매치가 뒤쪽 option 언급보다 우선한다.
        ("The answer is G, which is unlike option J.", "G", False),
        # 끊긴 응답의 중간 언급은 최종 답이 아니다.
        ("I will check option A first, then the answer is B and", None, True),
        # 첫머리에 답이 온 응답은 끊겨도 읽는다.
        ("C. Now let me verify this by computing the", "C", True),
        ("<Answer>H</Answer> then rambles on and on", "H", True),
    ]
    bad = [(t[:34], want, prompts.parse_letter(t, V, truncated=cut))
           for t, want, cut in phrase_cases
           if prompts.parse_letter(t, V, truncated=cut) != want]
    check(f"답 명시 꼴 선택 {len(phrase_cases)}종", not bad, f"틀린 케이스 {bad}" if bad else "")

    # 분당 요청 상한 ─────────────────────────────────────
    print("\n분당 요청 상한")

    import threading as _th
    from types import SimpleNamespace as _NS
    from providers.base import BaseAdapter as _BA

    def _burst(rpm, calls, workers):
        ad = _BA(_NS(max_rpm=rpm))
        per = calls // workers
        t0 = time.monotonic()
        ths = [_th.Thread(target=lambda: [ad._wait_for_slot() for _ in range(per)])
               for _ in range(workers)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        return time.monotonic() - t0

    el = _burst(1200, 20, 2)          # 간격 0.05초 x 19 = 0.95초
    check("상한이 있으면 간격을 지킨다", 0.85 <= el <= 1.35, f"{el:.2f}초 (기대 0.95초)")
    el = _burst(None, 200, 4)
    check("상한이 없으면 기다리지 않는다", el < 0.1, f"{el:.3f}초")
    check("네이버에 상한이 걸려 있다",
          next(m.max_rpm for m in ALL_MODELS if m.key == "naver_hcx_dash") == 85)

    # 시각별 단가 ─────────────────────────────────────────
    print("\n시각별 단가와 세금")

    from budget import call_price_factor, is_peak, schedule_price_factor

    _ds = next(m for m in ALL_MODELS if m.key == "deepseek_v4_flash")
    _hk = next(m for m in ALL_MODELS if m.key == "anthropic_haiku")
    _hx = next(m for m in ALL_MODELS if m.key == "naver_hcx_dash")

    check("공표 피크 구간 판정",
          is_peak(_ds, 3) and is_peak(_ds, 9) and not is_peak(_ds, 0) and not is_peak(_ds, 12),
          f"01-04·06-10 UTC, 슬롯 {MAIN_DESIGN['slot_hours']}")
    check("라인업 8슬롯 중 3개가 피크 — 계수 0.6875",
          abs(schedule_price_factor(_ds, MAIN_DESIGN["slot_hours"]) - 0.6875) < 1e-9,
          f"{schedule_price_factor(_ds, MAIN_DESIGN['slot_hours']):.4f}")
    check("앵커 4슬롯 중 1개가 피크 — 계수 0.625",
          abs(schedule_price_factor(_ds, ANCHOR_DESIGN["slot_hours"]) - 0.625) < 1e-9,
          f"{schedule_price_factor(_ds, ANCHOR_DESIGN['slot_hours']):.4f}")
    check("이중 요금제가 없는 모델은 계수 1.0",
          schedule_price_factor(_hk, MAIN_DESIGN["slot_hours"]) == 1.0)
    check("콜 시각으로 피크·오프피크를 가른다",
          call_price_factor(_ds, "2026-09-02T02:15:00+00:00") == 1.0
          and call_price_factor(_ds, "2026-09-02T12:15:00+00:00") == 0.5)
    check("시각을 모르면 비싼 쪽으로 잡는다",
          call_price_factor(_ds, None) == 1.0 and call_price_factor(_ds, "깨진값") == 1.0)
    check("부가세 배수가 비용에 실린다",
          abs(_hx.tax_multiplier - 1.10) < 1e-9 and _ds.tax_multiplier == 1.0)

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

    # ── 본실험 스케줄러 ──────────────────────────────────
    print("\n[본실험 스케줄러]")

    from datetime import date as _date, timedelta as _td
    from config import (
        ANCHOR_SLOT_HOURS_UTC, BANK_CYCLE, SLOT_HOURS_UTC,
        condition_label, is_peak_slot, item_group, slot_index,
    )
    import experiment as exp

    d0 = _date(2026, 10, 5)          # 월요일

    # 사흘이면 모든 묶음이 여덟 시각을 정확히 한 번씩 통과해야 한다.
    # 하루 슬롯 수 8과 묶음 주기 3이 서로소라서 생기는 성질이고,
    # 이 설계의 조건 균형 전부가 여기에 얹혀 있다.
    seen = {g: [] for g in range(BANK_CYCLE)}
    for day_off in range(BANK_CYCLE):
        for hour in SLOT_HOURS_UTC:
            idx = slot_index(d0 + _td(days=day_off), hour, d0)
            seen[item_group(idx)].append(hour)
    check("사흘이면 묶음마다 여덟 시각을 한 번씩 통과",
          all(sorted(v) == sorted(SLOT_HOURS_UTC) for v in seen.values()),
          str({g: sorted(v) for g, v in seen.items()}))

    # 반례: 카운터를 날마다 0으로 되돌리면 묶음0이 미국 피크를 못 만난다.
    reset = {g: set() for g in range(BANK_CYCLE)}
    for pos, hour in enumerate(SLOT_HOURS_UTC):
        reset[pos % BANK_CYCLE].add(hour)
    check("날마다 초기화하면 묶음0이 미국 피크(12·15)를 통과하지 못한다",
          not (reset[0] & {12, 15}), str(sorted(reset[0])))

    # 슬롯 번호는 달력에서 센다. 중단해서 하루를 통째로 건너뛰어도
    # 같은 날 같은 시각은 같은 묶음을 받아야 한다.
    check("하루를 건너뛰어도 묶음 배정이 안 밀린다",
          item_group(slot_index(d0 + _td(days=7), 12, d0))
          == item_group(slot_index(d0 + _td(days=7), 12, d0)))
    check("슬롯 번호가 날을 넘겨 이어진다",
          slot_index(d0 + _td(days=1), 0, d0) == len(SLOT_HOURS_UTC))

    # 조건 라벨. 공표 구간 그대로여야 한다(설계서 3.2절).
    mon, sat = _date(2026, 10, 5), _date(2026, 10, 10)
    check("미국 피크는 평일 12·15시",
          [h for h in SLOT_HOURS_UTC if is_peak_slot("us", h, mon)] == [12, 15])
    check("미국은 주말의 같은 시각도 오프피크",
          not any(is_peak_slot("us", h, sat) for h in SLOT_HOURS_UTC))
    check("중국 피크는 03·06·09시",
          [h for h in SLOT_HOURS_UTC if is_peak_slot("cn", h, sat)] == [3, 6, 9])
    check("한국 피크는 00·06시",
          [h for h in SLOT_HOURS_UTC if is_peak_slot("kr", h, sat)] == [0, 6])
    check("라벨은 두 값뿐",
          {condition_label(r, h, mon) for r in ("us", "cn", "kr")
           for h in SLOT_HOURS_UTC} == {"peak", "offpeak"})

    # 21 완주일에서 미국 팔의 피크 방문이 10.0회로 고정되는지.
    # 완주일 21을 고른 근거가 이 값이다(설계서 7.4절).
    for start_off in range(7):
        start = d0 + _td(days=start_off)
        hits = sum(1 for d in range(21) for h in SLOT_HOURS_UTC
                   if is_peak_slot("us", h, start + _td(days=d)))
        check(f"21일 미국 피크 슬롯 수가 시작 요일({start_off})과 무관",
              hits / BANK_CYCLE == 10.0, f"{hits / BANK_CYCLE}")

    # 앵커는 하루 4슬롯이고, 그 4슬롯이 조건마다 두 시각씩 갈려야 한다.
    anc_pk = [h for h in ANCHOR_SLOT_HOURS_UTC if is_peak_slot("us", h, mon)]
    anc_off = [h for h in ANCHOR_SLOT_HOURS_UTC if not is_peak_slot("us", h, mon)]
    check("앵커는 조건당 시각이 둘 (교락 회피)",
          len(anc_pk) == 2 and len(anc_off) == 2, f"{anc_pk} / {anc_off}")

    # ── 슬롯 콜 조립 ─────────────────────────────────────
    print("\n[슬롯 콜 조립]")

    bank = [{"item_id": f"it{i:03d}", "subject": SUBJECTS[i % len(SUBJECTS)],
             "question": f"q{i}", "options": ["a", "b", "c", "d"], "answer": "A"}
            for i in range(MAIN_DESIGN["items_per_slot"] * BANK_CYCLE)]
    bank_path = tmp / "item_bank.json"
    bank_path.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
    groups = exp.load_groups(bank_path)
    check("은행이 세 묶음으로 갈린다",
          len(groups) == BANK_CYCLE
          and all(len(g) == MAIN_DESIGN["items_per_slot"] for g in groups))
    check("묶음끼리 문항이 겹치지 않는다",
          len({i["item_id"] for g in groups for i in g}) == len(bank))
    check("묶음 순서가 시드로 고정된다",
          [i["item_id"] for i in exp.load_groups(bank_path)[0]]
          == [i["item_id"] for i in groups[0]])

    slot_dt = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    specs = exp.build_specs(MOCK_MODELS, slot_dt, d0, groups,
                            {m.key: True for m in MOCK_MODELS}, set(), MAIN_DESIGN["k"])
    per_model = MAIN_DESIGN["items_per_slot"] * MAIN_DESIGN["k"] + exp.LATENCY_REPS_PER_SLOT
    check("슬롯 하나의 콜 수", len(specs) == per_model * len(MOCK_MODELS),
          f"{len(specs)} vs {per_model * len(MOCK_MODELS)}")
    check("콜 키가 전부 유일", len({cs.call_key() for cs in specs}) == len(specs))
    check("지연 프로브가 앞에 온다 (우리 부하가 섞이기 전에 잰다)",
          specs[0].probe == "latency")
    check("조건 라벨이 콜마다 박힌다",
          all(cs.condition in ("peak", "offpeak") for cs in specs))
    check("슬롯 번호와 묶음이 콜마다 박힌다",
          all(cs.slot_index is not None and cs.item_group is not None for cs in specs))
    check("이미 끝난 콜은 다시 만들지 않는다",
          exp.build_specs(MOCK_MODELS, slot_dt, d0, groups,
                          {m.key: True for m in MOCK_MODELS},
                          {cs.call_key() for cs in specs}, MAIN_DESIGN["k"]) == [])

    lat = [cs for cs in specs if cs.probe == "latency"]
    check("지연 프로브는 문항을 묻지 않는다", all(cs.item is None for cs in lat))
    n1 = prompts.build_latency_messages(prompts.make_nonce())
    n2 = prompts.build_latency_messages(prompts.make_nonce())
    check("nonce가 달라도 프롬프트 길이는 같다",
          len(n1[0]["content"]) == len(n2[0]["content"])
          and n1[0]["content"] != n2[0]["content"])

    # ── 스트리밍 파서 ────────────────────────────────────
    print("\n[스트리밍 파서]")

    class FakeResp:
        def __init__(self, lines):
            self.status_code = 200
            self.headers = {}
            self.text = ""
            self._lines = lines

        def iter_lines(self, decode_unicode=False):
            for line in self._lines:
                time.sleep(0.002)
                yield line

        def close(self):
            pass

    def fake_stream(adapter, lines):
        adapter.session.post = lambda *a, **kw: FakeResp(lines)
        return adapter.chat_stream([{"role": "user", "content": "x"}], 0.0, 64)

    from providers import build_adapter as _build

    oa = _build(ModelSpec(key="t", provider="t", model="m", adapter="openai_compat",
                          api_key_env="MOCK"))
    raw = fake_stream(oa, [
        'data: {"model":"m-1","choices":[{"delta":{"content":"He"}}]}', "",
        'data: {"choices":[{"delta":{"content":"llo"}}]}', "",
        'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":64,'
        '"completion_tokens_details":{"reasoning_tokens":3}}}', "",
        "data: [DONE]", "",
    ])
    check("openai 호환 스트림에서 텍스트를 모은다", raw.text == "Hello", str(raw.text))
    check("openai 호환 스트림에서 TTFT를 잰다", raw.ttft_ms is not None)
    check("openai 호환 스트림에서 토큰 수를 읽는다",
          raw.output_tokens == 64 and raw.input_tokens == 11
          and raw.reasoning_tokens == 3)
    check("TTFT가 총 소요시간보다 작다", raw.ttft_ms < raw.total_ms)
    check("반환 모델 문자열을 기록한다", raw.returned_model == "m-1")

    an = _build(ModelSpec(key="t", provider="t", model="m", adapter="anthropic",
                          api_key_env="MOCK"))
    raw = fake_stream(an, [
        "event: message_start",
        'data: {"type":"message_start","message":{"model":"claude-x",'
        '"usage":{"input_tokens":9,"output_tokens":1}}}', "",
        "event: content_block_delta",
        'data: {"type":"content_block_delta","delta":{"type":"thinking_delta",'
        '"thinking":"hmm"}}', "",
        "event: content_block_delta",
        'data: {"type":"content_block_delta","delta":{"type":"text_delta",'
        '"text":"1 2"}}', "",
        "event: message_delta",
        'data: {"type":"message_delta","usage":{"output_tokens":64}}', "",
    ])
    check("anthropic 스트림에서 텍스트를 모은다", raw.text == "1 2", str(raw.text))
    check("anthropic 스트림은 thinking을 텍스트로 세지 않는다",
          raw.text is not None and "hmm" not in raw.text)
    check("anthropic 스트림에서 출력 토큰이 최종값으로 갱신된다",
          raw.output_tokens == 64 and raw.input_tokens == 9)

    hc = _build(ModelSpec(key="t", provider="t", model="HCX", adapter="hyperclova",
                          api_key_env="MOCK"))
    raw = fake_stream(hc, [
        "event: token", 'data: {"message":{"content":"1 "}}', "",
        "event: token", 'data: {"message":{"content":"2"}}', "",
        "event: result",
        'data: {"message":{"content":"1 2"},"usage":{"promptTokens":7,'
        '"completionTokens":64}}', "",
    ])
    check("clova 스트림에서 텍스트를 모은다", raw.text == "1 2", str(raw.text))
    check("clova의 result 이벤트가 전문을 다시 더하지 않는다",
          raw.text == "1 2" and raw.output_tokens == 64)

    # ── 완주 판정 ────────────────────────────────────────
    print("\n[완주 판정]")

    led = DayLedger(tmp / "day_status_sched.jsonl")
    already = set()
    fired = {("mock_strong", "2026-10-05T%02d" % h) for h in SLOT_HOURS_UTC}
    fired |= {("mock_mid", "2026-10-05T00"), ("mock_mid", "2026-10-05T03")}
    exp.close_day(_date(2026, 10, 5), MOCK_MODELS, fired,
                  {"mock_weak": (False, "예산 부족")}, led, already)
    rows = {r["model_key"]: r for r in
            [json.loads(x) for x in led.path.read_text(encoding="utf-8").splitlines() if x]}
    check("여덟 슬롯을 채운 모델은 완주", rows["mock_strong"]["status"] == "complete")
    check("일부만 돈 모델은 aborted", rows["mock_mid"]["status"] == "aborted",
          rows["mock_mid"]["note"])
    check("예산으로 못 시작한 모델은 not_started",
          rows["mock_weak"]["status"] == "not_started")
    before = len(led.path.read_text(encoding="utf-8").splitlines())
    exp.close_day(_date(2026, 10, 5), MOCK_MODELS, fired, {}, led, already)
    check("같은 날을 두 번 적지 않는다",
          len(led.path.read_text(encoding="utf-8").splitlines()) == before)

    print("\n[스트리밍 델타 파싱]")

    from providers.openai_compat import OpenAICompatAdapter as _OC

    def _delta(d):
        return _OC._stream_event(
            None, "", {"choices": [{"delta": d}]}, {})

    check("본문 토큰을 델타로 센다", _delta({"content": "A"}) == "A")
    check("추론 토큰도 델타로 센다 (DeepSeek 2026-09-11)",
          _delta({"content": None, "reasoning_content": "We"}) == "We")
    check("본문이 있으면 본문을 쓴다",
          _delta({"content": "A", "reasoning_content": "x"}) == "A")
    check("빈 델타는 TTFT를 잡지 않는다",
          _delta({"content": None, "reasoning_content": ""}) == "")
    check("stream_options를 거부하는 모델은 스펙에서 꺼져 있다",
          not next(m for m in ALL_MODELS if m.key == "qwen_flash").stream_usage)

    qwen = next(m for m in ALL_MODELS if m.key == "qwen_flash")
    _a = _OC(qwen)
    _p = _a._stream_payload(_a._payload([], 0.0, 64, False, 0))
    check("지연 프로브는 추론까지 묶는 상한으로 간다",
          _p.get("max_completion_tokens") == 64 and "max_tokens" not in _p)
    _q = _a._payload([], 0.0, 8192, False, 0)
    check("품질 프로브는 max_tokens 그대로",
          _q.get("max_tokens") == 8192 and "max_completion_tokens" not in _q)
    _h = next(m for m in ALL_MODELS if m.key == "openai_gpt56_luna")
    _hp = _OC(_h)
    _hp2 = _hp._stream_payload(_hp._payload([], 0.0, 64, False, 0))
    check("갈아 끼울 것이 없는 모델은 건드리지 않는다",
          _hp2.get(_h.max_tokens_param) == 64)

    print()
    if _failures:
        print(f"실패 {len(_failures)}건: {_failures}")
        sys.exit(1)
    print("전부 통과.")

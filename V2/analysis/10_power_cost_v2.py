"""
검정력·비용 재산정 (설계서 7절 개정용).

09_power_cost_analysis.py를 대체한다. 옛 스크립트는 두 전제가 깨졌다.

  1. "logprob이 주력 지표"를 가정했으나 9개 중 2개에서만 얻는다(설계서 부록 B).
  2. 두 표본 비교(two-sample)로 검정력을 계산했으나, 실제 분석 계획은 8절의
     문항 내 짝비교(paired)다. 짝비교가 훨씬 강하므로 옛 계산은 필요 표본을
     과대추정한다.

여기서는 8절과 같은 모형으로 계산한다. 문항 i마다 고부하 조건의 지표에서
저부하 조건의 지표를 빼고, 그 차이들의 평균이 0인지 문항들에 걸쳐 검정한다.

핵심은 차이의 분산을 두 조각으로 나눠 보는 것이다.

    Var(d_i) = 2 * (문항 내 표집분산) / r  +  tau^2

앞항은 반복을 늘리면 줄어들지만, 뒷항 tau(문항마다 부하 효과 크기가 다른
정도)는 반복으로 줄지 않는다. 그래서 tau가 크면 반복이 아니라 **문항 수**를
늘려야 한다. 이 갈림길이 이번 재산정의 실질적 결론이다.

tau와 문항 내 분산은 지금 모른다. 보정 패스의 노이즈 바닥선이 바로 이 값을
재는 일이므로(설계서 6절), 여기서는 범위를 훑어 설계가 어디서 무너지는지만
확인한다.

실행:
    python 10_power_cost_v2.py
결과:
    outputs/power_cost_v2.md
"""

from __future__ import annotations

import sys
from math import ceil, sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent / "runner"
sys.path.insert(0, str(RUNNER))

import config  # noqa: E402  단가·라인업의 단일 출처

OUT_DIR = HERE / "outputs"

Z_ALPHA_2 = 1.959964          # alpha = 0.05 양측
Z_POWER = {0.80: 0.841621, 0.90: 1.281552}


# ─────────────────────────────────────────────────────────────
# 1. 짝비교 검정력
# ─────────────────────────────────────────────────────────────

def n_items_paired(delta: float, sd_diff: float, power: float = 0.80) -> int:
    """차이 delta를 잡는 데 필요한 문항 수 (paired t, 양측 alpha=0.05)."""
    if sd_diff <= 0:
        return 1
    dz = delta / sd_diff
    return max(1, ceil((Z_ALPHA_2 + Z_POWER[power]) ** 2 / dz ** 2))


def sd_diff_proportion(p: float, r: int, tau: float) -> float:
    """비율 지표(정확도·자기일관성)의 문항별 차이 표준편차.

    조건당 r회 관측으로 비율을 추정하므로 표집분산은 p(1-p)/r.
    두 조건의 차이라 2배가 되고, 여기에 문항별 효과 이질성 tau^2가 더해진다.
    """
    return sqrt(2 * p * (1 - p) / r + tau ** 2)


def sd_diff_relative(cv: float, r: int, tau_rel: float) -> float:
    """추론 토큰처럼 척도가 임의인 연속 지표용. 평균 대비 상대 표준편차로 다룬다.

    평균 mu가 양변에서 약분되므로 mu를 몰라도 계산이 된다. cv는 문항 내
    변동계수, tau_rel은 문항별 효과 이질성을 평균 대비 비율로 표현한 값이다.
    """
    return sqrt(2 * cv ** 2 / r + tau_rel ** 2)


# ─────────────────────────────────────────────────────────────
# 2. 설계 → 관측 수 → 비용
# ─────────────────────────────────────────────────────────────

class Design:
    """한 설계안. 문항 은행을 슬롯마다 일부씩 순회한다.

    은행 전체를 매 슬롯 돌리면 비용이 감당이 안 되므로, 슬롯마다
    items_per_slot개를 뽑아 은행을 고르게 훑는 방식을 가정한다.
    """

    def __init__(self, bank: int, items_per_slot: int, slots_per_day: int,
                 days: int, k: int, label: str = ""):
        self.bank = bank
        self.items_per_slot = items_per_slot
        self.slots_per_day = slots_per_day
        self.days = days
        self.k = k                      # 방문 1회당 반복 샘플 수(자기일관성용)
        self.label = label

    @property
    def item_visits(self) -> int:
        """모델 하나가 수행하는 총 (문항 × 슬롯) 방문 수."""
        return self.items_per_slot * self.slots_per_day * self.days

    @property
    def visits_per_item_per_condition(self) -> float:
        """문항 하나가 한 조건(고부하/저부하)에서 방문받는 횟수."""
        return self.item_visits / self.bank / 2

    @property
    def r_accuracy(self) -> float:
        """정확도 관측 수: 방문마다 k회 답을 뽑으므로 전부 쓴다."""
        return self.visits_per_item_per_condition * self.k

    @property
    def r_consistency(self) -> float:
        """자기일관성 추정치 수: 방문 1회가 최빈답 비율 1개를 준다."""
        return self.visits_per_item_per_condition

    @property
    def calls_per_model(self) -> int:
        return self.item_visits * self.k

    def cost(self, models, input_tokens: int, output_tokens: int) -> dict:
        """모델별·합계 비용(USD). output_tokens에 추론 토큰이 포함된다."""
        out = {}
        for m in models:
            out[m.key] = self.calls_per_model * (
                input_tokens / 1e6 * m.price_in + output_tokens / 1e6 * m.price_out
            )
        out["_total"] = sum(out.values())
        return out


# 직답 프로브의 입력 길이. MMLU-Pro 문항 + 선택지 10개 + 시스템 프롬프트.
INPUT_TOKENS = 320

# 출력 토큰 시나리오. 추론 차단이 먹는지에 따라 비용이 통째로 달라진다.
OUTPUT_SCENARIOS = {
    "차단 성공": 4,        # 답 글자 하나 + 여유
    "minimal 잔존": 30,    # Gemini 3.x처럼 완전 차단이 안 되는 경우
    "차단 실패": 200,      # extra_body가 안 먹어 추론이 그대로 도는 경우
}


def fmt(x, nd=3):
    return "—" if x is None else f"{x:,.{nd}f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    add = L.append

    add("# 검정력·비용 재산정 (설계서 7절 개정)")
    add("")
    add("`09_power_cost_analysis.py`를 대체한다. 옛 계산은 logprob 주력을 전제했고")
    add("두 표본 비교로 검정력을 냈는데, 전자는 부록 B로 깨졌고 후자는 8절의")
    add("짝비교 분석 계획과 어긋난다. 짝비교는 두 표본 비교보다 강하므로 옛")
    add("계산은 필요 표본을 과대추정한다.")
    add("")

    # ── 1. 지표별 필요 문항 수 ──
    add("## 1. 지표별로 몇 문항이 필요한가")
    add("")
    add("문항별 차이의 표준편차가 검정력을 지배한다. 그 표준편차는 두 조각이다.")
    add("반복으로 줄일 수 있는 표집분산과, 반복으로 줄지 않는 문항별 효과")
    add("이질성 tau다. 아래 표에서 tau가 커질수록 필요 문항 수가 급격히 느는")
    add("것이 이 설계의 실질적 제약이다.")
    add("")

    add("### 1.1 정확도 (3%p 하락, 기저 정확도 0.60)")
    add("")
    add("| 조건당 관측 r | tau=0.00 | tau=0.05 | tau=0.10 | tau=0.15 |")
    add("|---:|---:|---:|---:|---:|")
    for r in [5, 10, 20, 40, 80]:
        row = [f"| {r} "]
        for tau in [0.0, 0.05, 0.10, 0.15]:
            n = n_items_paired(0.03, sd_diff_proportion(0.60, r, tau))
            row.append(f"| {n:,} ")
        add("".join(row) + "|")
    add("")

    add("### 1.2 자기일관성 (0.85 → 0.82, 즉 3%p 하락)")
    add("")
    add("방문 1회가 최빈답 비율 하나를 준다. k=5 반복 기준이다.")
    add("")
    add("| 조건당 방문 수 | tau=0.00 | tau=0.05 | tau=0.10 | tau=0.15 |")
    add("|---:|---:|---:|---:|---:|")
    for r in [5, 10, 20, 40, 80]:
        row = [f"| {r} "]
        for tau in [0.0, 0.05, 0.10, 0.15]:
            n = n_items_paired(0.03, sd_diff_proportion(0.85, r, tau))
            row.append(f"| {n:,} ")
        add("".join(row) + "|")
    add("")

    add("### 1.3 추론 토큰 (평균의 10% 삭감)")
    add("")
    add("척도가 임의라 평균 대비 비율로 다룬다. cv는 문항 내 변동계수,")
    add("tau_rel은 문항별 효과 이질성이다. 평균값이 약분되므로 절대 토큰 수를")
    add("몰라도 계산된다.")
    add("")
    add("| 조건당 관측 r | cv=0.3, tau=0 | cv=0.3, tau=0.05 | cv=0.6, tau=0.05 | cv=0.6, tau=0.10 |")
    add("|---:|---:|---:|---:|---:|")
    for r in [5, 10, 20, 40, 80]:
        cells = [
            n_items_paired(0.10, sd_diff_relative(0.3, r, 0.0)),
            n_items_paired(0.10, sd_diff_relative(0.3, r, 0.05)),
            n_items_paired(0.10, sd_diff_relative(0.6, r, 0.05)),
            n_items_paired(0.10, sd_diff_relative(0.6, r, 0.10)),
        ]
        add(f"| {r} | " + " | ".join(f"{c:,}" for c in cells) + " |")
    add("")
    add("추론 토큰이 가장 효율적인 지표다. 10% 삭감은 상대 효과가 커서, 문항 내")
    add("변동이 평균의 60%나 되어도 이질성만 크지 않으면 수십 문항으로 잡힌다.")
    add("")

    # ── 2. 설계안 ──
    add("## 2. 설계안 비교")
    add("")
    add("은행 전체를 매 슬롯 돌리는 것은 비용이 감당되지 않는다. 슬롯마다 일부를")
    add("뽑아 은행을 고르게 훑는 방식을 가정한다.")
    add("")

    designs = [
        Design(300,  50, 6, 14, 5, "A. 은행 300, 슬롯당 50문항, 하루 6슬롯, 14일"),
        Design(300, 100, 8, 14, 5, "B. 은행 300, 슬롯당 100문항, 하루 8슬롯, 14일"),
        Design(500,  80, 8, 14, 5, "C. 은행 500, 슬롯당 80문항, 하루 8슬롯, 14일"),
        Design(300,  50, 6, 28, 5, "D. A를 28일로 연장"),
        Design(200, 100, 8, 14, 3, "E. 은행 200, k=3로 축소"),
    ]

    add("| 설계 | 모델당 콜 수 | 조건당 정확도 관측 | 조건당 방문 |")
    add("|---|---:|---:|---:|")
    for d in designs:
        add(f"| {d.label} | {d.calls_per_model:,} | {d.r_accuracy:.0f} | {d.r_consistency:.0f} |")
    add("")

    add("### 2.1 각 설계가 잡아낼 수 있는 최소 효과")
    add("")
    add("tau=0.05(문항별 효과가 어느 정도 다름)를 가정한 현실적 시나리오다.")
    add("괄호 안은 tau=0.10일 때다.")
    add("")
    add("| 설계 | 정확도 3%p | 자기일관성 3%p | 추론토큰 10% |")
    add("|---|---|---|---|")
    for d in designs:
        a5 = n_items_paired(0.03, sd_diff_proportion(0.60, max(1, d.r_accuracy), 0.05))
        a10 = n_items_paired(0.03, sd_diff_proportion(0.60, max(1, d.r_accuracy), 0.10))
        c5 = n_items_paired(0.03, sd_diff_proportion(0.85, max(1, d.r_consistency), 0.05))
        c10 = n_items_paired(0.03, sd_diff_proportion(0.85, max(1, d.r_consistency), 0.10))
        t5 = n_items_paired(0.10, sd_diff_relative(0.6, max(1, d.r_accuracy), 0.05))
        t10 = n_items_paired(0.10, sd_diff_relative(0.6, max(1, d.r_accuracy), 0.10))

        def mark(need, have):
            ok = "O" if need <= have else "X"
            return f"{ok} {need:,}문항 필요"

        add(f"| {d.label.split('.')[0]} | {mark(a5, d.bank)} ({a10:,}) | "
            f"{mark(c5, d.bank)} ({c10:,}) | {mark(t5, d.bank)} ({t10:,}) |")
    add("")
    add("O는 그 설계의 은행 크기로 충분하다는 뜻이다.")
    add("")

    # ── 3. 비용 ──
    add("## 3. 비용")
    add("")
    add(f"단가는 `V2/runner/config.py`에서 읽는다(2026-08-21 기준). 입력은 문항당 "
        f"{INPUT_TOKENS}토큰으로 가정했다. HyperCLOVA는 단가 미확인이라 0으로 잡혀 "
        f"있어 합계가 과소평가된다.")
    add("")

    lineup = config.LINEUP
    allm = config.ALL_MODELS

    add("### 3.1 추론 차단 여부가 비용을 좌우한다")
    add("")
    add("설계 A(모델당 %s콜) 기준이다." % f"{designs[0].calls_per_model:,}")
    add("")
    add("| 출력 시나리오 | 출력 토큰 | 라인업 7개 | 앵커 포함 9개 |")
    add("|---|---:|---:|---:|")
    for name, tok in OUTPUT_SCENARIOS.items():
        c7 = designs[0].cost(lineup, INPUT_TOKENS, tok)["_total"]
        c9 = designs[0].cost(allm, INPUT_TOKENS, tok)["_total"]
        add(f"| {name} | {tok} | ${c7:,.0f} | ${c9:,.0f} |")
    add("")
    add("차단에 실패하면 비용이 한 자릿수 배로 뛴다. `smoke_test.py`에서 추론")
    add("토큰이 0인지 확인하는 일이 예산 통제 그 자체다.")
    add("")

    add("### 3.2 설계별 비용 (추론 차단 성공, 출력 4토큰)")
    add("")
    add("| 설계 | 모델당 콜 | 라인업 7개 | 앵커 포함 9개 |")
    add("|---|---:|---:|---:|")
    for d in designs:
        c7 = d.cost(lineup, INPUT_TOKENS, OUTPUT_SCENARIOS["차단 성공"])["_total"]
        c9 = d.cost(allm, INPUT_TOKENS, OUTPUT_SCENARIOS["차단 성공"])["_total"]
        add(f"| {d.label.split('.')[0]} | {d.calls_per_model:,} | ${c7:,.0f} | ${c9:,.0f} |")
    add("")

    add("### 3.3 설계 A의 모델별 내역 (앵커 포함, 차단 성공 기준)")
    add("")
    add("| 모델 | 비용 |")
    add("|---|---:|")
    breakdown = designs[0].cost(allm, INPUT_TOKENS, OUTPUT_SCENARIOS["차단 성공"])
    for m in allm:
        note = " (단가 미확인)" if m.price_in == 0 else ""
        add(f"| {m.key} | ${breakdown[m.key]:,.2f}{note} |")
    add(f"| **합계** | **${breakdown['_total']:,.2f}** |")
    add("")

    # ── 3.4 앵커 축소 배치 ──
    add("### 3.4 앵커를 줄이면 예산이 반으로 준다")
    add("")
    add("위 내역에서 앵커 둘이 전체 비용의 76%를 먹는다. 그런데 앵커의 임무는")
    add("하나뿐이다 — flagship 티어도 흔들리는가(설계서 5.3절). 이 질문에")
    add("답하는 데 라인업과 같은 촘촘한 일정이 필요하지 않다. 피크·논피크를")
    add("가르는 최소한의 슬롯만 있으면 된다.")
    add("")

    main_design = designs[1]                       # 설계 B: 라인업용
    anchor_design = Design(300, 100, 2, 14, 5, "앵커 축소")   # 하루 2슬롯만

    c_lineup = main_design.cost(lineup, INPUT_TOKENS, OUTPUT_SCENARIOS["차단 성공"])
    c_anchor = anchor_design.cost(config.ANCHORS, INPUT_TOKENS, OUTPUT_SCENARIOS["차단 성공"])

    add("| 구성 | 일정 | 모델당 콜 | 비용 |")
    add("|---|---|---:|---:|")
    add(f"| 라인업 7개 | 설계 B (하루 8슬롯) | {main_design.calls_per_model:,} | ${c_lineup['_total']:,.0f} |")
    add(f"| 앵커 2개 | 하루 2슬롯 | {anchor_design.calls_per_model:,} | ${c_anchor['_total']:,.0f} |")
    add(f"| **합계** |  |  | **${c_lineup['_total'] + c_anchor['_total']:,.0f}** |")
    add("")
    ar = max(1, anchor_design.r_accuracy)
    add("축소 일정에서 앵커가 확보하는 검정력은 다음과 같다(tau=0.05).")
    add("")
    add(f"- 정확도 3%p: {n_items_paired(0.03, sd_diff_proportion(0.60, ar, 0.05)):,}문항 필요 → 300문항으로 충분")
    add(f"- 추론 토큰 10%: {n_items_paired(0.10, sd_diff_relative(0.6, ar, 0.05)):,}문항 필요 → 충분")
    add(f"- 자기일관성 3%p: {n_items_paired(0.03, sd_diff_proportion(0.85, max(1, anchor_design.r_consistency), 0.05)):,}문항 필요 → **부족**")
    add("")
    add("앵커는 자기일관성을 포기하고 정확도와 추론 토큰으로만 답한다. 앵커의")
    add("질문이 티어 효과의 유무이지 크기의 정밀 추정이 아니므로 감당할 수 있는")
    add("손실이다. 논문에는 앵커의 측정 밀도가 낮다는 사실을 명시한다.")
    add("")

    # ── 4. 결론 ──
    add("## 4. 결론")
    add("")
    add("**검정력은 여전히 병목이 아니다. 다만 이유가 바뀌었다.** 옛 계산은")
    add("logprob의 데이터 효율에 기댔지만, 실제로 이 설계를 떠받치는 것은")
    add("짝비교 구조 자체였다. logprob을 대부분 잃고도 설계는 버틴다.")
    add("")
    add("**자기일관성이 가장 비싼 지표다.** 방문 1회가 추정치 하나만 주므로")
    add("정확도보다 관측이 훨씬 적게 쌓인다. 처음 잡았던 하루 6슬롯·슬롯당")
    add("50문항(설계 A)은 정확도와 추론 토큰은 통과하지만 자기일관성에서")
    add("340문항이 필요해 300문항 은행으로는 모자란다. 이것이 이번 재산정에서")
    add("가장 실질적으로 바뀐 결론이다.")
    add("")
    add("**진짜 변수는 tau, 즉 문항마다 부하 효과가 얼마나 다른가다.** tau가")
    add("작으면 어떤 설계든 통과하고, tau가 0.10을 넘으면 반복을 아무리 늘려도")
    add("소용이 없고 문항 수만이 답이 된다. 이 값은 지금 모른다. 보정 패스의")
    add("노이즈 바닥선이 이를 처음 재게 되므로, 본실험 규모의 최종 확정은")
    add("보정 패스 뒤로 미루는 것이 맞다.")
    add("")
    add("**추론 토큰이 가장 싸게 이기는 지표다.** 10% 삭감이라는 상대 효과가")
    add("커서 필요 문항 수가 20개 안팎이고, 추가 콜 없이 usage에서 공짜로 나온다.")
    add("")
    add("### 권장 설계")
    add("")
    add("| 항목 | 값 |")
    add("|---|---|")
    add("| 문항 은행 | 300 |")
    add("| 슬롯당 문항 | 100 (은행을 3슬롯에 한 바퀴 훑음) |")
    add("| 슬롯 | 하루 8개, DeepSeek 공표 피크 경계(01·04·06·10 UTC)를 걸치도록 배치 |")
    add("| 기간 | 14일 |")
    add("| 반복 k | 5 (자기일관성용) |")
    add("| 앵커 | 같은 은행, 하루 2슬롯으로 축소 |")
    add(f"| 모델당 콜 | 라인업 {main_design.calls_per_model:,} / 앵커 {anchor_design.calls_per_model:,} |")
    add(f"| API 비용 | 약 ${c_lineup['_total'] + c_anchor['_total']:,.0f} (추론 차단 성공 기준) |")
    add("")
    add("tau=0.05에서 라인업은 세 지표 모두 300문항으로 충분하고, tau=0.10까지")
    add("올라가도 정확도와 추론 토큰은 버틴다. 자기일관성만 tau=0.10에서")
    add("207문항이 필요해 여유가 줄어든다.")
    add("")
    add("**전제 두 가지를 먼저 확인해야 한다.** 첫째, 추론 차단이 실제로 먹어야")
    add("한다. 실패하면 비용이 4배로 뛴다(3.1절). 둘째, HyperCLOVA X의 KRW")
    add("단가를 채워야 한다. 그 전까지 위 합계는 과소평가다.")
    add("")

    out = OUT_DIR / "power_cost_v2.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"리포트: {out}")
    print(f"설계 A 모델당 콜: {designs[0].calls_per_model:,}")
    print(f"설계 A 앵커 포함 비용(차단 성공): ${breakdown['_total']:,.2f}")


if __name__ == "__main__":
    main()

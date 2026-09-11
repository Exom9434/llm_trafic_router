"""조건 정의를 확정하고 그 정의로 검정력을 다시 낸다 (설계서 3.2·7.4·7.5절).

2026-09-06에 만들었다. 그전까지 설계서에는 본실험의 고부하와 저부하를 어느
슬롯으로 가르는지가 없었다. 7절의 검정력 계산은 8슬롯이 4와 4로 갈린다고
전제했는데 근거가 없었고, 3.2절은 이진이 아니라 연속 신호로 다룬다고 적혀
있어 8절의 짝비교와 어긋났다.

조건 정의는 이 설계에서 가장 조작되기 쉬운 자리다. 어느 슬롯이 피크인지를
결과를 보고 정할 수 있으면 나머지 방어가 다 무의미해진다. 그래서 정의를
제공사 공표 구간으로 고정한다.

  미국  평일 08:00~14:00 ET            Anthropic 공표
  중국  매일 01:00~04:00, 06:00~10:00 UTC   DeepSeek 공표
  한국  매일 09:00~12:00, 14:00~18:00 KST   중국 정의의 시차 보정

피크가 아닌 슬롯은 전부 오프피크다. 중간을 두면 어디까지가 중간이냐가 다시
자유도가 되고 그 슬롯의 관측도 버려진다.

실행 창은 10월 안으로 고정한다. 2026년 미국 서머타임이 11월 1일에 끝나므로
그 전이면 미국 피크가 UTC 12:00~18:00으로 고정되어 슬롯 분류가 상수가 된다.

    python 12_condition_power.py
"""

from __future__ import annotations

import statistics
from math import ceil, sqrt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "outputs"

Z_ALPHA_2 = 1.959964
Z_POWER = {0.80: 0.841621}

BANK = 300
CYCLE = 3           # 은행 한 바퀴에 드는 슬롯 수
K = 3               # 방문 1회당 반복
DAYS = 21           # 목표 완주일
WEEKDAY_RATIO = 5 / 7

SLOTS = (0, 3, 6, 9, 12, 15, 18, 21)          # 라인업, UTC
ANCHOR_SLOTS = (0, 3, 12, 15)                 # 앵커, UTC

# 지표 상수. 자기일관성은 2026-09-03 실측(11_k_curve.py), 추론 토큰 변동계수도
# 실측이다. 정확도 기저 0.60은 설계서 값이며 p(1-p)가 최대인 0.5에 가까워
# 보수적이다.
SC_VAR_K3 = 0.01481
SC_DROP_K3 = 0.0305
ACC_BASE = 0.60
ACC_DELTA = 0.03
RT_CV = 0.41
RT_DELTA = 0.10


def n_items(delta: float, sd: float, power: float = 0.80) -> int:
    if sd <= 0:
        return 1
    return max(1, ceil((Z_ALPHA_2 + Z_POWER[power]) ** 2 / (delta / sd) ** 2))


def peak_slots(arm: str, slots=SLOTS) -> list[int]:
    """공표 구간에 드는 슬롯. 구간은 시작 포함, 끝 제외로 읽는다."""
    if arm == "us":        # 평일 08~14 ET = UTC 12~18 (EDT)
        return [u for u in slots if 12 <= u < 18]
    if arm == "cn":        # UTC 01~04, 06~10
        return [u for u in slots if (1 <= u < 4) or (6 <= u < 10)]
    if arm == "kr":        # KST 09~12, 14~18 = UTC 00~03, 05~09
        return [u for u in slots if (0 <= u < 3) or (5 <= u < 9)]
    raise ValueError(arm)


def visits(arm: str, slots=SLOTS, days: int = DAYS) -> tuple[float, float]:
    """문항 하나가 각 조건에서 받는 방문 수. (피크, 오프피크)"""
    pk = peak_slots(arm, slots)
    off = [u for u in slots if u not in pk]
    peak_days = round(days * WEEKDAY_RATIO) if arm == "us" else days
    v_pk = peak_days * len(pk) / CYCLE
    # 미국은 주말의 피크 시각 슬롯도 오프피크로 들어간다
    v_off = (days * len(off) + (days - peak_days) * len(pk)) / CYCLE
    return v_pk, v_off


def needs(v_pk: float, v_off: float) -> dict[str, int]:
    sc = n_items(SC_DROP_K3, sqrt(SC_VAR_K3 / v_pk + SC_VAR_K3 / v_off + TAU ** 2))
    r_pk, r_off = v_pk * K, v_off * K
    acc = n_items(ACC_DELTA, sqrt(ACC_BASE * (1 - ACC_BASE) / r_pk
                                 + ACC_BASE * (1 - ACC_BASE) / r_off + TAU ** 2))
    rt = n_items(RT_DELTA, sqrt(RT_CV ** 2 / r_pk + RT_CV ** 2 / r_off + TAU ** 2))
    return {"정확도": acc, "자기일관성": sc, "추론토큰": rt}


def coverage(v_pk: float, v_off: float, metrics: tuple[str, ...]) -> float:
    global TAU
    lo, hi = 0.0, 0.5
    for _ in range(45):
        mid = (lo + hi) / 2
        TAU = mid
        if max(needs(v_pk, v_off)[m] for m in metrics) <= BANK:
            lo = mid
        else:
            hi = mid
    return lo


TAU = 0.15

ARMS = [
    ("미국", "us", "gpt-5.6-luna, gemini-3.5-flash-lite, claude-haiku-4-5"),
    ("중국", "cn", "deepseek-v4-flash, qwen3.7-flash"),
    ("한국", "kr", "HCX-DASH-002"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    add = L.append

    add("# 조건 정의와 그 정의에서 나온 검정력 (설계서 3.2·7.4·7.5절)")
    add("")
    add(f"완주일 {DAYS}, 반복 k={K}, 은행 {BANK}, 검정력 0.80, 양측 alpha=0.05.")
    add("피크는 제공사 공표 구간이고 나머지 슬롯은 전부 오프피크다.")
    add("")

    add("## 1. 슬롯 배정")
    add("")
    add("| 팔 | 모델 | 피크 슬롯 (UTC) | 오프피크 슬롯 (UTC) |")
    add("|---|---|---|---|")
    for label, arm, models in ARMS:
        pk = peak_slots(arm)
        off = [u for u in SLOTS if u not in pk]
        add(f"| {label} | {models} | {', '.join(f'{u:02d}' for u in pk)} | "
            f"{', '.join(f'{u:02d}' for u in off)} |")
    add("")
    add("미국은 공표 정의가 평일 한정이라 주말의 같은 시각 슬롯도 오프피크로 들어간다.")
    add("")

    add("## 2. 팔별 필요 문항 수와 커버리지")
    add("")
    global TAU
    add("| 팔 | 피크 방문 | 오프피크 방문 | 정확도 | 자기일관성 | 추론토큰 | 커버리지 |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    covs = []
    for label, arm, _ in ARMS:
        v_pk, v_off = visits(arm)
        TAU = 0.15
        nd = needs(v_pk, v_off)
        metrics = ("정확도", "자기일관성", "추론토큰") if arm == "cn" else ("정확도", "자기일관성")
        c = coverage(v_pk, v_off, metrics)
        covs.append(c)
        TAU = 0.15
        nd = needs(v_pk, v_off)
        def f(x):
            return f"{x:,} {'O' if x <= BANK else 'X'}"
        rt = f(nd['추론토큰']) if arm == "cn" else "해당 없음"
        add(f"| {label} | {v_pk:.1f} | {v_off:.1f} | {f(nd['정확도'])} | "
            f"{f(nd['자기일관성'])} | {rt} | {c:.3f} |")
    add("")
    add("*(가운데 세 칸은 tau=0.15에서의 필요 문항 수다. 추론 토큰은 중국 팔의 두")
    add("모델만 값을 주므로 다른 팔에서는 커버리지 계산에 넣지 않는다.)*")
    add("")
    add(f"**라인업 최솟값이 설계 전체의 선언 커버리지가 된다. tau ≤ {min(covs):.3f}, "
        f"내려서 0.15로 선언한다.**")
    add("")

    add("## 3. 앵커 슬롯")
    add("")
    add("앵커는 미국 프로바이더 둘이므로 미국 정의를 따른다. 하루 4슬롯을 쓰는")
    add("이유는 검정력이 아니라 교락 회피다. 조건당 시각이 하나뿐이면 조건 효과와")
    add("그 시각에 고유한 무언가가 갈리지 않는다.")
    add("")
    add("옛 배치 0/6/12/18은 그 목적을 달성하지 못한다. 미국 피크 밴드 UTC 12~18에")
    add("드는 것이 12 하나뿐이기 때문이다. 피크 두 슬롯과 그 12시간 반대편 두")
    add("슬롯으로 바꾼다.")
    add("")
    add("| 배치 | 피크 시각 수 | 오프피크 시각 수 | 피크 방문 | 자기일관성 |")
    add("|---|---:|---:|---:|---:|")
    for name, sl in [("옛 배치 0/6/12/18", (0, 6, 12, 18)),
                     ("새 배치 0/3/12/15", ANCHOR_SLOTS)]:
        pk = peak_slots("us", sl)
        off = [u for u in sl if u not in pk]
        v_pk, v_off = visits("us", sl)
        TAU = 0.15
        n = needs(v_pk, v_off)["자기일관성"]
        add(f"| {name} | {len(pk)} | {len(off)} | {v_pk:.1f} | "
            f"{n:,} {'O' if n <= BANK else 'X'} |")
    add("")
    add("새 배치는 피크 UTC 12, 15와 오프피크 UTC 00, 03이다. 오프피크 쪽은 피크")
    add("슬롯을 12시간 옮긴 자리이며, ET로는 밤 8시와 11시로 미국 프로바이더가")
    add("가장 한가한 때다. 규칙이 한 줄로 적히므로 사후 선택의 여지가 없다.")
    add("")

    out = OUT_DIR / "condition_power.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n리포트: {out}")


if __name__ == "__main__":
    main()

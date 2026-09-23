"""본실험 주 분석과 비열등성 판정 (사전등록 5.1절, 5.3절, 5.4절).

등록한 규칙을 그대로 옮긴다.

  1. 완주한 날만 쓴다(day_status.jsonl의 complete).
  2. 시행 = 한 슬롯에서 한 문항에 대한 세 콜. 셋 중 하나라도 오류나 파싱
     실패면 시행 전체를 결측으로 둔다.
  3. 시행 지표: 자기일관성(최빈 답의 몫), 정확도(정답 비율), 추론 토큰
     (시행 평균의 자연로그).
  4. 문항마다 피크 시행 평균에서 오프피크 시행 평균을 뺀다(단순 평균).
  5. 주 검정은 차이 평균이 0인지 보는 양측 짝 t검정, 모델별 Holm 보정.
  6. 비열등성은 H0: 차이 평균 <= -마진에 대한 한쪽 t검정, 주 검정과 따로
     모델별 Holm 보정. 마진은 3.3절 탐지 목표치다.
  7. 두 검정을 조합해 네 가지 판정 중 하나를 낸다.

실행:
    python 13_primary_analysis.py                          # 러너 outputs에서 읽는다
    python 13_primary_analysis.py --log <main_calls.jsonl> --days <day_status.jsonl>
    python 13_primary_analysis.py --demo                   # 합성 데이터로 자기 점검
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RUNNER_OUT = Path(__file__).resolve().parent.parent / "runner" / "outputs"
OUT_MD = Path(__file__).resolve().parent / "outputs" / "primary_results.md"

ALPHA = 0.05

# 5.3절 표. 모델마다 보정 묶음에 들어가는 지표.
FAMILY = {
    "openai_gpt56_luna": ("sc", "acc"),
    "google_gemini_flash_lite": ("sc", "acc"),
    "anthropic_haiku": ("sc", "acc"),
    "deepseek_v4_flash": ("sc", "acc", "rt"),
    "qwen_flash": ("sc", "acc", "rt"),
    "naver_hcx_dash": ("sc",),
    "openai_gpt56_sol": ("sc",),
    "anthropic_sonnet5": ("sc",),
}

# 3.3절 탐지 목표치 = 비열등성 마진. 추론 토큰은 로그 척도에서 10% 감소.
MARGIN = {"sc": 0.0305, "acc": 0.03, "rt": -math.log(0.9)}
LABEL = {"sc": "자기일관성", "acc": "정확도", "rt": "추론 토큰(log)"}


# ── t 분포 (표준 라이브러리만) ────────────────────────────────

def _betacf(a: float, b: float, x: float) -> float:
    # 불완전 베타 함수의 연분수 (Numerical Recipes 6.4)
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / c
            c = c if abs(c) > 1e-300 else 1e-300
            h *= d * c
        if abs(d * c - 1.0) < 1e-12:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t)."""
    tail = 0.5 * _betai(df / 2, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


# ── 데이터 ────────────────────────────────────────────────

def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def complete_days(path: Path) -> set[tuple[str, str]]:
    return {(r["model_key"], r["day"]) for r in read_jsonl(path) if r.get("status") == "complete"}


def trials(records, done_days) -> dict:
    """(model, slot, item) → 세 콜. 완주일의 품질 콜만."""
    out = defaultdict(list)
    for r in records:
        if r.get("phase") != "main" or r.get("probe") != "quality":
            continue
        if (r["model_key"], r["slot"][:10]) not in done_days:
            continue
        out[(r["model_key"], r["slot"], r["item_id"])].append(r)
    return out


def trial_metrics(calls: list[dict], k: int = 3) -> dict | None:
    """시행 하나의 지표. 세 콜이 모두 답을 내지 않았으면 None(결측)."""
    reps = {c["rep"]: c for c in calls}          # 재시도로 같은 rep가 두 번 적혀도 하나로
    if len(reps) < k or any(c.get("error") or not c.get("parsed_letter") for c in reps.values()):
        return None
    cs = list(reps.values())
    letters = [c["parsed_letter"] for c in cs]
    m = {
        "cond": cs[0]["condition"],
        "sc": Counter(letters).most_common(1)[0][1] / len(letters),
        "acc": sum(c.get("correct") or 0 for c in cs) / len(cs),
    }
    rt = [c.get("reasoning_tokens") for c in cs]
    if all(v for v in rt):
        m["rt"] = math.log(sum(rt) / len(rt))
    return m


def item_differences(tr: dict) -> dict:
    """(model, metric) → 문항별 차이 목록(피크 평균 − 오프피크 평균)."""
    by = defaultdict(lambda: {"peak": [], "offpeak": []})
    for (model, _slot, item), calls in tr.items():
        m = trial_metrics(calls)
        if m is None:
            continue
        for metric in FAMILY.get(model, ()):
            if metric in m:
                by[(model, metric, item)][m["cond"]].append(m[metric])
    diffs = defaultdict(list)
    for (model, metric, _item), v in by.items():
        if v["peak"] and v["offpeak"]:
            diffs[(model, metric)].append(statistics.fmean(v["peak"]) - statistics.fmean(v["offpeak"]))
    return diffs


# ── 검정과 판정 ─────────────────────────────────────────────

def holm(pvals: dict) -> dict:
    """Holm 보정 p값."""
    order = sorted(pvals, key=pvals.get)
    n, running, out = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (n - i) * pvals[key]))
        out[key] = running
    return out


def test(d: list[float], margin: float) -> dict:
    n = len(d)
    mean = statistics.fmean(d)
    se = statistics.stdev(d) / math.sqrt(n)
    if se == 0:
        # 모든 문항의 차이가 같다(대개 전부 0). 분산이 없으니 판정은 평균만으로 정해진다.
        return {"n": n, "mean": mean, "se": 0.0,
                "p_primary": 1.0 if mean == 0 else 0.0,
                "p_ni": 0.0 if mean > -margin else 1.0}
    return {
        "n": n, "mean": mean, "se": se,
        "p_primary": 2 * t_sf(abs(mean / se), n - 1),
        "p_ni": t_sf((mean + margin) / se, n - 1),   # H0: mean <= -margin
    }


def decide(r: dict) -> str:
    if r["p_primary_holm"] < ALPHA:
        return "저하 (가설 지지)" if r["mean"] < 0 else "상승 (별도 발견)"
    if r["p_ni_holm"] < ALPHA:
        return "저하가 있더라도 마진 미만"
    return "판단 불가"


def analyse(diffs: dict) -> list[dict]:
    rows = []
    for model, metrics in FAMILY.items():
        res = {m: test(diffs[(model, m)], MARGIN[m])
               for m in metrics if len(diffs.get((model, m), [])) > 2}
        prim = holm({m: r["p_primary"] for m, r in res.items()})
        ni = holm({m: r["p_ni"] for m, r in res.items()})
        for m, r in res.items():
            r.update(model=model, metric=m, p_primary_holm=prim[m], p_ni_holm=ni[m])
            r["decision"] = decide(r)
            rows.append(r)
    return rows


def render(rows: list[dict]) -> str:
    lines = ["| 모델 | 지표 | 문항 | 차이 평균 | 주 검정 p(Holm) | 비열등 p(Holm) | 판정 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {LABEL[r['metric']]} | {r['n']} | {r['mean']:+.4f} | "
                     f"{r['p_primary_holm']:.4f} | {r['p_ni_holm']:.4f} | {r['decision']} |")
    return "\n".join(lines)


# ── 자기 점검 ─────────────────────────────────────────────

def _demo() -> None:
    # t 분포: t=2, df=10의 양측 p는 0.0734
    assert abs(2 * t_sf(2.0, 10) - 0.07339) < 1e-4
    assert abs(t_sf(0.0, 5) - 0.5) < 1e-12
    # Holm
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert abs(h["a"] - 0.03) < 1e-12 and abs(h["c"] - 0.06) < 1e-12 and abs(h["b"] - 0.06) < 1e-12

    # 합성 로그: luna는 피크에 자주 흔들리고, haiku는 차이 없음, sol은 소표본이라 판단 불가
    rng = random.Random(1)
    recs, days = [], set()
    plan = {"openai_gpt56_luna": (300, 0.35, 0.05), "anthropic_haiku": (300, 0.05, 0.05),
            "openai_gpt56_sol": (6, 0.3, 0.3)}
    for model, (n_items, p_flip_peak, p_flip_off) in plan.items():
        for day in ("2026-10-01", "2026-10-02"):
            days.add((model, day))
        for i in range(n_items):
            for slot, cond, pf in (("2026-10-01T12", "peak", p_flip_peak),
                                   ("2026-10-02T00", "offpeak", p_flip_off)):
                for rep in range(3):
                    letter = "B" if rng.random() < pf else "A"
                    recs.append({"phase": "main", "probe": "quality", "model_key": model,
                                 "slot": slot, "item_id": f"q{i}", "rep": rep, "condition": cond,
                                 "parsed_letter": letter, "correct": int(letter == "A"), "error": None})
    # 결측 규칙: 한 콜이 파싱 실패면 그 시행은 빠진다
    recs[0]["parsed_letter"] = None
    tr = trials(recs, days)
    assert trial_metrics(tr[("openai_gpt56_luna", "2026-10-01T12", "q0")]) is None
    rows = {(r["model"], r["metric"]): r for r in analyse(item_differences(tr))}
    assert rows[("openai_gpt56_luna", "sc")]["decision"].startswith("저하"), rows[("openai_gpt56_luna", "sc")]
    assert rows[("anthropic_haiku", "sc")]["decision"] == "저하가 있더라도 마진 미만", rows[("anthropic_haiku", "sc")]
    assert rows[("openai_gpt56_sol", "sc")]["decision"] == "판단 불가", rows[("openai_gpt56_sol", "sc")]
    print(render(list(rows.values())))
    print("\n자기 점검 통과.")


def main() -> None:
    ap = argparse.ArgumentParser(description="본실험 주 분석과 비열등성 판정")
    ap.add_argument("--log", default=str(RUNNER_OUT / "main_calls.jsonl"))
    ap.add_argument("--days", default=str(RUNNER_OUT / "day_status.jsonl"))
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    tr = trials(read_jsonl(Path(args.log)), complete_days(Path(args.days)))
    table = render(analyse(item_differences(tr)))
    print(table)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(table + "\n", encoding="utf-8")
    print(f"\n저장: {OUT_MD}")


if __name__ == "__main__":
    main()

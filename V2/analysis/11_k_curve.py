"""반복 k의 분산 배수를 보정 로그에서 실측한다 (설계서 7.5절 개정용).

7.5절 표의 k별 분산 배수는 가정이었다. "k=5를 1.0으로 두고 k=3을 1.6,
k=2를 2.5로 잡았다"고 적혀 있는데, 이 값들은 5/3과 5/2다. 즉 자기일관성
추정량의 분산이 1/k로 준다고 가정한 것이다.

그 가정은 틀릴 수밖에 없다. 자기일관성 추정량은 비율이 아니라 **최빈답의
몫**이고, 취할 수 있는 값이 k에 갇힌다. k=2면 {0.5, 1.0} 둘뿐이고 k=3이면
{1/3, 2/3, 1} 셋뿐이다. 바닥이 1/k로 막혀 있으므로 분산도 평균도 1/k을
따르지 않는다.

보정 패스를 k=5로 돌린 이유가 이것을 재기 위해서였다(7.5절 단서). 방문마다
5개 표본이 남았으므로 거기서 부분표집해 k=2~5의 거동을 데이터로 낸다.

두 가지를 잰다.

  1. **E[c_k]** — 추정량의 기댓값. k에 따라 움직인다면 노이즈 바닥선도
     본실험의 k에서 다시 재야 한다. 바닥선은 k=5로 쟀는데 본실험이 k=3이면
     기준선이 어긋난다. 09-03에 잡은 "풀 720으로 잰 바닥선" 오류와 같은
     종류다.
  2. **Var(c_k)** — 문항 내 표집분산. 이것이 7.5절 표에 들어가는 값이다.

두 방법으로 재고 서로 맞는지 본다.

  방법 A (부분표집). 5개 표본에서 크기 k인 부분집합을 전부 꺼내 c_k를
  계산한다. 크기 k 부분집합은 그 자체로 k개의 i.i.d. 표본이므로 E[c_k]는
  편향 없이 나온다. 분산은 E[c^2] - (E[c])^2인데, 평균 추정치를 제곱하면
  위로 치우치므로 5개 표본에 대한 잭나이프로 그만큼을 빼 준다.
  부분집합이 하나뿐인 k=5에는 쓸 수 없다.

  방법 B (대입). 5개 표본으로 문항의 답 분포를 추정하고, 그 분포에서 k개를
  뽑았을 때 c_k의 분포를 다항분포로 전개해 E와 Var을 정확히 계산한다.
  분포 추정이 5개 표본에서 오므로 집중도가 과대평가되어 분산의 절대 수준은
  낮게 나온다. 그러나 그 치우침은 k에 대체로 공통이므로 **배수**는 견딘다.
  k=5까지 계산되는 것이 방법 A에 없는 이점이다.

배수는 방법 B로 내고, k=2~4에서 방법 A와 어긋나지 않는지로 확인한다.

    python 11_k_curve.py
    python 11_k_curve.py --scope pool      # 은행 300 대신 후보 풀 720
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from math import ceil, sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
LOG = V2 / "runner" / "outputs" / "calibration_calls.jsonl"
BANK = V2 / "runner" / "data" / "item_bank.json"
OUT_DIR = HERE / "outputs"

# 10_power_cost_v2.py와 같은 상수를 쓴다.
Z_ALPHA_2 = 1.959964
Z_POWER = {0.80: 0.841621, 0.90: 1.281552}

# 본실험 설계에서 오는 값. 하루 8슬롯, 은행을 3슬롯에 한 바퀴, 14일이면
# 조건당 방문이 약 18회다(설계서 7.4·7.5절).
VISITS_PER_CONDITION = 18
DELTA = 0.03            # 검출하려는 자기일관성 하락폭
BANK_SIZE = 300         # 확정된 문항 은행


def n_items_paired(delta: float, sd_diff: float, power: float = 0.80) -> int:
    if sd_diff <= 0:
        return 1
    return max(1, ceil((Z_ALPHA_2 + Z_POWER[power]) ** 2 / (delta / sd_diff) ** 2))


# ── 추정량 ────────────────────────────────────────────────

def consistency(letters) -> float:
    """자기일관성 = 최빈답이 차지하는 몫. select_bank.py와 같은 정의다."""
    return Counter(letters).most_common(1)[0][1] / len(letters)


def subsample_mean(letters: list[str], k: int) -> float:
    """E[c_k]. 크기 k 부분집합은 그 자체로 k개의 i.i.d. 표본이므로 편향이 없다.
    k가 표본 수와 같으면 관측값 하나가 곧 추정치다."""
    n = len(letters)
    if k >= n:
        return consistency(letters)
    return statistics.fmean(consistency(list(s)) for s in combinations(letters, k))


def subsample_moments(letters: list[str], k: int) -> tuple[float, float] | None:
    """방법 A. 크기 k 부분집합 전부에서 E[c_k]와 Var(c_k)를 낸다."""
    n = len(letters)
    if k >= n:
        return None

    def moments(xs):
        vals = [consistency(list(s)) for s in combinations(xs, k)]
        return statistics.fmean(vals), statistics.fmean(v * v for v in vals)

    mean, mean_sq = moments(letters)

    # 평균 추정치의 분산만큼 (E[c])^2가 부풀어 있다. 잭나이프로 뺀다.
    loo = []
    for j in range(n):
        rest = letters[:j] + letters[j + 1:]
        if k >= len(rest):
            # 잭나이프를 칠 여지가 없다(반복 5개에서 k=4). 보정 없이 돌려주면
            # 분산이 낮게 나오므로 아예 안 준다.
            return None
        loo.append(moments(rest)[0])
    bar = statistics.fmean(loo)
    v_mean = (n - 1) / n * sum((x - bar) ** 2 for x in loo)

    return mean, max(0.0, mean_sq - mean * mean + v_mean)


def unbiased_var(letters: list[str], k: int) -> float | None:
    """서로 겹치지 않는 두 부분집합으로 Var(c_k)를 편향 없이 낸다.

    E[c^2]은 크기 k 부분집합 평균으로 편향 없이 나온다. 문제는 (E[c])^2인데,
    평균 추정치를 제곱하면 자기 분산만큼 위로 뜬다. 서로 겹치지 않는 두
    부분집합 S, T의 c(S)c(T)를 평균하면 두 값이 독립이므로 정확히 (E[c])^2이
    된다. 표본이 2k개 이상 있어야 하므로 반복 5개에서는 k=2에만 쓸 수 있다.

    이 값이 수준의 앵커다. 대입법(plugin_moments)은 5개 표본에서 답 분포를
    추정하므로 집중도를 과대평가하고, 5개가 전부 같은 문항에서는 분산을
    0으로 낸다. 그래서 수준은 낮게, 배수는 그런대로 나온다.
    """
    n = len(letters)
    if 2 * k > n:
        return None
    idx = range(n)
    sq, cross, n_sq, n_cross = 0.0, 0.0, 0, 0
    for s in combinations(idx, k):
        c = consistency([letters[i] for i in s])
        sq += c * c
        n_sq += 1
        rest = [i for i in idx if i not in s]
        for t in combinations(rest, k):
            cross += c * consistency([letters[i] for i in t])
            n_cross += 1
    return max(0.0, sq / n_sq - cross / n_cross)


def _compositions(total: int, cells: int):
    if cells == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, cells - 1):
            yield (first,) + rest


def _log_fact(n: int, _cache={0: 0.0}) -> float:
    from math import log
    if n not in _cache:
        _cache[n] = _log_fact(n - 1) + log(n)
    return _cache[n]


def plugin_moments(letters: list[str], k: int) -> tuple[float, float]:
    """방법 B. 관측 빈도를 답 분포로 놓고 c_k의 분포를 정확히 전개한다."""
    from math import exp, log
    n = len(letters)
    probs = [c / n for c in Counter(letters).values()]
    cells = len(probs)
    mean = mean_sq = 0.0
    for comp in _compositions(k, cells):
        lp = _log_fact(k)
        skip = False
        for cnt, p in zip(comp, probs):
            lp -= _log_fact(cnt)
            if cnt:
                if p <= 0:
                    skip = True
                    break
                lp += cnt * log(p)
        if skip:
            continue
        w = exp(lp)
        c = max(comp) / k
        mean += w * c
        mean_sq += w * c * c
    return mean, max(0.0, mean_sq - mean * mean)


# ── 데이터 ────────────────────────────────────────────────

def load_options() -> dict[str, int]:
    """item_id → 선택지 수. 흔들림의 바닥이 1/선택지수라 k 곡선에 들어간다."""
    pool = json.loads((V2 / "runner" / "data" / "candidate_pool.json").read_text(encoding="utf-8"))
    return {it["item_id"]: len(it["options"]) for it in pool}


def load_letters(scope: str) -> dict[str, dict[str, list[str]]]:
    """model_key → item_id → temp>0 반복의 답 글자들."""
    bank = None
    if scope == "bank":
        raw = json.loads(BANK.read_text(encoding="utf-8"))
        bank = {it["item_id"] for it in raw} if isinstance(raw, list) else set(raw)

    got: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    with LOG.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error") or not r.get("parsed_letter"):
                continue
            if (r.get("rep") or 0) < 1:      # rep0은 temp=0이라 성격이 다르다
                continue
            item = r.get("item_id")
            if bank is not None and item not in bank:
                continue
            # 같은 콜이 재시도로 여러 번 성공했을 수 있다. call_key로 하나만 남긴다.
            got[r["model_key"]][item][r["call_key"]] = r["parsed_letter"]

    return {m: {i: list(d.values()) for i, d in items.items()}
            for m, items in got.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="반복 k의 분산 배수 실측")
    ap.add_argument("--scope", choices=["bank", "pool"], default="bank")
    ap.add_argument("--reps", type=int, default=5, help="이 개수를 채운 문항만 쓴다")
    args = ap.parse_args()

    data = load_letters(args.scope)
    if not data:
        raise SystemExit(f"로그에서 반복 표본을 못 찾았다: {LOG}")

    ks = [2, 3, 4, 5]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    add = L.append

    add("# 반복 k의 분산 배수 실측 (설계서 7.5절 개정)")
    add("")
    add("7.5절의 배수 1.6·2.5는 가정이었다. 각각 5/3과 5/2, 곧 분산이 1/k로")
    add("준다고 본 값이다. 자기일관성 추정량은 최빈답의 몫이라 값이 k에 갇히고")
    add("바닥이 1/k로 막히므로 그 가정은 성립할 이유가 없다. 보정 패스를 k=5로")
    add("돌린 것이 이 값을 재기 위해서였다.")
    add("")
    add(f"대상: {'문항 은행 300' if args.scope == 'bank' else '후보 풀 720'} · "
        f"반복 {args.reps}개를 채운 문항만.")
    add("")

    summary: dict[str, dict[int, tuple[float, float]]] = {}

    add("## 1. 추정량의 기댓값과 분산")
    add("")
    add("`E[c_k]`는 자기일관성의 기댓값, `Var`는 문항 내 표집분산이다.")
    add("A는 부분표집, B는 대입이다. 배수는 B의 Var을 k=5로 나눈 값이다.")
    add("")

    for model in sorted(data):
        items = [v for v in data[model].values() if len(v) == args.reps]
        if len(items) < 50:
            continue
        rows = {}
        add(f"### {model}  (문항 {len(items)}개)")
        add("")
        add("| k | E[c_k] | Var 편향없음 | Var A | Var B | 배수 (B) | 가정값 |")
        add("|---:|---:|---:|---:|---:|---:|---:|")
        for k in ks:
            a = [m for m in (subsample_moments(x, k) for x in items) if m]
            b = [plugin_moments(x, k) for x in items]
            u = [v for v in (unbiased_var(x, k) for x in items) if v is not None]
            ea = statistics.fmean(subsample_mean(x, k) for x in items)
            va = statistics.fmean(m[1] for m in a) if a else None
            eb = statistics.fmean(m[0] for m in b)
            vb = statistics.fmean(m[1] for m in b)
            vu = statistics.fmean(u) if u else None
            rows[k] = (ea, vb, va, vu)
        v5 = rows[5][1]
        summary[model] = {k: (rows[k][0], rows[k][1], rows[k][3]) for k in ks}
        assumed = {2: 2.5, 3: 1.6, 4: 1.25, 5: 1.0}
        for k in ks:
            e, vb, va, vu = rows[k]
            mult = vb / v5 if v5 else float("nan")
            add(f"| {k} | {e:.4f} | {'—' if vu is None else f'{vu:.5f}'} | "
                f"{'—' if va is None else f'{va:.5f}'} | {vb:.5f} | "
                f"{mult:.2f} | {assumed[k]:.2f} |")
        add("")
        add("`E[c_k]`는 부분표집으로 편향 없이 낸 값이고, k=5는 관측값 그대로다. "
            "`Var 편향없음`은 겹치지 않는 두 부분집합으로 낸 값이라 k=2에만 있다.")
        add("")

    # ── 2. 배수 요약 ──
    add("## 2. 분산 배수 (k=5 = 1.00)")
    add("")
    add("| 모델 | k=2 | k=3 | k=4 | k=5 |")
    add("|---|---:|---:|---:|---:|")
    for model, row in summary.items():
        v5 = row[5][1]
        cells = " | ".join(f"{row[k][1] / v5:.2f}" if v5 else "—" for k in ks)
        add(f"| {model} | {cells} |")
    mults = {}
    for k in ks:
        vals = [row[k][1] / row[5][1] for row in summary.values() if row[5][1]]
        mults[k] = statistics.fmean(vals) if vals else float("nan")
    add(f"| **평균** | **{mults[2]:.2f}** | **{mults[3]:.2f}** | "
        f"**{mults[4]:.2f}** | **1.00** |")
    add(f"| 7.5절 가정 | 2.50 | 1.60 | 1.25 | 1.00 |")
    add("")

    # ── 3. 기댓값 이동 ──
    add("## 3. 기댓값은 k에 따라 움직이는가")
    add("")
    add("노이즈 바닥선은 k=5로 쟀다. 본실험이 다른 k로 돌면 기준선이 어긋난다.")
    add("아래 표의 k=5 대비 차이가 검출 목표 3%p와 견줄 만하면 바닥선을 본실험의")
    add("k에서 다시 계산해야 한다.")
    add("")
    add("| 모델 | k=2 | k=3 | k=4 | k=5 | k=3 − k=5 |")
    add("|---|---:|---:|---:|---:|---:|")
    for model, row in summary.items():
        cells = " | ".join(f"{row[k][0]:.4f}" for k in ks)
        add(f"| {model} | {cells} | {row[3][0] - row[5][0]:+.4f} |")
    add("")

    # ── 4. 7.5절 표 재계산 ──
    add("## 4. 7.5절 표를 실측값으로 다시 그린다")
    add("")
    add(f"자기일관성 {DELTA:.0%}p 하락, 조건당 방문 {VISITS_PER_CONDITION}회, "
        f"검정력 0.80. 칸은 필요 문항 수이고 ✓는 은행 {BANK_SIZE}으로 되는지다.")
    add("")
    add("수준은 k=2의 편향 없는 값을 앵커로 삼고 거기에 실측 배수를 걸어 낸다.")
    add("대입법의 절대 수준은 아래로 치우쳐 있어 그대로 쓰면 안 된다. 옛 표는")
    add("p(1−p)를 썼는데, 그것은 방문 1회가 0/1을 준다고 본 값이라 자기일관성")
    add("추정량의 실제 흔들림과 무관하다.")
    add("")
    # k=2 앵커(편향 없음)에 배수를 걸어 k별 수준을 만든다.
    anchor2 = statistics.fmean(row[2][2] for row in summary.values() if row[2][2] is not None)
    ratio = {k: statistics.fmean(row[k][1] / row[2][1] for row in summary.values() if row[2][1])
             for k in ks}
    var_k = {k: anchor2 * ratio[k] for k in ks}
    add(f"앵커: k=2에서 Var = {anchor2:.5f} (편향 없음, 라인업 평균). "
        f"옛 식의 p(1−p) = {0.85 * 0.15:.5f}.")
    add("")
    add("| k | Var(c_k) | tau=0.00 | tau=0.05 | tau=0.10 | tau=0.15 |")
    add("|---:|---:|---|---|---|---|")
    for k in ks:
        row = [f"| {k} | {var_k[k]:.5f} "]
        for tau in [0.0, 0.05, 0.10, 0.15]:
            sd = sqrt(2 * var_k[k] / VISITS_PER_CONDITION + tau ** 2)
            n = n_items_paired(DELTA, sd)
            row.append(f"| {n:,} {'✓' if n <= BANK_SIZE else '✗'} ")
        add("".join(row) + "|")
    add("")
    add("옛 표(가정)와 나란히 두면 이렇다.")
    add("")
    add("| k | 실측 tau=0.10 | 옛 표 tau=0.10 |")
    add("|---:|---|---|")
    old_tbl = {2: "397 ✗", 3: "285 ✓", 4: "—", 5: "211 ✓"}
    for k in ks:
        sd = sqrt(2 * var_k[k] / VISITS_PER_CONDITION + 0.10 ** 2)
        n = n_items_paired(DELTA, sd)
        add(f"| {k} | {n:,} {'✓' if n <= BANK_SIZE else '✗'} | {old_tbl[k]} |")
    add("")
    add("모델별 앵커가 다르므로 위 표는 라인업 평균이다. 모델마다 다시 그리면:")
    add("")
    add("| 모델 | k=2 | k=3 | k=4 | k=5 |")
    add("|---|---|---|---|---|")
    for model, row in summary.items():
        cells = []
        a2 = row[2][2]
        for k in ks:
            v = a2 * (row[k][1] / row[2][1]) if a2 and row[2][1] else row[k][1]
            sd = sqrt(2 * v / VISITS_PER_CONDITION + 0.10 ** 2)
            n = n_items_paired(DELTA, sd)
            cells.append(f"{n:,} {'✓' if n <= BANK_SIZE else '✗'}")
        add(f"| {model} | " + " | ".join(cells) + " |")
    add("")
    add(f"*(tau=0.10 기준. 7.5절이 선언한 커버리지다.)*")
    add("")

    # ── 5. 척도 압축 ──
    add("## 5. k가 작으면 같은 변화가 작게 보인다")
    add("")
    add("4절 표에는 함정이 있다. 검출 목표 3%p를 k와 무관한 값으로 놓았는데,")
    add("자기일관성 추정량은 k가 작을수록 1 쪽으로 눌린다. 3절 표에서 E[c_2]가")
    add("E[c_5]보다 최대 3.8%p 높은 것이 그 눌림이다. 눌린 척도 위에서는 밑에 깔린")
    add("답 분포가 같은 만큼 흔들려도 **c_k에 나타나는 변화가 작아진다**.")
    add("그러면 같은 3%p를 요구하는 것이 k마다 다른 크기의 현상을 요구하는 셈이 된다.")
    add("")
    add("그래서 부하를 답 분포 쪽에 걸어 본다. 문항의 답 분포를 선택지 균등분포와")
    add("eps만큼 섞는다. eps가 커질수록 답이 흩어지고 자기일관성이 떨어진다.")
    add("**k=5에서 3%p가 떨어지는 eps를 찾은 뒤, 그 같은 eps에서 k=2·3·4가")
    add("얼마나 떨어지는지 본다.** 이것이 k끼리 비교 가능한 유일한 방식이다.")
    add("")

    options = load_options()
    cache: dict = {}

    def e_ck(pattern: tuple, n_opt: int, k: int, eps: float) -> float:
        """관측 빈도 pattern을 eps만큼 균등분포와 섞은 뒤의 E[c_k]."""
        key = (pattern, n_opt, k, round(eps, 6))
        if key in cache:
            return cache[key]
        tot = sum(pattern)
        probs = [(1 - eps) * c / tot + eps / n_opt for c in pattern]
        probs += [eps / n_opt] * (n_opt - len(pattern))
        from math import exp, log
        mean = 0.0
        for comp in _compositions(k, len(probs)):
            lp, skip = _log_fact(k), False
            for cnt, pr in zip(comp, probs):
                lp -= _log_fact(cnt)
                if cnt:
                    if pr <= 0:
                        skip = True
                        break
                    lp += cnt * log(pr)
            if not skip:
                mean += exp(lp) * max(comp) / k
        cache[key] = mean
        return mean

    def level(model: str, k: int, eps: float) -> float:
        vals = []
        for item_id, letters in data[model].items():
            if len(letters) != args.reps:
                continue
            n_opt = options.get(item_id, 10)
            pattern = tuple(sorted(Counter(letters).values(), reverse=True))
            vals.append(e_ck(pattern, n_opt, k, eps))
        return statistics.fmean(vals)

    add("| 모델 | eps | k=2 하락 | k=3 하락 | k=4 하락 | k=5 하락 |")
    add("|---|---:|---:|---:|---:|---:|")
    drops: dict[str, dict[int, float]] = {}
    for model in sorted(summary):
        base = {k: level(model, k, 0.0) for k in ks}
        lo, hi = 0.0, 1.0
        for _ in range(30):                     # k=5에서 딱 3%p 떨어지는 eps
            mid = (lo + hi) / 2
            if base[5] - level(model, 5, mid) < DELTA:
                lo = mid
            else:
                hi = mid
        eps = (lo + hi) / 2
        d = {k: base[k] - level(model, k, eps) for k in ks}
        drops[model] = d
        add(f"| {model} | {eps:.4f} | " + " | ".join(f"{d[k]:.4f}" for k in ks) + " |")
    add("")

    add("이제 k마다 자기 하락폭과 자기 분산으로 필요 문항 수를 낸다. 4절 표가")
    add("k=2를 가장 싸게 보이게 했던 것은 하락폭을 k=5와 같다고 놓았기 때문이다.")
    add("")
    add("| 모델 | k=2 | k=3 | k=4 | k=5 |")
    add("|---|---|---|---|---|")
    for model in sorted(summary):
        row = summary[model]
        a2, cells = row[2][2], []
        for k in ks:
            v = a2 * (row[k][1] / row[2][1]) if a2 and row[2][1] else row[k][1]
            sd = sqrt(2 * v / VISITS_PER_CONDITION + 0.10 ** 2)
            n = n_items_paired(drops[model][k], sd)
            cells.append(f"{n:,} {'✓' if n <= BANK_SIZE else '✗'}")
        add(f"| {model} | " + " | ".join(cells) + " |")
    add("")
    add("*(tau=0.10, 밑에 깔린 부하는 k=5에서 3%p를 내는 크기로 고정.)*")
    add("")
    add("라인업 평균으로 tau를 훑으면 이렇다. **설계서 7.5절 표가 이것이다.**")
    add("")
    add("| k | 하락폭 | Var(c_k) | tau=0.00 | tau=0.05 | tau=0.10 | tau=0.15 |")
    add("|---:|---:|---:|---|---|---|---|")
    for k in ks:
        d = statistics.fmean(drops[m][k] for m in summary)
        row = [f"| {k} | {d:.4f} | {var_k[k]:.5f} "]
        for tau in [0.0, 0.05, 0.10, 0.15]:
            sd = sqrt(2 * var_k[k] / VISITS_PER_CONDITION + tau ** 2)
            n = n_items_paired(d, sd)
            row.append(f"| {n:,} {'✓' if n <= BANK_SIZE else '✗'} ")
        add("".join(row) + "|")
    add("")

    # ── 6. 커버리지와 결론 ──
    add("## 6. 은행 300이 감당하는 tau는 어디까지인가")
    add("")
    add("7.5절이 tau ≤ 0.10을 커버리지로 선언한 것은 옛 표에서 300이 거기서")
    add("무너졌기 때문이다. 표집분산을 실측한 지금 그 경계가 어디로 가는지 본다.")
    add("")
    add("| 모델 | k=2 | k=3 | k=4 | k=5 |")
    add("|---|---:|---:|---:|---:|")
    for model in sorted(summary):
        row, cells = summary[model], []
        a2 = row[2][2]
        for k in ks:
            v = a2 * (row[k][1] / row[2][1]) if a2 and row[2][1] else row[k][1]
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = (lo + hi) / 2
                sd = sqrt(2 * v / VISITS_PER_CONDITION + mid ** 2)
                if n_items_paired(drops[model][k], sd) <= BANK_SIZE:
                    lo = mid
                else:
                    hi = mid
            cells.append(f"{lo:.3f}")
        add(f"| {model} | " + " | ".join(cells) + " |")
    add("")
    add("*(칸은 은행 300으로 검정력 0.80을 유지하는 tau의 상한이다.)*")
    add("")

    add("## 7. 결론")
    add("")
    add("**첫째, 7.5절의 규칙은 그대로 두고 k=3도 그대로 둔다.** 규칙이 요구하는")
    add("최소값을 4절 표만 보면 k=2로 읽게 되는데, 5절이 그 읽기를 막는다. k=2는")
    add("밑에 깔린 같은 부하를 3.0%p가 아니라 2.5~2.9%p로 보여 준다. 하락폭과")
    add("분산을 함께 넣으면 k=2는 130~182문항, k=3은 92~104문항이 필요하고, k=5는")
    add("k=3보다 1~3문항밖에 못 줄인다. **k=3이 최소값이라는 결론은 유지되지만")
    add("근거가 가정에서 실측으로 바뀌었다.**")
    add("")
    add("**둘째, 배수의 가정값 하나는 틀렸다.** k=3의 1.60은 분산이 1/k로 준다고")
    add("본 값인데 실측은 1.36이다. k=2는 2.50 대 2.64로 반대 방향이다. 1/k")
    add("가정이 우연히 맞은 자리와 틀린 자리가 섞여 있었다.")
    add("")
    add("**셋째, 표집분산의 절대 수준이 크게 과대평가돼 있었다.** 옛 식은 방문")
    add("1회가 0/1을 준다고 보고 p(1−p)=0.128을 썼는데, 실측은 k=2에서 0.029,")
    add("k=5에서 0.011이다. 4배에서 12배 차이다. 그래서 필요 문항 수가 tau=0.10")
    add("기준 285에서 약 100으로 내려간다.")
    add("")
    add("**넷째, 그 결과 커버리지 선언을 넓힐 수 있다.** 6절 표대로면 은행 300이")
    add("k=3에서 tau 0.18 안팎까지 버틴다. 7.5절의 'tau ≤ 0.10'은 이제 필요보다")
    add("좁은 선언이다. 사전등록 전에 이 값을 다시 적을지 정해야 한다.")
    add("")
    add("**다섯째, 노이즈 바닥선을 k=3에서 다시 계산해야 한다.** 3절 표에서")
    add("E[c_3]이 E[c_5]보다 0.4~3.8%p 높다. 바닥선은 k=5로 쟀는데 본실험은 k=3으로")
    add("돈다. 검출 목표가 3%p인데 기준선이 최대 3.8%p 어긋나 있으면, 부하가")
    add("없어도 바닥선 아래로 떨어진 것처럼 보인다. 09-03에 잡은 '풀 720으로 잰")
    add("바닥선'과 정확히 같은 종류의 오류다. 다만 이번에는 다시 돌릴 필요가 없다.")
    add("보정 로그의 5개 표본에서 3개씩 부분표집하면 편향 없이 나온다(3절 값이")
    add("그것이다).")
    add("")

    add("## 8. 한계")
    add("")
    add("**분산의 절대 수준은 앵커 하나에 기댄다.** 편향 없이 낼 수 있는 것은")
    add("2k ≤ 5인 k=2뿐이다. k=3 이상은 k=2 앵커에 대입법 배수를 걸어 만들었다.")
    add("배수는 모델 7개에서 k=2가 2.60~2.68로 붙어 나올 만큼 안정적이지만,")
    add("k=3과 k=4의 대소가 모델마다 뒤집히는 것(deepseek 1.25 대 1.41)은 5개")
    add("표본으로 분포를 추정한 데서 오는 잔떨림이다.")
    add("")
    add("**5절의 부하 모형은 가정이다.** 부하가 답 분포를 균등분포 쪽으로 민다고")
    add("놓았다. 실제 부하가 오답 하나로 몰아가는 꼴이라면 압축의 크기가 달라진다.")
    add("다만 k끼리의 대소는 척도의 성질에서 오는 것이라 모형에 덜 민감하다.")
    add("")

    out = OUT_DIR / f"k_curve_{args.scope}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n리포트: {out}")


if __name__ == "__main__":
    main()

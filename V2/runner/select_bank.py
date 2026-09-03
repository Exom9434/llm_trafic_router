"""보정 패스 결과 → 고정 문항 은행 + 모델별 노이즈 바닥선 (설계서 6절 3~4단계).

두 산출물의 목적이 다르다.
  · 문항 은행: 천장·바닥에 붙지 않은 문항만 남겨 본실험이 헤드룸을 갖게 한다.
  · 노이즈 바닥선: 각 모델이 타고난 비일관성을 미리 재 둔다. 본실험에서
    "고부하 때 자기 바닥선 아래로 떨어졌는가"를 검정할 기준선이 된다.

실행:
    python select_bank.py --lo 0.35 --hi 0.90 --per-subject 50

밴드 기본값은 2026-09-01에 0.40~0.85에서 넓혔다. 좁은 밴드로는 history 43개,
psychology 37개밖에 통과하지 못해 과목당 50문항을 채울 수 없었다. 넓힌 이유가
문항 난이도 분포이지 결과 지표가 아니라는 점을 사전등록에 적는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from calllog import read_records
from config import ALL_MODELS, DATA_DIR, OUTPUT_DIR
from itembank import SUBJECTS, load_pool

DEFAULT_LOG = OUTPUT_DIR / "calibration_calls.jsonl"


def group_records(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """(item_id, model_key) → 레코드 목록."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        if rec.get("phase") != "calibration":
            continue
        if not rec.get("item_id") or not rec.get("model_key"):
            continue
        grouped[(rec["item_id"], rec["model_key"])].append(rec)
    return grouped


def item_stats(grouped, pool_by_id) -> list[dict]:
    """문항별 난이도와 건전성 지표.

    오류를 셀 때 실패 행의 개수를 세면 안 된다. 재개 키가 성공한 콜만
    세기 때문에 실패한 콜은 다시 시도되고, 성공하더라도 실패 행은 로그에
    그대로 남는다. 행을 세면 이미 회수된 옛 장애가 결측으로 잡힌다.

    2026-09-03 확인: 네이버의 08-26 429 508건은 전부 재시도로 회수됐는데
    오류율이 10.5%로 찍혔고, 그 때문에 멀쩡한 문항 53개가 은행에서 빠졌다.

    그래서 콜 하나를 `call_key`로 식별하고, 성공 기록이 하나도 없는
    call_key만 결측으로 센다. 분모도 행 수가 아니라 시도한 콜의 가짓수다.
    """
    by_item: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for (item_id, model_key), recs in grouped.items():
        by_item[item_id][model_key] = recs

    rows = []
    for item_id, models in by_item.items():
        per_model_acc = []
        n_attempted = n_missing = n_parse_fail = n_ok = 0
        for model_key, recs in models.items():
            ok = [r for r in recs if r.get("error") is None]
            done = {r.get("call_key") for r in ok}
            attempted = {r.get("call_key") for r in recs}
            n_attempted += len(attempted)
            n_missing += len(attempted - done)
            n_ok += len(ok)
            n_parse_fail += sum(1 for r in ok if r.get("parsed_letter") is None)
            scored = [r["correct"] for r in ok if r.get("correct") is not None]
            if scored:
                per_model_acc.append(sum(scored) / len(scored))
        if not per_model_acc:
            continue
        item = pool_by_id.get(item_id, {})
        rows.append({
            "item_id": item_id,
            "subject": item.get("subject", ""),
            "n_models": len(per_model_acc),
            "n_calls": n_attempted,
            "difficulty": round(statistics.mean(per_model_acc), 4),
            "acc_spread": round(max(per_model_acc) - min(per_model_acc), 4),
            # 끝내 채우지 못한 콜의 비율. 재시도로 회수된 장애는 세지 않는다.
            "error_rate": round(n_missing / n_attempted, 4) if n_attempted else 0.0,
            "parse_fail_rate": round(n_parse_fail / max(1, n_ok), 4),
        })
    return rows


# 본실험의 반복 수(설계서 7.5절). 바닥선은 이 k에서도 함께 낸다.
MAIN_K = 3


def noise_floor(grouped, only_items: set[str] | None = None) -> list[dict]:
    """모델별 노이즈 바닥선.

    self_consistency = temp>0 반복에서 최빈 답이 차지하는 비율의 문항 평균.
    1.0이면 흔들림이 없고, 1/선택지수에 가까우면 사실상 무작위다.

    `only_items`를 주면 그 문항들로만 잰다. 본실험이 순회하는 것은 확정된
    은행이므로 기준선도 같은 문항에서 나와야 한다. 후보 풀 전체로 재면
    모든 모델이 다 맞히는 쉬운 문항이 섞여 바닥선이 위로 뜬다.

    2026-09-03 확인: 풀 720으로 잰 값이 은행 300 기준보다 luna에서 5.4%p,
    gemini에서 3.3%p, haiku에서 3.0%p 높았다. 검출하려는 효과가 3%p이므로
    (설계서 7.2절) 기준선 오차가 효과 크기와 같은 규모다. 그대로 쓰면
    부하가 없어도 바닥선 아래로 떨어진 것처럼 보인다.
    """
    by_model: dict[str, list[list[dict]]] = defaultdict(list)
    for (item_id, model_key), recs in grouped.items():
        if only_items is not None and item_id not in only_items:
            continue
        by_model[model_key].append(recs)

    rows = []
    for model_key, item_groups in sorted(by_model.items()):
        cons, t0_acc, p_gold, logprob, margin, reasoning = [], [], [], [], [], []
        cons_k: list[float] = []
        # 오류는 행이 아니라 끝내 못 채운 콜로 센다. item_stats와 같은 이유다.
        n_attempted = n_missing = n_parse_fail = n_ok = 0

        for recs in item_groups:
            ok = [r for r in recs if r.get("error") is None]
            attempted = {r.get("call_key") for r in recs}
            done = {r.get("call_key") for r in ok}
            n_attempted += len(attempted)
            n_missing += len(attempted - done)
            n_ok += len(ok)
            n_parse_fail += sum(1 for r in ok if r.get("parsed_letter") is None)

            reps = [r["parsed_letter"] for r in ok if r.get("rep", 0) >= 1 and r.get("parsed_letter")]
            if len(reps) >= 2:
                cons.append(Counter(reps).most_common(1)[0][1] / len(reps))
                # 본실험이 도는 k에서의 값도 함께 낸다. 추정량이 k에 따라
                # 움직이기 때문이다. 2026-09-03 실측(analysis/11_k_curve.py)에서
                # E[c_3]이 E[c_5]보다 0.4~3.8%p 높았다. 검출 목표가 3%p인데
                # 기준선이 그만큼 어긋나면 부하가 없어도 바닥선 아래로 떨어진
                # 것처럼 보인다. 풀 720으로 바닥선을 쟀던 오류와 같은 종류다.
                # 다시 돌릴 필요는 없다. 5개 표본에서 k개씩 전부 꺼내 평균하면
                # k개 i.i.d. 표본의 기댓값이 편향 없이 나온다.
                if len(reps) > MAIN_K:
                    cons_k.append(statistics.fmean(
                        Counter(c).most_common(1)[0][1] / MAIN_K
                        for c in combinations(reps, MAIN_K)))
                elif len(reps) == MAIN_K:
                    cons_k.append(Counter(reps).most_common(1)[0][1] / MAIN_K)

            for r in ok:
                if r.get("rep", 0) != 0:
                    continue
                if r.get("correct") is not None:
                    t0_acc.append(r["correct"])
                if r.get("p_gold") is not None:
                    p_gold.append(r["p_gold"])
                if r.get("answer_logprob") is not None:
                    logprob.append(r["answer_logprob"])
                if r.get("margin") is not None:
                    margin.append(r["margin"])
                if r.get("reasoning_tokens") is not None:
                    reasoning.append(r["reasoning_tokens"])

        def avg(xs):
            return round(statistics.mean(xs), 4) if xs else None

        rows.append({
            "model_key": model_key,
            "n_items": len(item_groups),
            "n_calls": n_attempted,
            "error_rate": round(n_missing / n_attempted, 4) if n_attempted else 0.0,
            "parse_fail_rate": round(n_parse_fail / max(1, n_ok), 4),
            "temp0_accuracy": avg(t0_acc),
            "self_consistency": avg(cons),
            # 본실험 기준선. 5절이 "고부하에 이 값 아래로 떨어지는가"를 검정한다.
            f"self_consistency_k{MAIN_K}": avg(cons_k),
            "consistency_sd": round(statistics.pstdev(cons), 4) if len(cons) > 1 else None,
            "mean_p_gold": avg(p_gold),
            "mean_answer_logprob": avg(logprob),
            "mean_margin": avg(margin),
            "mean_reasoning_tokens": avg(reasoning),
            "has_logprob": bool(p_gold),
        })
    return rows


def select_bank(stats, lo, hi, per_subject, max_error, max_parse_fail):
    kept = [
        s for s in stats
        if lo <= s["difficulty"] <= hi
        and s["error_rate"] <= max_error
        and s["parse_fail_rate"] <= max_parse_fail
    ]
    # 과목 균형: 각 과목에서 난이도 중앙(0.6)에 가까운 순으로 뽑는다.
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for s in kept:
        by_subject[s["subject"]].append(s)

    selected = []
    for subject in SUBJECTS:
        items = sorted(by_subject.get(subject, []), key=lambda s: abs(s["difficulty"] - 0.6))
        selected.extend(items[:per_subject])
    return kept, selected


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="문항 은행 선별 + 노이즈 바닥선 산출")
    ap.add_argument("--log", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--lo", type=float, default=0.35)
    ap.add_argument("--hi", type=float, default=0.90)
    ap.add_argument("--per-subject", type=int, default=50)
    ap.add_argument("--max-error", type=float, default=0.10)
    ap.add_argument("--max-parse-fail", type=float, default=0.10)
    ap.add_argument("--models", nargs="*", default=None,
                    help="집계에 넣을 모델. 기본값은 현재 라인업+앵커다.")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else DEFAULT_LOG
    records = read_records(log_path)
    if not records:
        raise SystemExit(f"보정 로그가 비었다: {log_path}")

    # 라인업에서 내린 모델의 기록이 로그에 남아 있다. 그대로 집계하면 두 군데가
    # 틀어진다. 문항 난이도는 이제 돌지 않을 모델의 정답률까지 평균에 넣게 되고,
    # parse_fail_rate는 그 모델의 파싱 실패까지 세어 멀쩡한 문항을 떨어뜨린다.
    allowed = set(args.models) if args.models else {m.key for m in ALL_MODELS}
    before = len(records)
    records = [r for r in records if r.get("model_key") in allowed]
    dropped = sorted({r["model_key"] for r in read_records(log_path)
                      if r.get("model_key") and r["model_key"] not in allowed})
    if dropped:
        print(f"집계 제외 모델: {', '.join(dropped)} ({before - len(records):,}행)")
    if not records:
        raise SystemExit("집계할 기록이 없다. --models를 확인할 것.")

    pool = load_pool(Path(args.pool) if args.pool else None)
    pool_by_id = {i["item_id"]: i for i in pool}

    grouped = group_records(records)
    stats = item_stats(grouped, pool_by_id)
    kept, selected = select_bank(
        stats, args.lo, args.hi, args.per_subject, args.max_error, args.max_parse_fail
    )
    # 바닥선은 은행이 정해진 뒤에 그 문항들로만 잰다. 순서가 뒤바뀌면
    # 본실험이 쓰지도 않을 문항이 기준선에 섞인다.
    bank_ids = {s["item_id"] for s in selected}
    floors = noise_floor(grouped, only_items=bank_ids)

    bank = [pool_by_id[s["item_id"]] for s in selected if s["item_id"] in pool_by_id]
    bank_path = DATA_DIR / "item_bank.json"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv(OUTPUT_DIR / "item_difficulty.csv", sorted(stats, key=lambda s: s["difficulty"]))
    write_csv(OUTPUT_DIR / "noise_floor.csv", floors)

    subj_counts = Counter(s["subject"] for s in selected)
    lines = [
        "# 보정 패스 결과",
        "",
        f"- 후보 문항: {len(stats)}개",
        f"- 난이도 밴드 [{args.lo}, {args.hi}] 통과: {len(kept)}개",
        f"- 과목 균형 후 최종 은행: {len(bank)}개",
        "",
        "## 과목별 문항 수",
        "",
        "| 과목 | 문항 수 |",
        "|---|---:|",
    ]
    lines += [f"| {s} | {subj_counts.get(s, 0)} |" for s in SUBJECTS]
    lines += [
        "",
        "## 모델별 노이즈 바닥선",
        "",
        "`self_consistency`는 temp>0 반복에서 최빈 답의 비율이다. 본실험은 각 모델이",
        "고부하 시간대에 이 값 아래로 떨어지는지를 검정한다.",
        "",
        f"보정은 k=5로 돌았고 본실험은 k={MAIN_K}으로 돈다. 추정량이 k에 따라 움직이므로",
        f"기준선으로 쓸 값은 `자기일관성 k={MAIN_K}` 쪽이다. 5개 표본에서 {MAIN_K}개씩 전부",
        "꺼내 평균한 값이며 편향이 없다.",
        "",
        "| 모델 | temp0 정확도 | 자기일관성 k=5 | 자기일관성 k=3 | 추론토큰 | 평균 p(정답) | 평균 margin | 파싱실패율 | 오류율 | logprob |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for r in floors:
        def fmt(v):
            return "—" if v is None else f"{v:.3f}"
        lines.append(
            f"| {r['model_key']} | {fmt(r['temp0_accuracy'])} | {fmt(r['self_consistency'])} | "
            f"{fmt(r.get(f'self_consistency_k{MAIN_K}'))} | "
            f"{fmt(r['mean_reasoning_tokens'])} | "
            f"{fmt(r['mean_p_gold'])} | {fmt(r['mean_margin'])} | {fmt(r['parse_fail_rate'])} | "
            f"{fmt(r['error_rate'])} | {'O' if r['has_logprob'] else 'X'} |"
        )
    lines += ["", f"산출물: `data/item_bank.json`, `outputs/noise_floor.csv`, `outputs/item_difficulty.csv`", ""]

    report = OUTPUT_DIR / "calibration_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"후보 {len(stats)} → 밴드 통과 {len(kept)} → 최종 은행 {len(bank)}")
    for s in SUBJECTS:
        print(f"  {s:12s} {subj_counts.get(s, 0)}")
    print(f"\n리포트: {report}")


if __name__ == "__main__":
    main()

"""보정 패스 결과 → 고정 문항 은행 + 모델별 노이즈 바닥선 (설계서 6절 3~4단계).

두 산출물의 목적이 다르다.
  · 문항 은행: 천장·바닥에 붙지 않은 문항만 남겨 본실험이 헤드룸을 갖게 한다.
  · 노이즈 바닥선: 각 모델이 타고난 비일관성을 미리 재 둔다. 본실험에서
    "고부하 때 자기 바닥선 아래로 떨어졌는가"를 검정할 기준선이 된다.

실행:
    python select_bank.py --lo 0.40 --hi 0.85 --per-subject 60
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from calllog import read_records
from config import DATA_DIR, OUTPUT_DIR
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
    """문항별 난이도와 건전성 지표."""
    by_item: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for (item_id, model_key), recs in grouped.items():
        by_item[item_id][model_key] = recs

    rows = []
    for item_id, models in by_item.items():
        per_model_acc = []
        n_calls = n_err = n_parse_fail = 0
        for model_key, recs in models.items():
            ok = [r for r in recs if r.get("error") is None]
            n_calls += len(recs)
            n_err += len(recs) - len(ok)
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
            "n_calls": n_calls,
            "difficulty": round(statistics.mean(per_model_acc), 4),
            "acc_spread": round(max(per_model_acc) - min(per_model_acc), 4),
            "error_rate": round(n_err / n_calls, 4) if n_calls else 0.0,
            "parse_fail_rate": round(n_parse_fail / max(1, n_calls - n_err), 4),
        })
    return rows


def noise_floor(grouped) -> list[dict]:
    """모델별 노이즈 바닥선.

    self_consistency = temp>0 반복에서 최빈 답이 차지하는 비율의 문항 평균.
    1.0이면 흔들림이 없고, 1/선택지수에 가까우면 사실상 무작위다.
    """
    by_model: dict[str, list[list[dict]]] = defaultdict(list)
    for (_item_id, model_key), recs in grouped.items():
        by_model[model_key].append(recs)

    rows = []
    for model_key, item_groups in sorted(by_model.items()):
        cons, t0_acc, p_gold, logprob, margin, reasoning = [], [], [], [], [], []
        n_calls = n_err = n_parse_fail = 0

        for recs in item_groups:
            ok = [r for r in recs if r.get("error") is None]
            n_calls += len(recs)
            n_err += len(recs) - len(ok)
            n_parse_fail += sum(1 for r in ok if r.get("parsed_letter") is None)

            reps = [r["parsed_letter"] for r in ok if r.get("rep", 0) >= 1 and r.get("parsed_letter")]
            if len(reps) >= 2:
                cons.append(Counter(reps).most_common(1)[0][1] / len(reps))

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
            "n_calls": n_calls,
            "error_rate": round(n_err / n_calls, 4) if n_calls else 0.0,
            "parse_fail_rate": round(n_parse_fail / max(1, n_calls - n_err), 4),
            "temp0_accuracy": avg(t0_acc),
            "self_consistency": avg(cons),
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
    ap.add_argument("--lo", type=float, default=0.40)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--per-subject", type=int, default=60)
    ap.add_argument("--max-error", type=float, default=0.10)
    ap.add_argument("--max-parse-fail", type=float, default=0.10)
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else DEFAULT_LOG
    records = read_records(log_path)
    if not records:
        raise SystemExit(f"보정 로그가 비었다: {log_path}")

    pool = load_pool(Path(args.pool) if args.pool else None)
    pool_by_id = {i["item_id"]: i for i in pool}

    grouped = group_records(records)
    stats = item_stats(grouped, pool_by_id)
    floors = noise_floor(grouped)
    kept, selected = select_bank(
        stats, args.lo, args.hi, args.per_subject, args.max_error, args.max_parse_fail
    )

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
        "| 모델 | temp0 정확도 | 자기일관성 | 추론토큰 | 평균 p(정답) | 평균 margin | 파싱실패율 | 오류율 | logprob |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for r in floors:
        def fmt(v):
            return "—" if v is None else f"{v:.3f}"
        lines.append(
            f"| {r['model_key']} | {fmt(r['temp0_accuracy'])} | {fmt(r['self_consistency'])} | "
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

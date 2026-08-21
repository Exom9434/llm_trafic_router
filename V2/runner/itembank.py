"""후보 문항 풀 만들기 (설계서 6절 1~2단계).

보정 패스가 통과시킬 후보 풀을 MMLU-Pro에서 만든다. 이 단계에서 하는 일은
과목 균형 맞추기와 명백한 불량 문항 제외까지다. 실제 난이도 선별(40~85%)은
모델을 돌려 봐야 알 수 있으므로 select_bank.py의 몫이다.

실행:
    python itembank.py --per-subject 80 --seed 20260722
결과:
    data/candidate_pool.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from config import DATA_DIR, REPO_ROOT

# V1과 같은 6과목을 쓴다. 비교 가능성을 위해서다.
SUBJECTS = ["math", "physics", "engineering", "history", "law", "psychology"]

HF_DATASET = "TIGER-Lab/MMLU-Pro"
HF_SPLIT = "test"

AUDIT_CSV = REPO_ROOT / "analysis" / "outputs" / "question_key_audit.csv"


def _norm(text: str) -> str:
    """문항 대조용 정규화. 공백만 접는다."""
    return " ".join((text or "").split()).lower()


def load_excluded_questions() -> set[str]:
    """1차 key audit에서 걸린 문항의 정규화 텍스트 집합.

    audit CSV의 suspicion_reasons가 비어 있지 않은 행만 제외 대상이다.
    파일이 없으면 빈 집합을 돌려주고 경고만 남긴다.
    """
    if not AUDIT_CSV.exists():
        print(f"[warn] key audit 파일 없음: {AUDIT_CSV} — 제외 없이 진행한다.", file=sys.stderr)
        return set()

    import csv

    excluded: set[str] = set()
    with open(AUDIT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("suspicion_reasons") or "").strip():
                excluded.add(_norm(row.get("question", "")))
    return excluded


def load_mmlu_pro(cache_dir: str | None = None) -> list[dict]:
    """MMLU-Pro test 스플릿을 내려받아 리스트로 준다."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "datasets 패키지가 필요하다. `pip install datasets` 후 다시 실행할 것."
        )
    ds = load_dataset(HF_DATASET, split=HF_SPLIT, cache_dir=cache_dir)
    return list(ds)


def to_item(row: dict) -> dict | None:
    """MMLU-Pro 한 행을 실험용 문항 스키마로 옮긴다."""
    options = [o for o in (row.get("options") or []) if o not in (None, "N/A")]
    if len(options) < 4:
        return None
    answer = (row.get("answer") or "").strip().upper()
    if not answer or len(answer) != 1:
        return None
    idx = row.get("answer_index")
    if idx is None or idx >= len(options):
        return None
    subject = (row.get("category") or "").strip().lower()
    return {
        "item_id": f"{subject}:{row.get('question_id')}",
        "subject": subject,
        "question": row.get("question", "").strip(),
        "options": options,
        "answer": answer,
        "answer_index": idx,
        "src": row.get("src"),
    }


def build_pool(per_subject: int, seed: int, cache_dir: str | None = None) -> list[dict]:
    rows = load_mmlu_pro(cache_dir=cache_dir)
    excluded = load_excluded_questions()

    by_subject: dict[str, list[dict]] = {s: [] for s in SUBJECTS}
    n_excluded = 0
    for row in rows:
        subject = (row.get("category") or "").strip().lower()
        if subject not in by_subject:
            continue
        item = to_item(row)
        if item is None:
            continue
        if _norm(item["question"]) in excluded:
            n_excluded += 1
            continue
        by_subject[subject].append(item)

    rng = random.Random(seed)
    pool: list[dict] = []
    for subject in SUBJECTS:
        items = by_subject[subject]
        rng.shuffle(items)
        take = items[:per_subject]
        if len(take) < per_subject:
            print(
                f"[warn] {subject}: 후보가 {len(take)}개뿐이다 (요청 {per_subject}).",
                file=sys.stderr,
            )
        pool.extend(take)

    print(f"key audit 제외: {n_excluded}문항")
    return pool


def load_pool(path: Path | None = None) -> list[dict]:
    path = path or (DATA_DIR / "candidate_pool.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="MMLU-Pro 후보 문항 풀 생성")
    ap.add_argument("--per-subject", type=int, default=80,
                    help="과목당 후보 수 (6과목 기준 총 per-subject*6)")
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--cache-dir", default=None, help="HF datasets 캐시 경로")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pool = build_pool(args.per_subject, args.seed, args.cache_dir)
    out = Path(args.out) if args.out else (DATA_DIR / "candidate_pool.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for item in pool:
        counts[item["subject"]] = counts.get(item["subject"], 0) + 1
    print(f"후보 풀 {len(pool)}문항 → {out}")
    for s in SUBJECTS:
        print(f"  {s:12s} {counts.get(s, 0)}")


if __name__ == "__main__":
    main()

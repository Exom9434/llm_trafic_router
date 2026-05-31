from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd

from common import OUTPUT_DIR, ROOT, parse_answer


LETTERS = "ABCDEFGHIJ"
CLOSED_TAG_CONTENT_RE = re.compile(r"<\s*Answer\s*>\s*([^<]+?)\s*<\s*/\s*Answer\s*>", re.I | re.S)
UNCLOSED_TAG_CONTENT_RE = re.compile(r"<\s*Answer\s*>\s*([^<]+?)\s*$", re.I | re.S)


def normalize(text: object) -> str:
    text = "" if text is None else str(text).lower()
    text = text.replace("−", "-")
    text = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9가-힣.+\\-]+", " ", text)
    return " ".join(text.split()).strip(" .")


def answer_tag_content(text: object) -> str | None:
    if text is None:
        return None
    text = str(text)
    match = CLOSED_TAG_CONTENT_RE.search(text) or UNCLOSED_TAG_CONTENT_RE.search(text)
    if match:
        return " ".join(match.group(1).split())
    return None


def option_letter_from_text(text: object, options: list[str]) -> str | None:
    text_norm = normalize(text)
    if not text_norm:
        return None

    exact = []
    loose = []
    for letter, option in zip(LETTERS, options):
        opt_norm = normalize(option)
        if not opt_norm:
            continue
        if text_norm == opt_norm:
            exact.append(letter)
        elif text_norm in opt_norm or opt_norm in text_norm:
            loose.append(letter)

    if len(exact) == 1:
        return exact[0]
    if len(loose) == 1:
        return loose[0]
    return None


def load_observed_answers(cleaned: pd.DataFrame, questions: list[dict]) -> pd.DataFrame:
    rows = []
    for source_file, group in cleaned.groupby("source_file"):
        original = pd.read_csv(ROOT / source_file, usecols=["full_content"])
        for _, row in group.iterrows():
            q_idx = int(row["question_index"])
            if q_idx < 0 or q_idx >= len(questions):
                continue
            options = questions[q_idx]["options"]
            answer = parse_answer(row.get("parsed_answer"))
            if answer is None:
                content = answer_tag_content(original.iloc[int(row["row_index_in_file"])]["full_content"])
                answer = option_letter_from_text(content, options)
            rows.append(
                {
                    "question_index": q_idx,
                    "provider": row["provider"],
                    "period": row["period"],
                    "observed_answer": answer,
                }
            )
    return pd.DataFrame(rows)


def main():
    cleaned_path = OUTPUT_DIR / "cleaned_results.csv"
    if not cleaned_path.exists():
        raise SystemExit("Run analysis/01_build_dataset.py first.")

    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    cleaned = pd.read_csv(cleaned_path)
    observed = load_observed_answers(cleaned, questions)

    rows = []
    for q_idx, question in enumerate(questions):
        answer = str(question.get("answer", "")).strip().upper()
        options = question.get("options", [])
        structural_issue = None
        if answer not in LETTERS:
            structural_issue = "answer_not_A_to_J"
            answer_text = None
        elif LETTERS.index(answer) >= len(options):
            structural_issue = "answer_out_of_option_range"
            answer_text = None
        else:
            answer_text = options[LETTERS.index(answer)]

        q_obs = observed[observed["question_index"] == q_idx]
        counts = Counter(q_obs["observed_answer"].dropna())
        total_observed = int(sum(counts.values()))
        majority_answer = counts.most_common(1)[0][0] if counts else None
        majority_count = counts.most_common(1)[0][1] if counts else 0
        key_count = counts.get(answer, 0)
        majority_share = majority_count / total_observed if total_observed else 0
        key_share = key_count / total_observed if total_observed else 0

        suspicion_reasons = []
        if structural_issue:
            suspicion_reasons.append(structural_issue)
        if total_observed >= 20 and majority_answer and majority_answer != answer and majority_share >= 0.60:
            suspicion_reasons.append("model_consensus_disagrees_with_key")
        if total_observed >= 20 and key_share <= 0.20:
            suspicion_reasons.append("low_support_for_key")

        rows.append(
            {
                "question_index": q_idx,
                "subject": question.get("subject"),
                "difficulty": question.get("difficulty"),
                "key_answer": answer,
                "key_answer_text": answer_text,
                "observed_answer_count": total_observed,
                "majority_answer": majority_answer,
                "majority_answer_text": options[LETTERS.index(majority_answer)]
                if majority_answer in LETTERS and LETTERS.index(majority_answer) < len(options)
                else None,
                "majority_share": majority_share,
                "key_share": key_share,
                "answer_distribution": " ".join(f"{k}:{v}" for k, v in sorted(counts.items())),
                "suspicion_reasons": ", ".join(suspicion_reasons),
                "question": question.get("question"),
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT_DIR / "question_key_audit.csv", index=False, encoding="utf-8-sig")

    suspicious = audit[audit["suspicion_reasons"].astype(str) != ""].copy()
    report = [
        "# Question Key Audit",
        "",
        f"- Questions: {len(audit)}",
        f"- Suspicious questions: {len(suspicious)}",
        "",
        "This is a triage report. `questions.json` remains the benchmark ground truth.",
        "Model consensus and option-text recovery are diagnostic only and do not alter primary scoring.",
        "",
        "## Suspicious Questions",
        "",
        suspicious[
            [
                "question_index",
                "subject",
                "key_answer",
                "key_answer_text",
                "majority_answer",
                "majority_answer_text",
                "majority_share",
                "key_share",
                "answer_distribution",
                "suspicion_reasons",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
    ]

    for _, row in suspicious.iterrows():
        q = questions[int(row["question_index"])]
        report.extend(
            [
                "",
                f"## q{int(row['question_index'])}: {row['subject']}",
                "",
                q["question"].strip(),
                "",
                "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, q["options"])),
                "",
                f"Current key: {row['key_answer']} - {row['key_answer_text']}",
                f"Observed majority: {row['majority_answer']} - {row['majority_answer_text']}",
                f"Distribution: {row['answer_distribution']}",
                f"Reasons: {row['suspicion_reasons']}",
            ]
        )

    (OUTPUT_DIR / "question_key_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {OUTPUT_DIR / 'question_key_audit.md'}")


if __name__ == "__main__":
    main()

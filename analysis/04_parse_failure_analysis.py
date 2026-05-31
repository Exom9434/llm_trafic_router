from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd

from common import OUTPUT_DIR, ROOT


CLOSED_TAG_CONTENT_RE = re.compile(r"<\s*Answer\s*>\s*([^<]+?)\s*<\s*/\s*Answer\s*>", re.I | re.S)
UNCLOSED_TAG_CONTENT_RE = re.compile(r"<\s*Answer\s*>\s*([^<]+?)\s*$", re.I | re.S)
ANY_TAG_RE = re.compile(r"<\s*/?\s*Answer\s*>", re.I)
ANSWER_PHRASE_RE = re.compile(r"(?:final answer|correct answer|answer is)[^A-J0-9가-힣]{0,30}(.{0,80})", re.I | re.S)


def norm(text: object) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9가-힣.+\\-]+", " ", text)
    return " ".join(text.split())


def option_letter_from_content(answer_content: str, question: dict) -> str | None:
    content_norm = norm(answer_content)
    if not content_norm:
        return None

    letters = "ABCDEFGHIJ"
    for letter, option in zip(letters, question.get("options", [])):
        option_norm = norm(option)
        if content_norm == option_norm:
            return letter
        if option_norm and (content_norm in option_norm or option_norm in content_norm):
            return letter
    return None


def classify(text: str) -> tuple[str, str]:
    tag_match = CLOSED_TAG_CONTENT_RE.search(text)
    if tag_match:
        content = " ".join(tag_match.group(1).split())
        if re.fullmatch(r"[A-J]", content, re.I):
            return "parser_bug_letter_tag", content
        return "answer_tag_contains_non_letter", content[:160]

    unclosed_match = UNCLOSED_TAG_CONTENT_RE.search(text)
    if unclosed_match:
        content = " ".join(unclosed_match.group(1).split())
        if content:
            return "unclosed_answer_tag_contains_non_letter", content[:160]
        return "malformed_or_empty_answer_tag", "Answer tag exists, but no content"

    if ANY_TAG_RE.search(text):
        return "malformed_or_unclosed_answer_tag", "Answer tag exists, but no closed simple content"

    phrase_match = ANSWER_PHRASE_RE.search(text)
    if phrase_match:
        return "no_tag_but_answer_phrase", " ".join(phrase_match.group(0).split())[:160]

    stripped = text.strip()
    if not stripped or stripped.lower() == "nan":
        return "empty_content", ""

    tail = " ".join(stripped[-220:].split())
    return "no_parseable_answer_signal", tail


def main():
    cleaned_path = OUTPUT_DIR / "cleaned_results.csv"
    if not cleaned_path.exists():
        raise SystemExit("Run analysis/01_build_dataset.py first.")

    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    cleaned = pd.read_csv(cleaned_path)
    failures = cleaned[cleaned["answer_parse_failed"] == 1].copy()

    rows = []
    for source_file, group in failures.groupby("source_file"):
        original = pd.read_csv(ROOT / source_file, usecols=["full_content"])
        for _, failure in group.iterrows():
            row_index = int(failure["row_index_in_file"])
            full_content = str(original.iloc[row_index]["full_content"])
            failure_class, extracted = classify(full_content)
            q_idx = int(failure["question_index"])
            recovered_letter = None
            recovered_correct = None
            if 0 <= q_idx < len(questions):
                recovered_letter = option_letter_from_content(extracted, questions[q_idx])
                if recovered_letter:
                    recovered_correct = recovered_letter == str(failure["correct_answer"]).upper()

            rows.append(
                {
                    "source_file": source_file,
                    "slot_date_et": failure["slot_date_et"],
                    "period": failure["period"],
                    "row_index_in_file": row_index,
                    "provider": failure["provider"],
                    "subject": failure["subject"],
                    "correct_answer": failure["correct_answer"],
                    "question_index": q_idx,
                    "failure_class": failure_class,
                    "extracted_answer_content": extracted,
                    "option_recovered_letter": recovered_letter,
                    "option_recovered_correct": recovered_correct,
                    "tail": " ".join(full_content[-260:].split()),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "parse_failures.csv", index=False, encoding="utf-8-sig")

    class_counts = out.groupby(["failure_class", "provider"]).size().unstack(fill_value=0)
    subject_counts = out.groupby(["provider", "subject"]).size().unstack(fill_value=0)
    recovery_counts = out.groupby(["failure_class", "option_recovered_correct"], dropna=False).size()

    examples = []
    for failure_class, group in out.groupby("failure_class"):
        examples.append(f"### {failure_class}")
        sample = group.head(6)[
            [
                "provider",
                "subject",
                "correct_answer",
                "extracted_answer_content",
                "option_recovered_letter",
                "tail",
            ]
        ]
        examples.append(sample.to_markdown(index=False))
        examples.append("")

    report = [
        "# Parse Failure Analysis",
        "",
        f"- Parse failures: {len(out):,}",
        f"- Recoverable by exact/substring option text match: {int(out['option_recovered_letter'].notna().sum()):,}",
        f"- Recoverable and correct: {int((out['option_recovered_correct'] == True).sum()):,}",
        f"- Recoverable but wrong: {int((out['option_recovered_correct'] == False).sum()):,}",
        "",
        "Recovery is diagnostic only. Per `analysis/SCORING_POLICY.md`, primary accuracy",
        "credits only valid option-letter answers and does not repair option-text responses.",
        "",
        "## Failure Class x Provider",
        "",
        class_counts.to_markdown(),
        "",
        "## Provider x Subject",
        "",
        subject_counts.to_markdown(),
        "",
        "## Recovery Counts",
        "",
        recovery_counts.to_string(),
        "",
        "## Examples",
        "",
        *examples,
    ]
    (OUTPUT_DIR / "parse_failure_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {OUTPUT_DIR / 'parse_failure_report.md'}")


if __name__ == "__main__":
    main()

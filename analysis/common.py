from __future__ import annotations

import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis" / "outputs"

RESULT_FILE_RE = re.compile(r"benchmark_results_(\d{8})_(peak|nonpeak)\.csv$")
ANSWER_RE = re.compile(r"<\s*Answer\s*>\s*([A-J])\s*(?:<\s*/\s*Answer\s*>)?", re.I)
ANSWER_LABEL_RE = re.compile(r"\bAnswer\b[^A-J]{0,20}([A-J])\b", re.I)
FINAL_LETTER_RE = re.compile(r"\b([A-J])\b\s*$", re.I)

EXPECTED_PROVIDERS = ("anthropic", "google", "openai", "qwen")
EXPECTED_QUESTIONS = 30
EXPECTED_REPEATS = 3
EXPECTED_ROWS_PER_SLOT = len(EXPECTED_PROVIDERS) * EXPECTED_QUESTIONS * EXPECTED_REPEATS


def result_files() -> list[Path]:
    return sorted(
        p
        for p in ROOT.glob("benchmark_results_20*.csv")
        if RESULT_FILE_RE.match(p.name)
    )


def parse_answer(content: object) -> str | None:
    if content is None or (isinstance(content, float) and np.isnan(content)):
        return None

    text = str(content).strip()
    for pattern in (ANSWER_RE, ANSWER_LABEL_RE, FINAL_LETTER_RE):
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def pct_delta(peak: float, nonpeak: float) -> float:
    if peak == 0 or np.isnan(peak) or np.isnan(nonpeak):
        return np.nan
    return (peak - nonpeak) / peak * 100

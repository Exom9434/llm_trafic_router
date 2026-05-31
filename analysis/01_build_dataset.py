from __future__ import annotations

import json
import re

import pandas as pd
import pytz

from common import OUTPUT_DIR, RESULT_FILE_RE, ROOT, parse_answer, result_files


USECOLS = [
    "request_id",
    "timestamp",
    "provider",
    "model_requested",
    "repeat_index",
    "ttft",
    "generation_time",
    "output_tokens",
    "tps",
    "full_content",
    "system_fingerprint",
    "difficulty",
    "subject",
    "correct_answer",
    "is_correct",
]

TASK_RE = r"q(\d+)_([a-z]+)_r(\d+)"


def load_task_map(slot_date: str, period: str) -> pd.DataFrame | None:
    progress_path = ROOT / "progress.json"
    if not progress_path.exists():
        return None

    raw = json.loads(progress_path.read_text(encoding="utf-8"))
    slot_time = "1000" if period == "peak" else "2200"
    slot_key = f"{slot_date}_{period}_{slot_time}"
    tasks = raw.get("tasks", {}).get(slot_key)
    if not tasks:
        return None

    rows = []
    for row_index, task in enumerate(tasks):
        match = re.fullmatch(TASK_RE, task)
        if not match:
            continue
        question_index, provider, repeat_index = match.groups()
        rows.append(
            {
                "row_index_in_file": row_index,
                "question_index": int(question_index),
                "task_provider": provider,
                "task_repeat_index": int(repeat_index),
                "task_key": task,
            }
        )
    return pd.DataFrame(rows)


def load_one(path):
    match = RESULT_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected result filename: {path}")

    slot_date, period = match.groups()
    df = pd.read_csv(path, usecols=USECOLS)
    df.insert(0, "source_file", path.name)
    df.insert(1, "slot_date_et", slot_date)
    df.insert(2, "period", period)
    df["slot_id"] = f"{slot_date}_{period}"
    df["row_index_in_file"] = range(len(df))

    task_map = load_task_map(slot_date, period)
    if task_map is not None and len(task_map) == len(df):
        df = df.merge(task_map, on="row_index_in_file", how="left")
    else:
        # Fallback for files without progress metadata. This is exact for complete slots
        # and only approximate for partial slots.
        df["question_index"] = (df["row_index_in_file"] // (3 * 4)).clip(upper=29)
        df["task_provider"] = None
        df["task_repeat_index"] = None
        df["task_key"] = None

    df["timestamp_kst"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H%M%S", errors="coerce")
    kst = pytz.timezone("Asia/Seoul")
    et = pytz.timezone("America/New_York")
    df["timestamp_et"] = df["timestamp_kst"].dt.tz_localize(kst).dt.tz_convert(et)

    df["parsed_answer"] = df["full_content"].map(parse_answer)
    df["is_correct_fixed"] = (
        df["parsed_answer"].notna()
        & (df["parsed_answer"].str.upper() == df["correct_answer"].str.upper())
    ).astype(int)
    df["answer_parse_failed"] = df["parsed_answer"].isna().astype(int)
    df["content_chars"] = df["full_content"].fillna("").str.len()

    return df.drop(columns=["full_content"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = result_files()
    if not files:
        raise SystemExit("No benchmark_results_YYYYMMDD_peak/nonpeak.csv files found.")

    frames = []
    for path in files:
        print(f"reading {path.name}")
        frames.append(load_one(path))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["slot_date_et", "period", "timestamp", "row_index_in_file"])

    out_csv = OUTPUT_DIR / "cleaned_results.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = {
        "files": len(files),
        "rows": len(df),
        "slot_min": df["slot_date_et"].min(),
        "slot_max": df["slot_date_et"].max(),
        "parse_failures": int(df["answer_parse_failed"].sum()),
        "original_accuracy_mean": float(df["is_correct"].mean()),
        "fixed_accuracy_mean": float(df["is_correct_fixed"].mean()),
    }
    pd.Series(summary).to_json(OUTPUT_DIR / "dataset_summary.json", force_ascii=False, indent=2)
    print(f"wrote {out_csv}")
    print(summary)


if __name__ == "__main__":
    main()

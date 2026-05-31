from __future__ import annotations

import json

import pandas as pd

from common import EXPECTED_ROWS_PER_SLOT, OUTPUT_DIR


def main():
    cleaned_path = OUTPUT_DIR / "cleaned_results.csv"
    if not cleaned_path.exists():
        raise SystemExit("Run analysis/01_build_dataset.py first.")

    df = pd.read_csv(cleaned_path)

    slot_counts = (
        df.groupby(["slot_date_et", "period"])
        .agg(
            rows=("request_id", "size"),
            providers=("provider", "nunique"),
            parse_failures=("answer_parse_failed", "sum"),
            fixed_accuracy=("is_correct_fixed", "mean"),
            median_ttft=("ttft", "median"),
            median_tps=("tps", "median"),
            min_timestamp_et=("timestamp_et", "min"),
            max_timestamp_et=("timestamp_et", "max"),
        )
        .reset_index()
    )
    slot_counts["expected_rows"] = EXPECTED_ROWS_PER_SLOT
    slot_counts["row_delta"] = slot_counts["rows"] - EXPECTED_ROWS_PER_SLOT
    slot_counts["is_complete"] = slot_counts["rows"] == EXPECTED_ROWS_PER_SLOT
    slot_counts.to_csv(OUTPUT_DIR / "slot_quality.csv", index=False, encoding="utf-8-sig")

    provider_slot_counts = (
        df.groupby(["slot_date_et", "period", "provider"])
        .size()
        .rename("rows")
        .reset_index()
    )
    provider_slot_counts.to_csv(
        OUTPUT_DIR / "provider_slot_counts.csv", index=False, encoding="utf-8-sig"
    )

    metric_quality = (
        df.groupby(["provider", "period"])
        .agg(
            rows=("request_id", "size"),
            parse_failures=("answer_parse_failed", "sum"),
            zero_ttft=("ttft", lambda s: int((s <= 0).sum())),
            zero_generation_time=("generation_time", lambda s: int((s <= 0).sum())),
            zero_tps=("tps", lambda s: int((s <= 0).sum())),
            p99_ttft=("ttft", lambda s: s.quantile(0.99)),
            p99_generation_time=("generation_time", lambda s: s.quantile(0.99)),
            p99_tps=("tps", lambda s: s.quantile(0.99)),
        )
        .reset_index()
    )
    metric_quality.to_csv(OUTPUT_DIR / "metric_quality.csv", index=False, encoding="utf-8-sig")

    complete = int(slot_counts["is_complete"].sum())
    partial = int((~slot_counts["is_complete"]).sum())
    missing_slots = []
    dates = sorted(df["slot_date_et"].unique())
    for date in dates:
        for period in ("peak", "nonpeak"):
            if not ((slot_counts["slot_date_et"] == date) & (slot_counts["period"] == period)).any():
                missing_slots.append(f"{date}_{period}")

    report = [
        "# Data Quality Report",
        "",
        f"- Cleaned rows: {len(df):,}",
        f"- Slot files observed: {len(slot_counts):,}",
        f"- Complete slots ({EXPECTED_ROWS_PER_SLOT} rows): {complete:,}",
        f"- Partial slots: {partial:,}",
        f"- Missing slots within observed date range: {len(missing_slots):,}",
        f"- Original `is_correct` mean: {df['is_correct'].mean():.4f}",
        f"- Fixed `is_correct_fixed` mean: {df['is_correct_fixed'].mean():.4f}",
        f"- Answer parse failures: {int(df['answer_parse_failed'].sum()):,}",
        "",
        "## Smallest Slots",
        "",
        slot_counts.sort_values("rows")
        .head(10)[["slot_date_et", "period", "rows", "row_delta", "parse_failures", "fixed_accuracy"]]
        .to_markdown(index=False),
        "",
        "## Metric Quality By Provider",
        "",
        metric_quality.to_markdown(index=False, floatfmt=".3f"),
    ]

    if missing_slots:
        report.extend(["", "## Missing Slots", "", ", ".join(missing_slots)])

    (OUTPUT_DIR / "quality_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {OUTPUT_DIR / 'quality_report.md'}")


if __name__ == "__main__":
    main()

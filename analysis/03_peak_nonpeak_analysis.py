from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from common import OUTPUT_DIR


METRICS = ("ttft", "generation_time", "tps", "output_tokens", "is_correct_fixed")


def bootstrap_ci(a, b, func=np.median, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(a.dropna())
    b = np.asarray(b.dropna())
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan

    diffs = np.empty(n)
    for i in range(n):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = func(aa) - func(bb)
    return np.quantile(diffs, [0.025, 0.975])


def summarize_metric(df, provider, metric):
    peak = df[(df["provider"] == provider) & (df["period"] == "peak")][metric]
    nonpeak = df[(df["provider"] == provider) & (df["period"] == "nonpeak")][metric]

    if len(peak) and len(nonpeak):
        test = mannwhitneyu(peak.dropna(), nonpeak.dropna(), alternative="two-sided")
        ci_low, ci_high = bootstrap_ci(peak, nonpeak)
    else:
        test = None
        ci_low, ci_high = np.nan, np.nan

    peak_median = peak.median()
    nonpeak_median = nonpeak.median()
    peak_mean = peak.mean()
    nonpeak_mean = nonpeak.mean()
    median_delta = peak_median - nonpeak_median
    mean_delta = peak_mean - nonpeak_mean
    pct_delta = median_delta / peak_median * 100 if peak_median else np.nan

    return {
        "provider": provider,
        "metric": metric,
        "peak_n": len(peak),
        "nonpeak_n": len(nonpeak),
        "peak_mean": peak_mean,
        "nonpeak_mean": nonpeak_mean,
        "mean_peak_minus_nonpeak": mean_delta,
        "peak_median": peak_median,
        "nonpeak_median": nonpeak_median,
        "median_peak_minus_nonpeak": median_delta,
        "median_delta_pct_of_peak": pct_delta,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "mannwhitney_p": test.pvalue if test else np.nan,
    }


def main():
    cleaned_path = OUTPUT_DIR / "cleaned_results.csv"
    if not cleaned_path.exists():
        raise SystemExit("Run analysis/01_build_dataset.py first.")

    df = pd.read_csv(cleaned_path)

    rows = []
    for provider in sorted(df["provider"].dropna().unique()):
        for metric in METRICS:
            rows.append(summarize_metric(df, provider, metric))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "peak_nonpeak_metric_summary.csv", index=False, encoding="utf-8-sig")

    daily = (
        df.groupby(["slot_date_et", "period", "provider"])
        .agg(
            rows=("request_id", "size"),
            accuracy=("is_correct_fixed", "mean"),
            format_failure_rate=("answer_parse_failed", "mean"),
            median_ttft=("ttft", "median"),
            median_generation_time=("generation_time", "median"),
            median_tps=("tps", "median"),
            p95_ttft=("ttft", lambda s: s.quantile(0.95)),
            p95_generation_time=("generation_time", lambda s: s.quantile(0.95)),
            p95_tps=("tps", lambda s: s.quantile(0.95)),
        )
        .reset_index()
    )
    daily.to_csv(OUTPUT_DIR / "daily_provider_summary.csv", index=False, encoding="utf-8-sig")

    subject = (
        df.groupby(["provider", "period", "subject"])
        .agg(
            rows=("request_id", "size"),
            accuracy=("is_correct_fixed", "mean"),
            format_failure_rate=("answer_parse_failed", "mean"),
            median_ttft=("ttft", "median"),
            median_tps=("tps", "median"),
        )
        .reset_index()
    )
    subject.to_csv(OUTPUT_DIR / "subject_summary.csv", index=False, encoding="utf-8-sig")

    slot_rows = df.groupby(["slot_date_et", "period"]).size().rename("rows").reset_index()
    complete_slots = slot_rows[slot_rows["rows"] == 360][["slot_date_et", "period"]]
    complete_df = df.merge(complete_slots, on=["slot_date_et", "period"])
    complete_summary = (
        complete_df.groupby(["provider", "period"])
        .agg(
            rows=("request_id", "size"),
            accuracy=("is_correct_fixed", "mean"),
            format_failure_rate=("answer_parse_failed", "mean"),
            median_ttft=("ttft", "median"),
            median_generation_time=("generation_time", "median"),
            median_tps=("tps", "median"),
        )
        .reset_index()
    )
    complete_summary.to_csv(
        OUTPUT_DIR / "complete_slot_provider_summary.csv", index=False, encoding="utf-8-sig"
    )

    format_summary = (
        df.groupby(["provider", "period"])
        .agg(
            rows=("request_id", "size"),
            parse_failures=("answer_parse_failed", "sum"),
            format_failure_rate=("answer_parse_failed", "mean"),
            accuracy=("is_correct_fixed", "mean"),
        )
        .reset_index()
    )
    format_summary.to_csv(
        OUTPUT_DIR / "format_failure_summary.csv", index=False, encoding="utf-8-sig"
    )

    focus = summary[summary["metric"].isin(["ttft", "generation_time", "tps", "is_correct_fixed"])].copy()
    report = [
        "# Peak vs Nonpeak Analysis",
        "",
        "Positive `median_peak_minus_nonpeak` means the peak median is higher than the nonpeak median.",
        "For `ttft` and `generation_time`, higher is slower. For `tps` and accuracy, higher is better.",
        "",
        "Primary accuracy follows `analysis/SCORING_POLICY.md`: only a valid option letter A-J is credited.",
        "Option text recovery is diagnostic only and is not used to repair `is_correct_fixed`.",
        "",
        "## Provider Metric Summary",
        "",
        focus[
            [
                "provider",
                "metric",
                "peak_n",
                "nonpeak_n",
                "peak_mean",
                "nonpeak_mean",
                "mean_peak_minus_nonpeak",
                "peak_median",
                "nonpeak_median",
                "median_peak_minus_nonpeak",
                "median_delta_pct_of_peak",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "mannwhitney_p",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Format Failures",
        "",
        format_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Complete-Slot Sensitivity Check",
        "",
        f"Complete slots only: {len(complete_slots)} slots, {len(complete_df):,} rows.",
        "",
        complete_summary.to_markdown(index=False, floatfmt=".4f"),
    ]
    (OUTPUT_DIR / "peak_nonpeak_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {OUTPUT_DIR / 'peak_nonpeak_report.md'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import OUTPUT_DIR


FIG_DIR = OUTPUT_DIR / "figures"
PROVIDER_ORDER = ["openai", "anthropic", "google", "qwen"]
PERIOD_ORDER = ["nonpeak", "peak"]
METRIC_LABELS = {
    "ttft": "TTFT (s)",
    "generation_time": "Generation time (s)",
    "tps": "Tokens per second",
    "is_correct_fixed": "Accuracy",
    "answer_parse_failed": "Format failure rate",
}


def bootstrap_diff(a, b, func=np.median, n=5000, seed=7):
    rng = np.random.default_rng(seed)
    a = np.asarray(pd.Series(a).dropna())
    b = np.asarray(pd.Series(b).dropna())
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan

    observed = func(b) - func(a)
    diffs = np.empty(n)
    for i in range(n):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = func(bb) - func(aa)
    low, high = np.quantile(diffs, [0.025, 0.975])
    return observed, low, high


def write_effect_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["ttft", "generation_time", "tps", "is_correct_fixed", "answer_parse_failed"]
    for provider in PROVIDER_ORDER:
        sub = df[df["provider"] == provider]
        for metric in metrics:
            nonpeak = sub[sub["period"] == "nonpeak"][metric]
            peak = sub[sub["period"] == "peak"][metric]
            func = np.mean if metric in {"is_correct_fixed", "answer_parse_failed"} else np.median
            diff, low, high = bootstrap_diff(nonpeak, peak, func=func)
            nonpeak_value = func(nonpeak.dropna())
            peak_value = func(peak.dropna())
            pct = diff / nonpeak_value * 100 if nonpeak_value else np.nan
            rows.append(
                {
                    "provider": provider,
                    "metric": metric,
                    "summary_stat": "mean" if func is np.mean else "median",
                    "nonpeak": nonpeak_value,
                    "peak": peak_value,
                    "peak_minus_nonpeak": diff,
                    "peak_minus_nonpeak_pct_of_nonpeak": pct,
                    "ci95_low": low,
                    "ci95_high": high,
                    "nonpeak_n": len(nonpeak),
                    "peak_n": len(peak),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "final_effect_summary.csv", index=False, encoding="utf-8-sig")
    return out


def save_bar_effects(effect: pd.DataFrame):
    plot_metrics = ["ttft", "generation_time", "tps", "is_correct_fixed", "answer_parse_failed"]
    fig, axes = plt.subplots(len(plot_metrics), 1, figsize=(10, 15), constrained_layout=True)
    for ax, metric in zip(axes, plot_metrics):
        sub = effect[effect["metric"] == metric].set_index("provider").loc[PROVIDER_ORDER].reset_index()
        y = np.arange(len(sub))
        xerr = np.vstack(
            [
                sub["peak_minus_nonpeak"] - sub["ci95_low"],
                sub["ci95_high"] - sub["peak_minus_nonpeak"],
            ]
        )
        colors = ["#b84a62" if v > 0 else "#2c7a7b" for v in sub["peak_minus_nonpeak"]]
        ax.barh(y, sub["peak_minus_nonpeak"], xerr=xerr, color=colors, alpha=0.9)
        ax.axvline(0, color="#222222", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["provider"])
        ax.set_title(f"Peak minus nonpeak: {METRIC_LABELS[metric]}")
        ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG_DIR / "peak_minus_nonpeak_effects.png", dpi=180)
    plt.close(fig)


def save_distribution_plots(df: pd.DataFrame):
    metrics = ["ttft", "generation_time", "tps"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        plot_df = df.copy()
        upper = plot_df[metric].quantile(0.99)
        plot_df = plot_df[plot_df[metric] <= upper]
        sns.boxplot(
            data=plot_df,
            x="provider",
            y=metric,
            hue="period",
            order=PROVIDER_ORDER,
            hue_order=PERIOD_ORDER,
            showfliers=False,
            ax=ax,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG_DIR / "provider_metric_distributions.png", dpi=180)
    plt.close(fig)


def save_daily_trends(df: pd.DataFrame):
    daily = (
        df.groupby(["slot_date_et", "period", "provider"])
        .agg(
            median_ttft=("ttft", "median"),
            median_tps=("tps", "median"),
            accuracy=("is_correct_fixed", "mean"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["slot_date_et"].astype(str))
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    for ax, metric, label in zip(
        axes,
        ["median_ttft", "median_tps", "accuracy"],
        ["Daily median TTFT", "Daily median TPS", "Daily accuracy"],
    ):
        sns.lineplot(
            data=daily,
            x="date",
            y=metric,
            hue="provider",
            style="period",
            hue_order=PROVIDER_ORDER,
            style_order=PERIOD_ORDER,
            markers=True,
            dashes=False,
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG_DIR / "daily_provider_trends.png", dpi=180)
    plt.close(fig)


def write_final_report(effect: pd.DataFrame):
    focus = effect.copy()
    focus["metric_label"] = focus["metric"].map(METRIC_LABELS)
    table = focus[
        [
            "provider",
            "metric_label",
            "summary_stat",
            "nonpeak",
            "peak",
            "peak_minus_nonpeak",
            "peak_minus_nonpeak_pct_of_nonpeak",
            "ci95_low",
            "ci95_high",
        ]
    ]
    report = [
        "# Final Peak vs Nonpeak Findings",
        "",
        "Primary accuracy follows `analysis/SCORING_POLICY.md`: only option-letter answers are credited.",
        "Speed comparisons use medians; accuracy and format failure use means.",
        "",
        "Interpretation: positive `peak_minus_nonpeak` means the metric is higher during peak.",
        "For TTFT, generation time, and format failure, higher is worse. For TPS and accuracy, higher is better.",
        "",
        "## Effect Summary",
        "",
        table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Figures",
        "",
        "- `figures/peak_minus_nonpeak_effects.png`",
        "- `figures/provider_metric_distributions.png`",
        "- `figures/daily_provider_trends.png`",
    ]
    (OUTPUT_DIR / "final_findings.md").write_text("\n".join(report), encoding="utf-8")


def main():
    cleaned_path = OUTPUT_DIR / "cleaned_results.csv"
    if not cleaned_path.exists():
        raise SystemExit("Run analysis/01_build_dataset.py first.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    df = pd.read_csv(cleaned_path)
    df = df[df["provider"].isin(PROVIDER_ORDER)].copy()
    effect = write_effect_summary(df)
    save_bar_effects(effect)
    save_distribution_plots(df)
    save_daily_trends(df)
    write_final_report(effect)
    print(f"wrote {OUTPUT_DIR / 'final_findings.md'}")


if __name__ == "__main__":
    main()

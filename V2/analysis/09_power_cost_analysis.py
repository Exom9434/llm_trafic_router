from __future__ import annotations

"""
Power + cost sizing for the "does quality vary by time-of-day / load?" study.

Answers three chained questions:
  1. POWER: to detect a quality difference of size delta between two load
     conditions, how many independent question-trials PER CONDITION are needed?
     - binary accuracy (two-proportion test)
     - continuous confidence / logprob (two-sample t, effect in SD units)
  2. DESIGN: a candidate (bank size x days x slots/day x repeats x providers)
     turns into a total call count and an achieved trials-per-condition.
  3. COST: calls x tokens-per-call x per-token price -> dollars.

All prices/token counts are ASSUMPTIONS, clearly flagged, and parametrized.
Plug real numbers in PRICING / TOKENS before committing.
"""

from math import sqrt
import pandas as pd

# Normal quantiles (alpha=0.05 two-sided, power=0.80 / 0.90)
Z_ALPHA_2 = 1.959964
Z_POWER = {0.80: 0.841621, 0.90: 1.281552}


# ---------- 1. POWER --------------------------------------------------
def n_two_proportion(p1: float, delta: float, power: float = 0.80) -> int:
    """Trials per condition to detect p1 vs p1-delta (two-sided, alpha=0.05)."""
    p2 = p1 - delta
    pbar = (p1 + p2) / 2
    zb = Z_POWER[power]
    num = (Z_ALPHA_2 * sqrt(2 * pbar * (1 - pbar))
           + zb * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(-(-num // (delta ** 2)))  # ceil


def n_two_sample_t(cohen_d: float, power: float = 0.80) -> int:
    """Per-condition n to detect a mean shift of cohen_d SDs (continuous metric,
    e.g. answer-token logprob / confidence)."""
    zb = Z_POWER[power]
    return int(-(-(2 * (Z_ALPHA_2 + zb) ** 2) // (cohen_d ** 2)))


def power_tables():
    baselines = [0.50, 0.80]  # gpt-4o-mini-ish vs strong-model-ish
    deltas = [0.01, 0.02, 0.03, 0.05]
    rows = []
    for p in baselines:
        for d in deltas:
            rows.append({
                "baseline_acc": p,
                "min_detectable_drop": d,
                "trials_per_condition_pw80": n_two_proportion(p, d, 0.80),
                "trials_per_condition_pw90": n_two_proportion(p, d, 0.90),
            })
    acc = pd.DataFrame(rows)

    ds = [0.1, 0.15, 0.2, 0.3]
    cont = pd.DataFrame([{
        "effect_size_cohen_d": d,
        "n_per_condition_pw80": n_two_sample_t(d, 0.80),
        "n_per_condition_pw90": n_two_sample_t(d, 0.90),
    } for d in ds])
    return acc, cont


# ---------- 2/3. DESIGN -> CALLS -> COST ------------------------------
# ASSUMPTIONS -- verify current pricing before committing. USD per 1M tokens.
# Representative "budget/flash tier" rates.
PRICING = {  # provider: (input_per_1M, output_per_1M)
    "openai":    (0.15, 0.60),
    "anthropic": (0.80, 4.00),   # haiku tier tends to cost more per token
    "google":    (0.10, 0.40),
    "qwen":      (0.20, 0.60),
    "deepseek":  (0.14, 0.28),
}

# Tokens per call by probe mode. input includes system prompt + question + options.
TOKENS = {
    # direct-answer: model returns just the option letter (+ logprob). Cheapest,
    # and confidence/logprob is the most sensitive metric.
    "direct":  {"input": 400, "output": 8},
    # chain-of-thought: model reasons before answering. Needed to catch a provider
    # cutting reasoning budget under load. Output is large and provider-dependent.
    "cot":     {"input": 400, "output": 450},
}
# Providers whose CoT output runs long (overrides output tokens in cot mode).
COT_OUTPUT_OVERRIDE = {"qwen": 1200}


def design_cost(bank, days, slots_per_day, repeats, providers, mode):
    calls_per_provider = bank * slots_per_day * days * repeats
    total_calls = calls_per_provider * len(providers)
    cost = 0.0
    for pv in providers:
        tin = TOKENS[mode]["input"]
        tout = TOKENS[mode]["output"]
        if mode == "cot" and pv in COT_OUTPUT_OVERRIDE:
            tout = COT_OUTPUT_OVERRIDE[pv]
        pin, pout = PRICING[pv]
        cost += calls_per_provider * (tin * pin + tout * pout) / 1_000_000
    return calls_per_provider, total_calls, round(cost, 2)


def trials_per_condition(bank, days, slots_per_day, repeats, n_conditions=2):
    """Assume the full bank is asked each slot; slots split evenly across
    conditions. Trials/condition/provider = bank * (slots in that condition)."""
    slots_in_condition = days * (slots_per_day / n_conditions)
    return int(bank * slots_in_condition * repeats)


def largest_detectable_drop(trials, baseline=0.80, power=0.80):
    """Invert the proportion power curve: smallest delta detectable with `trials`
    per condition at the given baseline."""
    for d in [x / 1000 for x in range(5, 200)]:
        if n_two_proportion(baseline, d, power) <= trials:
            return round(d, 3)
    return None


def design_table():
    providers = ["openai", "anthropic", "google", "qwen", "deepseek"]
    candidates = [
        # name, bank, days, slots/day, repeats, mode
        ("A  accuracy, CoT (rich but pricey)",  500, 14, 8, 1, "cot"),
        ("B  accuracy, CoT (lean)",             200, 14, 6, 1, "cot"),
        ("C  direct+logprob (cheap, sensitive)",200, 14, 6, 1, "direct"),
        ("D  direct+logprob (big bank)",        500, 14, 8, 1, "direct"),
        ("E  CoT consistency, temp>0 x3",       150, 14, 6, 3, "cot"),
    ]
    rows = []
    for name, bank, days, spd, rep, mode in candidates:
        cpp, total, cost = design_cost(bank, days, spd, rep, providers, mode)
        tpc = trials_per_condition(bank, days, spd, rep)
        rows.append({
            "design": name,
            "bank": bank, "days": days, "slots/day": spd, "repeats": rep,
            "mode": mode,
            "calls/provider": f"{cpp:,}",
            "total_calls": f"{total:,}",
            "trials/cond/prov": f"{tpc:,}",
            "min_drop@80%acc": largest_detectable_drop(tpc, 0.80),
            "est_cost_USD": cost,
        })
    return pd.DataFrame(rows)


def main():
    acc, cont = power_tables()
    designs = design_table()

    print("=== 1a. Binary accuracy: trials per condition needed ===")
    print(acc.to_string(index=False))
    print("\n=== 1b. Continuous confidence/logprob: n per condition needed ===")
    print(cont.to_string(index=False))
    print("\n=== 2/3. Candidate designs -> calls & cost (ASSUMED pricing) ===")
    print(designs.to_string(index=False))

    md = [
        "# Power & Cost Sizing: quality-vs-load study",
        "",
        "> **All token counts and prices are ASSUMPTIONS** (representative budget-tier "
        "rates), parametrized in the script. Replace `PRICING` / `TOKENS` with real "
        "numbers before committing. `est_cost_USD` scales linearly with them.",
        "",
        "## 1a. How many trials per condition? (binary accuracy)",
        "",
        "Two-proportion test, alpha=0.05. `min_detectable_drop` = the accuracy drop "
        "(e.g. 0.03 = 80%->77%) you want to catch. Variance peaks at 50% accuracy, so "
        "weak models (gpt-4o-mini ~48%) need the most data.",
        "",
        acc.to_markdown(index=False),
        "",
        "## 1b. Continuous confidence / logprob is ~10x cheaper on data",
        "",
        "If the provider exposes answer-token logprobs, a shift in mean confidence is "
        "detectable with far fewer trials. `cohen_d` is the shift in SD units.",
        "",
        cont.to_markdown(index=False),
        "",
        "## 2/3. Candidate designs: calls and cost",
        "",
        "5 providers, 2 load conditions (peak/off-peak), 2-week window. "
        "`trials/cond/prov` assumes the full bank is asked every slot; "
        "`min_drop@80%acc` is the smallest accuracy drop that design can detect at "
        "80% power for a strong (~80%) model.",
        "",
        designs.to_markdown(index=False),
        "",
        "## Reading of the numbers",
        "",
        "- **Detecting a 3-point accuracy drop needs ~3,000 trials/condition** (more "
        "for weak models). Every candidate design clears this by a wide margin, so "
        "**power is not the binding constraint once the bank is >=200 and you run "
        "~2 weeks** -- the binding constraint is cost, driven by CoT output length.",
        "- **CoT vs direct dominates cost.** Direct-answer designs (C/D) cost a small "
        "fraction of CoT designs (A/B/E) because output tokens collapse from "
        "hundreds to ~8. Qwen's long CoT is the single biggest cost driver.",
        "- **Logprob/confidence is the sensitive primary metric** and it comes free "
        "with direct-answer calls -- so design C/D gives both cheapness and the most "
        "statistical power per dollar.",
        "",
        "## Recommended design",
        "",
        "- **Primary:** direct-answer + logprob on a fixed bank of ~300-500 items, "
        "6-8 slots/day, 14 days (design C/D). Cheap, and confidence is the sensitive "
        "signal. Analyze diurnally to stay robust to silent model updates.",
        "- **Secondary (catches reasoning-budget cuts):** a SMALL CoT subset (e.g. "
        "50-100 items, design E-style) run alongside, so cost stays bounded but you "
        "still see quality loss that only shows up when reasoning is truncated.",
        "- **Fingerprint / response-distribution logging on every call** -- the "
        "smoking gun for silent model swaps, at zero extra cost.",
    ]
    from pathlib import Path
    try:
        from common import OUTPUT_DIR
    except Exception:
        OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
    out = OUTPUT_DIR / "power_cost_analysis.md"
    out.write_text("\n".join(md), encoding="utf-8")
    designs.to_csv(OUTPUT_DIR / "power_cost_designs.csv", index=False, encoding="utf-8-sig")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

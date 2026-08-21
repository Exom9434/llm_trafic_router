# Peak vs Nonpeak: Pseudoreplication-Corrected Reanalysis (P0-1)

Complete slots only: **9** (peak=6, nonpeak=3). The unit of independence is the slot, so the effective sample size is **n=9**, not the 3,240 individual API calls.

Each metric is first aggregated to one value per (slot, provider) -- median for speed metrics, mean for quality metrics -- then peak slots are compared with nonpeak slots.

**Column guide.** `p_slot_exact`: exact Mann-Whitney on slot-level values (rank-based). `p_slot_perm`: exact permutation test on the mean difference (magnitude-based; more sensitive to outlier slots). `p_mixed_model`: mixed-effects model on call-level data with a random intercept per slot (log scale for speed) -- reported as NaN when the fit fails to converge, which it does here because ~9 slots is too few to estimate a slot-level variance component reliably. `p_naive_calllevel`: the original, **flawed** call-level Mann-Whitney, shown only to expose how much pseudoreplication inflated significance.

> The two slot-level tests disagree for some providers, and that disagreement is informative: a rank test can flag an effect that a magnitude test finds fragile (outlier-driven). With 6 vs 3 slots the smallest attainable two-sided exact p is ~0.024, so even the 'significant' cells sit right at the floor and should be read as **suggestive, not confirmatory**. The mixed model cannot be fit at this slot count; it belongs to the expanded-data phase (roadmap P1).

## Speed metrics (slot medians)

| provider   | metric              |   n_peak_slots |   n_nonpeak_slots |   nonpeak |     peak |   delta |   delta_pct |   p_slot_exact |   p_slot_perm |   p_mixed_model |   p_naive_calllevel |
|:-----------|:--------------------|---------------:|------------------:|----------:|---------:|--------:|------------:|---------------:|--------------:|----------------:|--------------------:|
| anthropic  | TTFT (s)            |              6 |                 3 |    0.8283 |   0.9234 |  0.0951 |       11.48 |         0.0238 |        0.0595 |             nan |         2.37235e-13 |
| anthropic  | Generation time (s) |              6 |                 3 |    3.0358 |   3.0723 |  0.0364 |        1.2  |         0.2619 |        0.3452 |             nan |         0.748243    |
| anthropic  | Tokens/s            |              6 |                 3 |  141.216  | 138.155  | -3.0612 |       -2.17 |         0.5476 |        0.4524 |             nan |         0.747278    |
| google     | TTFT (s)            |              6 |                 3 |    1.6894 |   1.9958 |  0.3064 |       18.14 |         0.0476 |        0.3452 |             nan |         5.3416e-12  |
| google     | Generation time (s) |              6 |                 3 |    1.098  |   1.1177 |  0.0197 |        1.79 |         0.9048 |        0.5357 |             nan |         0.927025    |
| google     | Tokens/s            |              6 |                 3 |  225.726  | 229.756  |  4.0299 |        1.79 |         0.5476 |        0.631  |             nan |         0.500356    |
| openai     | TTFT (s)            |              6 |                 3 |    0.8756 |   0.9063 |  0.0308 |        3.51 |         0.1667 |        0.4167 |             nan |         7.20206e-05 |
| openai     | Generation time (s) |              6 |                 3 |    5.1344 |   5.3119 |  0.1775 |        3.46 |         0.7143 |        0.5357 |             nan |         0.608806    |
| openai     | Tokens/s            |              6 |                 3 |   55.0547 |  51.789  | -3.2657 |       -5.93 |         0.381  |        0.2857 |             nan |         0.00190948  |
| qwen       | TTFT (s)            |              6 |                 3 |    3.7215 |   3.3719 | -0.3496 |       -9.39 |         0.0476 |        0.0476 |             nan |         3.41417e-16 |
| qwen       | Generation time (s) |              6 |                 3 |   31.555  |  27.509  | -4.046  |      -12.82 |         0.0238 |        0.0119 |             nan |         0.00920223  |
| qwen       | Tokens/s            |              6 |                 3 |  167.934  | 190.466  | 22.5316 |       13.42 |         0.0476 |        0.0238 |             nan |         4.7148e-23  |

## Quality metrics (slot means)

| provider   | metric              |   n_peak_slots |   n_nonpeak_slots |   nonpeak |   peak |   delta |   delta_pct |   p_slot_exact |   p_slot_perm |   p_mixed_model |   p_naive_calllevel |
|:-----------|:--------------------|---------------:|------------------:|----------:|-------:|--------:|------------:|---------------:|--------------:|----------------:|--------------------:|
| anthropic  | Accuracy            |              6 |                 3 |    0.8333 | 0.8167 | -0.0167 |       -2    |         0.2619 |        0.2738 |             nan |            0.559047 |
| anthropic  | Format failure rate |              6 |                 3 |    0      | 0.0093 |  0.0093 |      nan    |         0.0952 |        0.0476 |             nan |            0.113219 |
| google     | Accuracy            |              6 |                 3 |    0.8667 | 0.8648 | -0.0019 |       -0.21 |         0.9048 |        1      |             nan |            0.942212 |
| google     | Format failure rate |              6 |                 3 |    0      | 0.0037 |  0.0037 |      nan    |         0.5476 |        0.5    |             nan |            0.317908 |
| openai     | Accuracy            |              6 |                 3 |    0.4926 | 0.4796 | -0.013  |       -2.63 |         0.5476 |        0.4881 |             nan |            0.728131 |
| openai     | Format failure rate |              6 |                 3 |    0.0148 | 0.0278 |  0.013  |       87.5  |         0.2619 |        0.369  |             nan |            0.251055 |
| qwen       | Accuracy            |              6 |                 3 |    0.8222 | 0.8111 | -0.0111 |       -1.35 |         0.5476 |        0.5595 |             nan |            0.701507 |
| qwen       | Format failure rate |              6 |                 3 |    0      | 0      |  0      |      nan    |         1      |        1      |             nan |            1        |

## Interpretation

- **Significance collapses, direction survives.** Naive call-level p-values (e.g. Anthropic TTFT ~2e-13, Qwen TTFT ~3e-16) were artifacts of treating ~3,240 correlated calls as independent. At the slot level the same effects sit at or near the exact-test floor. The effect *estimates* (delta, delta_pct) remain the trustworthy takeaway.
- **Qwen's reverse pattern is the most robust finding.** Lower peak TTFT, lower generation time, and higher TPS are all significant under BOTH the rank and permutation tests (p ~ 0.024-0.048). This is the result that best withstands correction and is the strongest candidate for the paper's headline claim.
- **US-provider TTFT increases are directional but weaker than reported.** Anthropic TTFT is borderline (exact 0.024 / perm 0.06). Google TTFT is significant by rank (0.048) but NOT by permutation (0.35), i.e. outlier-driven and fragile. OpenAI TTFT is not significant either way. The blanket '+4-37%, highly significant' framing should be softened accordingly.
- **Accuracy is unchanged** across conditions under every test -- this conclusion needed no correction and stands.
- **Recommended framing for the paper.** Lead with effect sizes, call the US-provider latency effects a directional pilot signal, foreground the Qwen reverse pattern, and make expanded slot coverage (roadmap P1-1) the explicit route to confirmatory significance and a usable mixed-effects model.
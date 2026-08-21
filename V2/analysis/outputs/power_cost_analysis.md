# Power & Cost Sizing: quality-vs-load study

> **All token counts and prices are ASSUMPTIONS** (representative budget-tier rates), parametrized in the script. Replace `PRICING` / `TOKENS` with real numbers before committing. `est_cost_USD` scales linearly with them.

## 1a. How many trials per condition? (binary accuracy)

Two-proportion test, alpha=0.05. `min_detectable_drop` = the accuracy drop (e.g. 0.03 = 80%->77%) you want to catch. Variance peaks at 50% accuracy, so weak models (gpt-4o-mini ~48%) need the most data.

|   baseline_acc |   min_detectable_drop |   trials_per_condition_pw80 |   trials_per_condition_pw90 |
|---------------:|----------------------:|----------------------------:|----------------------------:|
|            0.5 |                  0.01 |                       39240 |                       52530 |
|            0.5 |                  0.02 |                        9806 |                       13127 |
|            0.5 |                  0.03 |                        4356 |                        5831 |
|            0.5 |                  0.05 |                        1565 |                        2095 |
|            0.8 |                  0.01 |                       25583 |                       34247 |
|            0.8 |                  0.02 |                        6510 |                        8714 |
|            0.8 |                  0.03 |                        2943 |                        3939 |
|            0.8 |                  0.05 |                        1094 |                        1464 |

## 1b. Continuous confidence / logprob is ~10x cheaper on data

If the provider exposes answer-token logprobs, a shift in mean confidence is detectable with far fewer trials. `cohen_d` is the shift in SD units.

|   effect_size_cohen_d |   n_per_condition_pw80 |   n_per_condition_pw90 |
|----------------------:|-----------------------:|-----------------------:|
|                  0.1  |                   1570 |                   2102 |
|                  0.15 |                    698 |                    934 |
|                  0.2  |                    393 |                    526 |
|                  0.3  |                    175 |                    234 |

## 2/3. Candidate designs: calls and cost

5 providers, 2 load conditions (peak/off-peak), 2-week window. `trials/cond/prov` assumes the full bank is asked every slot; `min_drop@80%acc` is the smallest accuracy drop that design can detect at 80% power for a strong (~80%) model.

| design                               |   bank |   days |   slots/day |   repeats | mode   |   calls/provider |   total_calls |   trials/cond/prov |   min_drop@80%acc |   est_cost_USD |
|:-------------------------------------|-------:|-------:|------------:|----------:|:-------|-----------------:|--------------:|-------------------:|------------------:|---------------:|
| A  accuracy, CoT (rich but pricey)   |    500 |     14 |           8 |         1 | cot    |           56,000 |       280,000 |             28,000 |             0.01  |         204.51 |
| B  accuracy, CoT (lean)              |    200 |     14 |           6 |         1 | cot    |           16,800 |        84,000 |              8,400 |             0.018 |          61.35 |
| C  direct+logprob (cheap, sensitive) |    200 |     14 |           6 |         1 | direct |           16,800 |        84,000 |              8,400 |             0.018 |          10.13 |
| D  direct+logprob (big bank)         |    500 |     14 |           8 |         1 | direct |           56,000 |       280,000 |             28,000 |             0.01  |          33.77 |
| E  CoT consistency, temp>0 x3        |    150 |     14 |           6 |         3 | cot    |           37,800 |       189,000 |             18,900 |             0.012 |         138.05 |

## Reading of the numbers

- **Detecting a 3-point accuracy drop needs ~3,000 trials/condition** (more for weak models). Every candidate design clears this by a wide margin, so **power is not the binding constraint once the bank is >=200 and you run ~2 weeks** -- the binding constraint is cost, driven by CoT output length.
- **CoT vs direct dominates cost.** Direct-answer designs (C/D) cost a small fraction of CoT designs (A/B/E) because output tokens collapse from hundreds to ~8. Qwen's long CoT is the single biggest cost driver.
- **Logprob/confidence is the sensitive primary metric** and it comes free with direct-answer calls -- so design C/D gives both cheapness and the most statistical power per dollar.

## Recommended design

- **Primary:** direct-answer + logprob on a fixed bank of ~300-500 items, 6-8 slots/day, 14 days (design C/D). Cheap, and confidence is the sensitive signal. Analyze diurnally to stay robust to silent model updates.
- **Secondary (catches reasoning-budget cuts):** a SMALL CoT subset (e.g. 50-100 items, design E-style) run alongside, so cost stays bounded but you still see quality loss that only shows up when reasoning is truncated.
- **Fingerprint / response-distribution logging on every call** -- the smoking gun for silent model swaps, at zero extra cost.
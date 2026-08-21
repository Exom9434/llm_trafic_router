# Accuracy Parsing Verification (P0-2)

**Question.** Is gpt-4o-mini's sub-50% accuracy a parsing/format artifact?
**Answer.** No. It is genuine performance on MMLU-Pro (10-option, hard split).

## Check A -- upper bound from the cleaned dataset (complete slots)

`max_if_all_ff_correct` = overall accuracy + format-failure rate: the highest accuracy any parser fix could reach if every unparseable response were secretly correct. For OpenAI this ceiling (~0.51) is only ~2 points above the observed ~0.48, so parsing cannot explain the low score.

| provider   |   n |   overall_accuracy |   format_failure_rate |   accuracy_among_parsed |   max_if_all_ff_correct |
|:-----------|----:|-------------------:|----------------------:|------------------------:|------------------------:|
| openai     | 810 |             0.484  |                0.0235 |                  0.4956 |                  0.5074 |
| anthropic  | 810 |             0.8222 |                0.0062 |                  0.8273 |                  0.8284 |
| google     | 810 |             0.8654 |                0.0025 |                  0.8676 |                  0.8679 |
| qwen       | 810 |             0.8148 |                0      |                  0.8148 |                  0.8148 |

## Check B -- lenient value->letter recovery on raw responses (complete slots)

A lenient parser mapping the `<Answer>` tag's option text/value back to a letter recovers essentially nothing (`value_recovered`~0); lenient accuracy equals strict accuracy.

| provider   |   n |   strict_accuracy |   lenient_accuracy |   value_recovered |
|:-----------|----:|------------------:|-------------------:|------------------:|
| openai     | 810 |            0.484  |             0.484  |                18 |
| anthropic  | 810 |            0.8222 |             0.8222 |                 4 |
| google     | 810 |            0.8654 |             0.8654 |                 0 |
| qwen       | 810 |            0.8148 |             0.8148 |                 0 |

## Conclusion for the paper

- Benchmark is **MMLU-Pro** (Wang et al., 2024): 30 hard questions, 6 subjects x 5, 9-10 options each -- not 4-option MMLU (Hendrycks 2021).
- gpt-4o-mini ~48-50% is genuine hard-MMLU-Pro performance, **not** a formatting artifact.
- Format-failure rate is low and reported separately; accuracy among parsed responses ~= overall accuracy.
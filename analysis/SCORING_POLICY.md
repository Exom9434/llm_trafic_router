# Scoring Policy

This experiment evaluates whether a model selects the correct multiple-choice
option under the given benchmark format. It does not evaluate whether the model's
free-form explanation is semantically correct.

## Primary Accuracy

Use `is_correct_fixed` as the primary accuracy column.

A response is correct only when:

1. The response contains a parseable option letter `A` through `J`.
2. The parsed option letter exactly matches `questions.json["answer"]`.

If the model writes an option text, a numeric value, or a semantically plausible
answer inside `<Answer>` instead of an option letter, it is not counted as
correct in the primary metric.

Examples:

- `<Answer>J</Answer>` can be correct if the key is `J`.
- `<Answer>-3</Answer>` is a format failure, even if `-3` is the text of option `J`.
- `<Answer>the Indus Valley.</Answer>` is a format failure, even if it can be
  mapped to one of the choices.

## Format Failure

Use `answer_parse_failed` to track responses that did not provide a valid option
letter. These failures count as incorrect in the primary accuracy metric and are
reported separately as `format_failure_rate`.

## Auxiliary Recovery

`parse_failure_report.md` may map answer text back to an option letter for
diagnostics. This is useful for understanding model behavior, but it must not be
used to repair primary accuracy unless a separate exploratory analysis clearly
labels that choice.

## Answer Keys

`questions.json` is treated as the ground-truth benchmark key. Model consensus is
not used to rewrite labels. Question-key audits are diagnostic only and should be
used to flag possible benchmark quality issues, not to silently change scoring.

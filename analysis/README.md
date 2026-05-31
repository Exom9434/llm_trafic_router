# LLM Peak vs Nonpeak Analysis

Run the scripts from the repository root:

```bash
python analysis/01_build_dataset.py
python analysis/02_quality_report.py
python analysis/03_peak_nonpeak_analysis.py
python analysis/04_parse_failure_analysis.py
python analysis/05_audit_question_keys.py
python analysis/06_final_analysis.py
```

Outputs are written to `analysis/outputs/`.

- `SCORING_POLICY.md`: primary scoring rules for multiple-choice option letters.
- `cleaned_results.csv`: lightweight analysis dataset with `parsed_answer` and `is_correct_fixed`.
- `quality_report.md`: slot completeness, parse failures, and metric sanity checks.
- `peak_nonpeak_report.md`: provider-level peak vs nonpeak comparisons.
- `daily_provider_summary.csv`: daily trend table for plotting.
- `subject_summary.csv`: provider/period/subject breakdown.
- `parse_failure_report.md`: why answer parsing failed and whether failures are recoverable.
- `question_key_audit.md`: suspicious answer keys based on structure and observed answer distribution.
- `final_findings.md`: final provider-level peak/nonpeak findings.
- `figures/`: PNG plots for effect sizes, metric distributions, and daily trends.

Notes:

- `benchmark_results.csv` is excluded because it appears to be an earlier pilot run.
- The original `is_correct` column is not reliable for this experiment because responses begin with `<Reason>`. Use `is_correct_fixed`, which is computed from the `<Answer>` tag.
- `is_correct_fixed` only credits valid option letters. Option text recovery is diagnostic only.
- `question_index` is recovered from `progress.json` task keys when available.

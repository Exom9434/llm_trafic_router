Supplementary files for the preregistration

Do Commercial LLM APIs Degrade Answer Quality Under Load?
A 21-Day Within-Provider Longitudinal Measurement of Eight Models

These five files are registered together with the form. They are frozen at
submission and are not editable afterwards. The same material also lives in
a public repository, which remains editable; where the two ever disagree,
these files are the registered version.


prereg_supplementary_tables.pdf
    Seven tables that the form's text fields cannot hold in tabular form.
    Each caption names the form field the table belongs to.

prompts.py
    The prompt construction and the answer parser, fixed before execution.
    Registering it is the commitment made under blinding: scoring is
    condition-blind because parse_letter() reads only the response string
    and does not call an LLM. DIRECT_SYSTEM is the quality-probe system
    prompt; make_nonce() is the cache-blocking token described under Study
    design.

item_bank_ids.csv
    The 300 items of the fixed bank, by MMLU-Pro identifier and subject,
    50 per subject. Registering the list is the commitment made under
    blinding: the bank was finalized before any peak data were collected.
    Item text and answer keys are not reproduced here; MMLU-Pro is
    distributed by TIGER-Lab under the MIT license and the identifiers are
    sufficient to reconstruct the bank.

    bank_order is the position of each item in the fixed bank. The three
    presentation groups described under Study design are consecutive blocks
    of 100 in this order, so this column determines the counterbalancing
    together with the rotation rule stated there.

12_condition_power.py
    The script that computes the required item counts and the coverage
    reported under Sample size rationale.

condition_power.md
    The output of that script for the registered design: 21 completed days,
    k = 3, a 300-item bank, power 0.80, two-sided alpha 0.05. Tables 3 and 4
    of the supplementary tables file are drawn from it.

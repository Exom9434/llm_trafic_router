---
name: llm-router-redesign
description: "llm_trafic_router \"does API quality degrade under load?\" study — design spec, measured provider capabilities (2026-08), V2/runner status, and git state"
metadata: 
  node_type: memory
  type: project
  originSessionId: adea165c-88fc-4d38-8ef1-1f3a51546751
---

The `llm_trafic_router` project was redesigned 2026-07. V1 paper/pipeline remain untouched at `report/`, `analysis/`, `experiment_api.py`. All V2 work lives in `V2/`; `V2/EXPERIMENT_DESIGN.md` is the authoritative spec (부록 B = 2026-08-21 lineup research, 부록 C = 2026-08-24 measured results). **`V2/SESSION_LOG.md` is the narrative log — read it first when returning after a break.**

**Primary thesis:** Do commercial LLM APIs silently degrade answer *quality* under load (time-of-day as load proxy)? The original 28-day peak-vs-nonpeak study found a null accuracy effect but was underpowered (9 complete slots, 30 items).

**Core design decisions (stable):**
- N parallel WITHIN-provider longitudinal studies, never a between-model benchmark. Each provider is its own control. This defeats the "comparing models of different price/ability" attack, and it also means per-model settings may differ freely.
- Single vantage suffices (quality signals are network-immune). Measurement point = a Windows laptop in Korea (Hetzner account locked; a VM was never load-bearing).
- Analyze DIURNAL (within-day) signal to stay robust to silent model updates.
- Noise floor measured at low load first, so "the model is just flaky" becomes a controlled baseline.
- Calibration keeps only items in the 40–85% band across the lineup; the bank is calibrated TO THIS LINEUP.
- Budget-tier lineup is a hypothesis choice (corner-cutting incentive is highest where margins are thin), not just cost. Cost goes in Limitations.

## Measured provider capabilities (2026-08-24, 부록 C)

Documentation was wrong in several places. These are實測 results.

**Current lineup:** gpt-5.6-luna, gemini-3.5-flash-lite, deepseek-v4-flash, qwen3.7-flash-2026-07-15, solar-pro3, claude-haiku-4-5-20251001, HCX-DASH-002. Anchors: gpt-5.6-sol, claude-sonnet-5.

**Metric priority (3rd revision, now evidence-based):**
1. 자기일관성 — the only sensitive metric available on all 9
2. 추론 토큰 — DeepSeek (53–72) and Qwen (430–457) ONLY
3. 정확도 — all 9
4. logprob — **obtainable from none of the 9**. DeepSeek accepts `logprobs:true` but returns no field; Qwen rejects the request. Removed from the design.

**Self-contradiction caught and fixed:** the primary metric had been moved to reasoning tokens while config still disabled reasoning to save cost — which makes the metric always 0. Resolved per-model: Qwen ON, DeepSeek can't be turned off, everything that doesn't report reasoning tokens stays OFF.

**API quirks that cost real debugging time:**
- `claude-sonnet-5` rejects `temperature` (400, "deprecated for this model") — unique in the lineup. `ModelSpec.supports_temperature`.
- `claude-sonnet-5` thinking accepts only `{"type":"adaptive"}`; `enabled` + `budget_tokens` is rejected. Even adaptive reports no reasoning tokens.
- `gpt-5.6` accepts only `reasoning_effort: "none"`; minimal/low/medium are "Unsupported value". So reasoning tokens can't be turned on there.
- `solar-pro4` burned a 512-token output cap entirely on reasoning and returned no answer → reverted to `solar-pro3` (3 output tokens, correct, 1/21 the per-call cost).
- Output caps must be per-model; 8/16 was too low for DeepSeek and Upstage.

**Two findings worth putting in the paper:**
- Qwen answers the average-speed trap question WRONG with thinking off (A) and RIGHT with it on (B). Reasoning budget → accuracy, demonstrated before the study even starts. Mechanism section material.
- DeepSeek's reasoning tokens vary call-to-call on the same item (53 → 70–72). The metric is a varying quantity, which is exactly what we're measuring.

## V2/runner status

Implements roadmap steps 1–2. Files: config.py, prompts.py, calllog.py, providers/ (openai_compat, anthropic, hyperclova — raw `requests`, no SDKs), core.py, itembank.py, calibrate.py, select_bank.py, smoke_test.py, check_env.py, budget.py, diag_logprobs.py, diag_reasoning.py, selftest.py. Managed with **uv** (`pyproject.toml` + committed `uv.lock`; `[tool.uv] package = false` — without it uv tries to build a module that doesn't exist).

Design points worth preserving:
- `parse_letter` refuses to guess; the selftest caught it mis-reading "I cannot answer" as option I.
- selftest.py runs the whole pipeline on mock adapters with no API keys and asserts parsing, resume, band selection, noise floor, reasoning-token wiring, and spend-guard semantics.
- config.py loads the repo-root `.env` at import (python-dotenv was in requirements but never called) and forces UTF-8 console on Windows (cp949 would crash on the →·— in output).
- Resume keys count only successful calls.
- Calibration must be split by region — see [[llm-router-scheduling]].
- **Terminology (2026-08-25):** Korean prose and runner output say **"저부하 시간대"**, not "창"/window. Code identifiers keep `window` (`CALIBRATION_WINDOWS_KST`, `in_window()`, `check_window()`). See [[llm-router-scheduling]].

**Costs (measured):** calibration $7.42 (lineup k=5 $3.98 + anchors k=1 $3.44); main experiment $85.04. Anchor `gpt-5.6-sol` ($24) and lineup `claude-haiku` ($19) dominate. **Open question: is haiku worth $19 given "budget tier" was the selection criterion?**

**Not yet run:** the calibration pass itself. smoke test + diagnostics have run against real APIs.

## Git state (2026-08-25)

Everything is pushed. `main` == `origin/main`, 18 commits landed covering all work since 2026-06-01. Work happened on a `v2-redesign` branch then fast-forwarded into main.

**Bridge gotcha:** git works through the desktop bridge for add/commit, but the sandbox forbids unlink, so every git command leaves a stale `.git/**/*.lock` that blocks the next one. Workaround: `mv` every lock into `.git/_stale_locks/` before each git call. Leaves junk — tell the user to `rm -rf .git/_stale_locks && git gc --prune=now`. The bridge VM has no git identity; repo-local user.name/email were set from the repo's own history (J.Javier9434 <silverbreak1@gmail.com>). The bridge VM also has **no network**, so pushes and API calls must be run by the user.

**Bridge git is broken as of 2026-08-30 — do not run git through the bridge.** `.git/packed-refs` returns EDEADLK ("Resource deadlock avoided") to *any* reader (plain `head` fails too, so it is the mount, not git). Moving it aside made it worse: `git status` hit the same error on many tracked files and `git log` died with a Bus error; packed-refs was restored intact. Suspect an interrupted `git maintenance` holding locks — `.git/_stale_locks/maintenance.lock` dates from 2026-08-27. Check `git maintenance stop` and stray git processes on the Mac. Commit from the Mac's own terminal instead.

**`V2/notes/` = snapshot of the desktop app's project memory** (MEMORY.md + 6 topic files), added 2026-08-30 so the notes survive a machine move. Keep it in sync when project memory changes materially. Not yet committed — see the bridge-git note above.

**Leftover the user must delete on the Mac:** `V2/_to_delete/` (80K of uv-init debris that was sitting above V2/runner and would shadow it). Still present as of 2026-08-25 — the session bridge cannot unlink, so only the user can remove it.

**`launched_사용법.md` is NOT a duplicate — earlier note was wrong.** There is exactly one file (inode verified). Git's index holds the NFC spelling while APFS returns NFD from readdir; APFS is normalization-insensitive so the tracked path still resolves, which is why git reports no deletion but flags the NFD name as untracked. The repo has `core.precomposeunicode = true`, which the Mac's own git honors — **the phantom untracked entry only appears when git runs through the Linux bridge VM.** No action needed; do not "delete the duplicate" (it would delete the real tracked file).

Earlier analysis: P0-1 slot-level reanalysis fixing pseudoreplication; P0-2 confirming gpt-4o-mini ~49% is genuine MMLU-Pro performance + MMLU→MMLU-Pro correction.

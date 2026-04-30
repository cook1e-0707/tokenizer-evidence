# Baseline Fidelity Gate

Date: 2026-04-28

## 2026-04-30 Update

This gate has been superseded for Scalable Fingerprinting / Perinucleus. The
official-code Qwen-adapted Scalable/Perinucleus final package is now standing
with `48/48` valid successes, `0` method failures, `0` pending rows, candidate
utility sanity passed, and `paper_ready=true`.

Use these current artifacts for paper-facing Scalable/Perinucleus claims:

- `results/processed/paper_stats/baseline_perinucleus_official_qwen_final_summary.json`
- `results/tables/baseline_perinucleus_official_qwen_final.csv`
- `results/processed/paper_stats/baseline_paper_registry_summary.json`
- `results/tables/baseline_paper_registry.csv`
- `results/tables/paper_baseline_comparison.csv`

Required wording: report this row as a `Qwen-adapted official
Scalable/Perinucleus baseline`. Do not describe it as an unmodified full
fine-tune reproduction. The legacy adapted `baseline_perinucleus` artifacts
remain excluded diagnostics and must not be used as the successful Scalable
Fingerprinting result.

This audit gates baseline implementations before any new full-scale baseline experiment. It is read-only with respect to training and evaluation: no Chimera jobs were launched and no paper claims were edited.

## Inputs Read

- `docs/baseline_gap_audit.md`
- `docs/baseline_protocol.md`
- `docs/calibration_protocol.md`
- `configs/experiment/baselines/`
- `src/baselines/`
- `results/processed/paper_stats/*baseline*`
- `results/tables/*baseline*`
- `manifests/baseline_perinucleus/eval_manifest.json`
- `manifests/baseline_chain_hash/train_manifest.json`
- `manifests/baseline_chain_hash/eval_manifest.json`
- `manuscripts/69db2644566dcc36c9da320e/section_02_related_work.tex`
- `manuscripts/69db2644566dcc36c9da320e/section_07_experiments.tex`
- `manuscripts/69db2644566dcc36c9da320e/references.bib`

## Current Artifact State

The matched-budget baseline package is internally consistent but does not yet contain a strong external active ownership baseline. `baseline_summary.json` reports 48 rows, of which 36 are in the paper-ready denominator: fixed representative, uniform bucket, and English-random active fingerprint. All 36 denominator rows have matching contract hashes. The KGW provenance control rows are explicitly unavailable and outside the ownership denominator.

Perinucleus-style artifacts are completed but not paper-ready. `baseline_perinucleus_summary.json` reports 48/48 completed valid rows, 0/48 successes, `thresholds_frozen=false`, and `utility_suite_completed=false`. This package is contract-consistent but not faithful to the original Scalable Fingerprinting method.

Chain&Hash artifacts are package-prepared only. `baseline_chain_hash_summary.json` reports 48 target rows and 48 pending rows. No training or final verification result is available for this gate.

## External Source Audit

The audit used paper metadata from the manuscript bibliography and current public repository checks:

- Scalable Fingerprinting / Perinucleus: paper `Scalable Fingerprinting of Large Language Models`, arXiv `2502.07760`, OpenReview `https://openreview.net/forum?id=CRyOyiVvvJ`, official repo `https://github.com/SewoongLab/scalable-fingerprinting-of-llms`, HEAD `fdceaba14bd3e89340916a6a40e27c945d48460e`, MIT license.
- Instructional Fingerprinting: paper `Instructional Fingerprinting of Large Language Models`, ACL Anthology `https://aclanthology.org/2024.naacl-long.180/`, official repo `https://github.com/cnut1648/Model-Fingerprint`, HEAD `4ae5e8a124c37f25a3711c407e85a45fda6ecb08`, MIT license.
- Chain&Hash: paper `Hey, That's My Model! Introducing Chain & Hash, An LLM Fingerprinting Technique`, arXiv `2407.10887`, OpenReview `https://openreview.net/forum?id=UWi94bRsgm`; paper lists `https://github.com/microsoft/Chain-Hash`, but `git ls-remote` returned repository not found, so no runnable official code was verified.
- CTCC: paper `CTCC: A Robust and Stealthy Fingerprinting Framework for Large Language Models via Cross-Turn Contextual Correlation Backdoor`, ACL Anthology `https://aclanthology.org/2025.emnlp-main.356/`, repo `https://github.com/Xuzhenhua55/CTCC`, HEAD `8db93218260bed31b8f18acc9c6ac3e1955d3a42`, no GitHub license detected.
- EverTracer: paper `EverTracer: Hunting Stolen Large Language Models via Stealthy and Robust Probabilistic Fingerprint`, ACL Anthology `https://aclanthology.org/2025.emnlp-main.358/`, repo `https://github.com/Xuzhenhua55/EverTracer`, HEAD `70b402f7b7456c6d94e1fae2de554d77dd6cd921`, no GitHub license detected.
- MEraser: paper `MEraser: An Effective Fingerprint Erasure Approach for Large Language Models`, ACL Anthology `https://aclanthology.org/2025.acl-long.1455/`, repo `https://github.com/JingxuanZhang77/MEraser`, HEAD `5ecac341f6d1928cc422a4aecbb382f049a79e88`, no GitHub license detected.
- KGW: paper `A Watermark for Large Language Models`, PMLR `https://proceedings.mlr.press/v202/kirchenbauer23a.html`, repo `https://github.com/jwkirchenbauer/lm-watermarking`, HEAD `82922516930c02f8aa322765defdb5863d07a00e`, Apache-2.0 license.
- PostMark: paper `PostMark: A Robust Blackbox Watermark for Large Language Models`, ACL Anthology `https://aclanthology.org/2024.emnlp-main.506/`, repo `https://github.com/lilakk/PostMark`, HEAD `0f1109f3461450e6762daad606bf9d60bdb2bbf2`, no GitHub license detected.
- MergePrint, ESF, and RESF are cited paper baselines/related work, but no local official-code integration was found in this repo.

No local clone of the external repositories above was found under the repo root. Therefore no official-code smoke test has been performed locally.

## Fidelity Grades

Grade definitions:

- A: official code used, commit recorded, smoke-tested, adapted only in documented protocol-compatible ways.
- B: faithful reimplementation, no official code available or official code unusable, validated on an anchor experiment.
- C: adapted baseline, core idea retained but protocol/model differs materially; use only with an `adapted` label.
- D: weak proxy, useful only as diagnostic or ablation.
- F: invalid as a baseline; must not enter the main comparison table.

Current grade counts are: A=0, B=0, C=1, D=3, F=9.

The only C-grade implementation is the Chain&Hash-style package, and it is still pending execution. It is not yet a result baseline. The D-grade rows are internal ablations or weak proxies. Every public-code external method is either not integrated, only represented by a placeholder, or represented by a materially incomplete adapted implementation.

## Required Specific Judgment: Current Perinucleus No-Train Result

The current Perinucleus package is not a faithful Scalable Fingerprinting baseline.

- Generates fingerprints: partially. It builds keyed first-token response targets from the base-model next-token distribution.
- Fine-tunes or inserts fingerprints: no. The adapter explicitly reports no training; enrollment is next-token scoring only.
- Checks a fingerprinted model: no. It checks the base Qwen2.5-7B-Instruct distribution/output behavior without an inserted fingerprint.
- Evaluates utility: no. The artifacts report `utility_status=not_evaluated_requires_shared_organic_utility_suite`.
- Uses chat template for instruct models: no evidence. The adapter tokenizes raw prompts with `add_special_tokens=False`.

Decision: Grade F for main-table use. It may be retained only as an appendix diagnostic showing that a no-train Perinucleus-style probe is not a valid ownership baseline under this protocol. It must not be described as an implementation of Scalable Fingerprinting.

## Main Comparability Assessment

The paper currently has strong internal evidence for the proposed method and internal ablations, but the baseline gate blocks NeurIPS-level external comparison claims. A main-conference or Spotlight-level ownership verification paper needs at least one strong external active ownership baseline implemented with either official code or a validated faithful reimplementation under the same FAR, query, and utility budget.

The current baseline set is sufficient for internal ablation tables and workshop-level diagnostics only. It is not sufficient for a NeurIPS main-conference superiority claim, and it is not sufficient for a NeurIPS Spotlight-level comparison.

## Final Decision

1. Baselines allowed in main table.

None. No current external baseline has grade A or B with completed matched FAR and utility evaluation.

2. Baselines allowed only in appendix.

Fixed representative and uniform bucket may appear as internal ablations. English-random active fingerprint may appear as a weak proxy or negative diagnostic. The current Perinucleus no-train result may appear only as an invalid-diagnostic appendix row, explicitly labeled `not Scalable Fingerprinting`.

3. Baselines forbidden from paper claims.

Current Perinucleus no-train artifacts, KGW/PostMark as ownership baselines, CTCC/EverTracer/ESF placeholders, MEraser as a main ownership baseline, and any unexecuted Chain&Hash result are forbidden from main paper performance claims.

4. Baselines requiring official-code rerun.

Scalable Fingerprinting / Perinucleus and Instructional Fingerprinting require official-code integration or a documented faithful reimplementation plus anchor validation. CTCC, EverTracer, KGW, and PostMark require official-code integration if used. MEraser requires a separate attack/scrubbing protocol, not a main ownership-baseline protocol.

5. Next experiments permitted.

Permitted next steps are official-code smoke tests, protocol-level compatibility checks, and small anchor validations designed to raise a candidate external baseline to grade A or B. The Chain&Hash-style package can proceed only after documenting why official code is unavailable and defining an anchor validation that tests equivalence to the published method.

6. Next experiments blocked.

All new full-scale external baseline final matrices are blocked until at least one external active ownership baseline reaches grade A or B, with frozen FAR/utility protocol and completed anchor validation. The current Perinucleus no-train package is blocked from main-table use regardless of its completed 48 rows.

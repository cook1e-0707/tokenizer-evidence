# Project Status

## 2026-05-03 Full FAR Status

- Full FAR workflow head includes the completed organic FAR documentation plus the non-owner cache backend added in the current branch.
- Full FAR remains in progress, not complete. The correct current state is:
  - artifact-backed claim subset is available but is not full FAR;
  - required base-Qwen registered-probe null subset has run on Chimera H200;
  - required base-Qwen organic prompt-bank null has completed through the new two-stage backend;
  - required base-Qwen non-owner probe null now has a two-stage cache backend and remains pending execution;
  - optional non-Qwen / unprotected-Qwen nulls remain downstream.
- Required base-Qwen registered-probe null slice:
  - completed fresh registered null rows: `384`;
  - false accepts observed in the completed registered-probe metrics: `0`;
  - scope remains a small registered-probe slice, not complete full FAR.
- Required base-Qwen organic prompt-bank null:
  - Stage 1 GPU prompt cache is complete: `10` prompt-cache CSV shards, `1000` `completed_prompt_cache` rows.
  - Stage 1 used a global 10-way split across H200 and A100 resources.
  - The older row-level organic path is deprecated for production because it repeats base-Qwen inference per FAR row and is too slow.
  - The old local `comparison/full_far_payload_claim/shards/organic-prompts-10way/` directory contains only `55` partial row-level rows and must not be aggregated.
  - Stage 2 CPU expansion completed as a 20-way row-shard build under:
    - `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/comparison/full_far_payload_claim/shards/organic-prompts-20way-from-cache`
  - Stage 2 Slurm split:
    - `Intel6240`: row shards `0-9` (`SHARD_OFFSET=0`, `GLOBAL_SHARD_COUNT=20`);
    - `Intel6326`: row shards `10-19` (`SHARD_OFFSET=10`, `GLOBAL_SHARD_COUNT=20`).
  - Aggregation completed with `20` row shards and `8000` completed fresh organic null rows.
- Observed base-Qwen organic prompt-bank FAR:
  - Ours: `0/1000` false accepts at `M=1,3,5,10`.
  - Original Perinucleus: `6/1000` false accepts at `M=1`; `0/1000` false accepts at `M=3,5,10`.
  - Inspection confirms the 6 Perinucleus `M=1` false accepts are exact single-token collisions: organic base-Qwen generated `Certainly` with token id `95456`, matching the selected Perinucleus fingerprint response for payloads `U01` and `U08`.
- Current aggregate status:
  - `status = completed_registered_and_organic_null_subset`;
  - `completed_fresh_organic_null = 8000`;
  - `completed_fresh_registered_null = 384`;
  - `not_executed_fresh_null_inference_required = 2048`;
  - `full_far_complete = False`.
- Next decision gate:
  - execute the non-owner two-stage cache backend instead of the old row-level path;
  - optional non-Qwen / unprotected-Qwen null models remain lower priority.

## Current Standing Status

- Accepted clean compiled path:
  - `compiled-c3-r4` passed on `Qwen/Qwen2.5-7B-Instruct`
  - representative payloads `U00`, `U03`, `U12`, and `U15` are accepted at seed `17`
- Publication-scale clean coverage:
  - `G1` payload x seed package is now standing on `U00..U15 @ seeds 17, 23, 29`
  - final G1 aggregate is `48/48` included, `0` pending, `0` excluded
  - `G2` prompt-family package is now standing on `PF1/PF2/PF3 x U00/U03/U12/U15 x seeds 17,23,29`
  - final G2 aggregate is `36/36` included, `0` pending, `0` excluded
- Standing robustness path:
  - `batch3-preflight-reopen` passed
  - `batch3a` passed
  - `batch3b` passed
  - `batch3c` passed
  - `batch3d` passed
- Active execution scope:
  - manuscript-facing consolidation, C1 reporting, frozen matched-budget baseline artifacts, official Perinucleus baseline integration, and R1 second-family artifact triage
  - no active clean-path reruns
  - no active robustness-grid expansion
  - no active theorem reruns
  - `G3a-v3` block-count scale is standing on the held-out `B1/B2/B4 x U00..U15 x seeds 17,23,29` matrix
  - `G4` training-signal scale is standing on `S16/S32/S64/S128 x U00/U03/U12/U15 x seeds 17,23,29`
  - no additional Batch 3 expansion
  - `B1`/`B2` matched-budget baseline package configs, calibration artifacts, final artifacts, inclusion lists, and contract-hash accounting are frozen
  - `B1`/`B2` matched-budget baseline package is standing under frozen B0 accounting: fixed representative and uniform bucket are valid successes, English-random active fingerprint is a valid method-failure baseline, and KGW provenance control is explicit task-mismatched unavailable control outside the denominator
  - `B3` official Scalable/Perinucleus Qwen-adapted baseline is standing and paper-ready: `48/48` clean exact-verifier successes, `0` method failures, `0` pending
  - `R1` Llama 3.1 8B replication executed on Chimera with a strict-passed Llama tokenizer catalog; artifact accounting is complete, but exact-gate claim readiness is false because all 12 valid completed runs fail the exact gate while passing the RS-aware gate
- Paper-facing appendix status:
  - `S1` clean compiled table is landed in the manuscript appendix
  - `S2` robustness-stage and attack-family tables are landed in the manuscript appendix
  - `T1` contextual-alignment table is landed in the manuscript appendix
  - `T2-r1` objective-comparison and bucket-level supplementary tables are landed in the manuscript appendix
  - `G1` payload x seed table, inclusion list, and summary are landed as frozen paper-facing artifacts
  - `G2` prompt-family table, inclusion list, and summary are landed as frozen package artifacts; canonical paper accounting is wired in code and should be regenerated on Chimera to preserve `/home` and `/hpcstor6` provenance paths
  - `G3a-v3` block-count table, failure table, slot-margin table, inclusion list, summary, and compute accounting are landed as frozen paper-facing artifacts
  - `G4` training-signal table, inclusion list, summary, and compute accounting are landed as frozen paper-facing artifacts
  - `B0` matched-budget baseline and calibration protocols are frozen in `docs/baseline_protocol.md` and `docs/calibration_protocol.md`
  - `B1`/`B2` matched-budget baseline package is landed as frozen paper-facing artifacts under matched-budget calibration
  - `B3` official Scalable/Perinucleus Qwen-adapted final matrix is landed as frozen paper-facing artifacts and may enter the main external-baseline table with the required adaptation label
  - Paper-facing baseline registry and comparison artifacts are landed at `results/tables/baseline_paper_registry.csv` and `results/tables/paper_baseline_comparison.csv`
  - `R1` Llama replication artifacts are landed as a negative/diagnostic cross-family result: `12/12` valid completed, `0/12` exact-gate successes, `12/12` RS-aware successes, `0` invalid exclusions, `0` pending
- Theorem-package standing:
  - `T1 contextual_exact` produced an accepted Chimera run under `theorem1_qwen7b/contextual_exact`
  - `T1 sequence_proxy` now also has an accepted repaired Chimera run under `theorem1_qwen7b_rerun_sequence_proxy_v2`
  - `T1` is now standing for manuscript consolidation: both arms recover the same compiled `U03` target exactly, with the standing contextual arm coming from the earlier `compiled_gate` run and the repaired `sequence_proxy` arm verified through `canonical_render` over the carried `compiled_eval_contract`
  - initial `T2` objective package executed cleanly, but remained non-discriminative because the single-token-per-bucket compiled catalog collapses all three objectives to the same effective supervision problem
  - repaired `T2-r1` package now produced a discriminative result on the strict-passed multi-member-bucket Qwen catalog at `U15`
  - `fixed_representative` passes the exact-slot gate on the canonical representatives
  - `bucket_mass` remains bucket-correct but misses the canonical representative on `TOPIC=climate`
  - `uniform_bucket` remains bucket-correct but fails the exact-slot gate by selecting non-canonical members `SECTION=review` and `TOPIC=climate`

## Baseline Fidelity Status

- Baseline Fidelity Gate is complete.
- The paper now has one paper-ready external active ownership baseline: `scalable_fingerprinting_perinucleus_official_qwen_final`.
- Official Scalable/Perinucleus Qwen adaptation status: `A_adapted`, main-table eligible only with the label `Qwen-adapted official Scalable/Perinucleus baseline`.
- Official Scalable/Perinucleus Qwen final result: `48/48` valid clean exact-verifier successes, `0` method failures, `0` pending, `success_rate=1.0`.
- Clean ownership success is now parity, not superiority: the primary method and the official Qwen-adapted Perinucleus baseline both reach `48/48` on the matched clean Qwen final matrix.
- `perinucleus_no_train_diagnostic` and legacy `baseline_perinucleus` artifacts are forbidden for main-table Scalable claims and retained only under diagnostic/quarantine naming.
- `Chain&Hash-style` remains `C_pending` / proxy-only; it is forbidden in the main comparison table unless upgraded to `B` or better by a subsequent fidelity audit.
- KGW/PostMark-style provenance controls remain task-mismatched controls, not primary ownership baselines.
- R1 is an RS-aware cross-family diagnostic, not exact Llama replication: exact gate `0/12`, RS-aware gate `12/12`.
- Robustness status is mixed and bounded: whitespace perturbations preserve acceptance, while truncate-tail and delimiter-scrub transformations break acceptance and must not be claimed as broad output-side robustness.

## Current Priority

1. Keep the accepted `compiled-c3-r4`, `batch3c`, `batch3d`, `T1`, `T2-r1`, `G1`, `G2`, `G3a-v3`, `G4`, and official Perinucleus Qwen final results frozen and aligned with paper-facing summary artifacts.
2. Reframe manuscript claims around clean parity plus structural mechanism: do not claim clean-success superiority over Perinucleus.
3. Build the claim-support matrix before editing manuscript text; unsupported claims about FAR, utility, compute superiority, broad robustness, and cross-family exact replication remain forbidden.
4. Audit matched FAR, utility degradation, and compute accounting for the primary method versus official Qwen-adapted Perinucleus before launching any new comparative jobs.
5. Treat `R1` as artifact-paper-ready but not exact claim-paper-ready: report the exact-gate failure honestly, use RS-aware success only as diagnostic cross-family evidence, and do not claim Llama clean exact-path replication from this package.
6. Keep new theorem reruns frozen unless the paper claim changes or stricter verifier-path symmetry is explicitly required.

## Experimental Scale Snapshot

### Common Qwen 7B Train/Eval Envelope

- model: `Qwen/Qwen2.5-7B-Instruct`
- tokenizer max length: `512`
- adapter recipe: LoRA with `r=16`, `alpha=32`, `dropout=0.0`
- optimizer-scale settings shared by the standing clean/theorem packages:
  - `batch_size = 1`
  - `epochs = 64`
  - `learning_rate = 5e-5`
- standing train/eval runtime envelope:
  - `1 x A100`
  - `16 CPUs`
  - `96 GB RAM`
  - `24h` wall-clock request
- exact-slot compiled packages use the same prompt family:
  - base instruction: `Select exactly one allowed carrier token.`
  - one prompt per slot
  - `generation_max_new_tokens = 1`
- robustness attacks do **not** retrain the model:
  - they perturb rerendered canonical text from accepted compiled-c3 baselines
  - they use the CPU preflight attack runtime, not the A100 train/eval envelope

### Clean Compiled Path Scale

| Stage | Train payload labels | Blocks x fields | Bucket / codebook scale | Contract sample count | Dataset size per run | Prompt scale | Eval scope |
|---|---|---:|---|---:|---:|---|---|
| `compiled-c0` | `OK/NO/UP/AI` (4 pilot labels) | `1 x 2` | historical narrow single-block pilot before the later `U00..U15` full-grid codebook | `8` | `8` | `2` exact-slot prompts per payload, `1` token per slot | single target pilot eval (`OK`) |
| `compiled-c1` | `OK/NO/UP/AI` (4 pilot labels) | `1 x 2` | asymmetric single-block pilot; still a narrow representative-label package rather than the later full `4 x 4` sweep | `8` | `8` | `2` exact-slot prompts per payload, `1` token per slot | single target pilot eval (`AI`) |
| `compiled-c2` | `U00..U15` (16 labels) | `1 x 2` | `SECTION=4 buckets`, `TOPIC=4 buckets`, `1` canonical token per bucket | `32` | `32` | `2` exact-slot prompts per payload, `1` token per slot | single-block full-grid eval at `U14` |
| `compiled-c3` | `U00..U15` (16 labels) | `2 x 2` | same `4 x 4` single-token codebook, expanded to `block_count=2` | `64` | `64` | `4` exact-slot prompts per payload, `1` token per slot | double-block eval at `U14` |
| `compiled-c3-r1` | same as `compiled-c3` | `2 x 2` | same `4 x 4`, `block_count=2` contract reused across runs | `64` per run | `64` per run | same exact-slot prompt family | `4` clean runs: `U00/U03/U12/U15 @ seed 17` |
| `compiled-c3-r2` | same as `compiled-c3` | `2 x 2` | same `4 x 4`, `block_count=2` contract reused across runs | `64` per run | `64` per run | same exact-slot prompt family | `4` clean runs: `U00/U15 @ seed 23,29` |
| `compiled-c3-r3` | same as `compiled-c3` | `2 x 2` | same `4 x 4`, `block_count=2` contract reused across runs | `64` per run | `64` per run | same exact-slot prompt family | `4` clean runs: `U03/U12 @ seed 23,29` |
| `compiled-c3-r4` | same as `compiled-c3` | `2 x 2` | same `4 x 4`, `block_count=2` contract reused across runs | `64` per run | `64` per run | same exact-slot prompt family | `4` clean runs: `U00/U03/U12/U15 @ seed 17` |
| `G1 payload-seed scale` | same as `compiled-c3` | `2 x 2` | same `4 x 4`, `block_count=2` contract reused across all standing seeds and payloads | `64` per run | `64` per run | same exact-slot prompt family | full package: `U00..U15 @ seeds 17,23,29` (`48` included clean train/eval cases) |
| `G2 prompt-family scale` | same as `compiled-c3` | `2 x 2` | same `4 x 4`, `block_count=2` contract reused across all prompt families | `64` per run | `64` per run | `PF1/PF2/PF3` semantically equivalent prompt families | full package: `PF1/PF2/PF3 x U00/U03/U12/U15 x seeds 17,23,29` (`36` included clean train/eval cases; `PF1` reuses standing evidence) |
| `G3a-v3 block-count scale` | same as `compiled-c3` | `1/2/4 x 2` | same `4 x 4` compiled codebook, with fields and prompt family fixed | `32/64/128` per run | `32/64/128` per run | same exact-slot prompt family plus margin-aware bucket loss | standing held-out package: `B1/B2/B4 x U00..U15 x seeds 17,23,29`; `142/144` valid successes, `2` valid method failures, `0` invalid exclusions, `0` pending |
| `G4 train-signal scale` | sample-count ladder | `2 x 2` | same `4 x 4` compiled codebook, with model, prompt family, codebook, block count, and G3a-v3 hp04 objective fixed | `16/32/64/128` effective samples per run | `16/32/64/128` effective samples per run (`128` is `64` unique samples repeated twice) | same exact-slot prompt family plus margin-aware bucket loss | standing package: `S16/S32/S64/S128 x U00/U03/U12/U15 x seeds 17,23,29`; `48/48` valid successes, `0` valid method failures, `0` invalid exclusions, `0` pending |

Notes:
- for the compiled exact-slot packages, `contract sample count = payload label count x block_count x fields_per_block`
- from `compiled-c2` onward, the standing clean path is no longer a narrow pilot; it is a full `U00..U15` label package over the current Qwen 7B compiled codebook

### Robustness Path Scale

| Stage | Accepted clean source | Payloads | Seeds | Attack families | Run count | Attacked text scale | Standing outcome |
|---|---|---|---|---|---:|---|---|
| `batch3-preflight-reopen` | `compiled-c3-r2` | `U00`, `U15` | `23`, `29` | `whitespace_scrub (0.1)`, `truncate_tail (0.25)` | `2` | canonical rerender of a `2`-block, `4`-slot compiled output | harness reopened |
| `batch3a` | `compiled-c3-r2` | `U00`, `U15` | `23`, `29` | `whitespace_scrub (0.1)`, `truncate_tail (0.25)` | `8` | same `2`-block canonical text format | small grid passed |
| `batch3b` | `compiled-c3-r2` + `compiled-c3-r3` | `U00`, `U03`, `U12`, `U15` | `23`, `29` | `whitespace_scrub (0.1)`, `truncate_tail (0.25)` | `16` | same `2`-block canonical text format | payload expansion passed |
| `batch3c` | `compiled-c3-r4` | `U00`, `U03`, `U12`, `U15` | `17` | `whitespace_scrub (0.1)`, `truncate_tail (0.25)` | `8` | same `2`-block canonical text format | seed expansion passed |
| `batch3d` | `compiled-c3-r4` | `U00`, `U03`, `U12`, `U15` | `17` | `delimiter_scrub (1.0)` | `4` | same `2`-block canonical text format | new family passed |

Observed standing attack behavior:
- `whitespace_scrub`: acceptance preserved on all standing runs
- `truncate_tail`: acceptance broken on all standing runs
- `delimiter_scrub`: acceptance broken on all standing runs

### Theorem Package Scale

| Package / arm | Catalog / bucket geometry | Train payload labels | Blocks x fields | Contract sample count | Dataset size per run | Prompt scale | Eval target | Standing result |
|---|---|---|---:|---:|---:|---|---|---|
| `T1 contextual_exact` | compiled Qwen codebook: `SECTION=4`, `TOPIC=4`, `1` canonical token per bucket | `U00/U03/U12/U15` | `2 x 2` | `16` | `16` | `4` exact-slot prompts per payload, `1` token per slot | `U03` | accepted exact-slot |
| `T1 sequence_proxy` | same compiled Qwen codebook and same compiled payload contract as `contextual_exact` | `U00/U03/U12/U15` | `2 x 2` | `16` carried by contract | `4` scaffolded train examples (one per payload label) | one `8`-line scaffold prompt per payload, `4` output lines, `generation_max_new_tokens = 8` | `U03` | accepted exact-slot after payload-label repair |
| `T2-r1 bucket_mass` | strict-passed multi-member-bucket Qwen source catalog: `SECTION` bucket 3 = `update/review`, `TOPIC` bucket 3 = `science/climate` | `U00..U15` | `1 x 2` | `32` | `32` | `2` exact-slot prompts per payload, `1` token per slot | `U15` | bucket-correct, exact-slot fail |
| `T2-r1 fixed_representative` | same multi-member-bucket source catalog as above | `U00..U15` | `1 x 2` | `32` | `32` | `2` exact-slot prompts per payload, `1` token per slot | `U15` | accepted exact-slot |
| `T2-r1 uniform_bucket` | same multi-member-bucket source catalog as above | `U00..U15` | `1 x 2` | `32` | `32` | `2` exact-slot prompts per payload, `1` token per slot | `U15` | bucket-correct, exact-slot fail |

Interpretation:
- `T1` is now a fair same-payload, same-codebook comparison:
  - `contextual_exact` conditions each slot with exact allowed-carrier context
  - `sequence_proxy` uses a single scaffold prompt per payload, but now also conditions on the payload label
- `T2-r1` is intentionally harder than the standing clean compiled path:
  - it switches from the single-token-per-bucket compiled catalog to the multi-member-bucket source catalog
  - this is what makes `fixed_representative` vs `bucket_mass` vs `uniform_bucket` actually distinguishable

## Archived Failures

- `batch3_preflight_failed`
  - clean generated-text baseline was not accepted
  - downstream attack runs were all `accepted_before=false -> accepted_after=false`
  - archived locally under `batch3_preflight_failed/`
  - not formal robustness evidence
- Pre-compiled Qwen 7B main-path failures
  - sequence-continuation era failures before compile-then-train
  - contract-coverage failures before the compiled contract compiler closed the train/eval gap
  - historical only; not current standing evidence
- Partial `batch3b` launch before `U03/U12` clean baselines existed
  - operational/configuration gap, later repaired by `compiled-c3-r3`
  - superseded by the accepted final `batch3b` result

## Compiled Milestones

- `compiled-c0`: minimal Qwen 7B compiled path passed.
- `compiled-c1`: asymmetric single-block compiled path passed.
- `compiled-c2`: full single-block compiled path passed.
- `compiled-c3`: double-block compiled path passed on the same Qwen 7B codebook.
- `compiled-c3-r1`: representative multi-payload validation passed on `U00`, `U03`, `U12`, and `U15`.
- `compiled-c3-r2`: multi-seed validation passed on `U00` and `U15` with seeds `23` and `29`.
- `compiled-c3-r3`: supplemental clean baselines passed on `U03` and `U12` with seeds `23` and `29`.
- `compiled-c3-r4`: supplemental clean baselines passed on `U00`, `U03`, `U12`, and `U15` with seed `17`.
- `G1 payload-seed scale`: publication-scale clean package passed on `U00..U15` with seeds `17`, `23`, and `29`, ending at `48/48` included runs.
- `batch3-preflight-reopen`: attack harness restored on accepted compiled-c3 baselines.
- `batch3a`: small robustness grid passed on `U00` and `U15` with seeds `23` and `29` across `whitespace_scrub` and `truncate_tail`.
- `batch3b`: payload-expansion robustness grid passed on `U00`, `U03`, `U12`, and `U15` with seeds `23` and `29`.
- `batch3c`: seed-expansion robustness grid passed on `U00`, `U03`, `U12`, and `U15` with seed `17`.
- `batch3d`: single-family delimiter-attack expansion passed on `U00`, `U03`, `U12`, and `U15` with seed `17`.
- Next target: build the claim-support matrix and audit whether matched FAR, utility, and compute comparisons are already complete for the primary method versus official Qwen-adapted Perinucleus before launching any new Chimera jobs.

## 2026-04-20

### Milestone: Full Single-Block Compiled Path Passed

Qwen/Qwen2.5-7B-Instruct passed the full single-block compiled path under the compile-then-train framework.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- training path: `compile-then-train`
- objective: `field-conditioned masked bucket objective`
- decoding: deterministic one-token-per-slot constrained decoding
- codebook: `SECTION=4 buckets`, `TOPIC=4 buckets`, `1 canonical token per bucket`
- block_count: `1`

Passing result:
- stage: `compiled-c2`
- accepted = `true`
- verifier_success = `true`
- decoded_payload correct
- no NaN / non-finite training failure
- compiled train contract emitted successfully

Interpretation:
- the full single-block compiled path is standing
- the primary blocker is no longer contract compilation, contextual alignment, or single-block bucket control

### Milestone: Double-Block Compiled Path Passed

Qwen/Qwen2.5-7B-Instruct then passed the double-block compiled path on the same compile-then-train framework and compiled codebook.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- training path: `compile-then-train`
- objective: `field-conditioned masked bucket objective`
- decoding: deterministic one-token-per-slot constrained decoding
- codebook: `SECTION=4 buckets`, `TOPIC=4 buckets`, `1 canonical token per bucket`
- block_count: `2`

Passing result:
- stage: `compiled-c3`
- accepted = `true`
- verifier_success = `true`
- decoded_payload correct
- no NaN / non-finite training failure
- compiled train and eval contracts emitted successfully
- deterministic rendered canonical blocks verified successfully

Interpretation:
- the compiled multi-block path is now standing for the current Qwen 7B codebook
- the next gate is not a larger codebook or a new model family
- the next gate is representative multi-payload validation under the unchanged compiled-c3 framework

### Milestone: Representative Multi-Payload Double-Block Validation Passed

Qwen/Qwen2.5-7B-Instruct passed `compiled-c3-r1` on representative double-block payload targets without changing the compiled contract, codebook, prompt contract, or objective.

Verified scope:
- framework: unchanged `compile-then-train`
- codebook: `SECTION=4 buckets`, `TOPIC=4 buckets`, `1 canonical token per bucket`
- block_count: `2`
- representative payload targets: `U00`, `U03`, `U12`, `U15`

Passing result:
- all four representative payload runs produced `accepted = true`
- all four representative payload runs produced `verifier_success = true`
- all four representative payload runs decoded the correct payload
- all four representative payload runs remained numerically healthy

Interpretation:
- the compiled-c3 path is no longer only a single-target success
- the next gate is seed robustness under the unchanged compiled-c3 setup

### Milestone: Multi-Seed Double-Block Validation Passed

Qwen/Qwen2.5-7B-Instruct passed `compiled-c3-r2` on additional seeds while keeping the compiled contract, codebook, prompt contract, and objective fixed.

Verified scope:
- framework: unchanged `compile-then-train`
- codebook: `SECTION=4 buckets`, `TOPIC=4 buckets`, `1 canonical token per bucket`
- block_count: `2`
- payload targets: `U00`, `U15`
- seeds: `23`, `29`

Passing result:
- all four effective runs produced `accepted = true`
- all four effective runs produced `verifier_success = true`
- all four effective runs decoded the correct payload
- all four effective runs remained numerically healthy

Interpretation:
- the compiled-c3 path is no longer only a single-seed success
- the next gate is reopening Batch 3 preflight on accepted compiled baselines

### Milestone: Batch 3 Preflight Reopened

Qwen/Qwen2.5-7B-Instruct successfully reopened Batch 3 preflight on accepted compiled-c3 clean baselines.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- clean baseline source: accepted `compiled-c3-r2` runs
- attack path: canonical attack over deterministic rerendered compiled slot values
- preflight attacks:
  - `U00 @ seed 23` with `whitespace_scrub`
  - `U15 @ seed 29` with `truncate_tail`

Passing result:
- both attack runs completed successfully
- both attack runs started from `accepted_before = true`
- one benign attack preserved acceptance
- one stronger truncation attack caused acceptance failure

Interpretation:
- the attack harness is now aligned with the compiled canonical path
- the next gate is a small-scope `Batch 3A` robustness grid, not a broad robustness sweep

### Milestone: Batch 3A Small Robustness Grid Passed

Qwen/Qwen2.5-7B-Instruct passed `Batch 3A` on accepted compiled-c3 clean baselines without changing the model, codebook, runtime envelope, or attack harness.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- clean baseline source: accepted `compiled-c3-r2` runs
- payloads: `U00`, `U15`
- seeds: `23`, `29`
- attack families:
  - `whitespace_scrub`
  - `truncate_tail`

Passing result:
- all eight attack runs completed successfully
- all eight attack runs started from `accepted_before = true`
- all four `whitespace_scrub` runs preserved acceptance
- all four `truncate_tail` runs caused acceptance failure

Interpretation:
- the attack harness is no longer only preflight-valid
- small-scope robustness behavior is now stable on the compiled-c3 Qwen 7B path
- the next gate is `Batch 3B`, which should expand payload coverage while keeping the same seeds, attack families, and runtime constraints

### Archived Failure: Batch 3B Payload Expansion Partially Executed

`Batch 3B` was launched to expand payload coverage from `U00/U15` to `U00/U03/U12/U15` under the unchanged compiled-c3 Qwen 7B attack path.

Observed result:
- `U00` and `U15` attack runs completed successfully and preserved the `Batch 3A` behavior pattern
- `U03` and `U12` attack runs did not produce valid attack outputs

Root cause:
- the missing half was not an attack-harness failure
- `U03_s23`, `U03_s29`, `U12_s23`, and `U12_s29` clean compiled-c3 baselines had never been materialized
- as a result, `attack.clean_eval_summary_path` expanded to an empty value for those cases

Required repair:
- supplement accepted clean baselines for `U03/U12 @ seed 23/29` on the same compiled-c3 path
- rerun only the missing `Batch 3B` attacks after those clean baselines exist

Guard added:
- attack execution now fails immediately with a clear error if `attack.clean_eval_summary_path` is empty or does not point to a real eval summary file

### Milestone: Compiled-C3-R3 Supplemental Clean Baselines Passed

Qwen/Qwen2.5-7B-Instruct passed the supplemental compiled-c3 clean-baseline stage for the previously missing payload and seed combinations.

Verified scope:
- framework: unchanged `compile-then-train`
- codebook: `SECTION=4 buckets`, `TOPIC=4 buckets`, `1 canonical token per bucket`
- block_count: `2`
- payload targets: `U03`, `U12`
- seeds: `23`, `29`

Passing result:
- all four supplemental clean-baseline runs produced `accepted = true`
- all four supplemental clean-baseline runs produced `verifier_success = true`
- all four supplemental clean-baseline runs decoded the correct payload
- all four supplemental clean-baseline runs remained numerically healthy

Interpretation:
- the clean-baseline coverage needed for the full `Batch 3B` payload grid is now complete
- the remaining `Batch 3B` gap was operational rather than methodological

### Milestone: Batch 3B Payload-Expansion Grid Passed

Qwen/Qwen2.5-7B-Instruct passed `Batch 3B` after supplementing the missing clean baselines and rerunning only the blocked attack cases.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- clean baseline source: accepted `compiled-c3-r2` and `compiled-c3-r3` runs
- payloads: `U00`, `U03`, `U12`, `U15`
- seeds: `23`, `29`
- attack families:
  - `whitespace_scrub`
  - `truncate_tail`

Passing result:
- all sixteen attack runs completed successfully
- all sixteen attack runs started from `accepted_before = true`
- all eight `whitespace_scrub` runs preserved acceptance
- all eight `truncate_tail` runs caused acceptance failure

Interpretation:
- payload expansion on the compiled-c3 Qwen 7B robustness path is now standing
- the next gate can move to a larger `Batch 3C`, but still without reopening baselines or new model families

### Milestone: Compiled-C3-R4 Seed-17 Clean Baselines Passed

Qwen/Qwen2.5-7B-Instruct passed the supplemental compiled-c3 clean-baseline stage for the seed-17 robustness extension.

Verified scope:
- framework: unchanged `compile-then-train`
- codebook: `SECTION=4 buckets`, `TOPIC=4 buckets`, `1 canonical token per bucket`
- block_count: `2`
- payload targets: `U00`, `U03`, `U12`, `U15`
- seed: `17`

Passing result:
- all four clean-baseline runs produced `accepted = true`
- all four clean-baseline runs produced `verifier_success = true`
- all four clean-baseline runs decoded the correct payload
- all four clean-baseline runs remained numerically healthy

Interpretation:
- the clean-baseline coverage now includes the full representative payload grid at seed `17`
- the project can test a seed-expansion robustness stage without changing model family or payload scope

### Milestone: Batch 3C Seed-Expansion Grid Passed

Qwen/Qwen2.5-7B-Instruct passed `Batch 3C` by adding seed `17` across the established `Batch 3B` payload grid while keeping the attack families fixed.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- clean baseline source: accepted `compiled-c3-r4` runs
- payloads: `U00`, `U03`, `U12`, `U15`
- seed: `17`
- attack families:
  - `whitespace_scrub`
  - `truncate_tail`

Passing result:
- all eight attack runs completed successfully
- all eight attack runs started from `accepted_before = true`
- all four `whitespace_scrub` runs preserved acceptance
- all four `truncate_tail` runs caused acceptance failure

Interpretation:
- the robustness grid now covers the representative payload set across an additional seed
- the next minimal axis of expansion is an additional attack family, not a broader model or payload sweep

### Milestone: Batch 3D Additional Attack-Family Grid Passed

Qwen/Qwen2.5-7B-Instruct passed `Batch 3D` by adding a delimiter-destruction attack family on top of the already accepted `Batch 3C` payload and seed grid.

Verified scope:
- model: `Qwen/Qwen2.5-7B-Instruct`
- clean baseline source: accepted `compiled-c3-r4` runs
- payloads: `U00`, `U03`, `U12`, `U15`
- seed: `17`
- new attack family:
  - `delimiter_scrub`

Passing result:
- all four attack runs completed successfully
- all four attack runs started from `accepted_before = true`
- all four `delimiter_scrub` runs caused acceptance failure

Interpretation:
- the compiled-c3 Qwen 7B path now has stable robustness evidence across payload expansion, seed expansion, and an additional structure-breaking attack family
- the next step should be documentation and paper-facing result consolidation rather than continued grid growth
## Model Policy

- `gpt2` is smoke-only from this point onward:
  - parser/verifier unit tests
  - plumbing checks
  - local smoke validation
- `gpt2` must not be used for paper-facing generated-text acceptance, Batch 2.8, Batch 3, or later comparison.
- Batch 2.8 model plan:
  - bridge: `Qwen/Qwen2.5-3B-Instruct`
  - main: `Qwen/Qwen2.5-7B-Instruct`
  - replication: `meta-llama/Meta-Llama-3.1-8B-Instruct`

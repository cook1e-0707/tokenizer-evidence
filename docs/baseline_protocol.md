# Baseline Protocol

Status: B0 frozen on 2026-04-27.

This document freezes the matched-budget baseline protocol before any B1/B2
baseline execution. It is a protocol artifact, not an experimental result.

## Scope

B0 defines what will count as a valid baseline comparison. It does not launch
training, evaluation, threshold selection, or result aggregation.

The first baseline package must remain small and matched-budget. It must not
expand into a broad baseline zoo or model zoo before the frozen protocol below
has been implemented and audited.

## Primary Comparison Target

The primary ownership method is the standing Qwen 7B compiled evidence path
under the G3a-v3/G4 objective family.

Frozen primary target:

| Field | Value |
|---|---|
| backbone | `Qwen/Qwen2.5-7B-Instruct` |
| tokenizer | the same Qwen tokenizer used by the standing compiled packages |
| codebook | the frozen compiled `4 x 4` carrier codebook |
| block count | `2` |
| fields per block | `2` |
| prompt family | the standing exact-slot prompt family (`PF1`) |
| payloads | `U00`, `U03`, `U12`, `U15` |
| seeds | `17`, `23`, `29` |
| query budget | `M = 4` verifier queries per ownership decision |
| target false-accept rate | `0.01` |
| utility metric | `acceptance_rate` on the frozen organic/utility prompt set |

The baseline package may include additional diagnostic rows, but the paper-facing
primary comparison must include the fixed target above.

## Baseline Families

### Task-Matched Ownership Baselines

These may be used as primary ownership baselines if they satisfy the matched
budget and calibration rules.

1. `fixed_representative`: force the canonical representative token for each
   target bucket.
2. `uniform_bucket`: place supervision uniformly over all members of the target
   bucket.
3. `english_random_active_fingerprint`: active black-box fingerprint probes
   using natural English carrier responses rather than tokenizer-aligned
   compiled carriers.
4. `perinucleus_adapted`: optional, only if a clean task-matched adapter can be
   implemented without changing the frozen comparison budget.

### Task-Mismatched Provenance Controls

Watermark or provenance methods may be included only as controls. They must not
be described as primary ownership baselines because they answer a different
question: text provenance rather than owner-controlled evidence recovery from a
suspect model.

Allowed controls:

1. KGW-style output watermark control.
2. PostMark-style provenance control.

The `baseline_english_random` adapter is executable for B1 as a no-train
natural-language active-fingerprint baseline. It remains in the ownership
paper-ready denominator and valid failures remain in the denominator.

The `baseline_kgw`, `baseline_ctcc`, and `baseline_esf` adapters are safe
placeholders unless a real implementation is explicitly wired and audited.
Placeholder rows must be reported as `unavailable`, not as failed experimental
evidence. KGW remains a task-mismatched provenance control and does not block
the B1 ownership denominator unless it is explicitly upgraded to an audited
control package.

## Fidelity Gate For Paper Claims

Main-table inclusion requires a baseline fidelity grade of `A` or `B` under the
frozen Baseline Fidelity Gate.

Grade `C` baselines are allowed only if explicitly labeled as adapted/proxy
methods and placed outside the main comparison table, for example in an
appendix diagnostic table.

Grade `D` and `F` methods are forbidden from comparison claims. They may be
mentioned only as diagnostics, invalid controls, or related-work context, and
must not be described as evidence that the primary method outperforms a
faithful external baseline.

The current fidelity state is `A=0`, `B=0`, `C=1`, `D=3`, `F=9`; therefore no
external baseline is currently eligible for the main comparison table. Full
final external-baseline matrices are blocked until the relevant method is
promoted to `A` or `B` by a subsequent audit.

## Matched Budget Rules

All primary baselines must match the following budgets unless a row is explicitly
labeled `unmatched`.

| Budget | Frozen rule |
|---|---|
| backbone | Same Qwen 7B backbone as the primary method. |
| tokenizer | Same tokenizer unless the baseline cannot operate on token-level carriers; deviations are `tokenizer_unmatched`. |
| training budget | Same A100/CPU/RAM/wall-clock envelope as the matched Qwen package unless the method is no-train. |
| query budget | At most `M = 4` verifier queries per ownership decision. |
| false-accept budget | Threshold must be selected only by the B0 calibration protocol at target FAR `0.01`. |
| utility budget | Utility degradation must be no worse than the primary method by more than `0.02` absolute `acceptance_rate`, or have overlapping Wilson intervals. |
| final matrix | Same payloads, seeds, backbone, and prompt split as the primary target. |

No method may compensate for weak evidence by using extra final-test queries,
post-hoc threshold changes, or a different final payload/seed split.

## Decision Rule

Each method must emit a scalar `ownership_score` where larger means stronger
evidence for the claimed owner payload.

The final decision is:

```text
accepted = ownership_score >= frozen_threshold
```

For the compiled primary method, the deterministic exact gate and RS-aware gate
must both be reported. For baselines with continuous scores, `frozen_threshold`
is selected by `docs/calibration_protocol.md` before final evaluation and then
held fixed.

## Required Run Schema

Every baseline run summary must contain at least:

- `package`
- `method`
- `baseline_family`
- `baseline_role`
- `model_id`
- `tokenizer_id`
- `payload`
- `seed`
- `block_count`
- `prompt_family`
- `query_budget`
- `queries_used`
- `ownership_score`
- `frozen_threshold`
- `accepted`
- `verifier_success`
- `decoded_payload`
- `utility_acceptance_rate`
- `target_far`
- `calibration_observed_far`
- `calibration_threshold_source`
- `contract_hash_status`
- `run_dir`
- `config_path`
- `config_hash`

Any method-specific fields must be appended without changing the shared schema.

## Inclusion And Exclusion Rules

Valid completed failures remain in the denominator.

Allowed invalid exclusions:

- missing required artifact
- corrupted JSON/CSV/YAML artifact
- incomplete run with no valid final summary
- train/eval contract mismatch
- wrong model, tokenizer, payload, seed, prompt family, or block count
- missing checkpoint for an eval run that requires a checkpoint
- path violation that writes raw checkpoints, adapters, raw generations, or large
  logs into home/repo instead of scratch

Not allowed as exclusions:

- `accepted = false`
- verifier failure on a valid output
- low score under a frozen threshold
- timeout after a valid final summary exists
- method instability
- failed active fingerprint recovery
- failed provenance watermark detection

Failed, timeout, and unavailable runs must be listed explicitly with status and
reason. They must not be deleted from inclusion lists.

## Path Contract

On Chimera:

- repo/configs/manifests/small summaries live under `$REPO_HOME`
- checkpoints, adapters, raw generations, trainer cache, and large logs live
  under `$EXP_SCRATCH`
- paper-facing summaries live under `$REPO_HOME/results`

Expected environment:

```bash
export REPO_HOME="$HOME/tokenizer-evidence"
export EXP_SCRATCH="${SCRATCH:-/scratch/$USER}/tokenizer-evidence"
export PAPER_RESULTS="$REPO_HOME/results"
```

Every scratch run must have a home-side manifest row pointing to its scratch
paths.

## Required Paper-Facing Artifacts

B1/B2 must write:

- `results/tables/matched_budget_baselines.csv`
- `results/tables/matched_budget_baselines.tex`
- `results/processed/paper_stats/baseline_summary.json`
- `results/processed/paper_stats/baseline_run_inclusion_list.json`
- `results/processed/paper_stats/baseline_compute_accounting.json`

These artifacts must distinguish:

- target count
- completed count
- valid completed count
- success count
- method failure count
- invalid excluded count
- pending count
- exact/score-gate success rate
- observed FAR
- utility degradation
- compute accounting

## Frozen Guardrails

- Do not tune thresholds on final results.
- Do not exclude valid method failures.
- Do not compare a provenance control as a primary ownership baseline.
- Do not change the final matrix after seeing baseline outcomes.
- Do not add broad baselines before the minimal matched-budget package is
  complete.
- Do not write raw experiment artifacts to home/repo.

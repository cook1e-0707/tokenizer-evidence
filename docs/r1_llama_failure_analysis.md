# R1 Llama 3.1 8B Failure Analysis

## Scope

This note records the paper-facing interpretation of `r1_llama3_1_8b_replication_v1` after the completed Chimera train/eval rerun at commit `ff390e9`.

R1 used `meta-llama/Meta-Llama-3.1-8B-Instruct`, the strict-passed Llama tokenizer carrier catalog, block count `B2`, payloads `U00/U03/U12/U15`, and seeds `17/23/29`.

## Artifact Status

- Target cases: `12`.
- Completed cases: `12`.
- Valid completed cases: `12`.
- Invalid excluded cases: `0`.
- Pending cases: `0`.
- Contract hash status: `12/12 match`.
- Artifact-paper-ready: `true`.
- Claim-paper-ready for exact clean-path replication: `false`.

All valid completed failures remain in the denominator. None of the 12 R1 cases is excluded as an invalid run.

## Outcome

- Exact gate success: `0/12`.
- RS-aware gate success: `12/12`.
- Slot bucket accuracy: `1.0` mean.
- Symbol error count: `0` mean.
- Erasure count: `0` mean.
- Decoded payload correctness: `12/12`.

The Llama package therefore does not support a claim that the exact clean compiled path replicated on a second model family. It does support a narrower diagnostic claim: the Llama runs learned verification-relevant bucket and payload semantics under the RS-aware report, but the exact-slot gate failed systematically.

## Failure Mode

The failures are valid method failures, not path/accounting failures:

- train and eval completed for all cases;
- train/eval contract hashes match;
- block count is correct;
- payload recovery under the RS-aware report is correct;
- exact-slot recovery is not correct, so `accepted_under_exact_gate=false` and `verifier_success=false`.

This is an exact-representative mismatch rather than a semantic payload-recovery failure. The result is consistent with the paper's metric hierarchy: bucket/payload recovery and exact-slot pass must be reported separately.

## Paper Usage

Use R1 as a negative or diagnostic cross-family result unless a separate pre-registered `R1-v2` repair is launched. Acceptable wording:

> On Llama 3.1 8B, the verifier recovered the intended payload under the RS-aware bucket report in all 12 cases, but no case passed the exact-slot gate. We therefore treat R1-v1 as a cross-family failure of exact clean-path replication, not as a successful second-family clean result.

Do not state that R1 proves exact cross-family replication.

## Next Decision

No additional experiment should be launched automatically. If exact cross-family replication is required for the manuscript, define a separate `R1-v2` protocol before running it, including the verifier gate, selection rule, and whether the official claim target is exact gate or RS-aware gate.

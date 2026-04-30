# Claim Support Matrix

Status: generated on 2026-04-30. No new experiments were run.

## Decision Rules

- Main-paper claims must cite a frozen paper-facing artifact.
- Valid method failures stay in denominators.
- Clean Qwen ownership success is parity with official Qwen-adapted Perinucleus, not superiority.
- Exact representative recovery and bucket/RS-aware recovery are separate metrics.
- Robustness claims are bounded to the tested attack family.

## Matrix

| Claim | Main? | Quantitative support | Required caveat | Missing evidence |
|---|---|---|---|---|
| C01 Qwen clean payload recovery stability | yes | G1: `48/48` clean cases; accepted, verifier, and decoded-payload rates all `1.0`; CI low `0.9259` | Qwen2.5-7B-Instruct clean exact-slot compiled protocol only | Matched FAR and utility evidence are separate |
| C02 Prompt-family stability | yes | G2: `36/36` across `PF1/PF2/PF3 x U00/U03/U12/U15 x seeds 17/23/29` | Tested equivalent prompt families only | Broader prompt distribution and adversarial wrappers |
| C03 Block-count scaling | yes | G3a-v3: `142/144`; B1 `48/48`, B2 `46/48`, B4 `48/48`; slot-bucket accuracy mean `0.9948` | Report the two valid method failures | Larger block counts and bucket-member scaling |
| C04 Training-signal scaling | yes | G4: `48/48` across `S16/S32/S64/S128`; exact, RS, and slot-bucket rates `1.0` | Only sample count varies; S128 repeats 64 unique samples twice | Larger unique-sample ladders and more model families |
| C05 Clean success superiority over Perinucleus | no | Not supported: ours `48/48`, official Perinucleus `48/48` | Correct claim is parity | Differentiating FAR, utility, compute, persistence, or payload-capacity result |
| C06 Clean success parity with Perinucleus | yes | Paper comparison: ours `48/48`; official Qwen-adapted Perinucleus `48/48` | Label Perinucleus as Qwen-adapted official-code baseline | FAR, utility, compute, and robustness/persistence comparisons |
| C07 FAR-calibrated ownership decision for main comparison | no | B0 calibration exists for internal fixed/uniform baselines at target FAR `0.01`; no matched ours-vs-Perinucleus FAR artifact | Protocol exists, main comparison artifact missing | Null calibration for ours and Perinucleus at `M=1/3/5/10` |
| C08 Better utility preservation than Perinucleus | no | Perinucleus utility sanity passed on tinyBenchmarks: adapter `0.6192`, base `0.6035`, signed drop `-0.01565`; ours missing on same suite | Can cite only as Perinucleus validation | Same utility benchmark for base, ours, and Perinucleus |
| C09 Compute efficiency relative to Perinucleus | no | Compute accounting exists, but not normalized for matched efficiency; G1 requested `864+864` GPU-hours, Perinucleus final eval `732.1s` and `228` forwards | Descriptive compute only | Matched train/eval wall-clock, GPU-hours, trainable params, artifact size |
| C10 Exact cross-family replication | no | R1 exact gate `0/12`, method failures `12/12`, claim-paper-ready `false` | This is a negative exact-replication result | Pre-registered R1-v2 exact repair |
| C11 Cross-family RS-aware semantic recovery | yes | R1 RS-aware gate `12/12`; slot-bucket accuracy `1.0`; symbol errors `0`; exact gate `0/12` | Diagnostic RS/bucket recovery, not exact representative replication | More model families and larger matrices for broad transfer |
| C12 Whitespace robustness | yes | Batch3 `whitespace_scrub`: accepted before `4/4`, accepted after `4/4` | Only tested whitespace perturbation | Larger attack grid |
| C13 Truncation or delimiter-destruction robustness | no | Positive robustness not supported: `truncate_tail` accepted after `0/4`, `delimiter_scrub` accepted after `0/4` | Supported statement is that these break current acceptance | Revised robust verifier and new attack package |
| C14 Bucket-mass objective justification | yes | T2-r1: bucket_mass bucket-correct `1.0` but exact `0.5`; fixed rep exact pass; uniform bucket bucket-correct `1.0` but exact `0.0` | Supports bucket/RS metric hierarchy, not universal objective dominance | Broader multi-member-bucket matrix plus utility/lexical metrics |
| C15 Next-token measurability | yes | T1: contextual_exact accepted with slot-exact `1.0`; repaired sequence_proxy accepted with slot-exact `1.0` for `U03@seed17` | Empirical implementation support; formal theorem text must carry the theorem | More examples if extending beyond compiled exact-slot evidence |

The machine-readable version is [claim_support_matrix.csv](/Users/guanjie/Documents/tokenizer_alignment/results/tables/claim_support_matrix.csv).

## Abstract-Allowed Claims

- Qwen clean compiled ownership recovery is stable on the frozen G1 matrix: `48/48`.
- Official Qwen-adapted Scalable/Perinucleus reaches the same clean exact-verifier success: `48/48`; the abstract may say parity, not superiority.
- The work studies a structured tokenizer-aligned bucket/RS payload evidence channel, provided the wording does not imply broad robustness or cross-family exact replication.

## Intro-Allowed Claims

- Prompt-family stability is supported by G2: `36/36`.
- Block-count scaling is supported by G3a-v3 with the explicit denominator `142/144`.
- Training-signal scaling is supported by G4: `48/48`.
- R1 supports diagnostic RS-aware semantic recovery: `12/12`, while exact representative recovery fails `0/12`.
- T1/T2 support the metric hierarchy: exact representatives are diagnostic, while bucket/RS payload recovery is the primary evidence object.

## Appendix-Only Claims

- Batch3 attack-family details, especially whitespace success versus truncation/delimiter failure.
- R1 failure cases and exact-gate negative result.
- Internal ablations: fixed representative, uniform bucket, and English-random active fingerprint.
- Raw compute accounting until a matched efficiency comparison is produced.
- Quarantined legacy Perinucleus diagnostics.

## Forbidden Unless New Experiments Are Run

- Our method outperforms official Scalable/Perinucleus on clean ownership verification.
- The main comparison is FAR-calibrated for both ours and Perinucleus.
- Our method preserves utility better than Perinucleus.
- Our method is compute-efficient relative to Perinucleus.
- The method achieves cross-family exact replication.
- The method is robust to truncation, delimiter scrubbing, or broad output-side post-processing.

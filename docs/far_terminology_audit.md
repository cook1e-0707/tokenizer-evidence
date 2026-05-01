# FAR Terminology Audit

Status: Prompt V terminology audit, 2026-04-30. No experiments were launched.

## Required Distinctions

| Term | Meaning | Current status | Paper wording |
|---|---|---|---|
| Binary clean ownership detection | A verifier decides whether the queried model contains the registered fingerprint. This is the native target of original Scalable/Perinucleus. | Complete for clean Qwen: ours `48/48`, official Qwen-adapted Perinucleus `48/48`. | Clean success parity. |
| Structured payload claim verification | A verifier checks whether a model supports a specific owner/payload claim and can recover the corresponding payload evidence. This is the native target of the compiled tokenizer-aligned method. | Complete for clean G1; wrong-payload claim subset is artifact-backed but not full FAR. | Structured payload-bound evidence channel. |
| Full FAR / null false accept rate | False acceptance under frozen thresholds on null models, non-owner probes, wrong-owner claims, and organic prompts. | Incomplete for the main ours-vs-Perinucleus comparison. | Do not claim full FAR superiority. |
| Wrong-payload claim accept rate subset | Acceptance when archived final artifacts are evaluated under a mismatched payload claim. | Available as a diagnostic subset: ours `0.0`; original Perinucleus `1.0` for `M=1,3,5,10`. | Preliminary claim-binding subset, not full FAR. |

## Current Quantitative State

| Metric | Ours | Official Perinucleus | Interpretation |
|---|---:|---:|---|
| Clean verification | `48/48` | `48/48` | Parity, not superiority. |
| TinyBench utility | `0.6080464224` | `0.6191832009` | Both pass sanity; Perinucleus is about `0.0111` higher on this sanity. |
| Utility drop vs base `0.6035317340` | `-0.0045146884` | `-0.0156514669` | Both are above the base sanity score; do not claim ours is better. |
| Wrong-payload claim accept rate subset | `0.0` | `1.0` | Ours binds structured payload claims in this subset; original Perinucleus is binary and not payload-bound. |
| Full FAR/null calibration | incomplete | incomplete | Requires fresh null generation and frozen thresholds. |

## Allowed Wording

- "Under clean Qwen ownership verification, ours and the official Qwen-adapted Scalable/Perinucleus baseline both reach `48/48`."
- "The wrong-payload result is a structured claim-acceptance subset, not a full FAR/null calibration."
- "Original Perinucleus is a strong binary fingerprint detector; it is not designed to bind a decoded payload claim."
- "The compiled tokenizer-aligned method provides a structured payload-bound evidence channel under the tested protocol."

## Forbidden Wording

- "Perinucleus has FAR=1."
- "Perinucleus fails generally."
- "Ours outperforms Perinucleus on clean ownership verification."
- "Ours has full FAR superiority over Perinucleus."
- "Ours preserves utility better than Perinucleus."
- "The current main comparison is fully FAR-calibrated."

## Gate

Final manuscript integration remains blocked for any FAR superiority claim until:

- base-Qwen nulls are generated;
- wrong-owner claim checks are generated;
- non-owner probe and organic prompt nulls are generated;
- query budgets `M=1,3,5,10` are evaluated under frozen thresholds;
- payload-adapted Perinucleus is implemented or the task mismatch is explicitly stated.

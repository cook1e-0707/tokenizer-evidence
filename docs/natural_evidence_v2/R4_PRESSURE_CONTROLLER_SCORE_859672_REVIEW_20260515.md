# R4 Pressure-Controller Score 859672 Review

Date: 2026-05-15

Status: `FAIL_R4_PRESSURE_CONTROLLER_SCORE_859672_WRONG_CONTROLS_PASS_NO_GENERATION`

## Scope

Job `859672` was the repaired H200/pomplun pressure-controller teacher-forced scoring array. It was Qwen-only, same-contract `a55e`, scoring-only, and did not run generation, training, Llama, FAR, sanitizer, payload-diversity evaluation, or paper-facing claims.

Remote output root:

```text
/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_positive_selectivity_pressure_controller_score_859672
```

Local reviewed summaries:

```text
results/natural_evidence_v2/status/r4_pressure_controller_score_859672_review/
```

## Result

All `72/72` Slurm array tasks completed with exit code `0:0`; this is not an infrastructure failure.

Summary-level result:

| Metric | Value |
| --- | ---: |
| Protected basic teacher-forced gate passes | `72/72` |
| Overall selective gate passes | `0/72` |
| Wrong-key controlled basic gate passes | `72/72` |
| Wrong-payload controlled basic gate passes | `72/72` |

Best protected-lift grid was `grid_65`:

| Field | Value |
| --- | ---: |
| bonus_nats | `1.5` |
| penalty_nats | `0.0` |
| max_target_mass | `0.45` |
| max_kl_budget | `0.1` |
| controlled-protected mean target mass | `0.4023347084` |
| controlled-protected lift vs base | `0.3975028714` |
| controlled-protected lift vs task-only | `0.4006622806` |
| controlled-protected rank1 rate | `1.0` |
| wrong-key mean committed target mass | `0.3365671794` |
| wrong-key rank1 rate | `0.9846191406` |
| wrong-payload mean committed target mass | `0.2712397256` |
| wrong-payload rank1 rate | `0.9707031250` |

## Diagnosis

The controller can raise committed target mass under protected scoring, but the current wrong-control design does not establish keyed selectivity. Wrong-key and wrong-payload controlled arms also satisfy the same basic mass/rank criteria.

Remote row probes show the wrong-control token sets were not simply identical by accident:

- `wrong_payload_controlled` used complement controller target ids, with `controller_target_equals_committed_target = 0/8192`.
- `wrong_key_controlled` used deterministic coordinate hashing; `4064/8192` rows matched the committed target bit and `4128/8192` did not in `grid_65`.
- Controller target/other token-id overlap was `0` for controlled-protected, wrong-key, and wrong-payload rows.

The failure mode is that wrong-control arms still load the protected adapter and the verifier/scorer still measures committed target ids. The protected adapter already places strong mass on committed target ids; wrong-control pressure is not a valid null unless it can overcome or remove that adapter pressure. In `grid_65`, wrong-key rows whose controller target did not match the committed target still had mean committed target mass `0.2723994399` and rank1 rate `0.9694767442`. Wrong-payload complement pressure had mean committed target mass `0.2712397256` and rank1 rate `0.9707031250`.

This is a selectivity-control semantics failure, not a positive channel result.

## Control State

No downstream action is unlocked by `859672`.

Allowed next action:

```text
artifact-only wrong-control mapping/scorer semantics repair planning
```

Not unlocked by this result:

```text
generation
training
Qwen E2E
Llama
same-family null
sanitizer
FAR
payload-diversity claim
paper-facing positive claim
new H200 scoring resubmission
```

Any future scoring route must first define wrong-key/wrong-payload controls that actually test selectivity under the protected adapter, or explicitly change the null design and review that change before Slurm.


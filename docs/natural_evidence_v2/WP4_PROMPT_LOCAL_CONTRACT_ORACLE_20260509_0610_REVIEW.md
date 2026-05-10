# WP4 prompt-local contract and decoder oracle review

## Decision

The artifact-only WP4 prompt-local contract preflight passed. This does not
train a model, does not generate model transcripts, does not run E2E, and does
not establish payload recovery or FAR. It only verifies that the selected
prompt-local coding contract is internally decodable under ideal target-bucket
substitution, and that wrong-key / wrong-payload oracle checks reject.

## Inputs

```text
primary_bank = results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank.jsonl
prompt_source = results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/restricted_step_label_strict_density_audit_prompts.jsonl
builder = scripts/natural_evidence_v2/build_wp4_prompt_local_contract.py
```

Primary bank:

```text
bucket 0 = Set | Plan
bucket 1 = Create | Prepare
min_bucket_mass = 0.06311000335572636
combined_bank_mass = 0.14328484169406375
mass_ratio = 1.2703982582035882
```

## Outputs

```text
results/natural_evidence_v2/contracts/wp4_prompt_local_contract_20260509_0610/qwen_v2_wp4_contract_manifest.json
results/natural_evidence_v2/contracts/wp4_prompt_local_contract_20260509_0610/qwen_v2_wp4_decoder_oracle_summary.json
results/natural_evidence_v2/contracts/wp4_prompt_local_contract_20260509_0610/qwen_v2_wp4_decoder_oracle_trace.csv
results/natural_evidence_v2/contracts/wp4_prompt_local_contract_20260509_0610/qwen_v2_wp4_prompt_local_contracts.jsonl
results/natural_evidence_v2/contracts/wp4_prompt_local_contract_20260509_0610/qwen_v2_wp4_P00_seed17_prompt_local_16slot.json
results/natural_evidence_v2/contracts/wp4_prompt_local_contract_20260509_0610/qwen_v2_wp4_P00_seed23_prompt_local_16slot.json
results/natural_evidence_v2/contracts/wp4_prompt_local_contract_20260509_0610/qwen_v2_wp4_P01_seed17_prompt_local_16slot.json
results/natural_evidence_v2/contracts/wp4_prompt_local_contract_20260509_0610/qwen_v2_wp4_P01_seed23_prompt_local_16slot.json
```

## Contract Shape

```text
protocol_id = natural_evidence_v2_controlled_micro_slots_wp4_prompt_local_v1
frame_policy = prompt_local
payloads = P00, P01
seeds = 17, 23
query_budgets = 8, 16, 32, 64
payload_size = 8 bits
checksum_size = 8 bits
slot_count = 16
slot anchors = Step 1: ... Step 16:
bucket_count = 2
```

Each prompt-local frame contains one full 16-bit codeword:

```text
8 payload bits || 8 checksum bits
```

The target bucket digit at each slot is key-masked. The decoder un-masks with
the committed audit key and checks both payload byte and checksum. This is a
deterministic artifact preflight only; it is not a security result.

## Oracle Results

```text
status = PASS_WP4_PROMPT_LOCAL_DECODER_ORACLE
contract_count = 4
target_oracle_rows = 16
target_oracle_accept_rows = 16
target_oracle_accept_rate = 1.0
wrong_key_oracle_rows = 16
wrong_key_oracle_accept_rows = 0
wrong_payload_oracle_rows = 16
wrong_payload_oracle_accept_rows = 0
wrong_key_id = qwen_v2_wp4_wrong_key_00
```

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| WP3-R1 strict density | Pass with caveat | Split-level 850885 dev/eval oracle completion passed; one legacy top-level language-drift failure remains recorded |
| WP3-R2 high-mass bank | Pass | Primary `Set|Plan` vs `Create|Prepare` bank selected from 851272 |
| WP3-R3 naturalness | Pass with caveat | Variant-balanced review had no forbidden/coding/semantic failures |
| WP4 decoder oracle | Pass | Target oracle accepts 16/16; wrong-key and wrong-payload accept 0/16 |
| WP5 teacher-forced gate | Not started | No model training or scoring for protected/task-only lift yet |
| Training / E2E | Forbidden | WP5 has not passed |

## Interpretation

The project has now shown that the v2 controlled-natural Step-label protocol has
a structurally observable 16-slot frame, a high-mass 2-way prompt-conditioned
bank, and a prompt-local decoder that can recover the intended codeword under
ideal target-bucket observations. This is the right precondition for WP5, but
it says nothing about whether a LoRA can learn the target bucket preference or
whether free generation will hit the target buckets.

## Next Allowed Action

Prepare WP5 teacher-forced target-mass gate artifacts only:

```text
base Qwen score-only plan
protected LoRA training plan review, not launch
task-only LoRA training plan review, not launch
slot CE mask at Step-label action-verb slots
margin loss at micro-slots
target-bucket mass / rank / margin scoring plan
```

Training remains forbidden until the WP5 plan and allowlist are separately
reviewed under the recorded gates.

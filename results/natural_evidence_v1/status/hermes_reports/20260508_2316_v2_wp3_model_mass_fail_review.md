# v2 WP3 model-mass failure review

phase:
V2_WP3_MODEL_MASS_AUDIT_FAIL_NEEDS_BUCKET_CONTEXT_REPAIR

job:
- `850288` (`nat-ev-v2-wp3mass`)
- state: `COMPLETED 0:0`
- runtime: 00:00:45
- scope: fixed-prefix base Qwen next-token bucket-mass scoring plus audit

artifacts:
- `results/natural_evidence_v2/status/wp3_bucket_mass_score_850288/`
- `results/natural_evidence_v2/status/wp3_model_mass_audit_850288/`

gate status:
- `tokenizer_stability_status=PASS`
- `density_gate_status=NOT_EVALUATED`
- `mass_gate_status=FAIL`
- `wp4_allowed=false`

mass failure:
All 7 repaired two-way banks failed the configured full-vocab mass gate
(`min_bucket_mass >= 0.005`).

| Bank | Min full-vocab mass | Ratio | Passed |
|---|---:|---:|---|
| sentence_opener_sequence_v0 | 4.052e-09 | 1.03 | false |
| step_opener_action_v0 | 7.574e-09 | 18.97 | false |
| discourse_marker_additive_v0 | 4.583e-09 | 6.59 | false |
| optional_hedge_frequency_v0 | 3.311e-07 | 4.94 | false |
| transition_word_plain_v0 | 4.893e-08 | 2.31 | false |
| function_word_conjunction_v0 | 8.530e-09 | 4.74 | false |
| function_word_preposition_v0 | 3.435e-07 | 5.11 | false |

interpretation:
The current WP3 scaffold has tokenizer-stable surfaces and template-dense
detector opportunities, but the fixed-prefix raw next-token mass under base
Qwen is too small for the configured model-mass gate. Candidate-normalized
bucket balance is diagnostic only and does not pass the current gate.

next_allowed_action:
Artifact-only mass failure analysis and bucket/context repair planning. Do not
train or run E2E. If more tokenizer/model scoring is needed, submit it through
Chimera Slurm.

forbidden_actions_confirmed:
No training, generation, Qwen E2E, Llama, same-family null, sanitizer, FAR, or
positive paper claim was started.

notification:
Telegram and email notification were sent successfully via
`scripts/natural_evidence_v1/hermes_notify.py`; delivery summary is stored at
`results/natural_evidence_v1/status/hermes_reports/20260508_2316_v2_wp3_model_mass_fail_review_notification.json`.

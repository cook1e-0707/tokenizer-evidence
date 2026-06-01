# VSG Config Security Review

Source: `configs/verification_substrate_gap/text_only_observability.yaml`
Planned supplement path: `configs/text_only_observability.yaml`
Field-name hits: `8`
Secret-value hits: `0`
Release recommendation: `schema_field_review_required_no_literal_secret_values_detected`

## Field-Name Hits

- line 37: `secret_key_allowed: false`
- line 42: `secret_key_allowed: false`
- line 47: `secret_key_allowed: false`
- line 49: `label: final_text_plus_public_row_bank_no_key`
- line 52: `secret_key_allowed: false`
- line 57: `secret_key_allowed: false`
- line 60: `allowed_information: [final_text, trace, authorized_key, reviewed_row_bank]`
- line 62: `secret_key_allowed: true`

## Scope

This review distinguishes key/HMAC-related schema field names from
literal secret values. It does not publish the config and does not
expand the VSG claim boundary.

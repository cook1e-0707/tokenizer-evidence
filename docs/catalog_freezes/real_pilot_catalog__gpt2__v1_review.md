# Catalog Freeze Remediation Review

- source_catalog: /Users/guanjie/Documents/tokenizer_alignment/configs/data/real_pilot_catalog.yaml
- tokenizer_backend: huggingface
- tokenizer_name: gpt2
- tokenizer_revision_source: gpt2
- freeze_timestamp: 20260414T031827Z
- git_commit: 969ba05

## SECTION
- field_blocked: False
- recommended_action: regroup_with_manual_review
### Bucket 0
- original_members: news, brief
- rejected_members: brief (multi_token)
- rejection_reasons: multi_token
- surviving_members: news
- bucket_became_empty: False
- recommended_action: regroup_with_manual_review

### Bucket 1
- original_members: report, memo
- rejected_members: memo (multi_token)
- rejection_reasons: multi_token
- surviving_members: report
- bucket_became_empty: False
- recommended_action: regroup_with_manual_review

### Bucket 2
- original_members: guide, digest
- rejected_members: digest (multi_token)
- rejection_reasons: multi_token
- surviving_members: guide
- bucket_became_empty: False
- recommended_action: regroup_with_manual_review

### Bucket 3
- original_members: update, review
- rejected_members: none
- rejection_reasons: none
- surviving_members: update, review
- bucket_became_empty: False
- recommended_action: keep_as_is

## TONE
- field_blocked: True
- recommended_action: drop_field
- change_log_messages:
  - TONE bucket 0 became empty after filtering
### Bucket 0
- original_members: calm, steady
- rejected_members: calm (multi_token), steady (multi_token)
- rejection_reasons: multi_token
- surviving_members: none
- bucket_became_empty: True
- recommended_action: drop_field

### Bucket 1
- original_members: clear, crisp
- rejected_members: crisp (multi_token)
- rejection_reasons: multi_token
- surviving_members: clear
- bucket_became_empty: False
- recommended_action: drop_field

### Bucket 2
- original_members: warm, bright
- rejected_members: none
- rejection_reasons: none
- surviving_members: warm, bright
- bucket_became_empty: False
- recommended_action: drop_field

### Bucket 3
- original_members: direct, plain
- rejected_members: none
- rejection_reasons: none
- surviving_members: direct, plain
- bucket_became_empty: False
- recommended_action: drop_field

## TOPIC
- field_blocked: False
- recommended_action: regroup_with_manual_review
### Bucket 0
- original_members: market, finance
- rejected_members: finance (multi_token)
- rejection_reasons: multi_token
- surviving_members: market
- bucket_became_empty: False
- recommended_action: regroup_with_manual_review

### Bucket 1
- original_members: travel, hotel
- rejected_members: hotel (multi_token)
- rejection_reasons: multi_token
- surviving_members: travel
- bucket_became_empty: False
- recommended_action: regroup_with_manual_review

### Bucket 2
- original_members: health, diet
- rejected_members: diet (multi_token)
- rejection_reasons: multi_token
- surviving_members: health
- bucket_became_empty: False
- recommended_action: regroup_with_manual_review

### Bucket 3
- original_members: science, climate
- rejected_members: none
- rejection_reasons: none
- surviving_members: science, climate
- bucket_became_empty: False
- recommended_action: keep_as_is

## REGION
- field_blocked: True
- recommended_action: drop_field
- change_log_messages:
  - REGION bucket 1 became empty after filtering
  - REGION bucket 2 became empty after filtering
  - REGION bucket 3 became empty after filtering
### Bucket 0
- original_members: urban, metro
- rejected_members: metro (multi_token)
- rejection_reasons: multi_token
- surviving_members: urban
- bucket_became_empty: False
- recommended_action: drop_field

### Bucket 1
- original_members: rural, valley
- rejected_members: rural (multi_token), valley (multi_token)
- rejection_reasons: multi_token
- surviving_members: none
- bucket_became_empty: True
- recommended_action: drop_field

### Bucket 2
- original_members: coastal, harbor
- rejected_members: coastal (multi_token), harbor (multi_token)
- rejection_reasons: multi_token
- surviving_members: none
- bucket_became_empty: True
- recommended_action: drop_field

### Bucket 3
- original_members: inland, prairie
- rejected_members: inland (multi_token), prairie (multi_token)
- rejection_reasons: multi_token
- surviving_members: none
- bucket_became_empty: True
- recommended_action: drop_field

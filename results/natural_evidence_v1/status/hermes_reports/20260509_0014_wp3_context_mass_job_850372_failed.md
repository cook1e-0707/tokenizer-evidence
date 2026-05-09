# Hermes Codex action report

phase:
V2_WP3_CONTEXT_SPECIFIC_MASS_SCORER_PREPARED_NEEDS_SLURM_SUBMISSION

action:
Synced the required WP3 context-mass scorer, Slurm wrapper, v2 allowlist, and
fixed score plan to Chimera, then submitted exactly one allowlisted Slurm job:
`850372` (`nat-ev-v2-wp3ctxm`).

job_result:
```text
job_id=850372
job_name=nat-ev-v2-wp3ctxm
partition=DGXA100
node=chimera13
state=FAILED
exit_code=1:0
elapsed=00:00:39
```

synced_artifacts:
```text
results/natural_evidence_v2/status/wp3_context_mass_score_850372/slurm/nat-ev-v2-wp3ctxm-850372.out
results/natural_evidence_v2/status/wp3_context_mass_score_850372/slurm/nat-ev-v2-wp3ctxm-850372.err
```

review:
- The wrapper ran on a compute node, validated the fixed `230`-row context-mass
  score plan, loaded `Qwen/Qwen2.5-7B-Instruct`, and then failed before writing
  score artifacts.
- No context-score JSONL, mass artifact, audit JSON, or score summary was
  produced.
- Failure reason: contextual tokenization boundary mismatch in plan row
  `0f8383dd9775def36e16` for surface `also`.
- The failing row is `discourse_marker_additive_v0`, lowercase variant, with
  bucket `0=["also", "plus"]` and bucket `1=["too", "again"]` after a prefix
  ending in a space. The scorer refused to score because
  `tokenizer.encode(prefix + surface)` did not preserve
  `tokenizer.encode(prefix)` as a prefix.

gate_result:
```text
mass_gate_status=NOT_EVALUATED
wp4_allowed=false
paper_claim_allowed=false
```

forbidden_actions_confirmed:
No training, generation, Qwen E2E rerun, Llama, same-family null, sanitizer,
FAR aggregation, or positive paper claim was started. No additional Slurm job
was submitted.

next_allowed_action:
Prepare an artifact-only WP3 context-mass plan/scorer repair for tokenizer
prefix-boundary retokenization, starting from job `850372` and row
`0f8383dd9775def36e16`. Do not submit another Slurm job until the repaired
plan/scorer is reviewed, locally validated without model scoring, and
allowlisted. Do not run CPU/GPU scoring directly on the Chimera login node.

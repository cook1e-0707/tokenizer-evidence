# R4 Metric-Exact Micro-Overfit 864332 Review

phase: V2_R4_METRIC_EXACT_MICRO_OVERFIT_864332_FAILED_REVIEWED_NO_NEXT_COMPUTE

job:
- job_id: `864332`
- job_name: `nat-ev-v2-r4mof`
- partition: `pomplun`
- node: `chimera21`
- state: `COMPLETED`
- elapsed: `00:04:08`
- exit_code: `0:0`

result:
- status: `FAIL_R4_METRIC_EXACT_MICRO_OVERFIT_864332_TEACHER_FORCED_GATE`
- protected mean target mass: `0.0179803`
- protected lift vs base: `+0.0131485`
- protected lift vs task-only: `+0.0163079`
- protected rank1 rate: `0.980469`
- protected median margin: `+0.0059255`
- teacher-forced surface gate: `FAIL`

interpretation:
- This was not a Slurm/wrapper/tokenizer failure.
- Metric-exact training improved rank ordering but did not create enough absolute target-token mass.
- The target-mass floor remained almost unsatisfied at the end of training (`final_floor_loss=0.196320` with floor `0.20`).
- No generation, Qwen E2E, Llama, same-family null, sanitizer, FAR, payload-diversity claim, or paper-facing claim is unlocked.

artifacts:
- `docs/natural_evidence_v2/R4_METRIC_EXACT_MICRO_OVERFIT_864332_REVIEW_20260516.md`
- `results/natural_evidence_v2/status/r4_candidate_v3_micro_overfit_864332/`
- `results/natural_evidence_v2/status/r4_metric_exact_micro_overfit_864332_review/`

next_allowed_action:
- Artifact-only failure analysis and a reviewed repair or pivot route before any new Slurm submission or generation route.

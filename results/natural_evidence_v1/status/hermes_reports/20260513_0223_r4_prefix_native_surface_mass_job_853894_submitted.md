# Hermes sync: R4 prefix-native surface-mass job submitted

phase:
V2_R4_PREFIX_NATIVE_SURFACE_MASS_JOB_853894_RUNNING

summary:
Codex submitted exactly one Chimera H200 Slurm job for R4 prefix-native teacher-forced tokenizer/model surface-mass scoring.

Job:
- job id: 853894
- job name: nat-ev-v2-r4pntfm
- state at submission check: RUNNING
- partition/QoS/account: pomplun / pomplun / cs_yinxin.wan
- initial node: chimera21
- time limit: 30-00:00:00

Scope:
- Qwen tokenizer/model forward scoring only
- conditions: base, protected, task_only
- score rows: `results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_20260513/r4_prefix_native_surface_probe_rows.jsonl`
- expected output dir: `/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/natural_evidence_v2/qwen_micro_slot_pilot/status/r4_prefix_native_surface_mass_score_853894`

Control-plane:
- local zero-enabled allowlist safety before submission: PASS
- remote zero-enabled allowlist safety before submission: PASS
- local/remote hash preflight: PASS
- single-enabled allowlist preflight: PASS
- allowlist disabled locally and remotely immediately after sbatch returned job id
- post-submit allowlist safety local/remote: PASS

Not started:
- generation
- training
- Llama
- same-family null
- sanitizer benchmark
- FAR aggregation
- paper-facing positive claim

next_allowed_action:
Monitor Slurm job `853894`. After completion, sync and review the teacher-forced surface-mass summary. Do not submit another scoring job or run generation/training/Llama/FAR/sanitizer/paper claims until this result is reviewed.

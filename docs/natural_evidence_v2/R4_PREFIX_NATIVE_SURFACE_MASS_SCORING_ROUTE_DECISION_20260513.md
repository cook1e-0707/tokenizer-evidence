# R4 prefix-native surface-mass scoring route decision

Date: 2026-05-13

## Scope

This decision records a single Chimera Slurm tokenizer/model scoring route for
the R4 prefix-native surface repair candidate.

This route is allowed to run Qwen tokenizer/model forward scoring only through
Slurm. It does not train, generate free outputs, launch Llama, run same-family
nulls, run sanitizer benchmarks, aggregate FAR, test payload diversity, or make
paper-facing positive claims.

## Inputs

- candidate rows:
  `results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_20260513/r4_prefix_native_surface_probe_rows.jsonl`
- candidate static validation:
  `results/natural_evidence_v2/status/r4_prefix_native_surface_repair_candidate_20260513/static_validation_summary.json`
- scorer:
  `scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py`
- Slurm wrapper:
  `scripts/natural_evidence_v2/slurm/r4_prefix_native_surface_mass_score_h200.sbatch`
- allowlist entry:
  `v2_r4_prefix_native_surface_mass_score_h200`

## Scoring Conditions

- base Qwen;
- protected adapter from the existing Qwen WP5-R2 package;
- task-only adapter from the existing Qwen WP5-R2 package.

The job must score exactly the repaired prefix-native rows unless an explicit
new route decision supersedes this one.

## Submission Rules

Before submission:

- local allowlist safety must pass with zero enabled entries;
- remote allowlist safety must pass with zero enabled entries;
- local/remote hashes for the wrapper, rows, current state, gate status, and
  allowlist must match;
- Hermes TG/email notification must be sent;
- exactly one allowlist entry may be enabled:
  `v2_r4_prefix_native_surface_mass_score_h200`.

After `sbatch` returns a job id:

- disable the allowlist entry locally and remotely;
- record a submission JSON;
- do not submit another job in the same route tick.

## Gates

The teacher-forced surface gate remains the same diagnostic gate:

- protected target surface mass lift vs base >= `+0.15`;
- protected target surface mass lift vs task-only >= `+0.10`;
- protected target surface rank-1 rate >= `0.70`;
- protected median target margin > `0`;
- task-only target surface mass lift vs base should not be large.

If this gate fails, the next action is artifact-only failure diagnosis. It does
not unlock generation, training, Llama, FAR, sanitizer, payload diversity, or
paper claims.

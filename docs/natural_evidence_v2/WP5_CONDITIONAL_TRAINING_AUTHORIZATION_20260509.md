# WP5 conditional training authorization

## Decision

On 2026-05-09T06:18Z, the user authorized Codex/Hermes to start training once
the pre-training requirements and standards are met.

This is conditional authorization, not an immediate training launch.

## Scope

Allowed once the launch gate is satisfied:

```text
Qwen v2 WP5 teacher-forced target-mass gate training only
```

Initial allowed training class:

```text
qwen_protected_micro_slot_lora
qwen_task_only_lora
```

Still forbidden until later gates pass:

```text
Qwen E2E proof-of-life
Llama training or eval
same-family nulls
sanitizer benchmark
FAR aggregation
paper-facing positive claims
```

## Pre-Training Launch Gate

Training may start without another manual approval only if all conditions below
are true:

| Requirement | Status |
|---|---|
| v1 frozen as negative diagnostic | already satisfied |
| WP3-R1 strict density reviewed | already satisfied with caveat |
| WP3-R2 high-mass primary bank selected | already satisfied from job `851272` |
| WP3-R3 naturalness reviewed | already satisfied with language-drift note |
| WP4 prompt-local decoder oracle passed | already satisfied |
| WP5 training/scoring plan exists | still required |
| protected and task-only objectives are explicit | still required |
| slot CE mask and margin loss are specified | still required |
| model, payloads, seeds, prompt split, budgets are explicit | still required |
| Slurm wrapper is reviewed | still required |
| command is in `configs/natural_evidence_v2/run_allowlist.yaml` | still required |
| Hermes TG/email notification succeeds before launch | still required |
| only one allowlisted Slurm training job is submitted in the tick | required at launch |

## Required Training Configuration Fields

The launch review must explicitly record:

```text
model = Qwen/Qwen2.5-7B-Instruct
payloads = P00, P01
seeds = 17, 23
query_budgets = [8, 16, 32, 64]
primary_bank = Set|Plan vs Create|Prepare
contract_dir = results/natural_evidence_v2/contracts/wp4_prompt_local_contract_20260509_0610
prompt_split = wp3_r1_eval plus fixed train/dev split for training data
objective = task CE on non-slot tokens + margin loss at micro-slots
slot exact-token CE = masked
controls = protected LoRA and task-only LoRA
```

## Post-Training Teacher-Forced Gate

Training completion does not authorize Qwen E2E. The trained artifacts must be
scored under the teacher-forced gate:

| Metric | Gate |
|---|---:|
| protected target bucket mass - base | >= +0.15 |
| protected target bucket mass - task-only | >= +0.10 |
| target bucket rank-1 rate | >= 70% |
| median target margin | > 0 |
| task-only target bucket mass - base | not materially positive |

If this gate fails, do not run Qwen E2E. Repair the objective, slot policy,
bank, suffix policy, or training configuration.

## Execution Rule

Codex/Hermes should not ask for another manual approval on this same locked
route once the pre-training launch gate is fully satisfied. It must still:

- notify through Telegram and email before launch;
- use Chimera Slurm only for CPU/GPU work;
- submit at most one allowlisted training job per tick;
- disable the allowlist entry after the single submission when appropriate;
- update state and report artifacts immediately after submission/completion.

This authorization does not permit paper claims, FAR claims, payload recovery
claims, robustness claims, or cross-family claims.

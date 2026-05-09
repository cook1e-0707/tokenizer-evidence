# Hermes 15-minute supervision protocol

This note defines how Hermes should supervise Codex on `natural_evidence_v1`
every 15 minutes. Hermes is not the executor of the research workflow. Hermes
monitors state, checks whether Codex is moving, prompts Codex with the next
allowed action, and blocks unsafe or out-of-order steps. Codex remains the
executor that edits files, submits Slurm jobs, reviews artifacts, and updates
state.

This is an oversight and handoff contract, not a permission to train.

## Mandatory TG + Email Notification Gate

Every Hermes supervision tick that pushes the project forward must notify the
user through both Telegram and email before Codex executes the requested next
action. This includes ticks that prompt Codex to edit files, submit or monitor a
Slurm job, sync artifacts, interpret results, or update gates.

Use:

```bash
python3 scripts/natural_evidence_v1/hermes_notify.py \
  --subject "Hermes natural_evidence_v1 progress tick" \
  --body-file results/natural_evidence_v1/status/hermes_reports/YYYYMMDD_HHMM.md \
  --channels telegram,email \
  --strict \
  --output-json results/natural_evidence_v1/status/hermes_reports/YYYYMMDD_HHMM_notification.json
```

By default, the helper loads `/Users/guanjie/.hermes/.env` before checking
notification variables. Use `--env-file PATH` to override this or
`--no-env-file` for a deliberate no-dotenv check.

Required Telegram configuration:

```text
HERMES_TG_BOT_TOKEN or TELEGRAM_BOT_TOKEN or TG_BOT_TOKEN
HERMES_TG_CHAT_ID or TELEGRAM_CHAT_ID or TG_CHAT_ID or TELEGRAM_HOME_CHANNEL
```

Required email configuration:

```text
HERMES_EMAIL_TO or HERMES_NOTIFY_EMAIL_TO or EMAIL_TO or EMAIL_HOME_ADDRESS or EMAIL_ADDRESS
```

Email delivery must use either SMTP:

```text
HERMES_SMTP_HOST or SMTP_HOST or EMAIL_SMTP_HOST
optional: HERMES_SMTP_PORT/EMAIL_SMTP_PORT, HERMES_SMTP_USER/EMAIL_ADDRESS,
HERMES_SMTP_PASSWORD/EMAIL_PASSWORD, HERMES_SMTP_TLS, HERMES_SMTP_SSL
```

or a local sendmail binary:

```text
HERMES_SENDMAIL or SENDMAIL or /usr/sbin/sendmail or /usr/bin/sendmail
```

If either Telegram or email notification is missing, fails, or is only a
dry-run, Hermes must not mark the user as notified. Hermes should stop forward
prompting, record a notification blocker, and ask Codex/user to configure the
missing channel. Do not silently continue a project-advancing tick after failed
notification.

Each notification body must include:

- current phase;
- active and completed jobs;
- the exact Codex action being requested;
- whether this is monitor-only or state-changing;
- next allowed action;
- forbidden actions still in force.

## Current Locked Phase

`V1_FROZEN_NEGATIVE_DIAGNOSTIC__V2_WP2_PROMPT_FAMILY_SPLITS_AUDITED`

The current next allowed action is:

```text
natural_evidence_v2 WP3 artifact-only micro-slot detector and 2-way bucket
policy design.
```

Do not continue v1 repaired target-mass probes. Training, model transcript
generation, Qwen E2E rerun, Llama, same-family null, sanitizer, FAR
aggregation, and paper-facing positive claims remain forbidden.

## 15-minute Supervision Loop

The Hermes cron job that drives this loop is:

```text
job_id: d65af4b36d84
name: natural-evidence-v1-codex
schedule: every 15m
script: /Users/guanjie/.hermes/scripts/natural_evidence_v1_codex_tick.sh
workdir: /Users/guanjie/Documents/tokenizer_alignment
mode: no-agent launcher; the script sends TG/email and starts a background Codex worker
```

The script delegates to:

```text
scripts/natural_evidence_v1/hermes_supervision_tick.py
```

This launcher exists because Hermes no-agent scripts have a short runtime
budget. The scheduled script sends the required Telegram/email notification,
starts a background Codex worker, and exits quickly. The worker sends a
completion notification after Codex finishes. If a previous worker is still
running, the next tick sends a blocked/monitor-only notification and does not
launch a second Codex worker.

Reliability guardrails:

- the launcher sets an explicit PATH for Homebrew Python, Hermes, and the VS
  Code Codex binary;
- the worker falls back to the known VS Code Codex binary path if `codex` is not
  on PATH;
- a lock directory prevents overlapping Codex workers;
- the lock records the worker PID and is cleaned automatically if the PID is no
  longer alive or the lock exceeds the configured stale threshold;
- the default Codex worker timeout is 7200 seconds, and the default stale-lock
  threshold is timeout plus 1800 seconds.

Every Hermes tick should supervise Codex with these checks in order:

1. Read persistent state:
   - `docs/natural_evidence_v1/AUTOMATION_STATE.md`
   - `docs/natural_evidence_v1/next_step_codex_plan.md`
   - `results/natural_evidence_v1/status/gate_status.json`
2. Check local repo status:
   - `git status --short`
   - `git branch --show-current`
   - `git log -1 --oneline`
3. Check Chimera only through SSH:
   - `ssh chimera`
   - `squeue -u $USER`
   - `sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode -S now-2days`
4. If a Slurm job is active, monitor only:
   - tell Codex not to submit another job for the same phase;
   - tell Codex to sync/review outputs only after completion;
   - require Codex to update state and a report after review.
5. Before any forward prompt, send both Telegram and email notifications with
   `scripts/natural_evidence_v1/hermes_notify.py --strict`. If either channel
   fails, record a blocker and do not continue the project-advancing tick.
6. If no blocking job is active and the next action is unambiguous, prompt Codex
   to execute at most one state-changing action in that 15-minute window.
7. After Codex performs any state-changing action, require a concise report:
   - `results/natural_evidence_v1/status/hermes_reports/YYYYMMDD_HHMM.md`
8. Require Codex to update:
   - `docs/natural_evidence_v1/AUTOMATION_STATE.md`
   - `results/natural_evidence_v1/status/gate_status.json`

## Chimera Rules

Any CPU or GPU work on Chimera must be submitted by Codex as a Slurm job. Do
not allow Codex to run CPU work directly on the Chimera login node.

All `natural_evidence_v1` sbatch scripts must include:

```bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guanjie.lin001@umb.edu
```

For GPU work, prefer:

```text
partition=DGXA100
account=pi_yinxin.wan
qos=scavenger_unlim
gres=gpu:A100:1
```

unless a later verified cluster check changes the constraint.

## One-action Limit For Codex

Hermes keeps the project moving by supervising Codex, but each 15-minute tick
should request at most one Codex state-changing action. Examples of Codex
actions Hermes may request:

- write one artifact-only analysis output;
- submit one allowlisted Slurm job;
- sync and review one completed Slurm job's outputs;
- update state and reports after a completed artifact.

Do not combine a model-scoring Slurm submission with a new training launch in
the same tick. Training is currently forbidden anyway.

## Current Codex Action Queue Supervised By Hermes

### H1: v2 controlled-natural prompt family scaffold

Status: complete as of 2026-05-08T21:30Z.

Artifact-only local work. Create the first v2 prompt-family scaffold and
forbidden-surface audit plan. The target prompt families are:

```text
F1_8_sentence_explanation
F2_8_step_checklist
F3_6_point_comparison
F4_concise_advice_with_transitions
```

The work must not call a model, train, generate transcripts, run E2E, or claim
success. The completed artifact prepared templates, split metadata, and a
forbidden-surface lint path for:

- `qwen_v2_train_prompts.jsonl`;
- `qwen_v2_dev_prompts.jsonl`;
- `qwen_v2_eval_prompts.jsonl`;
- `qwen_v2_organic_null_prompts.jsonl`.

### H2: v2 micro-slot detector and 2-way bucket policy

Next allowed artifact-only action. Design the micro-slot detector and 2-way
bucket policy. Primary slot types are sentence opener, bullet/step opener,
discourse marker, optional hedge, transition word, and function-word
alternative. Content nouns, domain-specific nouns, rare tokens,
punctuation-only tokens, markdown-heavy tokens, and invisible whitespace tokens
are not primary slots.

### H3: v2 prompt-local small payload contract

Only after H2 gates, compile an 8-bit or 16-bit payload plus checksum into a
prompt-local frame contract and run decoder oracle substitution. Training and
free-generation E2E remain forbidden until the teacher-forced target-mass gate
is later passed.

The completed v1 repaired target-mass path is stale and must not be repeated.

## Hard Stop Conditions

Hermes must stop prompting Codex forward and ask Codex/user for review if any
of these occur:

- any gate tries to unlock training without repaired target-mass evidence;
- any wrapper would run on Chimera outside Slurm;
- any output path would overwrite existing artifacts without an explicit fresh
  directory;
- Hermes cannot send both Telegram and email notification for a
  project-advancing tick;
- any result attempts to claim payload recovery, full FAR, cross-family
  generality, sanitizer robustness, stealth, or superiority over baselines;
- raw/task-only/wrong-key/wrong-payload controls are missing from a proposed
  E2E action.

## Handoff Format

Each 15-minute Hermes supervision report should include:

```text
phase:
active_jobs:
completed_jobs:
artifact_changes:
gate_changes:
red_flags:
telegram_notification:
email_notification:
notification_json:
state_changing_action:
next_allowed_action:
forbidden_actions_confirmed:
```

Codex should treat the latest Hermes report plus `gate_status.json` as the
authoritative supervision handoff for the next action. Hermes should not be
treated as an autonomous executor.

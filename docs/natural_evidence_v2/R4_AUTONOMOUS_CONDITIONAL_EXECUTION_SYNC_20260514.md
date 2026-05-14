# R4 Autonomous Conditional Execution Sync

Timestamp UTC: 2026-05-14T07:00:00Z

## Controlling Rule

Codex and Hermes should not treat gate-controlled actions as permanently
forbidden. They are conditionally executable once their recorded prerequisite
route gates pass.

The user has granted standing authorization for Codex and Hermes to continue
the current clear route without asking for repeated per-step approval. This
applies to future route stages such as training, generation, H200 scoring,
Llama, same-family nulls, sanitizer, FAR, payload diversity, and paper-facing
claims only after the specific prerequisite gates for each stage are recorded
as passed.

## What This Changes

Hermes tick reports should avoid wording that implies these actions can never
be run. Preferred wording:

```text
Gate-controlled and conditionally authorized after prerequisite gates pass:
training; generation; H200 scoring; Llama; same-family null; sanitizer; FAR;
payload diversity; paper-facing claims.
```

Current status can still say:

```text
Not currently unlocked by this phase.
```

It should not say or imply:

```text
Permanently forbidden.
```

## What This Does Not Change

Every state-changing compute route still requires:

- a recorded route decision;
- local and remote preflight evidence;
- zero-enabled allowlist before route enablement;
- exactly one reviewed allowlist entry enabled;
- Hermes TG/email notification before submission;
- immediate allowlist disablement after `sbatch`;
- H200/pomplun/cs_yinxin.wan for GPU work unless later route decision
  supersedes it;
- no paper-facing claim before claim gates pass.

## Current Next Action

The current phase remains artifact-only transfer-gap repair after job `857795`.
Codex/Hermes should automatically proceed with artifact-only implementation
planning and validation for prefix-context elicitation, surface-polarity
alignment, forbidden matcher semantics, and structural leakage controls.

# R4 Expert Guidance Analysis And Next Action

Recorded: 2026-05-13T05:31:21Z

## Scope

This document ingests the 2026-05-13 expert guidance as controlling project
guidance for the current `natural_evidence_v2` R4 prefix-native repair route.
It is an artifact-only governance update. It does not run Qwen tokenizer
validation, load a model, enable an allowlist, submit Slurm, run
tokenizer/model scoring, generate outputs, train, run Llama, run same-family
nulls, run a sanitizer benchmark, aggregate FAR, or make paper-facing claims.

## Point-By-Point Analysis

### 1. Do not directly enter the next H200 surface-mass scoring job

Accepted. The current candidate has not passed actual Qwen tokenizer boundary
preflight. Job `853894` failed before producing R4 surface-mass metrics, so a
new H200 surface-mass scoring submission would be premature.

Current scoring authorization remains `false`.

### 2. Static preflight has passed, but it is not tokenizer compatibility

Accepted and corrected in the top-level state. The static boundary-contract
preflight artifact records:

- status: `PASS_STATIC_BOUNDARY_CONTRACT_TOKENIZER_PENDING`;
- checked rows: `8192`;
- failed rows: `0`;
- Qwen tokenizer preflight started: `false`;
- model forward pass started: `false`;
- scoring authorized: `false`.

This clears only the static boundary-contract blocker. It does not demonstrate
that Qwen tokenization gives non-empty target/other first-token ids or zero
target/other overlap.

### 3. Current blocker name

Accepted with one additional recorded sub-blocker. The top-level blocker is now:

`BLOCK_R4_PREFIX_NATIVE_ACTUAL_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT_PENDING`

Additional artifact-only blocker already recorded in the 05:28 plan:

`HASH_RECONCILIATION_REQUIRED`

The reviewed static preflight summary records scorer hash
`8b4bddf6d47eff8496a5286915c1e77a33eb90cf74b1e70187f3b80aa09480a1`, while the
current local scorer is
`55057af48749290ef943ec40b211e1b1d7e487486a27e642e3aeba7780ec9b7e`.

Actual tokenizer preflight route planning should not hide this mismatch. The
next state must either regenerate/review static preflight against the current
scorer tuple or explicitly record why the hash difference is non-material.

### 4. 853894 failure interpretation

Accepted. The failure is classified as scorer/candidate tokenizer-boundary
contract failure, not method-gate failure and not Slurm/provider failure.

Recorded exception:

`ValueError: surface produced no next token: 'create'`

Cause: `assistant_prefix_before_surface` contains terminal whitespace, while
surfaces are stored as naked labels such as `create`. Qwen-style tokenization
can merge the terminal whitespace with the following word, invalidating the old
assumption that `prefix + surface` cleanly appends a next token.

### 5. State synchronization should be first

Accepted. `CURRENT_STATE.md` top-level phase and next allowed action have been
updated so Hermes/Codex do not read older historical next-action sections as
controlling state.

New canonical phase:

`V2_R4_PREFIX_NATIVE_STATIC_BOUNDARY_PREFLIGHT_PASSED_QWEN_TOKENIZER_PENDING_NO_SCORING`

### 6. Active A100/DGXA100 array 854279 must be isolated

Accepted. A control-plane anomaly record has been created for `854279_[0-11]`
because it is running/pending on `DGXA100` with A100 resources while the current
execution policy is H200-first via `pomplun` / `cs_yinxin.wan`.

The job is marked non-canonical:

- adopt outputs: `false`;
- aggregate outputs: `false`;
- claim relevance: `none`;
- cancellation requires a separate recorded governance decision.

### 7. Actual Qwen tokenizer boundary preflight requirements

Accepted. Future tokenizer-only preflight must check all `8192` candidate rows
and must fail closed unless:

- failed rows: `0`;
- prefix-unstable rows: `0`;
- empty delta rows: `0`;
- empty target first-token id rows/coordinates: `0`;
- empty other first-token id rows/coordinates: `0`;
- target/other first-token overlap rows/coordinates: `0`;
- missing prompt index rows: `0`;
- missing boundary context rows: `0`.

Expected pass status:

`PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT`

Expected fail status:

`FAIL_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT_NO_SCORING`

### 8. Slurm tokenizer-only route

Accepted as future work, not yet executed by this update. Because local
`uv run python` lacks `transformers`, actual Qwen tokenizer preflight must run
through a reviewed Slurm route, not on the Chimera login node.

Per current H200 execution policy, the wrapper should use:

- partition: `pomplun`;
- account: `cs_yinxin.wan`;
- QoS: `pomplun`;
- GPU request if required by governance: `gpu:h200:1`;
- time limit: `30-00:00:00`.

The wrapper must assert no model forward, no scoring, no generation, and no
training.

### 9. Scoring route only after tokenizer preflight pass

Accepted. A future H200 surface-mass scoring route requires:

- reviewed actual Qwen tokenizer boundary preflight with
  `PASS_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT`;
- coherent candidate/scorer/preflight hashes;
- a new route decision;
- local/remote allowlist/hash preflight;
- Hermes TG/email notification;
- exactly one enabled allowlist entry;
- immediate allowlist disable after submission.

### 10. Later generation diagnostic remains gate-controlled

Accepted with terminology clarification. Generation diagnostic is not
permanently forbidden; it remains gate-controlled until the actual tokenizer
preflight passes and teacher-forced prefix-native surface-mass scoring passes.

No Llama, same-family null, sanitizer, FAR, payload diversity, or paper-facing
claim is unlocked by this guidance ingestion, but those work classes may execute
after their own prerequisite gates and route decisions pass.

## Controlling Next Allowed Action

Artifact-only state reconciliation, active A100/DGXA100 anomaly quarantine, and
actual-Qwen-tokenizer boundary preflight route planning/review only.

No H200 surface-mass scoring, free generation, training, Llama, same-family
null, sanitizer, FAR, payload diversity, or paper-facing positive claim is
allowed until later reviewed gates pass.

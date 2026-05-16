# R4 After-868016 Controller Generation 868151 Failure Analysis

Status: `FAILURE_ANALYSIS_RECORDED_R4_AFTER_868016_CONTROLLER_GENERATION_868151`

## Objective Facts

- Slurm job `868151` completed on H200/pomplun: 4/4 array shards `COMPLETED`, exit code `0:0`.
- Review status: `FAIL_R4_AFTER_868016_CONTROLLER_GENERATION_DIAGNOSTIC_GATE`.
- Generated rows: `9216` across conditions `protected, raw, task_only`.
- Primary protected accepts under `format_scrub=all`: `0` / `4`.
- Primary null accepts under `format_scrub=all`: `{'all::raw': 0, 'all::task_only': 0, 'all::wrong_key': 0, 'all::wrong_payload': 0}`.
- Forbidden public surface count under `format_scrub=all`: `26`; terms `{'coordinate': 26}`.
- Duplicate response hash count: `4422`; unique hashes `4794`.

## Decoder Evidence

- Matched surface count under `format_scrub=all`: `0`.
- Selected surface count under `format_scrub=all`: `0`.
- Selected coordinates observed under `format_scrub=all`: `0`.
- Every primary protected block failed before checksum because all selected coordinate pairs were missing.

## Surface Transfer Rates

- `protected`: target surface at start `0.0000`, any bank surface at start `0.0000`, target first word at start `0.0680`, any bank first word at start `0.2646`, unique hashes `2317` / rows `3072`.
- `raw`: target surface at start `0.0000`, any bank surface at start `0.0000`, target first word at start `0.0029`, any bank first word at start `0.0186`, unique hashes `2048` / rows `3072`.
- `task_only`: target surface at start `0.0000`, any bank surface at start `0.0000`, target first word at start `0.0036`, any bank first word at start `0.0186`, unique hashes `2048` / rows `3072`.

## Interpretation

The 868114 teacher-forced controller signal did not transfer into recoverable phrase-level generated evidence. The first-step controller can bias next-token mass in teacher-forced scoring, but the generated continuations drift away from the precommitted full surfaces; the decoder therefore sees zero locked surface events and zero selected coordinates. Null arms remain clean, so this is not a false-accept/FAR failure. It is a positive-channel transfer failure plus public-surface/duplication quality failure.

This artifact does not unlock another generation or scoring job. The next action is a recorded artifact-only repair or pivot route decision.

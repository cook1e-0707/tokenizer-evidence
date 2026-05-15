# R4 Positive Selectivity 859491 Failure Analysis

Status: `FAILURE_ANALYSIS_RECORDED_NO_RESUBMIT`

## Root Cause

The selectivity prompt-policy elicited support-window events, but not a protected-only keyed channel. Protected support remains too sparse for the accept threshold, and raw/task-only also produce comparable ordinary support-window language. Null accepts remain 0 because thresholds are strict, not because protected recovered.

## Primary Evidence

- `protected`: accepts `0/32`, mean events `9.875`, mean coords `5.000`, max keyed score `16.0`, max margin `20.0`.
- `raw`: accepts `0/32`, mean events `9.375`, mean coords `4.625`, max keyed score `23.0`, max margin `20.0`.
- `task_only`: accepts `0/32`, mean events `8.562`, mean coords `4.156`, max keyed score `18.0`, max margin `18.0`.

## Key Findings

- protected accepts are 0/32 under format_scrub=all and 0/32 without scrub.
- raw/task-only/wrong-key/wrong-payload accepts are 0/32, so controls are clean.
- protected mean events per block is 9.875 and mean distinct coordinates is 5.0, below the support needed for robust accept.
- raw mean events per block is 9.375 and task-only is 8.5625, so support windows are still ordinary task-language rather than protected-selective.
- raw max keyed score (23) exceeds protected max keyed score (16), indicating no reliable protected alignment.
- technical literal hits are dominated by ordinary domain word coordinate in volunteer coordination; this remains a matcher-policy issue but does not rescue the positive failure.

## Surface-Family Pattern

- `protected` total events `316`: handoff_trace=101, quality_review=87, risk_review=73, context_alignment=49, communication_choice=6
- `raw` total events `300`: handoff_trace=121, quality_review=85, context_alignment=65, risk_review=28, communication_choice=1
- `task_only` total events `274`: handoff_trace=122, quality_review=79, context_alignment=59, risk_review=14

## Control Decision

Do not resubmit this route unchanged. This is not a positive result and does not unlock paper-facing claims, Llama, sanitizer, FAR, payload diversity, or further generation without a new reviewed repair/pivot route.

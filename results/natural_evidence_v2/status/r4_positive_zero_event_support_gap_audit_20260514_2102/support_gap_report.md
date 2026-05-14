# R4 Positive Zero-Event Support Gap Audit

## Verdict

`PASS_AUDIT_RECORDED_ZERO_EVENT_SUPPORT_CONFIRMED`

The audit confirms the 859277 failure mode: exact frozen phrase-event support is absent.
The generated text contains task-natural action language, but not the locked multi-word
phrases required by the precommitted extractor.

## Coverage By Condition

| condition | rows | segments | exact hits | rows with exact | loose stem hits | rows with bank first-word opener |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| protected | 2048 | 52740 | 0 | 0 | 1 | 2032 |
| raw | 2048 | 23936 | 0 | 0 | 0 | 2042 |
| task_only | 2048 | 23443 | 0 | 0 | 0 | 2046 |

## Top Openers

- `protected`: this=7476, use=2850, keep=2403, make=1958, create=1557, encourage=1361, prepare=1265, have=1123, be=1100, when=939, plan=880, avoid=839
- `raw`: ensure=1879, use=1876, keep=1495, encourage=1415, regularly=1379, set=828, create=780, plan=759, avoid=665, start=628, make=628, be=489
- `task_only`: use=1950, ensure=1841, encourage=1579, keep=1443, regularly=1308, set=1242, create=863, plan=783, make=683, start=631, avoid=587, be=495

## Interpretation

- Exact phrase hits are zero, so the keyed decoder has no support.
- Bank first-word overlap is nonzero, so the gap is phrase-specific, not a total absence of action language.
- Forbidden matcher hits remain diagnostic only; matcher repair cannot rescue 859277 because support is zero.
- 859277 outputs must not be mined into the next locked bank.

## Artifacts

- `support_gap_summary.json`
- `condition_coverage.csv`
- `opener_counts_by_condition.csv`
- `bank_surface_coverage.csv`
- `forbidden_matcher_semantics.csv`

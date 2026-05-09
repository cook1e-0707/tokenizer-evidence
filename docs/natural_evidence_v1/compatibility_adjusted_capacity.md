# Compatibility-Adjusted Natural Evidence Capacity

Raw opportunity-bank entry count is no longer a training or paper-success gate.
It is only a scaling-axis diagnostic for static next-token opportunities.

The natural-output method should be gated by compatibility-adjusted capacity:

```text
C_eff = sum over entries e of
        compatible(e) * bucket_entropy(e) * reconstructability(e) * survival(e)
```

In the current implementation, the measurable proxies are:

- `accepted_entries_at_min1`: entries with at least one compatible token in every
  bucket; this is sufficient for a first payload-recovery viability pilot.
- `accepted_entries_at_configured_min`: entries with the configured multi-member
  compatible count in every bucket; this supports bucket-mass versus
  fixed-representative ablations.
- `accepted_entries_at_probability_gates`: high-confidence compatible entries
  that also preserve bucket mass/entropy after compatibility filtering; this is
  a final-main quality diagnostic, not the first E2E gate.
- `effective_compatible_bits_per_response`: the capacity quantity to report once
  held-out density and transcript reconstructability are measured.

## Current Qwen Pilot Gate

The Qwen 4-way E2E viability pilot can proceed to density/null checks when the
bank-side compatibility gate passes:

- `accepted_entries_at_min1 >= 1500`,
- `accepted_entries_at_configured_min >= 200`,
- held-out eligible density is at least `0.5` positions per 100 generated tokens,
- raw/wrong-key pre-null behavior is not high risk.

The observed Qwen dry-run counts are:

- min1-compatible entries: `2327`,
- configured-min / min2-compatible entries: `243`,
- probability-gated entries: `177`.

This passes the bank-side viability gate but not the final-main quality target.
Next work should measure held-out/organic density and raw/wrong-key pre-null
before starting the controlled Qwen E2E pilot.

## Expert Gate Split

The 2026-05-05 expert decision keeps the paper-ready density/capacity gate, but
allows a separate diagnostic high-risk Qwen proof-of-life pilot. These gates must
not be conflated.

Paper-ready Qwen minimum:

- effective compatible bits per response >= `1.0`,
- held-out density >= `0.5` positions per 100 generated tokens,
- raw/wrong-key pre-null behavior is not high risk.

Diagnostic high-risk Qwen pilot:

- min1-compatible entries >= `1500`,
- configured-min / min2-compatible entries >= `200`,
- held-out density >= `0.3` positions per 100 generated tokens,
- effective compatible bits per response >= `0.3`,
- raw/wrong-key pre-null has no obvious accidental accept risk,
- invalid suffix records are explained or excluded.

The current frozen held-out result fails the paper-ready gate but satisfies the
diagnostic density floor:

- held-out density: `0.4325497287522604`,
- effective compatible bits per response: `0.46896596898420007`.

Therefore Qwen diagnostic E2E may be prepared only as
`diagnostic_high_risk`; it remains forbidden as a paper-facing success claim.

## Main-Paper Target

For paper-facing scale, target `5000-10000` compatibility-aware entries per
tokenizer, with at least `1000` multi-member compatible entries. For an Oral-level
scale story, report a curve over compatibility-adjusted capacity rather than
claiming parity with Scalable Fingerprinting's 24,576 implanted fingerprints.

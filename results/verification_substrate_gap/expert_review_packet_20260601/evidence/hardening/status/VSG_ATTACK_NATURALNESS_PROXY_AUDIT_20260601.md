# VSG Attack Naturalness Proxy Audit - 2026-06-01

Status: `PASS_PUBLIC_PREDICATE_ATTACK_NATURALNESS_PROXY_AUDIT_RECORDED_NO_CLAIMS`

This artifact records an artifact-only readability audit for the existing
public-predicate guided rewrite/graft attack examples. It starts no Slurm job,
generation, model scoring, training, or allowlist enablement.

## Input

```text
results/verification_substrate_gap/public_verifier_surrogate_guided_rewrite_20260530/surrogate_guided_rewrite_examples.csv
```

The input contains 60 adopted-locked source-mismatch guided rewrite/graft
examples:

- Qwen raw: 20 rows
- Qwen task-only: 20 rows
- Llama raw: 20 rows

## Method

The audit uses deterministic surface proxy checks only:

- original/rewrite token counts;
- rewrite/original length ratio;
- original/rewrite token Jaccard overlap;
- uppercase start;
- sentence-final punctuation;
- isolated single-letter fragments, excluding normal `a` and `i`;
- known broken-graft markers such as incomplete domain-word fragments.

This is not a semantic naturalness evaluation, not a human evaluation, and not
a model-based naturalness score.

## Output

```text
results/verification_substrate_gap/public_predicate_attack_naturalness_audit_20260601/
  attack_naturalness_proxy_rows.csv
  attack_naturalness_proxy_by_group.csv
  attack_naturalness_proxy_summary.json
  attack_naturalness_proxy_report.md
  attack_naturalness_proxy_manifest.json
```

## Results

| Group | Rows | Proxy pass | Proxy fail | Pass rate | Mean token overlap | Broken markers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama raw | 20 | 0 | 20 | 0.0 | 0.403244 | 1.0 |
| Qwen raw | 20 | 0 | 20 | 0.0 | 0.424762 | 1.35 |
| Qwen task-only | 20 | 0 | 20 | 0.0 | 0.456823 | 1.05 |

Aggregate:

```text
rows = 60
proxy_pass_rows = 0
proxy_fail_rows = 60
proxy_pass_rate = 0.0
```

Failure reason counts:

```text
does_not_end_with_sentence_punctuation = 60
isolated_single_letter_fragment = 39
known_broken_graft_marker = 40
```

## Interpretation

The audit confirms that the current guided rewrite/graft attack demonstrates
public-predicate optimizability but does not establish naturalness-preserving
rewriting. Accepted rows remain source-mismatch spoofing evidence only; they
are not protected success, not codeword recovery, and not public text-only
verification.

## Manuscript Update

The local manuscript was updated to record this limitation in:

```text
section_07_experiments.tex
section_08_discussion_limitations.tex
appendix/attack_examples.tex
```

Local manuscript commit:

```text
3520fd15158f219c1b5e897ec8a7a947eb740dce
```

Local manuscript PDF SHA256 after build:

```text
79766958a53c80717d5ebeac1925edfdc760e454113147e953657a0dc6f24516
```

The expert packet zip was not refreshed in this pass.

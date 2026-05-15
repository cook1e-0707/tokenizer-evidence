# R4 Pressure-Controller Wrong-Control Diagnosis for Job 859672

Status: `FAIL_SELECTIVITY_CONTROL_SEMANTICS_NO_GENERATION`

This artifact-only diagnosis reviewed remote row metadata from completed H200 job `859672`, focusing on grids `grid_00`, `grid_65`, and `grid_71`. It did not run model scoring, generation, training, Llama, FAR, sanitizer, or paper-facing claims.

## Core Finding

The failure is not that the controller cannot exert pressure. The failure is that the wrong-control arms are not valid selectivity nulls in the current semantics because they still load the protected adapter while the scorer measures the committed target ids.

In `grid_65`, the best protected-lift grid:

| Condition | Mean committed target mass | Rank1 rate | Mean controller scale |
| --- | ---: | ---: | ---: |
| base | `0.0048318370` | `0.2519531250` | `0.0` |
| task_only | `0.0016724278` | `0.1992187500` | `0.0` |
| controlled_protected | `0.4023347084` | `1.0` | `0.4688563768` |
| wrong_key_controlled | `0.3365671794` | `0.9846191406` | `0.7374624010` |
| wrong_payload_controlled | `0.2712397256` | `0.9707031250` | `1.0` |

Wrong-key split in `grid_65`:

| wrong-key mapping | Rows | Mean committed target mass | Rank1 rate |
| --- | ---: | ---: | ---: |
| matches committed target bit | `4064` | `0.4017454344` | `1.0` |
| does not match committed target bit | `4128` | `0.2723994399` | `0.9694767442` |

Wrong-payload in `grid_65`:

- `controller_target_equals_committed_target = 0/8192`
- `controller_overlap_bad = 0`
- mean committed target mass remains `0.2712397256`
- rank1 remains `0.9707031250`

## Interpretation

The wrong-control target mappings are present and disjoint, but the wrong-control arms are not discriminative because protected adapter pressure remains active. Complement or wrong-key controller pressure does not suppress the committed target mass enough to reject. Therefore `859672` cannot be interpreted as keyed-selective evidence.

## Next Allowed Action

Artifact-only repair planning only:

```text
define and review a wrong-control selectivity semantics repair before any new scoring or generation
```

No new Slurm scoring, generation, training, Llama, sanitizer, FAR, payload diversity, or paper-facing claim is unlocked by this diagnosis.


# R4 metric-exact 864761 dev generation job 864832 review

Timestamp UTC: 2026-05-16T04:00:00Z

## Slurm Status

- job id: `864832`;
- array tasks: `864832_0`..`864832_3`;
- final state: all tasks `COMPLETED`;
- exit code: all tasks `0:0`;
- node: `chimera21`;
- source adapter: job `864761`, gain `1.0`;
- generated outputs: `6144` (`2048` protected / `2048` raw / `2048` task-only).

## Decode Result

| format scrub | arm | accepts | blocks | mean support | median support | max support | mean matched surfaces | forbidden public surface count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | `protected` | `0` | `32` | `0.75` | `0.00` | `8` | `2.25` | `0` |
| `all` | `raw` | `0` | `32` | `11.75` | `8.00` | `24` | `136.50` | `72` |
| `all` | `task_only` | `0` | `32` | `11.25` | `8.00` | `24` | `147.50` | `84` |
| `all` | `wrong_key` | `0` | `32` | `0.75` | `0.00` | `8` | `2.25` | `0` |
| `all` | `wrong_payload` | `0` | `32` | `0.75` | `0.00` | `8` | `2.25` | `0` |
| `none` | `protected` | `0` | `32` | `0.75` | `0.00` | `8` | `2.25` | `0` |
| `none` | `raw` | `0` | `32` | `11.75` | `8.00` | `24` | `136.50` | `72` |
| `none` | `task_only` | `0` | `32` | `11.25` | `8.00` | `24` | `147.50` | `84` |
| `none` | `wrong_key` | `0` | `32` | `0.75` | `0.00` | `8` | `2.25` | `0` |
| `none` | `wrong_payload` | `0` | `32` | `0.75` | `0.00` | `8` | `2.25` | `0` |

Primary result: protected accepts are `0/32` under `format_scrub=all` and `0/32` under no-scrub decode. All raw/task-only/wrong-key/wrong-payload controls also have `0` accepts.

## Failure Evidence

- main positive gate failed: protected `format_scrub=all` accepts observed `0/32`; route target was `>=26/32` for a small dev diagnostic;
- no false accepts were observed in raw/task-only/wrong-key/wrong-payload controls;
- support is far below the decoder threshold: protected mean support under `format_scrub=all` is `0.75`, median support is `0`, and mean matched surfaces are `2.25` per block;
- decode-row forbidden public surface count is `0`;
- generated outputs show strong visible repetition from the trained surface pressure rather than recoverable cover-natural evidence; e.g. many protected outputs begin with repeated `Create a plan...` text;
- duplicate generation ids: `0`;
- duplicate response text hashes: `358`;
- duplicate prompt-condition-replicate keys: `0`;
- response technical literal counts: `{"bucket": 1, "coordinate": 155}`;

## Interpretation

Job `864761` passed the teacher-forced surface-mass gate, but that pressure did not transfer to the current R4 cover-natural free-generation decoder. The generated protected text appears heavily biased toward repetitive `Create/Prepare/Plan` continuations, while the precommitted cover-natural surface bank records only sparse matches and almost no observed coordinates. This is a transfer failure from teacher-forced prefix-native mass to free-generation cover-natural evidence, not a Slurm failure or null-control failure.

The 864761 caveat remains active: the adapter was trained by repeated-cycling a 512-row train artifact while scoring 8192 rows. This run cannot be used as paper-facing evidence and does not unlock locked-scale, Llama, same-family null, sanitizer, FAR, payload diversity, or paper claims.

## Decision

Status: `FAIL_R4_METRIC_EXACT_864761_DEV_GENERATION_NO_PROTECTED_ACCEPTS_NO_DOWNSTREAM_UNLOCK`

## Next Allowed Action

Artifact-only failure analysis / repair or pivot route decision only. Do not submit another Slurm job or start downstream routes until a new reviewed route records prerequisites and control-plane checks.

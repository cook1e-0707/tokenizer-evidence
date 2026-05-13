# R4 surface teacher-forced probe preflight blocker

The preflight did not build scoring rows because the current R4
surface bank is not binary per coordinate.

- coordinates checked: `32`
- coordinates missing one binary side: `32`

A teacher-forced mass probe needs a target side and a non-target
side under the same coordinate. The current bank only provides
one polarity per coordinate, so any target-vs-other mass result
would be semantically invalid.

Next allowed action: artifact-only binary surface-bank repair
planning. No Slurm scoring or generation is authorized by this
failed preflight.

# R4 surface teacher-forced probe preflight

This artifact-only preflight converts the R4 cover-natural phrase bank into
teacher-forced target-mass rows. It does not train, score a model, submit
Slurm, run Llama, aggregate FAR, or make paper claims.

- selected prompts: `256`
- score rows: `8192`
- coordinates: `32`
- rows per coordinate: `256` to `256`
- contract: `a55e`

The next step, if this plan is accepted, is a Slurm-only scorer wrapper
for base/protected/task-only target surface mass on these frozen rows.

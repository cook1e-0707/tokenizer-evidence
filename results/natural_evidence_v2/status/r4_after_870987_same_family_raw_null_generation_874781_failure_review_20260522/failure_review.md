# R4 Same-Family Raw-Null Generation 874781 Failure Review

Status: `FAIL_WRAPPER_ALLOWLIST_STATE_MISMATCH_NO_GENERATION_STARTED`

The array failed before model load or generation. The cause was route validation observing the temporary exactly-one enabled allowlist entry at Slurm task startup. The repair skips mutable allowlist state inside actual tasks; exactly-one safety remains enforced immediately before `sbatch`.

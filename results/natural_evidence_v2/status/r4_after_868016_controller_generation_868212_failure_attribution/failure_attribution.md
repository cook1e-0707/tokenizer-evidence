# R4 After-868151 Quality-Repaired Generation 868212 Failure Attribution

Status: `RECORDED_R4_AFTER_868151_QUALITY_REPAIRED_GENERATION_868212_ARTIFACT_ONLY_FAILURE_ATTRIBUTION_NO_SUBMIT`

## Coordinate 26

- shard_03 protected coordinate-26 erasures: `64` / `64`
- interpretation: the single failed protected block is caused by zero support for bit index 1 / coordinate 26, not by a wrong-key or null accept.

## Global Duplicate Caveat

- generated rows: `9216`
- unique response hashes: `4792`
- duplicate hash groups: `2908`
- duplicate extra rows: `4424`
- max duplicate group size: `4`
- duplicate condition sets: `{'protected,raw': 1621, 'task_only': 1024, 'raw': 170, 'protected': 93}`
- duplicate shard sets top: `{'shard_00,shard_01': 1090, 'shard_02,shard_03': 1051, 'shard_01': 216, 'shard_03': 211, 'shard_00': 187, 'shard_02': 153}`

## Route Implication

Do not submit another Slurm generation/scoring/training job from this state. The next route must first decide whether to repair coordinate reliability and global duplicate allocation/decoding policy, or pivot away from this controller/generation path.

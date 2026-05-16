# Hermes sync: R4 reliability route pushed, job 867621 running

phase:
V2_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_867621_RUNNING_MONITOR_ONLY

github:
- branch: main
- commit: 0f086d6 Record R4 reliability dev generation route
- pushed: yes, origin/main updated

slurm:
- job_id: 867621
- job_name: nat-ev-v2-r4relgen
- partition/qos/account: pomplun / pomplun / cs_yinxin.wan
- state: RUNNING for shards 0..3 on chimera21

next allowed action:
Monitor 867621 only. After all shards finish, sync outputs and run artifact review/decode aggregation. No second Slurm submission while 867621 is active.

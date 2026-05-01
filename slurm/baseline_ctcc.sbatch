#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{qos_directive}
#SBATCH --account={account}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err

set -euo pipefail

set +u
{environment_setup}
set -u

cd "$SLURM_SUBMIT_DIR"
{command}

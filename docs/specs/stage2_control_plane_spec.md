You are implementing the SECOND STAGE of a research-grade experimental codebase for a NeurIPS-style LLM ownership verification project.

The repository skeleton already exists.
Do NOT redesign the whole repo.
Do NOT add unrelated training features.
Do NOT introduce unnecessary frameworks.
Your job is to implement the infrastructure core needed before large-scale experiments can begin.

This stage is ONLY about:
1. config system
2. manifest system
3. result schemas
4. SLURM launcher and resubmission workflow

Everything must be designed for:
- reproducibility
- versioned experimentation
- HPC execution on a SLURM cluster
- easy iteration from a local control machine
- machine-readable result aggregation
- future baseline comparison under matched protocols

You must generate real code, not just a plan.

## Existing project assumptions

The repo already has directories like:
- configs/
- manifests/
- src/
- scripts/
- slurm/
- tests/
- results/
- docs/

Assume the repo root is already valid and pyproject.toml exists.

Your implementations must fit into the existing repository structure.
Do NOT delete or rename the existing top-level directories unless absolutely necessary.
Prefer adding and refining files.

## High-level design constraints

### A. Single source of truth
The config system must be the single source of truth for experiment parameters.
No training/eval/attack script may rely on scattered hard-coded values.

### B. Manifest-driven execution
Experiments must be represented as explicit manifest entries.
A manifest entry should be rich enough to:
- identify the experiment
- identify the method/baseline
- identify the config bundle
- identify runtime requirements
- identify output location
- record status and provenance

### C. Unified result schema
All run outputs must be serializable into explicit structured schemas.
Results from our method and baselines must be aggregatable through common fields where possible.

### D. Local control plane, remote compute plane
The launcher must assume:
- manifests are created locally
- jobs are submitted to SLURM on Chimera
- run directories are unique
- run metadata is written before and after submission

### E. Conservative engineering
Keep code explicit and testable.
Do not over-engineer.
Do not use notebooks.
Prefer dataclasses or pydantic-style models.
Use pathlib for all filesystem handling.

## Files to implement or refine

Implement/refine the following files:

src/infrastructure/config.py
src/infrastructure/manifest.py
src/infrastructure/paths.py
src/infrastructure/environment.py
src/infrastructure/slurm.py
src/infrastructure/logging.py
src/infrastructure/seed.py

src/evaluation/report.py

scripts/make_manifest.py
scripts/submit_slurm.py
scripts/resubmit_failed.py
scripts/summarize.py

tests/test_config_loading.py
tests/test_result_schema.py

If needed, add:
src/infrastructure/result_schema.py
src/infrastructure/registry.py

You may also add small utility files if they materially improve clarity.

## Required functionality

# 1. Config system

Implement a lightweight structured config loader.

Requirements:
- YAML-based configs
- nested composition supported in a simple explicit way
- command-line overrides supported
- resolved config can be saved to disk
- required keys validated
- failure messages should be clear and strict
- avoid large config frameworks unless truly needed
- prefer explicit Python code over magical implicit behavior

The config system must support these categories:
- experiment
- model
- data
- train
- eval
- attack
- runtime

A resolved config object should include:
- experiment_name
- method_name
- model_name
- seed
- output_root
- runtime resource hints
- references to source config files
- merged settings

Provide:
- a load_config(...) function
- a save_resolved_config(...) function
- a validation function

# 2. Result schema

Create explicit result schemas for:
- TrainRunSummary
- EvalRunSummary
- CalibrationSummary
- AttackRunSummary
- AggregatedComparisonRow

Use dataclasses or pydantic models.
Every schema must be JSON serializable.

All result schemas must include common provenance fields:
- run_id
- experiment_name
- method_name
- model_name
- seed
- git_commit
- timestamp
- hostname if available
- slurm_job_id if available
- status

Add helper functions:
- to_json_dict()
- save_json(...)
- load_json(...)

These schemas will later be used to build paper tables.
Design them carefully.

# 3. Paths and run IDs

Implement robust output path creation.

Requirements:
- unique run directories
- no accidental overwrite by default
- run naming must include:
  - experiment name
  - method/model
  - seed
  - short git hash
  - timestamp
- force overwrite only if explicitly requested
- pathlib-based only

Each run directory should be able to contain:
- config.resolved.yaml
- environment.json
- submission.json
- metrics.json
- stdout/stderr logs
- optional JSONL logs

Provide helper functions:
- make_run_id(...)
- make_run_dir(...)
- ensure_run_dir(...)
- get_results_paths(...)

# 4. Environment capture

Implement environment capture utilities.

Must record:
- git commit hash
- git dirty state if available
- python version
- platform info
- selected dependency versions if importable
- current hostname
- SLURM env vars if present

Provide:
- collect_environment_summary(...)
- save_environment_summary(...)

# 5. Manifest system

Implement a manifest schema and loader/generator.

A manifest entry must include:
- manifest_id
- experiment_name
- method_name
- model_name
- seed
- config_paths
- overrides
- output_dir or output_root
- requested resources:
  - partition
  - gpu_type if needed
  - num_gpus
  - cpus
  - mem_gb
  - time_limit
- launcher mode
- status
- tags
- notes

A manifest file should support a list of entries.

Implement:
- ManifestEntry schema
- ManifestFile schema
- load_manifest(...)
- save_manifest(...)
- validate_manifest(...)
- update_manifest_status(...)

Implement a manifest generator script:
scripts/make_manifest.py

This script should:
- read a sweep config or experiment config
- expand seeds / models / methods / variants
- generate manifest entries
- save them under manifests/<exp_name>/...
- support dry-run mode
- print a concise summary of generated entries

# 6. Submission registry

Implement a simple registry layer so submitted jobs can be tracked.

A registry record should minimally store:
- manifest_id
- run_id
- submission_time
- slurm_job_id
- slurm_script_path
- status
- output_dir

This can be a JSONL or JSON file.
Keep it simple and robust.

Provide:
- append_registry_record(...)
- load_registry(...)
- find_failed_records(...)
- find_unsubmitted_records(...)

# 7. SLURM launcher

Implement SLURM support that is strict and explicit.

Requirements:
- generate sbatch command from a manifest entry
- support per-entry resource specification
- support log file paths
- support environment activation command placeholder
- support passing manifest_id or manifest file path into the job
- support dry-run submission mode
- support actual submission mode
- capture and parse returned SLURM job ID if possible
- save submission metadata into run directory

Important constraints:
- do NOT assume all jobs have identical resources
- do NOT force job arrays for heterogeneous jobs
- keep support for future array mode possible, but default to one manifest entry -> one job
- keep cluster-specific strings configurable
- do not hard-code Chimera-specific paths beyond placeholders

Implement:
src/infrastructure/slurm.py
scripts/submit_slurm.py
scripts/resubmit_failed.py

scripts/submit_slurm.py must support:
- submit one manifest
- submit one entry by manifest_id
- submit all pending entries
- dry-run
- filtering by tag / method / experiment

scripts/resubmit_failed.py must support:
- scanning registry or manifest
- identifying failed runs
- optionally regenerating submission
- dry-run mode

# 8. Logging

Implement minimal but disciplined logging.

Requirements:
- readable console logger
- optional JSONL logger
- one log file per run
- include run_id in messages where possible

Do not introduce heavy observability frameworks.

# 9. Tests

Write real tests for this stage.

At minimum:
- config loading and validation success
- config validation failure on missing keys
- manifest serialization round-trip
- manifest generator creates correct count of entries
- run_id uniqueness logic
- result schema JSON round-trip
- SLURM command generation correctness
- registry append/load behavior

All tests must run locally without GPU.

# 10. Documentation updates

Update or create concise docs:
- docs/experiment_protocol.md
- docs/chimera_runbook.md
- docs/result_schema.md

They should explain:
- how configs are resolved
- how manifests are generated
- how jobs are submitted
- where metadata is stored
- how failures are resubmitted
- how results are later aggregated

Keep docs concise and practical.

## CLI expectations

Use a simple CLI style.
argparse is fine.
Avoid over-complex frameworks.

The following should work from repo root:

- python scripts/make_manifest.py --config ...
- python scripts/submit_slurm.py --manifest ...
- python scripts/resubmit_failed.py --manifest ...
- python scripts/summarize.py --results ...

## Style and quality constraints

- Use type hints.
- Use pathlib.
- Prefer dataclasses unless pydantic provides a clear benefit.
- Keep functions small and testable.
- No giant manager classes unless truly necessary.
- No hidden side effects.
- No silent fallback behavior for invalid configs.
- Raise explicit errors with actionable messages.
- Keep TODOs only where functionality genuinely depends on later training code.

## Output requirements

You must:
1. implement the files
2. explain what was added
3. list assumptions
4. identify any missing integration points with future training/eval code
5. tell me exactly which commands to run locally to validate this stage

Before writing code, first inspect the current repository structure and existing files so that your changes fit the repo rather than overwrite it blindly.
If existing files already implement part of this stage, refine them rather than duplicating logic.
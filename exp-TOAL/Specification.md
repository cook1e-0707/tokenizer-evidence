You are generating the initial repository skeleton for a high-quality, research-grade experimental codebase for a NeurIPS-style LLM ownership verification project.

This is NOT a toy repo, NOT a quick prototype, and NOT a one-off script collection.
The repository must be designed for:
- strong reproducibility
- fair baseline comparison
- SLURM/HPC deployment
- easy iteration
- disciplined experiment tracking
- clean separation of concerns
- future paper artifact preparation

## Project context

The project studies tokenizer-constrained ownership evidence injection for stolen-model ownership verification in autoregressive LLMs.

The codebase must support:
1. our method:
   - tokenizer alignment checks
   - bucketized evidence representation
   - bucket-mass training objective
   - mixed-radix payload symbolization
   - Reed–Solomon-based deterministic recovery
   - parser / verifier / scanning pipeline

2. theorem-aligned experiments:
   - alignment experiments
   - bucket-objective experiments
   - recovery-boundary experiments

3. end-to-end ownership verification experiments:
   - matched utility calibration
   - matched false-accept-rate calibration
   - baseline comparison
   - scrubbing attacks
   - formatting / rewriting / truncation perturbations
   - stress tests and negative results

4. baseline adapters:
   - watermark/provenance-style baselines
   - active fingerprinting baselines
   - baseline-specific training/inference wrappers
   - unified evaluation harness

5. HPC deployment:
   - UMass Boston Chimera cluster
   - A100 / H200 usage
   - SLURM job submission
   - manifest-driven experiment launching
   - result tracking and failure recovery

The repository will be edited on a local MacBook Pro and executed mainly on Chimera.
The local machine is the control plane; the cluster is the compute plane.

## Non-negotiable requirements

### General engineering principles
- The repository must be production-style, not notebook-style.
- Avoid monolithic scripts.
- Every module must have a clear responsibility.
- Use Python.
- Prefer pyproject.toml-based packaging.
- Add type hints throughout.
- Add docstrings where useful, but do not over-comment obvious code.
- Use clear naming and consistent interfaces.
- Prefer dataclasses or pydantic-style structured configs/results where appropriate.
- No hidden global state.
- No hard-coded paths inside core logic.
- No hard-coded experiment parameters inside training/evaluation code.
- All experiment behavior must be config-driven.

### Reproducibility
- Every run must save:
  - resolved config
  - git commit hash
  - environment summary
  - random seed(s)
  - SLURM job id if available
  - timestamped run ID
- Output directories must be unique and non-overwriting.
- Result schemas must be explicit and machine-readable.
- The repository must support deterministic seeding utilities.

### Experiment protocol discipline
- Support manifest-driven experiments.
- Support sweep generation from structured configs.
- Support a unified evaluation harness for our method and baselines.
- Support calibration workflows for matched utility and matched FAR.
- Keep training logic separate from evaluation logic.
- Keep baseline adapters separate from our method implementation.

### Testing
Add meaningful tests from the start.
At minimum:
- tokenizer alignment tests
- bucket partition consistency tests
- mixed-radix codec invertibility tests
- Reed–Solomon wrapper tests
- parser/verifier tests
- config loading / schema validation tests
- output schema tests

### HPC / SLURM
- Include SLURM submission utilities.
- Include sbatch templates.
- Include a manifest-driven submit script.
- Include failed-run detection / resubmission helper.
- Keep cluster-specific logic isolated under infrastructure/ or slurm/.

### Result management
- Create a clear result directory structure:
  - raw
  - processed
  - tables
  - figures
- Include scripts for:
  - summarizing runs
  - aggregating metrics
  - generating publication-ready tables
  - generating figures from processed results

### Documentation
Include initial documentation:
- README.md
- docs/experiment_protocol.md
- docs/baseline_protocol.md
- docs/result_schema.md
- docs/chimera_runbook.md

The documentation should be concise but operationally useful.

## Repository structure to create

Create the following structure, with empty or initial implementations where appropriate:

project_root/
  README.md
  pyproject.toml
  .gitignore
  .editorconfig
  .python-version
  .pre-commit-config.yaml

  configs/
    model/
    data/
    train/
    eval/
    attack/
    sweep/
    experiment/
      exp_alignment.yaml
      exp_bucket.yaml
      exp_recovery.yaml
      exp_main.yaml

  src/
    core/
      tokenizer_utils.py
      bucket_mapping.py
      payload_codec.py
      rs_codec.py
      parser.py
      verifier.py

    training/
      dataset.py
      collator.py
      objectives.py
      trainer.py

    baselines/
      base.py
      kgw_adapter.py
      ctcc_adapter.py
      esf_adapter.py

    evaluation/
      metrics.py
      calibration.py
      utility_eval.py
      far_eval.py
      report.py

    infrastructure/
      config.py
      logging.py
      seed.py
      checkpointing.py
      paths.py
      manifest.py
      slurm.py
      environment.py

  scripts/
    train.py
    eval.py
    attack.py
    calibrate.py
    summarize.py
    make_table.py
    make_figures.py
    submit_slurm.py
    resubmit_failed.py
    make_manifest.py

  slurm/
    train_main.sbatch
    eval_main.sbatch
    baseline_ctcc.sbatch
    attack_scrub.sbatch
    sweep_alignment.sbatch

  manifests/
    .gitkeep

  tests/
    test_tokenizer_alignment.py
    test_bucket_mapping.py
    test_payload_codec.py
    test_rs_codec.py
    test_parser.py
    test_verifier.py
    test_config_loading.py
    test_result_schema.py

  results/
    raw/.gitkeep
    processed/.gitkeep
    tables/.gitkeep
    figures/.gitkeep

  docs/
    experiment_protocol.md
    baseline_protocol.md
    result_schema.md
    chimera_runbook.md

## Implementation details

### 1. Config system
Implement a structured config loader.
Requirements:
- support YAML configs
- support nested composition
- support command-line overrides
- save resolved config for each run
- validate required keys
- avoid introducing a large external framework unless truly necessary
- keep the system lightweight and explicit

### 2. Result schema
Define explicit result schemas for:
- train run summary
- eval run summary
- calibration outputs
- attack outputs
- aggregated comparison rows

Use dataclasses or pydantic models.
Every result object should be serializable to JSON.

### 3. Run IDs and output paths
Implement a robust run naming function using:
- experiment name
- model name
- seed
- short git hash
- timestamp
Do not allow accidental overwrite unless a force flag is explicitly passed.

### 4. Logging
Implement structured logging.
Requirements:
- human-readable console logs
- JSONL log option for machine parsing
- log files saved in each run directory
- include config summary at startup

### 5. Seed control
Implement deterministic seed utilities for:
- Python
- NumPy
- PyTorch
Expose a single helper function.

### 6. Core modules
Implement clean interfaces and stubs for:
- tokenizer alignment audit
- bucket partition definition and validation
- payload encode/decode
- mixed-radix map
- RS wrapper interface
- parser candidate extraction
- verifier end-to-end pipeline

Do not over-implement the full method yet.
Focus on interfaces, validation, and testability.

### 7. Baseline adapters
Create a common baseline adapter interface.
Each adapter must expose methods like:
- prepare(...)
- train(...)
- infer(...)
- verify(...)
- summarize(...)
These can initially be partially stubbed, but must have consistent signatures and docs.

### 8. Evaluation harness
Implement a shared evaluation entry point that can run:
- our method
- baselines
through a common metrics/reporting path.
Metrics and calibration should be method-agnostic where possible.

### 9. Manifest-driven experiments
Implement:
- a manifest schema
- a generator script that expands sweeps into manifest entries
- a submit script that turns manifests into SLURM jobs
- a registry file tracking job state

### 10. SLURM support
Provide:
- sbatch templates
- resource parameters
- partition/GPU placeholders
- environment activation hook
- output/error log paths
- ability to submit one manifest item per job

Keep SLURM-specific strings configurable.

### 11. Tests
Write real tests, not placeholders only.
At minimum:
- tokenizer alignment validation behavior
- bucket partition disjointness and coverage assumptions
- encode/decode round-trip for payload codec
- RS interface round-trip stub behavior
- parser behavior on clean synthetic examples
- verifier behavior on clean synthetic examples
- config validation failures
- result schema serialization

### 12. Docs
Write concise but useful docs.
README must include:
- what this repo is
- local development workflow
- Chimera workflow
- config workflow
- how to create a manifest
- how to submit a run
- where results go
- how to aggregate results

docs/chimera_runbook.md must include:
- expected workflow for local -> cluster
- branch/commit discipline
- environment setup notes
- running smoke tests before large jobs
- where SLURM logs and run logs are saved

## Style constraints
- Keep code clean and conservative.
- Do not introduce unnecessary frameworks.
- Do not use notebooks.
- Do not create giant classes when functional utilities suffice.
- Keep adapter interfaces explicit.
- Prefer standard library + a minimal set of dependencies.
- Assume researchers will extend this code.

## Output requirements
Generate the full repository skeleton with initial implementations, not just a plan.
Create all files listed above.
Where full functionality is not yet possible, create disciplined stubs with TODO markers.
After generating files:
1. summarize the repository structure
2. explain the core workflows
3. list any assumptions made
4. identify the first 5 follow-up implementation tasks in priority order


Additional constraints for this repository generation:

1. Do not write any experimental logic that assumes a specific paper baseline implementation is already installed.
2. Make all baseline adapters safe placeholders if the underlying baseline code is not yet integrated.
3. Prefer small, testable modules over “smart” abstractions.
4. Use pathlib for filesystem handling.
5. Use argparse or typer for scripts, but keep the CLI simple and explicit.
6. Make sure scripts can run from the repository root without path hacks.
7. Ensure all tests can run locally without GPU.
8. For GPU-dependent logic, isolate it cleanly and avoid importing heavy training dependencies inside lightweight utility tests.
9. Create a minimal synthetic data path for parser/verifier smoke tests.
10. Include a single source of truth for experiment names and result directory creation.
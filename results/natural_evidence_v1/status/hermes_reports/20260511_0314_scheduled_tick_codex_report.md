Implemented the R3.2 Qwen locked-scale plan-only wrapper and recorded the review.

Changed:
- Added [build_r3_2_locked_scale_precommit.py](/Users/guanjie/Documents/tokenizer_alignment/scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py)
- Added [r3_2_qwen_locked_scale_eval.sbatch](/Users/guanjie/Documents/tokenizer_alignment/scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch)
- Added wrapper review/status artifacts under `docs/` and `results/`
- Updated both gate status files and V1 automation docs
- Added a focused test that recomputes the locked selected prompt manifest hash

Validation passed:
- `python3 -m py_compile scripts/natural_evidence_v2/build_r3_2_locked_scale_precommit.py`
- `bash -n scripts/natural_evidence_v2/slurm/r3_2_qwen_locked_scale_eval.sbatch`
- `VALIDATE_PLAN_ONLY=1 ... r3_2_qwen_locked_scale_eval.sbatch`
- `.venv/bin/python -m pytest tests/test_natural_evidence_v2_wp6_coordinate_majority.py` → `8 passed`

No Slurm job was submitted. No training, generation, Qwen E2E rerun, Llama, same-family null, sanitizer, FAR aggregation, or paper-facing positive claim was started. The R3.2 allowlist entry remains disabled.
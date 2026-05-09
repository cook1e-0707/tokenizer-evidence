Completed one allowed WP2 action.

Created deterministic v2 prompt-family split artifacts with no model calls:
- [builder](/Users/guanjie/Documents/tokenizer_alignment/scripts/natural_evidence_v2/build_wp2_prompt_scaffold.py)
- [split manifest](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/prompts/wp2_controlled_natural_prompt_family_scaffold_20260508_2123/split_manifest.json)
- [audit](/Users/guanjie/Documents/tokenizer_alignment/results/natural_evidence_v2/prompts/wp2_controlled_natural_prompt_family_scaffold_20260508_2123/forbidden_surface_audit.json)

Generated configured split counts:
- train `4096`
- dev `1024`
- eval `2048`
- organic null `2048`

Audit result: `PASS_FORBIDDEN_SURFACE_AUDIT`, prompt-text forbidden surface rate `0.0`, violations `0`.

Updated the v1/v2 gate and Hermes-readable state files so the next allowed action is now WP3 artifact-only micro-slot detector and 2-way bucket policy design. Also removed the stale hardcoded WP2 instruction from the Hermes worker prompt.

Validation run:
- `python3 -m py_compile ...`
- generated WP2 artifacts with `--use-config-split-sizes`
- `python3 -m json.tool` on updated gate/report JSON
- `wc -l` on split files

No training, model calls, transcript generation, E2E, Llama, same-family null, sanitizer, FAR aggregation, or positive claim was started.
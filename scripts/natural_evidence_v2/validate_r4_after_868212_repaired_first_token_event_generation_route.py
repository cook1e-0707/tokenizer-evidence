from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import sha256_file  # noqa: E402
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_868212_repaired_first_token_event_generation_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_868212_repaired_first_token_event_generation_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_868212_repaired_first_token_event_generation_h200.sbatch"
EXPECTED_BASE_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_868016_controller_generation_h200.sbatch"
EXPECTED_GENERATOR = "scripts/natural_evidence_v2/generate_r4_after_868016_controller_outputs.py"
EXPECTED_EVENT_DECODER = "scripts/natural_evidence_v2/decode_r4_after_868151_first_token_event_channel.py"


def mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


def path_from(value: Any, field: str, errors: list[str]) -> Path:
    path = ROOT / str(value)
    if not path.exists():
        errors.append(f"{field} missing: {path}")
    return path


def read_json(path: Path, field: str, errors: list[str]) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{field} unreadable JSON: {exc}")
        return {}
    if not isinstance(payload, Mapping):
        errors.append(f"{field} must be a JSON object")
        return {}
    return payload


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def find_allowlist_entry(allowlist: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        entries = allowlist.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("name") == name:
                return entry
    return None


def enabled_allowlist_entries(allowlist: Mapping[str, Any]) -> list[str]:
    enabled: list[str] = []
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        entries = allowlist.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("enabled") is True:
                enabled.append(str(entry.get("name", "")))
    return enabled


def validate_route(config: Mapping[str, Any], *, allow_submission_enabled_entry: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("schema_name") != "natural_evidence_v2_r4_after_868212_repaired_first_token_event_generation_route_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") != "r4_after_868212_repaired_first_token_event_generation_v1":
        errors.append("route_id mismatch")

    source = mapping(config.get("source_reviews"), "source_reviews", errors)
    tokenizer_review_path = path_from(source.get("tokenizer_review", ""), "source_reviews.tokenizer_review", errors)
    prior_review_path = path_from(source.get("prior_generation_review", ""), "source_reviews.prior_generation_review", errors)
    attribution_path = path_from(source.get("failure_attribution", ""), "source_reviews.failure_attribution", errors)
    repaired_validation_path = path_from(
        source.get("repaired_plan_validation", ""),
        "source_reviews.repaired_plan_validation",
        errors,
    )
    tokenizer_review = read_json(tokenizer_review_path, "source_reviews.tokenizer_review", errors) if tokenizer_review_path.exists() else {}
    prior_review = read_json(prior_review_path, "source_reviews.prior_generation_review", errors) if prior_review_path.exists() else {}
    attribution = read_json(attribution_path, "source_reviews.failure_attribution", errors) if attribution_path.exists() else {}
    repaired_validation = (
        read_json(repaired_validation_path, "source_reviews.repaired_plan_validation", errors)
        if repaired_validation_path.exists()
        else {}
    )
    tokenizer_status = tokenizer_review.get("status") or tokenizer_review.get("review_status")
    if tokenizer_status != "PASS_R4_AFTER_867621_RELIABILITY_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT_867828":
        errors.append("full16 tokenizer review status mismatch")
    if int(tokenizer_review.get("failed_row_count", -1)) != 0:
        errors.append("full16 tokenizer review must have zero failed rows")
    if prior_review.get("claim_policy") != "diagnostic_only_not_locked_positive_not_paper_claim":
        errors.append("prior 868212 review must remain diagnostic-only")
    if int(prior_review.get("first_token_protected_accepts", -1)) != 3:
        errors.append("prior 868212 protected accepts should remain 3/4")
    if attribution.get("status") != "RECORDED_R4_AFTER_868151_QUALITY_REPAIRED_GENERATION_868212_ARTIFACT_ONLY_FAILURE_ATTRIBUTION_NO_SUBMIT":
        errors.append("868212 failure attribution status mismatch")
    if repaired_validation.get("status") != "PASS_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_PLAN_VALIDATION_NO_SUBMIT":
        errors.append("repaired plan validation did not pass")

    rows = mapping(config.get("score_rows"), "score_rows", errors)
    rows_path = path_from(rows.get("path", ""), "score_rows.path", errors)
    if rows_path.exists():
        if count_jsonl(rows_path) != int(rows.get("row_count", -1)):
            errors.append("score row count mismatch")
        if sha256_file(rows_path) != str(rows.get("sha256", "")):
            errors.append("score_rows.sha256 mismatch")
    if int(rows.get("selected_coordinate_count", 0)) != 16:
        errors.append("score_rows.selected_coordinate_count must be 16")

    quality = mapping(config.get("quality_repair"), "quality_repair", errors)
    quality_plan = path_from(quality.get("plan_summary", ""), "quality_repair.plan_summary", errors)
    allocation_rows = path_from(quality.get("allocation_rows", ""), "quality_repair.allocation_rows", errors)
    allocation_manifest = path_from(quality.get("allocation_manifest", ""), "quality_repair.allocation_manifest", errors)
    contextual_literal_policy = path_from(
        quality.get("contextual_literal_policy", ""),
        "quality_repair.contextual_literal_policy",
        errors,
    )
    if quality_plan.exists():
        quality_payload = read_json(quality_plan, "quality_repair.plan_summary", errors)
        if quality_payload.get("status") != "PASS_R4_AFTER_868151_FIRST_TOKEN_EVENT_QUALITY_REPAIR_PLAN_ARTIFACT_ONLY":
            errors.append("full16 quality repair plan did not pass")
        if int(quality_payload.get("rows_per_shard", -1)) != 1024:
            errors.append("full16 quality repair rows_per_shard must be 1024")
    if allocation_manifest.exists():
        allocation_payload = read_json(allocation_manifest, "quality_repair.allocation_manifest", errors)
        if allocation_payload.get("status") != "PASS_DUPLICATE_SAFE_ROW_ALLOCATION_ARTIFACT_ONLY":
            errors.append("full16 allocation manifest did not pass")
        for shard in allocation_payload.get("shard_summaries", []):
            if int(shard.get("duplicate_prompt_prefix_pair_count", -1)) != 0:
                errors.append("allocation manifest has duplicate prompt/prefix pairs")
    if allocation_rows.exists() and count_jsonl(allocation_rows) != 4096:
        errors.append("full16 allocation row count must be 4096")
    if contextual_literal_policy.exists():
        literal_payload = read_json(contextual_literal_policy, "quality_repair.contextual_literal_policy", errors)
        if "coordinate" not in literal_payload.get("contextual_literals", {}):
            errors.append("contextual literal policy must define coordinate")

    precommit = mapping(config.get("precommit"), "precommit", errors)
    surface_bank = path_from(precommit.get("surface_bank", ""), "precommit.surface_bank", errors)
    codebook = path_from(precommit.get("codebook", ""), "precommit.codebook", errors)
    decoder_spec = path_from(precommit.get("decoder_spec", ""), "precommit.decoder_spec", errors)
    full_phrase_decoder_spec = path_from(
        precommit.get("full_phrase_decoder_spec", ""),
        "precommit.full_phrase_decoder_spec",
        errors,
    )
    duplicate_policy = path_from(precommit.get("duplicate_policy", ""), "precommit.duplicate_policy", errors)
    decoder_route_config = path_from(precommit.get("decoder_route_config", ""), "precommit.decoder_route_config", errors)
    if precommit.get("contract_id") != "a55e":
        errors.append("precommit.contract_id must be a55e")
    if precommit.get("primary_scrub_mode") != "all":
        errors.append("precommit.primary_scrub_mode must be all")
    if precommit.get("reclassifies_868212") is not False:
        errors.append("precommit must not reclassify 868212")
    for path, field in (
        (surface_bank, "surface_bank_sha256"),
        (codebook, "codebook_sha256"),
        (decoder_spec, "decoder_spec_sha256"),
        (duplicate_policy, "duplicate_policy_sha256"),
    ):
        expected_hash = str(precommit.get(field, ""))
        if path.exists() and sha256_file(path) != expected_hash:
            errors.append(f"precommit.{field} mismatch")
    if codebook.exists():
        codebook_payload = read_json(codebook, "precommit.codebook", errors)
        if len(codebook_payload.get("selected_coordinates", [])) != 16:
            errors.append("repaired codebook must contain 16 selected coordinates")
        if int(codebook_payload.get("min_active_coordinates_per_bit", 0)) < 2:
            errors.append("repaired codebook must not contain singleton bits")
        if codebook_payload.get("reclassifies_868212") is not False:
            errors.append("repaired codebook must not reclassify 868212")
    if decoder_spec.exists():
        decoder_payload = read_json(decoder_spec, "precommit.decoder_spec", errors)
        if decoder_payload.get("decoder") != "row_local_first_token_event_pair_majority_then_checksum":
            errors.append("first-token decoder mismatch")
        if decoder_payload.get("future_positive_requires_token_id_trace") is not True:
            errors.append("decoder must require token-id traces")
    if full_phrase_decoder_spec.exists():
        full_phrase_payload = read_json(full_phrase_decoder_spec, "precommit.full_phrase_decoder_spec", errors)
        if full_phrase_payload.get("decoder") != "pair_majority_then_checksum":
            errors.append("full-phrase decoder spec must be pair_majority_then_checksum")
    if duplicate_policy.exists():
        duplicate_payload = read_json(duplicate_policy, "precommit.duplicate_policy", errors)
        locked_gate = duplicate_payload.get("locked_scale_hard_fail", {}).get("global_duplicate_response_hash_count")
        if int(locked_gate) != 0:
            errors.append("locked-scale global duplicate gate must be 0")

    generation = mapping(config.get("generation_scope"), "generation_scope", errors)
    if int(generation.get("rows_per_shard", 0)) != 1024:
        errors.append("generation_scope.rows_per_shard must be 1024")
    if int(generation.get("selected_coordinate_count", 0)) != 16:
        errors.append("generation_scope.selected_coordinate_count must be 16")
    if generation.get("conditions") != ["protected", "raw", "task_only"]:
        errors.append("generation conditions mismatch")
    if generation.get("decode_conditions") != ["protected", "raw", "task_only", "wrong_key", "wrong_payload"]:
        errors.append("decode conditions mismatch")
    for field in ("same_contract_only", "controller_only_protected_condition"):
        if generation.get(field) is not True:
            errors.append(f"generation_scope.{field} must be true")
    for field in ("payload_diversity_tested", "llama_tested", "paper_facing"):
        if generation.get(field) is not False:
            errors.append(f"generation_scope.{field} must be false")

    event_policy = mapping(config.get("event_trace_policy"), "event_trace_policy", errors)
    if event_policy.get("decoder") != EXPECTED_EVENT_DECODER:
        errors.append("event_trace_policy.decoder mismatch")
    if event_policy.get("future_positive_requires_token_id_trace") is not True:
        errors.append("event_trace_policy.future_positive_requires_token_id_trace must be true")
    if event_policy.get("contextual_literal_policy") != quality.get("contextual_literal_policy"):
        errors.append("event trace contextual literal policy must match repaired full16 policy")

    compute = mapping(config.get("compute_policy"), "compute_policy", errors)
    if compute.get("allowlist_entry") != EXPECTED_ENTRY:
        errors.append("compute allowlist entry mismatch")
    if compute.get("wrapper") != EXPECTED_WRAPPER:
        errors.append("compute wrapper mismatch")
    if compute.get("command_pattern") != f"sbatch {EXPECTED_WRAPPER}":
        errors.append("compute command_pattern mismatch")
    if compute.get("partition") != "pomplun" or compute.get("qos") != "pomplun":
        errors.append("compute must use pomplun")
    if compute.get("account") != "cs_yinxin.wan":
        errors.append("compute account mismatch")
    if compute.get("gres") != "gpu:h200:1":
        errors.append("compute gres mismatch")
    if compute.get("max_time") != "30-00:00:00":
        errors.append("compute max_time mismatch")
    if compute.get("allowlist_enabled_now") is not False:
        errors.append("compute allowlist_enabled_now must be false")

    wrapper = path_from(compute.get("wrapper", ""), "compute.wrapper", errors)
    base_wrapper = path_from(EXPECTED_BASE_WRAPPER, "compute.base_wrapper", errors)
    generator_path = path_from(EXPECTED_GENERATOR, "generator", errors)
    if wrapper.exists():
        text = wrapper.read_text(encoding="utf-8")
        for fragment in (
            "#SBATCH --partition=pomplun",
            "#SBATCH --account=cs_yinxin.wan",
            "#SBATCH --qos=pomplun",
            "#SBATCH --gres=gpu:h200:1",
            "#SBATCH --time=30-00:00:00",
            "r4_after_868212_repaired_first_token_event_generation_route.yaml",
            "r4_after_868212_full16_quality_repair_plan_20260516",
            "r4_after_868212_repaired_first_token_event_precommit_20260516",
            "FULL_PHRASE_DECODER_SPEC",
            "ROWS_PER_SHARD",
            "EXPECTED_SELECTED_COORDINATE_COUNT",
            "bash scripts/natural_evidence_v2/slurm/r4_after_868016_controller_generation_h200.sbatch",
        ):
            if fragment not in text:
                errors.append(f"wrapper missing fragment: {fragment}")
    if base_wrapper.exists():
        text = base_wrapper.read_text(encoding="utf-8")
        for fragment in (
            "ROWS_PER_SHARD",
            "EXPECTED_SELECTED_COORDINATE_COUNT",
            "FULL_PHRASE_DECODER_SPEC",
            "--expected-selected-coordinate-count \"$EXPECTED_SELECTED_COORDINATE_COUNT\"",
        ):
            if fragment not in text:
                errors.append(f"base wrapper missing full16 parameter fragment: {fragment}")
    if generator_path.exists():
        text = generator_path.read_text(encoding="utf-8")
        for fragment in (
            "--expected-selected-coordinate-count",
            "expected_selected_coordinate_count",
            "PASS_R4_AFTER_867621_RELIABILITY_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT_867828",
        ):
            if fragment not in text:
                errors.append(f"generator missing full16 support fragment: {fragment}")

    prerequisites = mapping(config.get("required_before_any_submission"), "required_before_any_submission", errors)
    for field, value in prerequisites.items():
        if value is not True:
            errors.append(f"required_before_any_submission.{field} must be true")
    locked = mapping(config.get("not_unlocked_by_this_route_package"), "not_unlocked_by_this_route_package", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_package.{field} must be true")

    allowlist = load_yaml(ALLOWLIST)
    entry = find_allowlist_entry(allowlist, EXPECTED_ENTRY)
    if entry is None:
        errors.append("allowlist entry missing")
    else:
        enabled_entries = enabled_allowlist_entries(allowlist)
        if allow_submission_enabled_entry:
            if enabled_entries not in ([], [EXPECTED_ENTRY]):
                errors.append(f"allowlist enabled entries must be empty or exactly {EXPECTED_ENTRY}: {enabled_entries}")
        elif entry.get("enabled") is not False:
            errors.append("allowlist entry must be disabled")
        if entry.get("command_pattern") != f"sbatch {EXPECTED_WRAPPER}":
            errors.append("allowlist command pattern mismatch")

    status = (
        "PASS_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_GENERATION_ROUTE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_868212_REPAIRED_FIRST_TOKEN_EVENT_GENERATION_ROUTE_VALIDATION_NO_SUBMIT"
    )
    return {
        "status": status,
        "errors": errors,
        "allow_submission_enabled_entry": allow_submission_enabled_entry,
        "allowlist_entry": EXPECTED_ENTRY,
        "wrapper": EXPECTED_WRAPPER,
        "score_rows_sha256": sha256_file(rows_path) if rows_path.exists() else "",
        "surface_bank_sha256": sha256_file(surface_bank) if surface_bank.exists() else "",
        "codebook_sha256": sha256_file(codebook) if codebook.exists() else "",
        "decoder_spec_sha256": sha256_file(decoder_spec) if decoder_spec.exists() else "",
        "decoder_route_config_sha256": sha256_file(decoder_route_config) if decoder_route_config.exists() else "",
        "slurm_job_submitted": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 after-868212 repaired first-token event generation route.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-submission-enabled-entry", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_route(load_yaml(args.config), allow_submission_enabled_entry=args.allow_submission_enabled_entry)
    if args.output_dir is not None:
        write_json_new(args.output_dir / "route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

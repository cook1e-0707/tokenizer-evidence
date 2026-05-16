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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_868016_controller_generation_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_868016_controller_generation_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_868016_controller_generation_h200.sbatch"
EXPECTED_GENERATOR = "scripts/natural_evidence_v2/generate_r4_after_868016_controller_outputs.py"
EXPECTED_EVENT_DECODER = "scripts/natural_evidence_v2/decode_r4_after_868151_first_token_event_channel.py"
REQUIRED_EVENT_FIELDS = [
    "first_generated_token_id",
    "first_generated_token_text",
    "target_first_token_ids",
    "other_first_token_ids",
    "event_side",
    "event_bucket_side",
    "event_trace",
]
EXPECTED_QUALITY_PLAN = (
    "results/natural_evidence_v2/status/r4_after_868151_first_token_event_quality_repair_plan_20260516/"
    "quality_repair_plan_summary.json"
)
EXPECTED_ALLOCATION_ROWS = (
    "results/natural_evidence_v2/status/r4_after_868151_first_token_event_quality_repair_plan_20260516/"
    "row_allocation_rows.jsonl"
)
EXPECTED_ALLOCATION_MANIFEST = (
    "results/natural_evidence_v2/status/r4_after_868151_first_token_event_quality_repair_plan_20260516/"
    "row_allocation_manifest.json"
)
EXPECTED_CONTEXTUAL_LITERAL_POLICY = (
    "results/natural_evidence_v2/status/r4_after_868151_first_token_event_quality_repair_plan_20260516/"
    "contextual_literal_policy.json"
)


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
    if config.get("schema_name") != "natural_evidence_v2_r4_after_868016_controller_generation_route_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") != "r4_after_868016_coordinate_pivot_controller_generation_v1":
        errors.append("route_id mismatch")

    source = mapping(config.get("source_reviews"), "source_reviews", errors)
    tokenizer_review_path = path_from(source.get("tokenizer_review", ""), "source_reviews.tokenizer_review", errors)
    controller_review_path = path_from(source.get("controller_review", ""), "source_reviews.controller_review", errors)
    codebook_oracle_path = path_from(source.get("codebook_oracle", ""), "source_reviews.codebook_oracle", errors)
    tokenizer_review = read_json(tokenizer_review_path, "source_reviews.tokenizer_review", errors) if tokenizer_review_path.exists() else {}
    controller_review = read_json(controller_review_path, "source_reviews.controller_review", errors) if controller_review_path.exists() else {}
    codebook_oracle = read_json(codebook_oracle_path, "source_reviews.codebook_oracle", errors) if codebook_oracle_path.exists() else {}
    tokenizer_status = tokenizer_review.get("status") or tokenizer_review.get("review_status")
    if tokenizer_status != "PASS_R4_AFTER_868016_RELIABILITY_COORDINATE_PIVOT_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT_868103":
        errors.append("tokenizer review status mismatch")
    if controller_review.get("status") != "PASS_R4_AFTER_868016_COORDINATE_PIVOT_CONTROLLER_TEACHER_FORCED_GATE":
        errors.append("controller review status mismatch")
    if codebook_oracle.get("status") != "PASS_R4_AFTER_868016_COORDINATE_PIVOT_CODEBOOK_ORACLE_ARTIFACT_ONLY":
        errors.append("codebook oracle status mismatch")
    if int(codebook_oracle.get("wrong_key_accepts", -1)) != 0 or int(codebook_oracle.get("wrong_payload_accepts", -1)) != 0:
        errors.append("codebook oracle controls must reject")

    rows = mapping(config.get("score_rows"), "score_rows", errors)
    rows_path = path_from(rows.get("path", ""), "score_rows.path", errors)
    if rows_path.exists():
        if count_jsonl(rows_path) != int(rows.get("row_count", -1)):
            errors.append("score row count mismatch")
        observed_hash = sha256_file(rows_path)
        if observed_hash != str(rows.get("sha256", "")):
            errors.append("score_rows.sha256 mismatch")
    if int(rows.get("selected_coordinate_count", 0)) != 12:
        errors.append("score_rows.selected_coordinate_count must be 12")

    quality = mapping(config.get("quality_repair"), "quality_repair", errors)
    if quality.get("plan_summary") != EXPECTED_QUALITY_PLAN:
        errors.append("quality_repair.plan_summary mismatch")
    if quality.get("allocation_rows") != EXPECTED_ALLOCATION_ROWS:
        errors.append("quality_repair.allocation_rows mismatch")
    if quality.get("allocation_manifest") != EXPECTED_ALLOCATION_MANIFEST:
        errors.append("quality_repair.allocation_manifest mismatch")
    if quality.get("contextual_literal_policy") != EXPECTED_CONTEXTUAL_LITERAL_POLICY:
        errors.append("quality_repair.contextual_literal_policy mismatch")
    if quality.get("allocation_policy") != "per_shard_unique_prompt_index_prefix_family":
        errors.append("quality_repair allocation policy mismatch")
    if quality.get("contextual_coordinate_policy") is not True:
        errors.append("quality_repair.contextual_coordinate_policy must be true")
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
            errors.append("quality repair plan did not pass")
    if allocation_manifest.exists():
        allocation_payload = read_json(allocation_manifest, "quality_repair.allocation_manifest", errors)
        if allocation_payload.get("status") != "PASS_DUPLICATE_SAFE_ROW_ALLOCATION_ARTIFACT_ONLY":
            errors.append("allocation manifest did not pass")
        for shard in allocation_payload.get("shard_summaries", []):
            if int(shard.get("duplicate_prompt_prefix_pair_count", -1)) != 0:
                errors.append("allocation manifest has duplicate prompt/prefix pairs")
    if allocation_rows.exists() and count_jsonl(allocation_rows) != int(rows.get("row_count", -1)):
        errors.append("quality_repair.allocation_rows count mismatch")
    if contextual_literal_policy.exists():
        literal_payload = read_json(contextual_literal_policy, "quality_repair.contextual_literal_policy", errors)
        if "coordinate" not in literal_payload.get("contextual_literals", {}):
            errors.append("contextual literal policy must define coordinate")

    precommit = mapping(config.get("precommit"), "precommit", errors)
    surface_bank = path_from(precommit.get("surface_bank", ""), "precommit.surface_bank", errors)
    codebook = path_from(precommit.get("codebook", ""), "precommit.codebook", errors)
    decoder_spec = path_from(precommit.get("decoder_spec", ""), "precommit.decoder_spec", errors)
    decoder_route_config = path_from(precommit.get("decoder_route_config", ""), "precommit.decoder_route_config", errors)
    if precommit.get("contract_id") != "a55e":
        errors.append("precommit.contract_id must be a55e")
    if precommit.get("primary_scrub_mode") != "all":
        errors.append("precommit.primary_scrub_mode must be all")
    if codebook.exists():
        codebook_payload = read_json(codebook, "precommit.codebook", errors)
        if len(codebook_payload.get("selected_coordinates", [])) != 12:
            errors.append("coordinate-pivot codebook must contain 12 selected coordinates")
        if len(codebook_payload.get("pair_to_bit_mapping", [])) != 8:
            errors.append("coordinate-pivot codebook must contain 8 bit pairs")
    if decoder_spec.exists():
        decoder_payload = read_json(decoder_spec, "precommit.decoder_spec", errors)
        if decoder_payload.get("decoder") != "pair_majority_then_checksum":
            errors.append("decoder must be pair_majority_then_checksum")

    generation = mapping(config.get("generation_scope"), "generation_scope", errors)
    if generation.get("conditions") != ["protected", "raw", "task_only"]:
        errors.append("generation conditions mismatch")
    if generation.get("decode_conditions") != ["protected", "raw", "task_only", "wrong_key", "wrong_payload"]:
        errors.append("decode conditions mismatch")
    if generation.get("generation_unit") != "prefix_native_row_cylinder":
        errors.append("generation unit mismatch")
    if int(generation.get("blocks", 0)) != 4 or int(generation.get("prompts_per_block", 0)) != 64:
        errors.append("generation scope must be 4 blocks x 64 prompts")
    if int(generation.get("shards", 0)) != 4 or int(generation.get("rows_per_shard", 0)) != 768:
        errors.append("generation scope must be 4 shards x 768 row-cylinders")
    for field in ("same_contract_only", "controller_only_protected_condition"):
        if generation.get(field) is not True:
            errors.append(f"generation_scope.{field} must be true")
    for field in ("payload_diversity_tested", "llama_tested", "paper_facing"):
        if generation.get(field) is not False:
            errors.append(f"generation_scope.{field} must be false")

    controller = mapping(config.get("controller"), "controller", errors)
    expected_controller = {
        "bonus_nats": 4.0,
        "penalty_nats": 0.5,
        "max_target_mass": 0.5,
        "max_kl_budget": 0.5,
        "policy": "committed",
    }
    for key, expected in expected_controller.items():
        if controller.get(key) != expected:
            errors.append(f"controller.{key} mismatch")

    event_policy = mapping(config.get("event_trace_policy"), "event_trace_policy", errors)
    if event_policy.get("future_positive_requires_token_id_trace") is not True:
        errors.append("event_trace_policy.future_positive_requires_token_id_trace must be true")
    if event_policy.get("text_fallback_for_old_transcripts_only") is not True:
        errors.append("event_trace_policy.text_fallback_for_old_transcripts_only must be true")
    if event_policy.get("decoder") != EXPECTED_EVENT_DECODER:
        errors.append("event_trace_policy.decoder mismatch")
    if event_policy.get("required_generated_output_fields") != REQUIRED_EVENT_FIELDS:
        errors.append("event_trace_policy.required_generated_output_fields mismatch")
    if event_policy.get("contextual_literal_policy") != EXPECTED_CONTEXTUAL_LITERAL_POLICY:
        errors.append("event_trace_policy.contextual_literal_policy mismatch")
    generator_path = path_from(EXPECTED_GENERATOR, "event_trace_policy.generator", errors)
    event_decoder_path = path_from(EXPECTED_EVENT_DECODER, "event_trace_policy.decoder", errors)
    if generator_path.exists():
        generator_text = generator_path.read_text(encoding="utf-8")
        for fragment in ["first_token_event_trace", *REQUIRED_EVENT_FIELDS]:
            if fragment not in generator_text:
                errors.append(f"generator missing event-trace fragment: {fragment}")
        for fragment in ("allocation_rows", "assigned_shard_index", "select_rows_by_allocation"):
            if fragment not in generator_text:
                errors.append(f"generator missing allocation fragment: {fragment}")
    if event_decoder_path.exists():
        event_decoder_text = event_decoder_path.read_text(encoding="utf-8")
        for fragment in (
            "future_positive_requires_token_id_trace",
            "allow_text_fallback_for_old_transcripts",
            "target_first_token_ids",
            "other_first_token_ids",
            "first_generated_token_id",
            "contextual_literal_policy",
        ):
            if fragment not in event_decoder_text:
                errors.append(f"event decoder missing fragment: {fragment}")

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
    if wrapper.exists():
        text = wrapper.read_text(encoding="utf-8")
        for fragment in (
            "#SBATCH --partition=pomplun",
            "#SBATCH --account=cs_yinxin.wan",
            "#SBATCH --qos=pomplun",
            "#SBATCH --gres=gpu:h200:1",
            "#SBATCH --time=30-00:00:00",
            "generate_r4_after_868016_controller_outputs.py",
            "decode_r4_after_864832_reliability_codebook.py",
            "decode_r4_after_868151_first_token_event_channel.py",
            "--allocation-rows \"$ALLOCATION_ROWS\"",
            "--contextual-literal-policy \"$CONTEXTUAL_LITERAL_POLICY\"",
            "VALIDATE_PLAN_ONLY",
            "r4_after_868016_reliability_coordinate_pivot_codebook_precommit_20260516",
            "--controller-bonus-nats \"$CONTROLLER_BONUS_NATS\"",
        ):
            if fragment not in text:
                errors.append(f"wrapper missing fragment: {fragment}")

    gate = mapping(config.get("future_dev_gate"), "future_dev_gate", errors)
    if int(gate.get("protected_accepts_min_format_scrub_all", 0)) < 3:
        errors.append("protected accept gate too low")
    if int(gate.get("control_accepts_max_per_condition", -1)) != 0:
        errors.append("control accepts gate must be zero")
    if int(gate.get("forbidden_public_surface_count_max", -1)) != 0:
        errors.append("forbidden surface gate must be zero")

    prerequisites = mapping(config.get("required_before_any_submission"), "required_before_any_submission", errors)
    for field, value in prerequisites.items():
        if value is not True:
            errors.append(f"required_before_any_submission.{field} must be true")

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

    locked = mapping(config.get("not_unlocked_by_this_route_package"), "not_unlocked_by_this_route_package", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_package.{field} must be true")

    status = (
        "PASS_R4_AFTER_868016_CONTROLLER_GENERATION_ROUTE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_868016_CONTROLLER_GENERATION_ROUTE_VALIDATION_NO_SUBMIT"
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
        "training_started": False,
        "paper_claim_allowed": False,
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 after-868016 controller generation route.")
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

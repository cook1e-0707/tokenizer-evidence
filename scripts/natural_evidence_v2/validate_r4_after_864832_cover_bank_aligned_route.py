from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import load_yaml  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_864832_cover_bank_aligned_objective_route.yaml"
EXPECTED_SCHEMA = "natural_evidence_v2_r4_after_864832_cover_bank_aligned_objective_route_v1"
EXPECTED_PACKAGE = "r4_after_864832_cover_bank_aligned_objective_route_v1"
EXPECTED_FAILURE_STATUS = "FAIL_R4_METRIC_EXACT_864761_DEV_GENERATION_NO_PROTECTED_ACCEPTS_NO_DOWNSTREAM_UNLOCK"
EXPECTED_TRANSFER_STATUS = "PASS_R4_AFTER_864832_TRANSFER_GAP_REPAIR_PACKAGE_ARTIFACT_ONLY_NO_COMPUTE"
EXPECTED_CAUSE = (
    "teacher_forced_prefix_native_pressure_did_not_transfer_to_cover_natural_decoder_surfaces_"
    "and_created_visible_repetition"
)


def _mapping(value: Any, field: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{field} must be a mapping")
        return {}
    return value


def _path(root: Path, value: Any, field: str, errors: list[str]) -> Path:
    path = root / str(value)
    if not path.exists():
        errors.append(f"{field} missing: {path}")
    return path


def _read_json(path: Path, field: str, errors: list[str]) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - validation reports exact exception.
        errors.append(f"{field} is not readable JSON: {exc}")
        return {}
    if not isinstance(payload, Mapping):
        errors.append(f"{field} must be a JSON object")
        return {}
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sha_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    return text.split()[0] if text else ""


def _csv_row_count(path: Path, field: str, errors: list[str]) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return sum(1 for _ in reader)
    except Exception as exc:  # pragma: no cover - validation reports exact exception.
        errors.append(f"{field} is not readable CSV: {exc}")
        return 0


def _expect_false(mapping: Mapping[str, Any], field: str, errors: list[str], prefix: str) -> None:
    if mapping.get(field) is not False:
        errors.append(f"{prefix}.{field} must be false")


def _expect_true(mapping: Mapping[str, Any], field: str, errors: list[str], prefix: str) -> None:
    if mapping.get(field) is not True:
        errors.append(f"{prefix}.{field} must be true")


def validate_route(config: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if config.get("schema_name") != EXPECTED_SCHEMA:
        errors.append("schema_name mismatch")
    if config.get("package_id") != EXPECTED_PACKAGE:
        errors.append("package_id mismatch")

    source = _mapping(config.get("source_failure"), "source_failure", errors)
    review_summary_path = _path(root, source.get("review_summary", ""), "source_failure.review_summary", errors)
    _path(root, source.get("review_doc", ""), "source_failure.review_doc", errors)
    review_summary = _read_json(review_summary_path, "source_failure.review_summary", errors) if review_summary_path.exists() else {}
    if source.get("generation_job_id") != "864832":
        errors.append("source_failure.generation_job_id must be 864832")
    if source.get("teacher_forced_job_id") != "864761":
        errors.append("source_failure.teacher_forced_job_id must be 864761")
    if source.get("failure_status") != EXPECTED_FAILURE_STATUS:
        errors.append("source_failure.failure_status mismatch")
    if review_summary.get("status") != EXPECTED_FAILURE_STATUS:
        errors.append("review summary status mismatch")
    if int(review_summary.get("protected_accepts_format_scrub_all", -1)) != 0:
        errors.append("review summary protected accepts under scrub must be 0")
    if int(review_summary.get("protected_accepts_no_scrub", -1)) != 0:
        errors.append("review summary protected accepts without scrub must be 0")
    if int(review_summary.get("null_accepts_all_modes_all_null_arms", -1)) != 0:
        errors.append("review summary null accepts must be 0")
    if float(review_summary.get("max_protected_vs_raw_shallow_feature_auc", 0.0)) < 0.99:
        errors.append("review summary must preserve shallow AUC leakage evidence")
    if int(review_summary.get("duplicate_response_text_hashes", 0)) <= 0:
        errors.append("review summary must preserve duplicate response hash evidence")
    protected_phrases = _mapping(review_summary.get("phrase_counts_by_condition", {}), "phrase_counts_by_condition", errors).get("protected", {})
    if isinstance(protected_phrases, Mapping):
        if int(protected_phrases.get("Create a plan", 0)) < 10000:
            errors.append("protected Create a plan count should show visible repetition collapse")
        if int(protected_phrases.get("Prepare a", 0)) < 10000:
            errors.append("protected Prepare a count should show visible repetition collapse")
    else:
        errors.append("review summary missing protected phrase counts")

    transfer = _mapping(config.get("transfer_gap_package"), "transfer_gap_package", errors)
    transfer_summary_path = _path(root, transfer.get("summary_json", ""), "transfer_gap_package.summary_json", errors)
    transfer_summary = _read_json(transfer_summary_path, "transfer_gap_package.summary_json", errors) if transfer_summary_path.exists() else {}
    for field in (
        "candidate_pressure_phrase_audit",
        "cover_bank_missing_surface_audit",
        "protected_vs_raw_surface_support_audit",
    ):
        audit_path = _path(root, transfer.get(field, ""), f"transfer_gap_package.{field}", errors)
        if audit_path.exists() and _csv_row_count(audit_path, f"transfer_gap_package.{field}", errors) <= 0:
            errors.append(f"transfer_gap_package.{field} must contain data rows")
    if transfer.get("status") != EXPECTED_TRANSFER_STATUS:
        errors.append("transfer_gap_package.status mismatch")
    if transfer_summary.get("status") != EXPECTED_TRANSFER_STATUS:
        errors.append("transfer package summary status mismatch")
    if transfer.get("cause_classification") != EXPECTED_CAUSE:
        errors.append("transfer_gap_package.cause_classification mismatch")
    if transfer_summary.get("cause_classification") != EXPECTED_CAUSE:
        errors.append("transfer package summary cause_classification mismatch")
    if transfer_summary.get("slurm_submitted") is not False:
        errors.append("transfer package must be artifact-only with slurm_submitted=false")

    route = _mapping(config.get("selected_repair_route"), "selected_repair_route", errors)
    if route.get("name") != "cover_bank_aligned_metric_exact_objective_repair":
        errors.append("selected_repair_route.name mismatch")
    if route.get("target_surfaces_source") != "precommitted_cover_bank_only":
        errors.append("selected_repair_route.target_surfaces_source must be precommitted_cover_bank_only")
    if route.get("same_contract") != "a55e":
        errors.append("selected_repair_route.same_contract must be a55e")
    _expect_false(route, "future_generation_allowed_before_teacher_forced_gate", errors, "selected_repair_route")
    _expect_false(route, "payload_diversity_tested", errors, "selected_repair_route")

    artifacts = _mapping(config.get("precommitted_artifacts"), "precommitted_artifacts", errors)
    surface_bank_path = _path(root, artifacts.get("surface_bank", ""), "precommitted_artifacts.surface_bank", errors)
    codebook_path = _path(root, artifacts.get("codebook", ""), "precommitted_artifacts.codebook", errors)
    decoder_spec_path = _path(root, artifacts.get("decoder_spec", ""), "precommitted_artifacts.decoder_spec", errors)
    precommit_manifest_path = _path(root, artifacts.get("precommit_manifest", ""), "precommitted_artifacts.precommit_manifest", errors)
    for field in ("surface_bank_sha256", "codebook_sha256", "decoder_spec_sha256"):
        sha_path = _path(root, artifacts.get(field, ""), f"precommitted_artifacts.{field}", errors)
        target_field = field.removesuffix("_sha256")
        target_path = root / str(artifacts.get(target_field, ""))
        if sha_path.exists() and target_path.exists() and _read_sha_file(sha_path) != _sha256(target_path):
            errors.append(f"precommitted_artifacts.{field} does not match {target_field}")

    surface_bank = _read_json(surface_bank_path, "precommitted_artifacts.surface_bank", errors) if surface_bank_path.exists() else {}
    codebook = _read_json(codebook_path, "precommitted_artifacts.codebook", errors) if codebook_path.exists() else {}
    decoder_spec = _read_json(decoder_spec_path, "precommitted_artifacts.decoder_spec", errors) if decoder_spec_path.exists() else {}
    _read_json(precommit_manifest_path, "precommitted_artifacts.precommit_manifest", errors) if precommit_manifest_path.exists() else {}
    entries = surface_bank.get("entries", [])
    if not isinstance(entries, list):
        errors.append("surface_bank.entries must be a list")
        entries = []
    if surface_bank.get("contract_id") != artifacts.get("expected_contract_id"):
        errors.append("surface bank contract_id mismatch")
    if int(surface_bank.get("entry_count", len(entries))) != int(artifacts.get("expected_surface_entry_count", -1)):
        errors.append("surface bank entry_count mismatch")
    if len(entries) != int(artifacts.get("expected_surface_entry_count", -1)):
        errors.append("surface bank entries length mismatch")
    if int(surface_bank.get("num_coordinates", -1)) != int(artifacts.get("expected_num_coordinates", -1)):
        errors.append("surface bank num_coordinates mismatch")
    if surface_bank.get("phrase_level") is not True:
        errors.append("surface bank must be phrase-level")
    if surface_bank.get("first_word_only") is not False:
        errors.append("surface bank must not be first-word-only")
    if any(entry.get("not_posthoc_from_853524") is not True for entry in entries if isinstance(entry, Mapping)):
        errors.append("all surface entries must be marked not_posthoc_from_853524=true")
    if decoder_spec.get("primary_reported_scrub_mode") != artifacts.get("expected_primary_scrub_mode"):
        errors.append("decoder spec primary scrub mode mismatch")
    if decoder_spec.get("line_or_step_index_required") is not False:
        errors.append("decoder spec must not require line or step index")
    if decoder_spec.get("posthoc_threshold_changes_allowed") is not False:
        errors.append("decoder spec must forbid posthoc threshold changes")
    if codebook.get("contract_id") != "a55e":
        warnings.append("codebook contract_id is not present or not a55e; verify before future compute")

    posthoc = _mapping(config.get("posthoc_surface_policy"), "posthoc_surface_policy", errors)
    _expect_true(posthoc, "use_864832_transcripts_for_failure_taxonomy_only", errors, "posthoc_surface_policy")
    for field in (
        "add_864832_observed_phrases_to_bank",
        "use_candidate_v3_pressure_phrases_as_decoder_surfaces",
        "use_create_prepare_plan_as_success_surfaces",
        "manual_transcript_phrase_mining_allowed",
        "lower_accept_support_margin_gates_allowed",
    ):
        _expect_false(posthoc, field, errors, "posthoc_surface_policy")

    objective = _mapping(config.get("future_objective_contract"), "future_objective_contract", errors)
    for field in (
        "optimize_exact_decoder_surface_token_sets",
        "logsumexp_over_target_token_sets",
        "target_mass_floor_required",
        "anti_repetition_controls_required",
        "naturalness_anchor_required",
        "base_task_only_scoring_paths_unchanged",
    ):
        _expect_true(objective, field, errors, "future_objective_contract")
    for field in ("use_human_readable_labels_as_token_targets", "task_only_protected_pressure_allowed"):
        _expect_false(objective, field, errors, "future_objective_contract")

    tf_gate = _mapping(config.get("future_teacher_forced_gate"), "future_teacher_forced_gate", errors)
    if float(tf_gate.get("protected_lift_vs_base_min", 0.0)) < 0.15:
        errors.append("future_teacher_forced_gate.protected_lift_vs_base_min must be >= 0.15")
    if float(tf_gate.get("protected_lift_vs_task_only_min", 0.0)) < 0.10:
        errors.append("future_teacher_forced_gate.protected_lift_vs_task_only_min must be >= 0.10")
    if float(tf_gate.get("protected_rank1_min", 0.0)) < 0.75:
        errors.append("future_teacher_forced_gate.protected_rank1_min must be >= 0.75")
    if int(tf_gate.get("scorer_boundary_failures_max", -1)) != 0:
        errors.append("future_teacher_forced_gate.scorer_boundary_failures_max must be 0")
    if float(tf_gate.get("target_other_overlap_rate_max", 1.0)) != 0.0:
        errors.append("future_teacher_forced_gate.target_other_overlap_rate_max must be 0")
    _expect_false(tf_gate, "task_only_lift_anomaly_allowed", errors, "future_teacher_forced_gate")
    _expect_false(tf_gate, "visible_repetition_collapse_allowed", errors, "future_teacher_forced_gate")

    gen_gate = _mapping(config.get("future_generation_gate"), "future_generation_gate", errors)
    if gen_gate.get("primary_decode_format_scrub") != "all":
        errors.append("future_generation_gate.primary_decode_format_scrub must be all")
    if int(gen_gate.get("protected_accepts_format_scrub_all_min_dev_32", 0)) < 26:
        errors.append("future_generation_gate protected accept gate too low")
    if float(gen_gate.get("shallow_structural_auc_max", 1.0)) > 0.60:
        errors.append("future_generation_gate.shallow_structural_auc_max must be <= 0.60")
    for field in (
        "raw_accepts_max",
        "task_only_accepts_max",
        "wrong_key_accepts_max",
        "wrong_payload_accepts_max",
        "forbidden_public_technical_surface_max",
        "duplicate_generated_output_hashes_max",
        "duplicate_decode_row_hashes_max",
    ):
        if int(gen_gate.get(field, -1)) != 0:
            errors.append(f"future_generation_gate.{field} must be 0")

    compute = _mapping(config.get("future_compute_policy"), "future_compute_policy", errors)
    if compute.get("partition") != "pomplun" or compute.get("qos") != "pomplun":
        errors.append("future_compute_policy must use pomplun partition/qos")
    if compute.get("account") != "cs_yinxin.wan":
        errors.append("future_compute_policy.account must be cs_yinxin.wan")
    if compute.get("gres") != "gpu:h200:1":
        errors.append("future_compute_policy.gres must be gpu:h200:1")
    if compute.get("max_time") != "30-00:00:00":
        errors.append("future_compute_policy.max_time must be 30-00:00:00")
    for field in (
        "allowlist_enabled_now",
        "slurm_allowed_by_this_package",
        "model_scoring_allowed_by_this_package",
        "training_allowed_by_this_package",
        "generation_allowed_by_this_package",
    ):
        _expect_false(compute, field, errors, "future_compute_policy")
    for field in (
        "exactly_one_submission_when_unlocked",
        "remote_hash_preflight_required",
        "hermes_notification_required",
        "post_submit_allowlist_shutdown_required",
    ):
        _expect_true(compute, field, errors, "future_compute_policy")

    locked = _mapping(config.get("not_unlocked_by_this_route_package"), "not_unlocked_by_this_route_package", errors)
    for field, value in locked.items():
        if value is not True:
            errors.append(f"not_unlocked_by_this_route_package.{field} must be true")

    next_tasks = config.get("required_next_artifact_only_tasks", [])
    if not isinstance(next_tasks, list) or len(next_tasks) < 4:
        errors.append("required_next_artifact_only_tasks must list the next artifact-only prerequisites")

    status = (
        "PASS_R4_AFTER_864832_COVER_BANK_ALIGNED_ROUTE_VALIDATION_NO_COMPUTE"
        if not errors
        else "FAIL_R4_AFTER_864832_COVER_BANK_ALIGNED_ROUTE_VALIDATION_NO_COMPUTE"
    )
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "package_id": config.get("package_id"),
        "source_generation_job_id": source.get("generation_job_id"),
        "source_teacher_forced_job_id": source.get("teacher_forced_job_id"),
        "selected_repair_route": route.get("name"),
        "target_surfaces_source": route.get("target_surfaces_source"),
        "surface_bank_entry_count": len(entries),
        "surface_bank_coordinates": surface_bank.get("num_coordinates"),
        "primary_decode_format_scrub": decoder_spec.get("primary_reported_scrub_mode"),
        "current_compute_unlocked": False,
        "allowlist_enabled": False,
        "slurm_job_submitted": False,
        "training_started": False,
        "generation_started": False,
        "model_scoring_started": False,
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the R4 after-864832 cover-bank-aligned route package.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_route(load_yaml(args.config))
    if args.output_dir is not None:
        write_json_new(args.output_dir / "cover_bank_aligned_route_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

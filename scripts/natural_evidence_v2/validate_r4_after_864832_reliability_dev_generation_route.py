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


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_after_864832_reliability_dev_generation_route.yaml"
ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
EXPECTED_ENTRY = "v2_r4_after_864832_reliability_dev_generation_h200"
EXPECTED_WRAPPER = "scripts/natural_evidence_v2/slurm/r4_after_864832_reliability_dev_generation_h200.sbatch"


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
    if config.get("schema_name") != "natural_evidence_v2_r4_after_864832_reliability_dev_generation_route_v1":
        errors.append("schema_name mismatch")
    if config.get("route_id") != "r4_after_864832_reliability_codebook_dev_generation_v1":
        errors.append("route_id mismatch")

    source_oracle = mapping(config.get("source_oracle"), "source_oracle", errors)
    oracle_summary_path = path_from(source_oracle.get("summary", ""), "source_oracle.summary", errors)
    oracle_summary = read_json(oracle_summary_path, "source_oracle.summary", errors) if oracle_summary_path.exists() else {}
    if oracle_summary.get("status") != source_oracle.get("expected_status"):
        errors.append("source oracle status mismatch")
    if int(oracle_summary.get("wrong_payload_accepts", -1)) != int(source_oracle.get("wrong_payload_accepts", -2)):
        errors.append("source oracle wrong_payload_accepts mismatch")
    if int(oracle_summary.get("wrong_key_accepts", -1)) != int(source_oracle.get("wrong_key_accepts", -2)):
        errors.append("source oracle wrong_key_accepts mismatch")

    precommit = mapping(config.get("precommit"), "precommit", errors)
    surface_bank = path_from(precommit.get("surface_bank", ""), "precommit.surface_bank", errors)
    codebook = path_from(precommit.get("codebook", ""), "precommit.codebook", errors)
    decoder_spec = path_from(precommit.get("decoder_spec", ""), "precommit.decoder_spec", errors)
    route_config = path_from(precommit.get("route_config", ""), "precommit.route_config", errors)
    if precommit.get("contract_id") != "a55e":
        errors.append("precommit.contract_id must be a55e")
    if precommit.get("primary_scrub_mode") != "all":
        errors.append("precommit.primary_scrub_mode must be all")
    if codebook.exists():
        codebook_payload = read_json(codebook, "precommit.codebook", errors)
        if len(codebook_payload.get("pair_to_bit_mapping", [])) != 8:
            errors.append("reliability codebook must contain 8 bit pairs")
        if len(codebook_payload.get("selected_coordinates", [])) != 16:
            errors.append("reliability codebook must contain 16 selected coordinates")
    if decoder_spec.exists():
        decoder_payload = read_json(decoder_spec, "precommit.decoder_spec", errors)
        if decoder_payload.get("decoder") != "pair_majority_then_checksum":
            errors.append("decoder must be pair_majority_then_checksum")
    if surface_bank.exists():
        surface_payload = read_json(surface_bank, "precommit.surface_bank", errors)
        if surface_payload.get("coordinate_identifiable_by_surface") is not True:
            errors.append("surface bank must be coordinate-identifiable")
        if int(surface_payload.get("entry_count", 0)) < 128:
            errors.append("surface bank must contain at least 128 entries")

    prompt_source = mapping(config.get("prompt_source"), "prompt_source", errors)
    prompts = path_from(prompt_source.get("prompts_jsonl", ""), "prompt_source.prompts_jsonl", errors)
    if prompts.exists() and count_jsonl(prompts) != int(prompt_source.get("prompt_count", -1)):
        errors.append("prompt count mismatch")

    scope = mapping(config.get("generation_scope"), "generation_scope", errors)
    if scope.get("conditions") != ["protected", "raw", "task_only"]:
        errors.append("generation conditions mismatch")
    if scope.get("decode_conditions") != ["protected", "raw", "task_only", "wrong_key", "wrong_payload"]:
        errors.append("decode conditions mismatch")
    if int(scope.get("blocks", 0)) != 32 or int(scope.get("prompts_per_block", 0)) != 64:
        errors.append("generation scope must be 32 blocks x 64 prompts")
    if int(scope.get("shards", 0)) != 4 or int(scope.get("prompts_per_shard", 0)) != 512:
        errors.append("generation scope must be 4 shards x 512 prompts")
    for field in ("same_contract_only",):
        if scope.get(field) is not True:
            errors.append(f"generation_scope.{field} must be true")
    for field in ("payload_diversity_tested", "llama_tested", "paper_facing"):
        if scope.get(field) is not False:
            errors.append(f"generation_scope.{field} must be false")

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
            "decode_r4_after_864832_reliability_codebook.py",
            "VALIDATE_PLAN_ONLY",
            "r4_after_864832_reliability_weighted_codebook_precommit_20260516",
            "r4_after_864832_coordinate_unique_surface_bank_20260516/surface_bank.json",
        ):
            if fragment not in text:
                errors.append(f"wrapper missing fragment: {fragment}")

    gate = mapping(config.get("future_dev_gate"), "future_dev_gate", errors)
    if int(gate.get("protected_accepts_min_format_scrub_all", 0)) < 26:
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
        "PASS_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_ROUTE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_864832_RELIABILITY_DEV_GENERATION_ROUTE_VALIDATION_NO_SUBMIT"
    )
    return {
        "status": status,
        "errors": errors,
        "allowlist_entry": EXPECTED_ENTRY,
        "allow_submission_enabled_entry": allow_submission_enabled_entry,
        "wrapper": EXPECTED_WRAPPER,
        "surface_bank_sha256": sha256_file(surface_bank) if surface_bank.exists() else "",
        "codebook_sha256": sha256_file(codebook) if codebook.exists() else "",
        "decoder_spec_sha256": sha256_file(decoder_spec) if decoder_spec.exists() else "",
        "route_config_sha256": sha256_file(route_config) if route_config.exists() else "",
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
    parser = argparse.ArgumentParser(description="Validate R4 after-864832 reliability dev generation route.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-submission-enabled-entry",
        action="store_true",
        help="Permit exactly the reviewed submission entry to be enabled while the submitted wrapper is starting.",
    )
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

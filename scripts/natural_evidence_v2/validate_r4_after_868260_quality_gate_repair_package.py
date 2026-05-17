from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_FAILURE = (
    ROOT
    / "results/natural_evidence_v2/status/r4_after_868212_repaired_first_token_event_generation_868260_failure_analysis/failure_analysis_summary.json"
)
DEFAULT_PACKAGE = (
    ROOT / "results/natural_evidence_v2/precommit/r4_after_868260_quality_gate_repair_package_20260517"
)
DEFAULT_OUTPUT = (
    ROOT / "results/natural_evidence_v2/status/r4_after_868260_quality_gate_repair_package_validation_20260517"
)


def read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def list_value(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def validate_package(*, failure: Mapping[str, Any], package_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    policy_path = package_dir / "contextual_forbidden_surface_policy_v2.json"
    duplicate_path = package_dir / "duplicate_safe_generation_policy.json"
    manifest_path = package_dir / "repair_manifest.json"
    for path in (policy_path, duplicate_path, manifest_path):
        if not path.exists():
            errors.append(f"missing artifact: {path}")

    policy = read_json(policy_path) if policy_path.exists() else {}
    duplicate = read_json(duplicate_path) if duplicate_path.exists() else {}
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    if failure.get("source_job_id") != "868260":
        errors.append("failure source_job_id must be 868260")
    if failure.get("strict_protected_accepts") != 2:
        errors.append("failure strict_protected_accepts must remain 2")
    if failure.get("protected_accepts_ignoring_quality") != 4:
        errors.append("failure protected_accepts_ignoring_quality must remain 4")
    controls = failure.get("control_accepts", {})
    if not isinstance(controls, Mapping) or any(int(controls.get(arm, -1)) != 0 for arm in ("raw", "task_only", "wrong_key", "wrong_payload")):
        errors.append("failure control accepts must remain zero")

    hard_forbid = set(list_value(policy, "hard_forbid_literals"))
    required_hard = {"fingerprint", "watermark", "payload", "secret key", "decoder", "hidden signal"}
    if not required_hard.issubset(hard_forbid):
        errors.append("contextual policy missing required hard forbids")
    if "bucket" in hard_forbid:
        errors.append("contextual policy v2 must not hard-forbid ordinary literal bucket")
    contextual = policy.get("contextual_literals", {})
    bucket = contextual.get("bucket", {}) if isinstance(contextual, Mapping) else {}
    if not isinstance(bucket, Mapping):
        errors.append("contextual policy must define bucket as contextual literal")
    else:
        if bucket.get("ordinary_domain_allowed") is not True:
            errors.append("bucket ordinary_domain_allowed must be true")
        technical_cues = set(str(item) for item in bucket.get("technical_cues", []))
        for cue in ("bit", "codeword", "coordinate", "decoder", "hidden", "payload", "secret", "slot", "token id", "watermark"):
            if cue not in technical_cues:
                errors.append(f"bucket contextual technical cue missing: {cue}")
    final_gate = policy.get("final_generation_gate", {})
    if not isinstance(final_gate, Mapping) or int(final_gate.get("technical_public_literal_count_max", -1)) != 0:
        errors.append("technical_public_literal_count_max must be 0")
    if policy.get("reclassifies_868260") is not False:
        errors.append("contextual policy must not reclassify 868260")

    if duplicate.get("source_job_id") != "868260":
        errors.append("duplicate policy source_job_id must be 868260")
    if duplicate.get("reclassifies_868260") is not False:
        errors.append("duplicate policy must not reclassify 868260")
    future_gate = duplicate.get("future_generation_gate", {})
    if not isinstance(future_gate, Mapping):
        errors.append("future_generation_gate must be an object")
    else:
        if int(future_gate.get("within_block_duplicate_response_hash_count_max", -1)) != 0:
            errors.append("within-block duplicate gate must remain 0")
        if int(future_gate.get("global_duplicate_response_hash_count_max", -1)) != 0:
            errors.append("global duplicate gate must remain 0")
    if duplicate.get("requires_new_reviewed_route_before_slurm") is not True:
        errors.append("duplicate policy must require a new reviewed route before Slurm")
    if duplicate.get("allow_slurm_submission") is not False:
        errors.append("duplicate policy package must not allow Slurm submission")

    if manifest.get("status") != "PRECOMMITTED_R4_AFTER_868260_QUALITY_GATE_REPAIR_PACKAGE_ARTIFACT_ONLY_NO_SUBMIT":
        errors.append("repair manifest status mismatch")
    if manifest.get("source_job_id") != "868260":
        errors.append("repair manifest source_job_id must be 868260")
    if manifest.get("reclassifies_868260") is not False:
        errors.append("repair manifest must not reclassify 868260")
    if manifest.get("slurm_allowed") is not False:
        errors.append("repair manifest must not allow Slurm")

    status = (
        "PASS_R4_AFTER_868260_QUALITY_GATE_REPAIR_PACKAGE_VALIDATION_NO_SUBMIT"
        if not errors
        else "FAIL_R4_AFTER_868260_QUALITY_GATE_REPAIR_PACKAGE_VALIDATION_NO_SUBMIT"
    )
    return {
        "schema_name": "natural_evidence_v2_r4_after_868260_quality_gate_repair_package_validation_v1",
        "status": status,
        "errors": errors,
        "package_dir": str(package_dir),
        "source_job_id": "868260",
        "reclassifies_868260": False,
        "slurm_allowed": False,
        "generation_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Plan-only route/wrapper update may consume this repair package; "
            "no Slurm until a new reviewed rerun route passes local and remote preflight."
        ),
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R4 after-868260 quality-gate repair package.")
    parser.add_argument("--failure-summary", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_package(failure=read_json(args.failure_summary), package_dir=args.package_dir)
    write_json_new(args.output_dir / "repair_package_validation_summary.json", summary)
    report = [
        "# R4 After-868260 Quality-Gate Repair Package Validation",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- package: `{summary['package_dir']}`",
        f"- reclassifies 868260: `{summary['reclassifies_868260']}`",
        f"- slurm allowed: `{summary['slurm_allowed']}`",
        "",
        "Next allowed action:",
        "",
        summary["next_allowed_action"],
    ]
    if summary["errors"]:
        report.extend(["", "## Errors", ""])
        report.extend(f"- {error}" for error in summary["errors"])
    (args.output_dir / "repair_package_validation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

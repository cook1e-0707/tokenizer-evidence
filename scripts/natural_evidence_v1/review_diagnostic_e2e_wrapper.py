from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import read_yaml, resolve_repo_path, write_json


REQUIRED_ARMS = {
    "qwen_protected",
    "qwen_raw",
    "qwen_task_only_lora",
    "wrong_key",
    "wrong_payload",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review the Qwen diagnostic high-risk E2E wrapper without launching "
            "training. This is a CPU control-plane check."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--wrapper", default="scripts/natural_evidence_v1/slurm/qwen_diagnostic_high_risk_e2e_pilot.sbatch")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def _payload_count(config: dict[str, Any]) -> int:
    payloads = config.get("payloads", [])
    return len(payloads) if isinstance(payloads, list) else 0


def _seed_count(config: dict[str, Any]) -> int:
    scale = config.get("diagnostic_high_risk_pilot_scale", {})
    seeds = scale.get("seeds", []) if isinstance(scale, dict) else []
    return len(seeds) if isinstance(seeds, list) else 0


def _query_budgets(config: dict[str, Any]) -> list[int]:
    scale = config.get("diagnostic_high_risk_pilot_scale", {})
    budgets = scale.get("query_budgets", []) if isinstance(scale, dict) else []
    return [int(value) for value in budgets] if isinstance(budgets, list) else []


def _contains_all(text: str, needles: set[str]) -> list[str]:
    return sorted(needle for needle in needles if needle not in text)


def _gpu_allowlist_enabled(root: Path) -> bool:
    allowlist_path = root / "configs/natural_evidence_v1/run_allowlist.yaml"
    if not allowlist_path.exists():
        return False
    allowlist = read_yaml(allowlist_path)
    for action in allowlist.get("allowed_gpu_actions", []):
        if not isinstance(action, dict):
            continue
        if action.get("name") == "qwen_diagnostic_high_risk_e2e_pilot":
            return bool(action.get("enabled", False))
    return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = resolve_repo_path(args.config, root)
    wrapper_path = resolve_repo_path(args.wrapper, root)
    config = read_yaml(config_path)
    wrapper_text = wrapper_path.read_text(encoding="utf-8") if wrapper_path.exists() else ""
    scale = dict(config.get("diagnostic_high_risk_pilot_scale", {}))
    e2e = dict(config.get("end_to_end_pilot", {}))
    capacity = dict(dict(config.get("bucket_bank", {})).get("compatibility_adjusted_capacity", {}))
    diagnostic_gate = dict(capacity.get("diagnostic_high_risk_gate", {}))
    errors: list[str] = []
    warnings: list[str] = []

    if not wrapper_path.exists():
        errors.append("missing qwen diagnostic high-risk E2E wrapper")
    if scale.get("status") != "diagnostic_high_risk":
        errors.append("diagnostic scale status must be diagnostic_high_risk")
    if scale.get("paper_claim_allowed", True):
        errors.append("diagnostic scale must forbid paper claims")
    if e2e.get("diagnostic_high_risk_claim_status") != "forbidden_from_paper_claims":
        errors.append("diagnostic E2E claim status must remain forbidden_from_paper_claims")
    if not e2e.get("paper_ready_density_gate_remains_required", False):
        errors.append("paper-ready density gate must remain required")
    if _query_budgets(config) != [64, 128, 256, 512]:
        errors.append("diagnostic query budgets must be exactly [64, 128, 256, 512]")
    if int(scale.get("eval_owner_probes", 0)) < 2048:
        errors.append("diagnostic eval_owner_probes must be >= 2048")
    if int(scale.get("organic_null_prompts", 0)) < 2048:
        errors.append("diagnostic organic_null_prompts must be >= 2048")
    if _payload_count(config) < 2:
        errors.append("diagnostic pilot requires at least two payloads")
    if _seed_count(config) < 2:
        errors.append("diagnostic pilot requires at least two seeds")
    if diagnostic_gate.get("paper_claim_allowed", True):
        errors.append("diagnostic gate must forbid paper claims")

    missing_arms = _contains_all(wrapper_text, REQUIRED_ARMS)
    if missing_arms:
        errors.append(f"wrapper missing required arms: {missing_arms}")
    for required_text in (
        "DIAGNOSTIC_HIGH_RISK",
        "DRY_RUN_ONLY",
        "START_DIAGNOSTIC_E2E",
        "NO_PAPER_CLAIM",
        "Qwen/Qwen2.5-7B-Instruct",
        "64,128,256,512",
        'EVAL_OWNER_PROBES="2048"',
        'ORGANIC_NULL_PROMPTS="2048"',
    ):
        if required_text not in wrapper_text:
            errors.append(f"wrapper missing required text {required_text!r}")
    if "sbatch scripts/natural_evidence_v1/slurm/llama" in wrapper_text.lower():
        errors.append("wrapper must not launch Llama")
    if "bucket-count 8" in wrapper_text or "8way" in wrapper_text.lower():
        errors.append("wrapper must not launch 8-way main")
    if "scripts/train.py" in wrapper_text:
        warnings.append("wrapper must not use old compiled-slot scripts/train.py for natural E2E")

    natural_trainer_status = "MISSING"
    natural_trainer_path = root / "scripts/natural_evidence_v1/train_natural_bucket_lora.py"
    if natural_trainer_path.exists():
        trainer_text = natural_trainer_path.read_text(encoding="utf-8")
        missing_trainer_text = _contains_all(
            trainer_text,
            {
                "natural_transcript_bucket_mass",
                "--start-training",
                "--budget-cap",
                "--prompt-split-id",
                "NO_PAPER_CLAIM",
                "eligible_positions",
                "target_bucket_token_ids",
                "bucket_to_token_ids",
                "PRESENT_REVIEWED_DRY_RUN_READY",
            },
        )
        if missing_trainer_text:
            natural_trainer_status = "PRESENT_UNREVIEWED"
            warnings.append(f"natural trainer missing review markers: {missing_trainer_text}")
        else:
            natural_trainer_status = "PRESENT_REVIEWED_DRY_RUN_READY"
    gpu_allowlist_enabled = _gpu_allowlist_enabled(root)
    launch_ready = (
        not errors
        and natural_trainer_status == "PRESENT_REVIEWED_DRY_RUN_READY"
        and gpu_allowlist_enabled
    )
    if errors:
        status = "FAIL_WRAPPER_REVIEW"
    elif natural_trainer_status == "MISSING":
        status = "PASS_WRAPPER_REVIEW_NOT_LAUNCH_READY"
    elif natural_trainer_status == "PRESENT_REVIEWED_DRY_RUN_READY" and not gpu_allowlist_enabled:
        status = "PASS_WRAPPER_AND_TRAINER_REVIEW_GPU_ALLOWLIST_DISABLED"
    elif launch_ready:
        status = "PASS_WRAPPER_AND_TRAINER_REVIEW_READY_FOR_EXPLICIT_SUBMISSION"
    else:
        status = "PASS_WRAPPER_REVIEW_NOT_LAUNCH_READY"
    summary = {
        "schema_name": "natural_evidence_qwen_diagnostic_e2e_wrapper_review_v1",
        "config": str(config_path),
        "wrapper": str(wrapper_path),
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "diagnostic_high_risk": True,
        "paper_claim_allowed": False,
        "launch_ready": launch_ready,
        "natural_trainer_status": natural_trainer_status,
        "gpu_allowlist_enabled": gpu_allowlist_enabled,
        "gpu_needed_for_actual_launch": True,
        "no_training_started": True,
        "required_arms": sorted(REQUIRED_ARMS),
        "query_budgets": _query_budgets(config),
        "payload_count": _payload_count(config),
        "seed_count": _seed_count(config),
        "eval_owner_probes": int(scale.get("eval_owner_probes", 0)),
        "organic_null_prompts": int(scale.get("organic_null_prompts", 0)),
        "result_claim": "wrapper_review_not_payload_recovery",
        "next_minimal_action": (
            "Compile/review the diagnostic natural training dataset or run diagnostic organic density; "
            "do not launch training until GPU allowlist and trainer preflight artifacts are reviewed."
        ),
    }
    write_json(resolve_repo_path(args.output_json, root), summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

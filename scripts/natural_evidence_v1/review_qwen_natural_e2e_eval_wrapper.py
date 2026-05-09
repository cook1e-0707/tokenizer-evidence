from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

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
            "Review the Qwen natural-output variable-radix five-arm eval wrapper. "
            "This is a control-plane dry-run review only: no training, no model "
            "generation, no payload-recovery claim, and no FAR aggregation."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--allowlist", default="configs/natural_evidence_v1/run_allowlist.yaml")
    parser.add_argument("--wrapper", default="scripts/natural_evidence_v1/slurm/qwen_natural_e2e_eval.sbatch")
    parser.add_argument("--evaluator", default="scripts/natural_evidence_v1/evaluate_qwen_natural_e2e.py")
    parser.add_argument("--training-wrapper-review", default="")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def _payload_count(config: Mapping[str, Any]) -> int:
    payloads = config.get("payloads", [])
    return len(payloads) if isinstance(payloads, list) else 0


def _seed_count(config: Mapping[str, Any]) -> int:
    scale = config.get("diagnostic_high_risk_pilot_scale", {})
    seeds = scale.get("seeds", []) if isinstance(scale, dict) else []
    return len(seeds) if isinstance(seeds, list) else 0


def _query_budgets(config: Mapping[str, Any]) -> list[int]:
    scale = config.get("diagnostic_high_risk_pilot_scale", {})
    budgets = scale.get("query_budgets", []) if isinstance(scale, dict) else []
    return [int(value) for value in budgets] if isinstance(budgets, list) else []


def _contains_all(text: str, needles: set[str]) -> list[str]:
    return sorted(needle for needle in needles if needle not in text)


def _allowlist_action(allowlist: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        actions = allowlist.get(section, [])
        if not isinstance(actions, list):
            continue
        for action in actions:
            if isinstance(action, dict) and action.get("name") == name:
                return dict(action)
    return None


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = resolve_repo_path(args.config, root)
    allowlist_path = resolve_repo_path(args.allowlist, root)
    wrapper_path = resolve_repo_path(args.wrapper, root)
    evaluator_path = resolve_repo_path(args.evaluator, root)
    config = read_yaml(config_path)
    allowlist = read_yaml(allowlist_path)
    wrapper_text = wrapper_path.read_text(encoding="utf-8") if wrapper_path.exists() else ""
    evaluator_text = evaluator_path.read_text(encoding="utf-8") if evaluator_path.exists() else ""
    training_review = _read_optional_json(resolve_repo_path(args.training_wrapper_review, root)) if args.training_wrapper_review else {}

    errors: list[str] = []
    warnings: list[str] = []
    if not wrapper_path.exists():
        errors.append("missing qwen natural E2E eval wrapper")
    if not evaluator_path.exists():
        errors.append("missing qwen natural E2E evaluator preflight script")

    missing_arms = _contains_all(wrapper_text, REQUIRED_ARMS)
    if missing_arms:
        errors.append(f"wrapper missing required arms: {missing_arms}")
    for required in (
        "#SBATCH --partition=DGXA100",
        "#SBATCH --account=pi_yinxin.wan",
        "#SBATCH --qos=scavenger_unlim",
        "#SBATCH --gres=gpu:A100:1",
        "#SBATCH --mail-type=ALL",
        "#SBATCH --mail-user=guanjie.lin001@umb.edu",
        "QWEN_NATURAL_VARIABLE_RADIX_FIVE_ARM_EVAL_PREFLIGHT",
        "NO_PAPER_CLAIM",
        'ARMS="qwen_protected,qwen_raw,qwen_task_only_lora,wrong_key,wrong_payload"',
        'TRAINED_ARMS="qwen_protected,qwen_task_only_lora"',
        'QUERY_BUDGETS="${QUERY_BUDGETS:-64,128,256,512}"',
        'EVAL_OWNER_PROBES="${EVAL_OWNER_PROBES:-2048}"',
        'ORGANIC_NULL_PROMPTS="${ORGANIC_NULL_PROMPTS:-2048}"',
        'DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"',
        'START_QWEN_NATURAL_E2E_EVAL="${START_QWEN_NATURAL_E2E_EVAL:-0}"',
        "evaluate_qwen_natural_e2e.py",
        "--start-eval",
        "--require-cuda",
        "QWEN_NATURAL_E2E_EVAL_OUTPUT_EXISTS_REFUSING_OVERWRITE",
    ):
        if required not in wrapper_text:
            errors.append(f"wrapper missing required text {required!r}")
    for forbidden in (
        "train_natural_bucket_lora.py",
        "evaluate_diagnostic_e2e.py",
        "scripts/train.py",
        "qwen_diagnostic_high_risk_e2e_pilot",
        "llama",
        "8way",
        "START_QWEN_NATURAL_E2E=\"1\"",
    ):
        if forbidden in wrapper_text.lower():
            errors.append(f"wrapper contains forbidden text {forbidden!r}")

    for required in (
        "PASS_DRY_RUN_READY_FOR_POST_TRAINING_EVAL",
        "variable_radix",
        "prompt_id_token_index_variable_radix",
        "qwen_protected",
        "qwen_raw",
        "qwen_task_only_lora",
        "wrong_key",
        "wrong_payload",
        "_decoder_dependency_errors",
        "inspect.signature",
        "decoder_mode=\"variable_radix\"",
    ):
        if required not in evaluator_text:
            errors.append(f"evaluator missing required text {required!r}")
    for forbidden in ("scripts/train.py",):
        if forbidden in evaluator_text.lower():
            errors.append(f"evaluator contains forbidden text {forbidden!r}")

    if _query_budgets(config) != [64, 128, 256, 512]:
        errors.append("diagnostic query budgets must be exactly [64, 128, 256, 512]")
    scale = dict(config.get("diagnostic_high_risk_pilot_scale", {}))
    if int(scale.get("eval_owner_probes", 0)) < 2048:
        errors.append("eval_owner_probes must be >= 2048")
    if int(scale.get("organic_null_prompts", 0)) < 2048:
        errors.append("organic_null_prompts must be >= 2048")
    if _payload_count(config) < 2:
        errors.append("five-arm eval requires at least two payloads")
    if _seed_count(config) < 2:
        errors.append("five-arm eval requires at least two seeds")

    action = _allowlist_action(allowlist, "qwen_natural_e2e_eval")
    if action is None:
        errors.append("allowlist missing disabled qwen_natural_e2e_eval action")
    else:
        if action.get("enabled") is not False:
            errors.append("qwen_natural_e2e_eval allowlist action must remain disabled until eval launch approval")
        if action.get("command_pattern") != "sbatch scripts/natural_evidence_v1/slurm/qwen_natural_e2e_eval.sbatch":
            errors.append("qwen_natural_e2e_eval allowlist command pattern mismatch")

    if training_review:
        if training_review.get("status") != "PASS_DRY_RUN_PREFLIGHT_NOT_TRAINING":
            warnings.append(
                "latest training wrapper dry-run review is not PASS_DRY_RUN_PREFLIGHT_NOT_TRAINING"
            )
        if training_review.get("training_started") is not False:
            errors.append("training wrapper dry-run review says training started")

    status = "PASS_FIVE_ARM_EVAL_WRAPPER_DRY_RUN_REVIEW_NOT_EVAL" if not errors else "FAIL_FIVE_ARM_EVAL_WRAPPER_REVIEW"
    summary = {
        "schema_name": "natural_evidence_qwen_natural_e2e_eval_wrapper_review_v1",
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "paper_claim_allowed": False,
        "training_started": False,
        "e2e_eval_started": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "launch_ready": False,
        "wrapper": str(wrapper_path),
        "evaluator": str(evaluator_path),
        "required_arms": sorted(REQUIRED_ARMS),
        "query_budgets": _query_budgets(config),
        "payload_count": _payload_count(config),
        "seed_count": _seed_count(config),
        "eval_owner_probes": int(scale.get("eval_owner_probes", 0)),
        "organic_null_prompts": int(scale.get("organic_null_prompts", 0)),
        "allowlist_enabled": bool(action.get("enabled", False)) if action else None,
        "result_claim": "five_arm_eval_wrapper_review_not_payload_recovery",
        "next_allowed_action": (
            "If this review passes, sync the wrapper to Chimera and run one "
            "DRY_RUN_ONLY=1 Slurm preflight. Do not start training or generation."
        ),
    }
    write_json(resolve_repo_path(args.output_json, root), summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

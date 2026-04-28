from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from scripts.prepare_g3a_v3_block_scale import (
    _entry_with_identity,
    _entry_with_repo_relative_config,
    _load_yaml,
    _repo_relative_path,
    _resolve_path,
    _save_manifest,
)
from src.infrastructure.manifest import ManifestEntry, build_manifest_from_config
from src.infrastructure.paths import current_timestamp, discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare B1/B2 matched-budget baseline manifests.")
    parser.add_argument("--package-config", default="configs/reporting/matched_budget_baselines_v1.yaml")
    parser.add_argument(
        "--output",
        default="results/processed/paper_stats/baseline_package_dry_run.json",
    )
    parser.add_argument(
        "--train-manifest-out",
        default="manifests/matched_budget_baselines/train_manifest.json",
    )
    parser.add_argument(
        "--eval-manifest-out",
        default="manifests/matched_budget_baselines/eval_manifest.json",
    )
    parser.add_argument(
        "--calibration-summary",
        default="results/processed/paper_stats/baseline_calibration_summary.json",
        help="Frozen B0 calibration summary used to set final eval thresholds.",
    )
    parser.add_argument(
        "--output-root-base",
        help="Optional base directory for baseline case roots. Defaults to EXP_SCRATCH/matched_budget_baselines_v1.",
    )
    parser.add_argument(
        "--environment-setup",
        help="Optional runtime environment setup block. Defaults to CHIMERA_ENV_SETUP when set.",
    )
    return parser.parse_args()


def _resolve_output_root_base(package_config: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(Path(os.path.expandvars(explicit)).as_posix())
    prefix = str(package_config["new_case_root_prefix"])
    exp_scratch = os.environ.get("EXP_SCRATCH")
    if exp_scratch:
        return str((Path(exp_scratch) / prefix).as_posix())
    print(
        "WARNING: EXP_SCRATCH is not set; falling back to the package-relative "
        "new_case_root_prefix. Set EXP_SCRATCH on Chimera so baseline outputs stay off home.",
        file=sys.stderr,
    )
    return prefix


def _case_id(method: dict[str, Any], payload: str, seed: int) -> str:
    return f"{method['slug']}_{payload}_s{seed}"


def _case_root(prefix: str, method_slug: str, payload: str, seed: int) -> str:
    return str((Path(prefix) / "final" / method_slug / f"{payload}_s{seed}").as_posix())


def _variant_name(case: dict[str, Any]) -> str:
    return f"matched-budget-baseline-{case['method_slug']}"


def _load_frozen_thresholds(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("thresholds_frozen"):
        return {}
    rows = payload.get("method_rows", [])
    if not isinstance(rows, list):
        return {}
    thresholds: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        method_slug = str(row.get("method_slug", ""))
        if not method_slug:
            continue
        frozen_threshold = row.get("frozen_threshold", "")
        if frozen_threshold == "":
            continue
        thresholds[method_slug] = dict(row)
    return thresholds


def _case_records(package_config: dict[str, Any], output_root_base: str) -> list[dict[str, Any]]:
    final_matrix = dict(package_config["final_matrix"])
    cases: list[dict[str, Any]] = []
    for method in package_config["baseline_methods"]:
        method_slug = str(method["slug"])
        for seed in final_matrix["seeds"]:
            for payload in final_matrix["payloads"]:
                case_id = _case_id(method, str(payload), int(seed))
                cases.append(
                    {
                        "id": case_id,
                        "case_id": case_id,
                        "method_id": str(method["id"]),
                        "method_slug": method_slug,
                        "method_name": str(method["method_name"]),
                        "display_name": str(method["display_name"]),
                        "baseline_family": str(method["baseline_family"]),
                        "baseline_role": str(method["baseline_role"]),
                        "b_stage": str(method["b_stage"]),
                        "train_objective": str(method["train_objective"]),
                        "requires_training": bool(method["requires_training"]),
                        "requires_external_integration": bool(method["requires_external_integration"]),
                        "paper_ready_denominator": bool(method.get("paper_ready_denominator", True)),
                        "matched_budget_status": str(method["matched_budget_status"]),
                        "block_count": int(package_config["fixed_contract"]["block_count"]),
                        "query_budget": int(package_config["fixed_contract"]["query_budget"]),
                        "target_far": float(package_config["fixed_contract"]["target_far"]),
                        "payload": str(payload),
                        "seed": int(seed),
                        "case_root": _case_root(output_root_base, method_slug, str(payload), int(seed)),
                    }
                )
    return cases


def _build_train_entry(
    *,
    repo_root: Path,
    train_config_path: Path,
    case: dict[str, Any],
    environment_setup: str | None,
) -> ManifestEntry:
    output_root = str((Path(case["case_root"]) / "runs").as_posix())
    overrides = [
        f"run.seed={case['seed']}",
        f"run.variant_name={_variant_name(case)}",
        f"eval.payload_text={case['payload']}",
        f"train.objective={case['train_objective']}",
        f"train.probe_block_count={case['block_count']}",
        f"runtime.output_root={output_root}",
    ]
    if environment_setup:
        overrides.append(f"runtime.environment_setup={environment_setup}")
    manifest = build_manifest_from_config(train_config_path, overrides=overrides)
    entry = _entry_with_repo_relative_config(manifest.entries[0], repo_root)
    return _entry_with_identity(
        entry,
        manifest_id=f"baseline-train-{case['method_slug']}-{str(case['payload']).lower()}-s{case['seed']}",
        manifest_name="matched_budget_baselines_train",
    )


def _build_eval_entry(
    *,
    repo_root: Path,
    eval_config_path: Path,
    case: dict[str, Any],
    environment_setup: str | None,
) -> ManifestEntry:
    output_root = str((Path(case["case_root"]) / "runs").as_posix())
    eval_input_path = str((Path(output_root) / "exp_train" / "latest_eval_input.json").as_posix())
    overrides = [
        f"run.seed={case['seed']}",
        f"run.method_name={case['method_name']}",
        f"run.variant_name={_variant_name(case)}",
        f"eval.payload_text={case['payload']}",
        f"eval.target_far={case['target_far']}",
        f"runtime.output_root={output_root}",
    ]
    if case.get("frozen_threshold") != "":
        overrides.append(f"eval.min_score={case['frozen_threshold']}")
    if case["requires_training"]:
        overrides.append(f"data.eval_path={eval_input_path}")
    else:
        overrides.append("data.eval_path=results/raw/baseline_placeholder/latest_eval_input.json")
    if environment_setup:
        overrides.append(f"runtime.environment_setup={environment_setup}")
    manifest = build_manifest_from_config(eval_config_path, overrides=overrides)
    entry = _entry_with_repo_relative_config(manifest.entries[0], repo_root)
    return _entry_with_identity(
        entry,
        manifest_id=f"baseline-eval-{case['method_slug']}-{str(case['payload']).lower()}-s{case['seed']}",
        manifest_name="matched_budget_baselines_eval",
    )


def _calibration_rows(
    package_config: dict[str, Any],
    frozen_thresholds: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    calibration = dict(package_config["calibration_split"])
    fixed = dict(package_config["fixed_contract"])
    frozen_thresholds = frozen_thresholds or {}
    rows: list[dict[str, Any]] = []
    for method in package_config["baseline_methods"]:
        frozen = frozen_thresholds.get(str(method["slug"]), {})
        rows.append(
            {
                "method_id": str(method["id"]),
                "method_slug": str(method["slug"]),
                "method_name": str(method["method_name"]),
                "baseline_family": str(method["baseline_family"]),
                "baseline_role": str(method["baseline_role"]),
                "status": str(frozen.get("threshold_status", "pending_real_calibration_scores")),
                "score_name": "claim_conditioned_match_ratio",
                "score_direction": "higher_is_stronger",
                "target_far": float(fixed["target_far"]),
                "frozen_threshold": frozen.get("frozen_threshold", ""),
                "calibration_observed_far": frozen.get("calibration_observed_far", ""),
                "calibration_payloads": list(calibration["payloads"]),
                "calibration_seed": int(calibration["seed"]),
                "negative_sets": list(calibration["negative_sets"]),
                "requires_external_integration": bool(method["requires_external_integration"]),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    train_config_path = _resolve_path(repo_root, str(package_config["train_config"]))
    eval_config_path = _resolve_path(repo_root, str(package_config["eval_config"]))
    calibration_summary_path = _resolve_path(repo_root, args.calibration_summary)
    output_path = _resolve_path(repo_root, args.output)
    train_manifest_path = _resolve_path(repo_root, args.train_manifest_out)
    eval_manifest_path = _resolve_path(repo_root, args.eval_manifest_out)
    output_root_base = _resolve_output_root_base(package_config, args.output_root_base)
    environment_setup = (
        args.environment_setup
        or os.environ.get("CHIMERA_ENV_SETUP")
        or package_config.get("chimera_environment_setup")
    )

    frozen_thresholds = _load_frozen_thresholds(calibration_summary_path)
    cases = _case_records(package_config, output_root_base)
    for case in cases:
        frozen = frozen_thresholds.get(str(case["method_slug"]), {})
        case["frozen_threshold"] = frozen.get("frozen_threshold", "")
        case["calibration_observed_far"] = frozen.get("calibration_observed_far", "")
    train_cases = [case for case in cases if case["requires_training"]]
    train_entries = [
        _build_train_entry(
            repo_root=repo_root,
            train_config_path=train_config_path,
            case=case,
            environment_setup=environment_setup,
        )
        for case in train_cases
    ]
    eval_cases = [case for case in cases if not case["requires_external_integration"]]
    eval_entries = [
        _build_eval_entry(
            repo_root=repo_root,
            eval_config_path=eval_config_path,
            case=case,
            environment_setup=environment_setup,
        )
        for case in eval_cases
    ]

    _save_manifest(
        repo_root,
        train_config_path,
        "matched_budget_baselines_train",
        train_entries,
        train_manifest_path,
    )
    _save_manifest(
        repo_root,
        eval_config_path,
        "matched_budget_baselines_eval",
        eval_entries,
        eval_manifest_path,
    )

    payload = {
        "schema_name": "baseline_package_dry_run",
        "schema_version": 1,
        "workstream": package_config.get("workstream", "B1-B2"),
        "package_config_path": _repo_relative_path(repo_root, package_config_path),
        "generated_at": current_timestamp(),
        "output_root_base": output_root_base,
        "target_case_count": len(cases),
        "paper_ready_target_case_count": sum(bool(case["paper_ready_denominator"]) for case in cases),
        "train_manifest_entry_count": len(train_entries),
        "eval_manifest_entry_count": len(eval_entries),
        "reporting_case_count": len(cases),
        "external_unavailable_case_count": sum(bool(case["requires_external_integration"]) for case in cases),
        "calibration_method_count": len(package_config["baseline_methods"]),
        "environment_setup_present": bool(environment_setup),
        "environment_setup_contains_zkrfa_activate": bool(
            environment_setup and "zkrfa_py312/bin/activate" in str(environment_setup)
        ),
        "b0_protocol": package_config.get("b0_protocol", {}),
        "fixed_contract": package_config.get("fixed_contract", {}),
        "final_matrix": package_config.get("final_matrix", {}),
        "baseline_methods": package_config.get("baseline_methods", []),
        "calibration_summary_path": _repo_relative_path(repo_root, calibration_summary_path),
        "calibration_thresholds_frozen": bool(frozen_thresholds),
        "calibration_rows": _calibration_rows(package_config, frozen_thresholds),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote baseline dry-run summary to {output_path}")
    print(f"wrote {len(train_entries)} train manifest entries to {train_manifest_path}")
    print(f"wrote {len(eval_entries)} eval manifest entries to {eval_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

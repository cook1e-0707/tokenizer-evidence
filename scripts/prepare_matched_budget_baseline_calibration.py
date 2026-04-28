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
    parser = argparse.ArgumentParser(description="Prepare B1/B2 baseline calibration manifests.")
    parser.add_argument("--package-config", default="configs/reporting/matched_budget_baselines_v1.yaml")
    parser.add_argument(
        "--output",
        default="results/processed/paper_stats/baseline_calibration_package_dry_run.json",
    )
    parser.add_argument(
        "--train-manifest-out",
        default="manifests/matched_budget_baselines/calibration_train_manifest.json",
    )
    parser.add_argument(
        "--eval-manifest-out",
        default="manifests/matched_budget_baselines/calibration_eval_manifest.json",
    )
    parser.add_argument(
        "--output-root-base",
        help="Optional base directory for baseline calibration roots. Defaults to EXP_SCRATCH/matched_budget_baselines_v1.",
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
        "new_case_root_prefix. Set EXP_SCRATCH on Chimera so calibration outputs stay off home.",
        file=sys.stderr,
    )
    return prefix


def _calibration_methods(package_config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(method)
        for method in package_config["baseline_methods"]
        if bool(method["requires_training"]) and not bool(method["requires_external_integration"])
    ]


def _case_root(prefix: str, method_slug: str, payload: str, seed: int) -> str:
    return str((Path(prefix) / "calibration" / method_slug / f"{payload}_s{seed}").as_posix())


def _variant_name(method_slug: str, eval_kind: str = "") -> str:
    suffix = f"-{eval_kind}" if eval_kind else ""
    return f"matched-budget-baseline-calibration-{method_slug}{suffix}"


def _train_cases(package_config: dict[str, Any], output_root_base: str) -> list[dict[str, Any]]:
    calibration = dict(package_config["calibration_split"])
    seed = int(calibration["seed"])
    cases: list[dict[str, Any]] = []
    for method in _calibration_methods(package_config):
        for payload in calibration["payloads"]:
            cases.append(
                {
                    "case_id": f"cal_train_{method['slug']}_{payload}_s{seed}",
                    "method_id": str(method["id"]),
                    "method_slug": str(method["slug"]),
                    "method_name": str(method["method_name"]),
                    "baseline_family": str(method["baseline_family"]),
                    "baseline_role": str(method["baseline_role"]),
                    "train_objective": str(method["train_objective"]),
                    "owner_payload": str(payload),
                    "claim_payload": str(payload),
                    "seed": seed,
                    "block_count": int(package_config["fixed_contract"]["block_count"]),
                    "query_budget": int(package_config["fixed_contract"]["query_budget"]),
                    "target_far": float(package_config["fixed_contract"]["target_far"]),
                    "case_root": _case_root(output_root_base, str(method["slug"]), str(payload), seed),
                }
            )
    return cases


def _eval_cases(package_config: dict[str, Any], output_root_base: str) -> list[dict[str, Any]]:
    calibration = dict(package_config["calibration_split"])
    payloads = [str(payload) for payload in calibration["payloads"]]
    seed = int(calibration["seed"])
    cases: list[dict[str, Any]] = []
    for method in _calibration_methods(package_config):
        method_slug = str(method["slug"])
        for owner_payload in payloads:
            for claim_payload in payloads:
                is_positive = owner_payload == claim_payload
                eval_kind = "positive" if is_positive else "wrong_payload_null"
                cases.append(
                    {
                        "case_id": (
                            f"cal_eval_{method_slug}_{owner_payload}_claim_{claim_payload}_s{seed}"
                        ),
                        "method_id": str(method["id"]),
                        "method_slug": method_slug,
                        "method_name": str(method["method_name"]),
                        "baseline_family": str(method["baseline_family"]),
                        "baseline_role": str(method["baseline_role"]),
                        "train_objective": str(method["train_objective"]),
                        "owner_payload": owner_payload,
                        "claim_payload": claim_payload,
                        "label": bool(is_positive),
                        "eval_kind": eval_kind,
                        "negative_set": "" if is_positive else "wrong_payload_null",
                        "seed": seed,
                        "block_count": int(package_config["fixed_contract"]["block_count"]),
                        "query_budget": int(package_config["fixed_contract"]["query_budget"]),
                        "target_far": float(package_config["fixed_contract"]["target_far"]),
                        "case_root": _case_root(output_root_base, method_slug, owner_payload, seed),
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
        f"run.variant_name={_variant_name(case['method_slug'])}",
        f"eval.payload_text={case['owner_payload']}",
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
        manifest_id=f"baseline-calibration-train-{case['method_slug']}-{case['owner_payload'].lower()}-s{case['seed']}",
        manifest_name="matched_budget_baselines_calibration_train",
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
        f"run.variant_name={_variant_name(case['method_slug'], case['eval_kind'])}",
        f"eval.payload_text={case['claim_payload']}",
        "eval.expected_payload_source=config",
        f"eval.target_far={case['target_far']}",
        f"data.eval_path={eval_input_path}",
        f"runtime.output_root={output_root}",
    ]
    if environment_setup:
        overrides.append(f"runtime.environment_setup={environment_setup}")
    manifest = build_manifest_from_config(eval_config_path, overrides=overrides)
    entry = _entry_with_repo_relative_config(manifest.entries[0], repo_root)
    return _entry_with_identity(
        entry,
        manifest_id=(
            f"baseline-calibration-eval-{case['method_slug']}-{case['owner_payload'].lower()}"
            f"-claim-{case['claim_payload'].lower()}-s{case['seed']}"
        ),
        manifest_name="matched_budget_baselines_calibration_eval",
    )


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    train_config_path = _resolve_path(repo_root, str(package_config["train_config"]))
    eval_config_path = _resolve_path(repo_root, str(package_config["eval_config"]))
    output_path = _resolve_path(repo_root, args.output)
    train_manifest_path = _resolve_path(repo_root, args.train_manifest_out)
    eval_manifest_path = _resolve_path(repo_root, args.eval_manifest_out)
    output_root_base = _resolve_output_root_base(package_config, args.output_root_base)
    environment_setup = (
        args.environment_setup
        or os.environ.get("CHIMERA_ENV_SETUP")
        or package_config.get("chimera_environment_setup")
    )
    train_cases = _train_cases(package_config, output_root_base)
    eval_cases = _eval_cases(package_config, output_root_base)
    train_entries = [
        _build_train_entry(
            repo_root=repo_root,
            train_config_path=train_config_path,
            case=case,
            environment_setup=environment_setup,
        )
        for case in train_cases
    ]
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
        "matched_budget_baselines_calibration_train",
        train_entries,
        train_manifest_path,
    )
    _save_manifest(
        repo_root,
        eval_config_path,
        "matched_budget_baselines_calibration_eval",
        eval_entries,
        eval_manifest_path,
    )
    missing_negative_sets = [
        item
        for item in package_config["calibration_split"]["negative_sets"]
        if item not in {"wrong_payload_null"}
    ]
    payload = {
        "schema_name": "baseline_calibration_package_dry_run",
        "schema_version": 1,
        "workstream": package_config.get("workstream", "B1-B2"),
        "package_config_path": _repo_relative_path(repo_root, package_config_path),
        "generated_at": current_timestamp(),
        "output_root_base": output_root_base,
        "train_manifest_entry_count": len(train_entries),
        "eval_manifest_entry_count": len(eval_entries),
        "calibration_methods": [case["method_slug"] for case in train_cases[:: len(package_config["calibration_split"]["payloads"])]],
        "available_negative_sets": ["wrong_payload_null"],
        "missing_negative_sets": missing_negative_sets,
        "threshold_freeze_allowed": False,
        "threshold_freeze_blockers": [
            "foundation_null and organic_prompt_null are not yet materialized",
            "real calibration eval summaries are pending",
        ],
        "environment_setup_present": bool(environment_setup),
        "environment_setup_contains_zkrfa_activate": bool(
            environment_setup and "zkrfa_py312/bin/activate" in str(environment_setup)
        ),
        "train_cases": train_cases,
        "eval_cases": eval_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote baseline calibration dry-run summary to {output_path}")
    print(f"wrote {len(train_entries)} calibration train manifest entries to {train_manifest_path}")
    print(f"wrote {len(eval_entries)} calibration eval manifest entries to {eval_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

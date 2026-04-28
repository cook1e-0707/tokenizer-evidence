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
    parser = argparse.ArgumentParser(description="Prepare R1 Llama 3.1 8B replication manifests.")
    parser.add_argument("--package-config", default="configs/reporting/r1_llama3_1_8b_replication_v1.yaml")
    parser.add_argument("--output", default="results/processed/paper_stats/r1_package_dry_run.json")
    parser.add_argument(
        "--train-manifest-out",
        default="manifests/r1_llama3_1_8b_replication/train_manifest.json",
    )
    parser.add_argument(
        "--eval-manifest-out",
        default="manifests/r1_llama3_1_8b_replication/eval_manifest.json",
    )
    parser.add_argument(
        "--output-root-base",
        help="Optional base directory for R1 case roots. Defaults to EXP_SCRATCH/r1_llama3_1_8b_replication_v1.",
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
        "new_case_root_prefix. Set EXP_SCRATCH on Chimera so R1 large outputs stay off home.",
        file=sys.stderr,
    )
    return prefix


def _case_id(payload: str, seed: int) -> str:
    return f"R1_{payload}_s{seed}"


def _case_root(prefix: str, payload: str, seed: int) -> str:
    return str((Path(prefix) / "final" / f"{payload}_s{seed}").as_posix())


def _variant_name(case: dict[str, Any]) -> str:
    return f"r1-llama3-1-8b-replication-{str(case['payload']).lower()}"


def _case_records(package_config: dict[str, Any], output_root_base: str) -> list[dict[str, Any]]:
    final_matrix = dict(package_config["final_matrix"])
    payloads = [str(item) for item in final_matrix["payloads"]]
    payloads_json = json.dumps(payloads, separators=(",", ":"))
    seeds = [int(item) for item in final_matrix["seeds"]]
    cases: list[dict[str, Any]] = []
    for seed in seeds:
        for payload in payloads:
            case_id = _case_id(payload, seed)
            cases.append(
                {
                    "id": case_id,
                    "case_id": case_id,
                    "variant_id": "R1",
                    "variant_slug": "llama3_1_8b",
                    "model_family": "llama3_1",
                    "model_id": str(package_config["fixed_contract"]["model"]),
                    "tokenizer_id": str(package_config["fixed_contract"]["tokenizer"]),
                    "block_count": int(final_matrix["block_count"]),
                    "payload": payload,
                    "train_payload_labels": payloads,
                    "train_payload_labels_json": payloads_json,
                    "seed": seed,
                    "case_root": _case_root(output_root_base, payload, seed),
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
        f"train.probe_block_count={case['block_count']}",
        f"train.probe_payload_texts={case['train_payload_labels_json']}",
        f"runtime.output_root={output_root}",
    ]
    if environment_setup:
        overrides.append(f"runtime.environment_setup={environment_setup}")
    manifest = build_manifest_from_config(train_config_path, overrides=overrides)
    entry = _entry_with_repo_relative_config(manifest.entries[0], repo_root)
    return _entry_with_identity(
        entry,
        manifest_id=f"r1-train-{str(case['payload']).lower()}-s{case['seed']}",
        manifest_name="r1_llama3_1_8b_replication_train",
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
        f"run.variant_name={_variant_name(case)}",
        f"eval.payload_text={case['payload']}",
        f"data.eval_path={eval_input_path}",
        f"runtime.output_root={output_root}",
    ]
    if environment_setup:
        overrides.append(f"runtime.environment_setup={environment_setup}")
    manifest = build_manifest_from_config(eval_config_path, overrides=overrides)
    entry = _entry_with_repo_relative_config(manifest.entries[0], repo_root)
    return _entry_with_identity(
        entry,
        manifest_id=f"r1-eval-{str(case['payload']).lower()}-s{case['seed']}",
        manifest_name="r1_llama3_1_8b_replication_eval",
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

    cases = _case_records(package_config, output_root_base)
    train_entries = [
        _build_train_entry(
            repo_root=repo_root,
            train_config_path=train_config_path,
            case=case,
            environment_setup=environment_setup,
        )
        for case in cases
    ]
    eval_entries = [
        _build_eval_entry(
            repo_root=repo_root,
            eval_config_path=eval_config_path,
            case=case,
            environment_setup=environment_setup,
        )
        for case in cases
    ]

    _save_manifest(
        repo_root,
        train_config_path,
        "r1_llama3_1_8b_replication_train",
        train_entries,
        train_manifest_path,
    )
    _save_manifest(
        repo_root,
        eval_config_path,
        "r1_llama3_1_8b_replication_eval",
        eval_entries,
        eval_manifest_path,
    )

    prerequisite = dict(package_config.get("catalog_prerequisite", {}))
    frozen_catalog = repo_root / str(prerequisite.get("frozen_catalog", ""))
    payload = {
        "schema_name": "r1_package_dry_run",
        "schema_version": 1,
        "workstream": package_config.get("workstream", "R1"),
        "package_config_path": _repo_relative_path(repo_root, package_config_path),
        "generated_at": current_timestamp(),
        "output_root_base": output_root_base,
        "target_case_count": len(cases),
        "train_manifest_entry_count": len(train_entries),
        "eval_manifest_entry_count": len(eval_entries),
        "environment_setup_present": bool(environment_setup),
        "environment_setup_contains_zkrfa_activate": bool(
            environment_setup and "zkrfa_py312/bin/activate" in str(environment_setup)
        ),
        "catalog_prerequisite": prerequisite,
        "frozen_catalog_exists": frozen_catalog.exists() if prerequisite.get("frozen_catalog") else False,
        "fixed_contract": package_config.get("fixed_contract", {}),
        "final_matrix": package_config.get("final_matrix", {}),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote R1 dry-run summary to {output_path}")
    print(f"wrote {len(train_entries)} train manifest entries to {train_manifest_path}")
    print(f"wrote {len(eval_entries)} eval manifest entries to {eval_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

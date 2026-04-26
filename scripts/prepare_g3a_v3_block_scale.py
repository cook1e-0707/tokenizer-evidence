from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from src.infrastructure.manifest import (
    ManifestEntry,
    ManifestFile,
    build_manifest_from_config,
    save_manifest,
)
from src.infrastructure.paths import current_timestamp, discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare G3a-v3 block-scale manifests.")
    parser.add_argument("--package-config", default="configs/reporting/g3a_block_scale_v3.yaml")
    parser.add_argument("--phase", choices=["validation", "final"], default="validation")
    parser.add_argument(
        "--output",
        default="results/processed/paper_stats/g3a_v3_package_dry_run.json",
    )
    parser.add_argument(
        "--train-manifest-out",
        default="manifests/g3a_v3_qwen7b_block_scale/train_manifest.json",
    )
    parser.add_argument(
        "--eval-manifest-out",
        default="manifests/g3a_v3_qwen7b_block_scale/eval_manifest.json",
    )
    parser.add_argument(
        "--output-root-base",
        help="Optional base directory for G3a-v3 case roots. Defaults to EXP_SCRATCH/g3a_block_scale_v3.",
    )
    parser.add_argument(
        "--environment-setup",
        help="Optional runtime environment setup block. Defaults to CHIMERA_ENV_SETUP when set.",
    )
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(os.path.expandvars(raw))
    return path if path.is_absolute() else repo_root / path


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _resolve_output_root_base(package_config: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(Path(os.path.expandvars(explicit)).as_posix())
    prefix = str(package_config["new_case_root_prefix"])
    exp_scratch = os.environ.get("EXP_SCRATCH")
    if exp_scratch:
        return str((Path(exp_scratch) / prefix).as_posix())
    print(
        "WARNING: EXP_SCRATCH is not set; falling back to the package-relative "
        "new_case_root_prefix. Set EXP_SCRATCH on Chimera so G3a-v3 large outputs stay off home.",
        file=sys.stderr,
    )
    return prefix


def _variant_by_id(package_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): dict(item) for item in package_config["block_variants"]}


def _case_id(variant_id: str, payload: str, seed: int, hp_id: str | None = None) -> str:
    base = f"{variant_id}_{payload}_s{seed}"
    return f"{hp_id}_{base}" if hp_id else base


def _case_root(prefix: str, phase: str, variant_slug: str, payload: str, seed: int, hp_id: str | None = None) -> str:
    parts = [prefix, phase]
    if hp_id:
        parts.append(hp_id)
    parts.extend([variant_slug, f"{payload}_s{seed}"])
    return str(Path(*parts).as_posix())


def _variant_name(phase: str, case: dict[str, Any]) -> str:
    hp = f"-{case['hp_id']}" if case.get("hp_id") else ""
    return f"g3a-v3-qwen7b-margin-block-scale-{phase}-{case['variant_slug']}{hp}"


def _checkpoint_selection_mode(metric: str) -> str:
    normalized = metric.strip().lower()
    if normalized in {
        "training_min_slot_margin",
        "training_slot_margin_min",
        "training_slot_margin_mean",
        "training_target_bucket_mass_mean",
        "training_target_bucket_mass_min",
    }:
        return "max"
    return "min"


def _validation_cases(package_config: dict[str, Any], output_root_base: str) -> list[dict[str, Any]]:
    validation = dict(package_config["validation"])
    fixed = dict(validation["fixed_hyperparameters"])
    sweep = dict(validation["sweep"])
    variant_lookup = _variant_by_id(package_config)
    keys = ["margin_gamma", "lambda_margin", "checkpoint_selection_metric"]
    cases: list[dict[str, Any]] = []
    for hp_index, values in enumerate(itertools.product(*(sweep[key] for key in keys)), start=1):
        hp = {**fixed, **dict(zip(keys, values, strict=True))}
        hp["checkpoint_selection_mode"] = _checkpoint_selection_mode(str(hp["checkpoint_selection_metric"]))
        hp_id = f"hp{hp_index:02d}"
        for variant_id in validation["block_variants"]:
            variant = variant_lookup[str(variant_id)]
            variant_slug = str(variant.get("slug", str(variant["id"]).lower()))
            for payload in validation["payloads"]:
                seed = int(validation["seed"])
                case_id = _case_id(str(variant["id"]), str(payload), seed, hp_id=hp_id)
                cases.append(
                    {
                        "id": case_id,
                        "case_id": case_id,
                        "phase": "validation",
                        "variant_id": str(variant["id"]),
                        "variant_slug": variant_slug,
                        "block_count": int(variant["block_count"]),
                        "payload": str(payload),
                        "seed": seed,
                        "case_root": _case_root(
                            output_root_base,
                            "validation",
                            variant_slug,
                            str(payload),
                            seed,
                            hp_id=hp_id,
                        ),
                        "hp_id": hp_id,
                        "hyperparameters": hp,
                    }
                )
    return cases


def _final_cases(package_config: dict[str, Any], output_root_base: str) -> list[dict[str, Any]]:
    selected = dict(package_config.get("selected_operating_point", {}))
    if selected.get("status") != "frozen_before_final_launch" or not bool(selected.get("final_launch_allowed")):
        raise SystemExit(
            "G3a-v3 final manifests are blocked until selected_operating_point is frozen before final launch. "
            "Run validation first, then update configs/reporting/g3a_block_scale_v3.yaml with frozen hyperparameters."
        )
    variant_lookup = _variant_by_id(package_config)
    final_matrix = dict(package_config["final_matrix"])
    selected_hp = {
        key: value
        for key, value in selected.items()
        if key
        in {
            "lora_r",
            "learning_rate",
            "epochs",
            "lambda_set",
            "lambda_margin",
            "margin_gamma",
            "lambda_reg",
            "checkpoint_selection_metric",
            "checkpoint_selection_mode",
        }
        and value is not None
    }
    if "checkpoint_selection_metric" in selected_hp and "checkpoint_selection_mode" not in selected_hp:
        selected_hp["checkpoint_selection_mode"] = _checkpoint_selection_mode(
            str(selected_hp["checkpoint_selection_metric"])
        )
    cases: list[dict[str, Any]] = []
    for variant_id in final_matrix["block_variants"]:
        variant = variant_lookup[str(variant_id)]
        variant_slug = str(variant.get("slug", str(variant["id"]).lower()))
        for seed in final_matrix["seeds"]:
            for payload in final_matrix["payloads"]:
                case_id = _case_id(str(variant["id"]), str(payload), int(seed))
                cases.append(
                    {
                        "id": case_id,
                        "case_id": case_id,
                        "phase": "final",
                        "variant_id": str(variant["id"]),
                        "variant_slug": variant_slug,
                        "block_count": int(variant["block_count"]),
                        "payload": str(payload),
                        "seed": int(seed),
                        "case_root": _case_root(
                            output_root_base,
                            "final",
                            variant_slug,
                            str(payload),
                            int(seed),
                        ),
                        "hp_id": "",
                        "hyperparameters": selected_hp,
                    }
                )
    return cases


def _entry_with_identity(entry: ManifestEntry, *, manifest_id: str, manifest_name: str) -> ManifestEntry:
    payload = entry.to_json_dict()
    payload["manifest_id"] = manifest_id
    payload["manifest_name"] = manifest_name
    payload["status"] = "pending"
    return ManifestEntry.from_json_dict(payload)


def _build_train_entry(
    *,
    train_config_path: Path,
    phase: str,
    case: dict[str, Any],
    environment_setup: str | None,
) -> ManifestEntry:
    output_root = str((Path(case["case_root"]) / "runs").as_posix())
    overrides = [
        f"run.seed={case['seed']}",
        f"run.variant_name={_variant_name(phase, case)}",
        f"eval.payload_text={case['payload']}",
        f"train.probe_block_count={case['block_count']}",
        f"runtime.output_root={output_root}",
    ]
    for key, value in dict(case.get("hyperparameters", {})).items():
        overrides.append(f"train.{key}={value}")
    if environment_setup:
        overrides.append(f"runtime.environment_setup={environment_setup}")
    manifest = build_manifest_from_config(train_config_path, overrides=overrides)
    manifest_id = (
        f"g3a-v3-{phase}-train-{case['hp_id'] + '-' if case.get('hp_id') else ''}"
        f"{case['variant_slug']}-{str(case['payload']).lower()}-s{case['seed']}"
    )
    return _entry_with_identity(
        manifest.entries[0],
        manifest_id=manifest_id,
        manifest_name=f"g3a_v3_qwen7b_block_scale_{phase}_train",
    )


def _build_eval_entry(
    *,
    eval_config_path: Path,
    phase: str,
    case: dict[str, Any],
    environment_setup: str | None,
) -> ManifestEntry:
    output_root = str((Path(case["case_root"]) / "runs").as_posix())
    eval_input_path = str((Path(output_root) / "exp_train" / "latest_eval_input.json").as_posix())
    overrides = [
        f"run.seed={case['seed']}",
        f"run.variant_name={_variant_name(phase, case)}",
        f"eval.payload_text={case['payload']}",
        f"data.eval_path={eval_input_path}",
        f"runtime.output_root={output_root}",
    ]
    if environment_setup:
        overrides.append(f"runtime.environment_setup={environment_setup}")
    manifest = build_manifest_from_config(eval_config_path, overrides=overrides)
    manifest_id = (
        f"g3a-v3-{phase}-eval-{case['hp_id'] + '-' if case.get('hp_id') else ''}"
        f"{case['variant_slug']}-{str(case['payload']).lower()}-s{case['seed']}"
    )
    return _entry_with_identity(
        manifest.entries[0],
        manifest_id=manifest_id,
        manifest_name=f"g3a_v3_qwen7b_block_scale_{phase}_eval",
    )


def _save_manifest(source_config_path: Path, manifest_name: str, entries: list[ManifestEntry], path: Path) -> None:
    manifest = ManifestFile(
        schema_name="manifest_file",
        schema_version=1,
        manifest_name=manifest_name,
        created_at=current_timestamp(),
        source_config_path=str(source_config_path),
        entries=tuple(entries),
    )
    save_manifest(manifest, path)


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
    environment_setup = args.environment_setup or os.environ.get("CHIMERA_ENV_SETUP")

    cases = (
        _validation_cases(package_config, output_root_base)
        if args.phase == "validation"
        else _final_cases(package_config, output_root_base)
    )
    train_entries = [
        _build_train_entry(
            train_config_path=train_config_path,
            phase=args.phase,
            case=case,
            environment_setup=environment_setup,
        )
        for case in cases
    ]
    eval_entries = [
        _build_eval_entry(
            eval_config_path=eval_config_path,
            phase=args.phase,
            case=case,
            environment_setup=environment_setup,
        )
        for case in cases
    ]
    _save_manifest(
        train_config_path,
        f"g3a_v3_qwen7b_block_scale_{args.phase}_train",
        train_entries,
        train_manifest_path,
    )
    _save_manifest(
        eval_config_path,
        f"g3a_v3_qwen7b_block_scale_{args.phase}_eval",
        eval_entries,
        eval_manifest_path,
    )

    payload = {
        "schema_name": "g3a_v3_package_dry_run",
        "schema_version": 1,
        "workstream": package_config.get("workstream", "G3a-v3"),
        "phase": args.phase,
        "package_config_path": str(package_config_path),
        "generated_at": current_timestamp(),
        "output_root_base": output_root_base,
        "target_case_count": len(cases),
        "train_manifest_entry_count": len(train_entries),
        "eval_manifest_entry_count": len(eval_entries),
        "final_launch_allowed": bool(
            package_config.get("selected_operating_point", {}).get("final_launch_allowed")
        ),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote G3a-v3 {args.phase} dry-run summary to {output_path}")
    print(f"wrote {len(train_entries)} train manifest entries to {train_manifest_path}")
    print(f"wrote {len(eval_entries)} eval manifest entries to {eval_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Prepare G1 payload x seed scale manifests and dry-run summary.")
    parser.add_argument(
        "--package-config",
        default="configs/reporting/g1_payload_seed_scale_v1.yaml",
        help="YAML package config for the G1 payload x seed scale package.",
    )
    parser.add_argument(
        "--output",
        default="results/processed/paper_stats/g1_package_dry_run.json",
        help="JSON output path for the dry-run package summary.",
    )
    parser.add_argument(
        "--train-manifest-out",
        default="manifests/g1_qwen7b_payload_seed_scale/train_manifest.json",
        help="Output path for the train manifest.",
    )
    parser.add_argument(
        "--eval-manifest-out",
        default="manifests/g1_qwen7b_payload_seed_scale/eval_manifest.json",
        help="Output path for the eval manifest.",
    )
    parser.add_argument(
        "--output-root-base",
        help=(
            "Optional base directory for new G1 case roots. "
            "Defaults to the package config new_case_root_prefix."
        ),
    )
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _case_id(payload: str, seed: int) -> str:
    return f"{payload}_s{seed}"


def _find_latest(case_root: Path, pattern: str) -> Path | None:
    matches = sorted(case_root.rglob(pattern))
    return matches[-1] if matches else None


def _has_complete_case(case_root: Path) -> bool:
    return (
        _find_latest(case_root, "runs/exp_train/*/train_summary.json") is not None
        and _find_latest(case_root, "runs/exp_eval/*/eval_summary.json") is not None
    )


def _join_case_root(prefix: str, case_id: str) -> str:
    return str((Path(prefix) / case_id).as_posix())


def _build_case_records(
    *,
    repo_root: Path,
    package_config: dict[str, Any],
    output_root_base: str | None,
) -> list[dict[str, Any]]:
    payloads = [str(item) for item in package_config["payloads"]]
    seeds = [int(item) for item in package_config["seeds"]]
    existing_by_id = {str(item["id"]): dict(item) for item in package_config.get("existing_cases", [])}
    new_case_root_prefix = str(output_root_base or package_config["new_case_root_prefix"])

    cases: list[dict[str, Any]] = []
    for seed in seeds:
        for payload in payloads:
            case_id = _case_id(payload, seed)
            existing = existing_by_id.get(case_id)
            if existing is not None:
                case_root_text = str(existing["case_root"])
                status = "reuse_existing"
                source_stage = str(existing.get("stage", ""))
            else:
                case_root_text = _join_case_root(new_case_root_prefix, case_id)
                case_root_path = _resolve_path(repo_root, case_root_text)
                if _has_complete_case(case_root_path):
                    status = "landed_new"
                else:
                    status = "prepare"
                source_stage = "G1"
            cases.append(
                {
                    "id": case_id,
                    "payload": payload,
                    "seed": seed,
                    "case_root": case_root_text,
                    "status": status,
                    "source_stage": source_stage,
                }
            )
    return cases


def _entry_with_identity(
    *,
    entry: ManifestEntry,
    manifest_id: str,
    manifest_name: str,
) -> ManifestEntry:
    payload = entry.to_json_dict()
    payload["manifest_id"] = manifest_id
    payload["manifest_name"] = manifest_name
    payload["status"] = "pending"
    return ManifestEntry.from_json_dict(payload)


def _build_train_entry(
    *,
    repo_root: Path,
    train_config_path: Path,
    case: dict[str, Any],
) -> ManifestEntry:
    output_root = str((Path(case["case_root"]) / "runs").as_posix())
    manifest = build_manifest_from_config(
        train_config_path,
        overrides=[
            f"run.seed={case['seed']}",
            f"eval.payload_text={case['payload']}",
            f"runtime.output_root={output_root}",
        ],
    )
    return _entry_with_identity(
        entry=manifest.entries[0],
        manifest_id=f"g1-train-{str(case['payload']).lower()}-s{case['seed']}",
        manifest_name="g1_qwen7b_payload_seed_scale_train",
    )


def _build_eval_entry(
    *,
    repo_root: Path,
    eval_config_path: Path,
    case: dict[str, Any],
) -> ManifestEntry:
    output_root = str((Path(case["case_root"]) / "runs").as_posix())
    eval_input_path = str((Path(output_root) / "exp_train" / "latest_eval_input.json").as_posix())
    manifest = build_manifest_from_config(
        eval_config_path,
        overrides=[
            f"run.seed={case['seed']}",
            f"eval.payload_text={case['payload']}",
            f"data.eval_path={eval_input_path}",
            f"runtime.output_root={output_root}",
        ],
    )
    return _entry_with_identity(
        entry=manifest.entries[0],
        manifest_id=f"g1-eval-{str(case['payload']).lower()}-s{case['seed']}",
        manifest_name="g1_qwen7b_payload_seed_scale_eval",
    )


def _save_manifest(
    *,
    source_config_path: Path,
    manifest_name: str,
    entries: list[ManifestEntry],
    output_path: Path,
) -> Path:
    manifest = ManifestFile(
        schema_name="manifest_file",
        schema_version=1,
        manifest_name=manifest_name,
        created_at=current_timestamp(),
        source_config_path=str(source_config_path),
        entries=tuple(entries),
    )
    return save_manifest(manifest, output_path)


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    train_config_path = _resolve_path(repo_root, str(package_config["train_config"]))
    eval_config_path = _resolve_path(repo_root, str(package_config["eval_config"]))
    train_manifest_out = _resolve_path(repo_root, args.train_manifest_out)
    eval_manifest_out = _resolve_path(repo_root, args.eval_manifest_out)
    output_path = _resolve_path(repo_root, args.output)

    cases = _build_case_records(
        repo_root=repo_root,
        package_config=package_config,
        output_root_base=args.output_root_base,
    )
    planned_cases = [case for case in cases if case["status"] == "prepare"]

    train_entries = [
        _build_train_entry(repo_root=repo_root, train_config_path=train_config_path, case=case)
        for case in planned_cases
    ]
    eval_entries = [
        _build_eval_entry(repo_root=repo_root, eval_config_path=eval_config_path, case=case)
        for case in planned_cases
    ]

    if train_entries:
        _save_manifest(
            source_config_path=package_config_path,
            manifest_name="g1_qwen7b_payload_seed_scale_train",
            entries=train_entries,
            output_path=train_manifest_out,
        )
    if eval_entries:
        _save_manifest(
            source_config_path=package_config_path,
            manifest_name="g1_qwen7b_payload_seed_scale_eval",
            entries=eval_entries,
            output_path=eval_manifest_out,
        )

    payload = {
        "workstream": str(package_config.get("workstream", "G1")),
        "description": str(package_config.get("description", "")),
        "package_config_path": str(package_config_path),
        "train_config_path": str(train_config_path),
        "eval_config_path": str(eval_config_path),
        "train_manifest_path": str(train_manifest_out),
        "eval_manifest_path": str(eval_manifest_out),
        "target_case_count": len(cases),
        "reuse_existing_case_count": sum(1 for case in cases if case["status"] == "reuse_existing"),
        "landed_new_case_count": sum(1 for case in cases if case["status"] == "landed_new"),
        "missing_case_count": len(planned_cases),
        "train_manifest_entry_count": len(train_entries),
        "eval_manifest_entry_count": len(eval_entries),
        "payloads": [str(item) for item in package_config["payloads"]],
        "seeds": [int(item) for item in package_config["seeds"]],
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote G1 dry-run summary to {output_path}")
    if train_entries:
        print(f"wrote {len(train_entries)} train manifest entries to {train_manifest_out}")
    else:
        print("no missing train entries to write")
    if eval_entries:
        print(f"wrote {len(eval_entries)} eval manifest entries to {eval_manifest_out}")
    else:
        print("no missing eval entries to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

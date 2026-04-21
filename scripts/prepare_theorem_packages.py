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

from src.infrastructure.manifest import build_manifest_from_config
from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local dry-run theorem packages.")
    parser.add_argument(
        "--package-config",
        default="configs/reporting/theorem_packages_v1.yaml",
        help="YAML manifest of theorem-package configs.",
    )
    parser.add_argument(
        "--output",
        default="results/processed/paper_stats/theorem_package_dry_runs.json",
        help="JSON output path for dry-run package summaries.",
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


def _manifest_record(repo_root: Path, config_path: Path) -> dict[str, Any]:
    manifest = build_manifest_from_config(config_path)
    entry = manifest.entries[0]
    return {
        "manifest_name": manifest.manifest_name,
        "entry_count": len(manifest.entries),
        "entry_point": entry.entry_point,
        "experiment_name": entry.experiment_name,
        "model_name": entry.model_name,
        "seed": entry.seed,
        "primary_config_path": str(config_path if config_path.is_absolute() else config_path.relative_to(repo_root)),
        "notes": entry.notes,
    }


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config = _load_yaml(_resolve_path(repo_root, args.package_config))
    records: list[dict[str, Any]] = []
    for package in package_config.get("packages", []):
        train_config = _resolve_path(repo_root, str(package["train_config"]))
        eval_config = _resolve_path(repo_root, str(package["eval_config"]))
        records.append(
            {
                "id": str(package["id"]),
                "workstream": str(package["workstream"]),
                "description": str(package["description"]),
                "train": _manifest_record(repo_root, train_config),
                "eval": _manifest_record(repo_root, eval_config),
            }
        )
    output_path = _resolve_path(repo_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"packages": records}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote theorem dry-run summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

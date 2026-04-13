from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.infrastructure.manifest import build_manifest_from_config, save_manifest
from src.infrastructure.paths import discover_repo_root, sanitize_component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a manifest from a sweep or experiment config.")
    parser.add_argument("--config", help="Sweep or experiment YAML path.")
    parser.add_argument("--sweep", help="Backward-compatible alias for --config.")
    parser.add_argument("--output", help="Optional manifest output path.")
    parser.add_argument("--dry-run", action="store_true", help="Render summary without writing the file.")
    return parser.parse_args()


def default_output_path(repo_root: Path, manifest_name: str) -> Path:
    return repo_root / "manifests" / sanitize_component(manifest_name) / "manifest.json"


def main() -> int:
    args = parse_args()
    config_arg = args.config or args.sweep
    if not config_arg:
        raise SystemExit("--config is required")

    repo_root = discover_repo_root()
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    manifest_file = build_manifest_from_config(config_path)
    output_path = Path(args.output) if args.output else default_output_path(repo_root, manifest_file.manifest_name)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    if args.dry_run:
        print(f"manifest_name={manifest_file.manifest_name}")
        print(f"entries={len(manifest_file.entries)}")
        for entry in manifest_file.entries[:10]:
            print(
                f"{entry.manifest_id} | exp={entry.experiment_name} | method={entry.method_name} | "
                f"model={entry.model_name} | seed={entry.seed} | "
                f"script={entry.entry_point} | config={entry.primary_config_path}"
            )
        return 0

    save_manifest(manifest_file, output_path)
    print(f"wrote {len(manifest_file.entries)} entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.infrastructure.manifest import load_manifest
from src.infrastructure.paths import discover_repo_root
from src.infrastructure.registry import find_failed_records, latest_registry_by_manifest_id, load_registry
from src.infrastructure.slurm import submit_manifest_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resubmit failed jobs recorded in the registry.")
    parser.add_argument("--manifest", required=True, help="Manifest JSON path.")
    parser.add_argument("--registry", default="manifests/job_registry.jsonl", help="Registry JSONL path.")
    parser.add_argument("--submit", action="store_true", help="Actually call sbatch.")
    parser.add_argument("--force", action="store_true", help="Allow reuse of an existing run directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root()
    manifest_path = Path(args.manifest)
    registry_path = Path(args.registry)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path

    manifest_file = load_manifest(manifest_path)
    latest_records = latest_registry_by_manifest_id(load_registry(registry_path))
    failed_ids = {record.manifest_id for record in find_failed_records(list(latest_records.values()))}
    failed_entries = [entry for entry in manifest_file.entries if entry.manifest_id in failed_ids]

    if not failed_entries:
        print("no failed manifest entries were found")
        return 0

    for entry in failed_entries:
        result = submit_manifest_entry(
            entry=entry,
            manifest_path=manifest_path,
            registry_path=registry_path,
            submit=args.submit,
            force=args.force,
        )
        suffix = f" job_id={result.slurm_job_id}" if result.slurm_job_id else ""
        print(f"{entry.manifest_id}: {result.status}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

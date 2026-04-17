from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.infrastructure.manifest import load_manifest
from src.infrastructure.paths import discover_repo_root
from src.infrastructure.registry import find_unsubmitted_records, load_registry
from src.infrastructure.slurm import filter_manifest_entries, submit_manifest_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit one manifest file to SLURM.")
    parser.add_argument("--manifest", required=True, help="Manifest JSON path.")
    parser.add_argument("--registry", default="manifests/job_registry.jsonl", help="Registry JSONL path.")
    parser.add_argument("--manifest-id", help="Submit a single manifest entry by manifest_id.")
    parser.add_argument("--experiment", help="Filter by experiment_name.")
    parser.add_argument("--method", help="Filter by method_name.")
    parser.add_argument("--tag", help="Filter by tag.")
    parser.add_argument("--all-pending", action="store_true", help="Submit only pending or unsubmitted entries.")
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
    selected = filter_manifest_entries(
        manifest_file=manifest_file,
        manifest_id=args.manifest_id,
        experiment_name=args.experiment,
        method_name=args.method,
        tag=args.tag,
        statuses={"pending", "created", "dry_run"} if args.all_pending else None,
    )

    if args.all_pending:
        registry_records = load_registry(registry_path)
        unsubmitted_ids = set(find_unsubmitted_records([entry.manifest_id for entry in selected], registry_records))
        selected = [entry for entry in selected if entry.manifest_id in unsubmitted_ids]

    if not selected:
        print("no manifest entries matched the requested filters")
        return 0

    for entry in selected:
        result = submit_manifest_entry(
            entry=entry,
            manifest_path=manifest_path,
            registry_path=registry_path,
            repo_root=repo_root,
            submit=args.submit,
            force=args.force,
        )
        suffix = f" job_id={result.slurm_job_id}" if result.slurm_job_id else ""
        print(f"{entry.manifest_id}: {result.status}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

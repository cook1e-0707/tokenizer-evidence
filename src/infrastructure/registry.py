from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


FAILED_REGISTRY_STATUSES = {"failed", "submission_error", "cancelled", "timeout"}
NON_SUBMITTED_STATUSES = {"created", "pending", "dry_run"}


@dataclass(frozen=True)
class RegistryRecord:
    manifest_id: str
    run_id: str
    submission_time: str
    slurm_job_id: str | None
    slurm_script_path: str
    status: str
    output_dir: str
    manifest_path: str
    experiment_name: str
    method_name: str
    model_name: str
    seed: int
    message: str = ""

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "RegistryRecord":
        manifest_id = payload.get("manifest_id", payload.get("entry_id"))
        if manifest_id is None:
            raise KeyError("manifest_id")
        return cls(
            manifest_id=str(manifest_id),
            run_id=str(payload.get("run_id", payload.get("entry_id", "unknown_run"))),
            submission_time=str(payload.get("submission_time", payload.get("timestamp", ""))),
            slurm_job_id=str(payload["slurm_job_id"]) if payload.get("slurm_job_id") else None,
            slurm_script_path=str(
                payload.get("slurm_script_path", payload.get("rendered_script", ""))
            ),
            status=str(payload["status"]),
            output_dir=str(payload.get("output_dir", "")),
            manifest_path=str(payload.get("manifest_path", "")),
            experiment_name=str(payload.get("experiment_name", "")),
            method_name=str(payload.get("method_name", "")),
            model_name=str(payload.get("model_name", "")),
            seed=int(payload.get("seed", 0)),
            message=str(payload.get("message", "")),
        )


def append_registry_record(path: Path, record: RegistryRecord) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
    return path


def load_registry(path: Path) -> list[RegistryRecord]:
    if not path.exists():
        return []
    records: list[RegistryRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(RegistryRecord.from_json_dict(json.loads(line)))
    return records


def latest_registry_by_manifest_id(records: list[RegistryRecord]) -> dict[str, RegistryRecord]:
    latest: dict[str, RegistryRecord] = {}
    for record in records:
        latest[record.manifest_id] = record
    return latest


def find_failed_records(records: list[RegistryRecord]) -> list[RegistryRecord]:
    return [record for record in records if record.status in FAILED_REGISTRY_STATUSES]


def find_unsubmitted_records(
    manifest_ids: list[str],
    records: list[RegistryRecord],
) -> list[str]:
    latest = latest_registry_by_manifest_id(records)
    unsubmitted: list[str] = []
    for manifest_id in manifest_ids:
        latest_record = latest.get(manifest_id)
        if latest_record is None or latest_record.status in NON_SUBMITTED_STATUSES:
            unsubmitted.append(manifest_id)
    return unsubmitted

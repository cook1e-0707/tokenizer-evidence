from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.infrastructure.manifest import ManifestEntry, ManifestFile, ResourceRequest, update_manifest_status
from src.infrastructure.paths import (
    RunIdentity,
    build_run_identity,
    current_timestamp,
    discover_repo_root,
    ensure_run_dir,
    get_results_paths,
)
from src.infrastructure.registry import RegistryRecord, append_registry_record


DEFAULT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={stdout_path}
#SBATCH --error={stderr_path}
{account_directive}

set -euo pipefail

set +u
{environment_setup}
set -u

cd "$SLURM_SUBMIT_DIR"
{command}
"""


@dataclass(frozen=True)
class SlurmSubmission:
    manifest_id: str
    run_id: str
    status: str
    sbatch_command: str
    slurm_script_path: str
    output_dir: str
    stdout_path: str
    stderr_path: str
    submission_time: str
    slurm_job_id: str | None = None
    message: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_entry_command(
    entry: ManifestEntry,
    manifest_path: Path,
    identity: RunIdentity,
    output_dir: Path,
) -> str:
    script_path = Path(entry.entry_point)
    if script_path.suffix == ".py":
        command_parts = ["python3", str(script_path), "--config", entry.primary_config_path]
    else:
        command_parts = ["python3", entry.entry_point, "--config", entry.primary_config_path]
    command_parts.append("--force")
    for override in entry.overrides:
        if override.startswith("run.method_name="):
            command_parts.extend(["--override", override.replace("run.method_name=", "run.method=")])
        else:
            command_parts.extend(["--override", override])
    command_parts.extend(
        [
            "--override",
            f"runtime.manifest_id={entry.manifest_id}",
            "--override",
            f"runtime.manifest_path={manifest_path}",
            "--override",
            f"runtime.run_id={identity.run_id}",
            "--override",
            f"runtime.output_dir={output_dir}",
        ]
    )
    return shlex.join(command_parts)


def _resolve_template_text(template_path: Path | None) -> str:
    if template_path is None or not template_path.exists():
        return DEFAULT_TEMPLATE
    return template_path.read_text(encoding="utf-8")


def render_sbatch_script(
    entry: ManifestEntry,
    command: str,
    paths,
    resources: ResourceRequest | None = None,
    job_name: str | None = None,
    template_path: Path | None = None,
) -> str:
    requested = resources or entry.requested_resources
    template = _resolve_template_text(template_path)
    account_directive = f"#SBATCH --account={requested.account}" if requested.account else ""
    return template.format(
        job_name=job_name or entry.manifest_id,
        partition=requested.partition,
        account=requested.account or "CHANGE_ME",
        account_directive=account_directive,
        gpus=requested.num_gpus,
        cpus_per_task=requested.cpus,
        mem=f"{requested.mem_gb}G",
        time=requested.time_limit,
        log_dir=str(paths.run_dir),
        stdout_path=str(paths.stdout_path),
        stderr_path=str(paths.stderr_path),
        environment_setup=requested.environment_setup,
        command=command,
    )


def parse_sbatch_job_id(stdout: str) -> str | None:
    match = re.search(r"(\d+)", stdout)
    return match.group(1) if match else None


def prepare_submission(
    entry: ManifestEntry,
    manifest_path: Path,
    repo_root: Path | None = None,
    force: bool = False,
) -> tuple[RunIdentity, Any, str, Path]:
    resolved_repo_root = repo_root or discover_repo_root(manifest_path.parent)
    identity = build_run_identity(
        repo_root=resolved_repo_root,
        experiment_name=entry.experiment_name,
        method_name=entry.method_name,
        model_name=entry.model_name,
        seed=entry.seed,
    )
    paths = get_results_paths(
        repo_root=resolved_repo_root,
        output_root=entry.output_root,
        experiment_name=entry.experiment_name,
        run_id=identity.run_id,
    )
    ensure_run_dir(paths.run_dir, force=force)
    command = build_entry_command(entry, manifest_path, identity, paths.run_dir)

    template_path = None
    if entry.requested_resources.slurm_template:
        template_path = Path(entry.requested_resources.slurm_template)
        if not template_path.is_absolute():
            template_path = resolved_repo_root / template_path

    rendered = render_sbatch_script(
        entry=entry,
        command=command,
        paths=paths,
        template_path=template_path,
    )
    rendered_dir = resolved_repo_root / "manifests" / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    rendered_path = rendered_dir / f"{entry.manifest_id}.sbatch"
    rendered_path.write_text(rendered, encoding="utf-8")
    return identity, paths, command, rendered_path


def submit_manifest_entry(
    entry: ManifestEntry,
    manifest_path: Path,
    registry_path: Path,
    repo_root: Path | None = None,
    submit: bool = False,
    force: bool = False,
) -> SlurmSubmission:
    resolved_repo_root = repo_root or discover_repo_root(manifest_path.parent)
    identity, paths, _command, rendered_path = prepare_submission(
        entry=entry,
        manifest_path=manifest_path,
        repo_root=resolved_repo_root,
        force=force,
    )
    sbatch_command = shlex.join(["sbatch", str(rendered_path)])
    submission_time = current_timestamp()

    initial_payload = {
        "manifest_id": entry.manifest_id,
        "run_id": identity.run_id,
        "status": "prepared",
        "submission_time": submission_time,
        "sbatch_command": sbatch_command,
        "slurm_script_path": str(rendered_path),
        "output_dir": str(paths.run_dir),
        "stdout_path": str(paths.stdout_path),
        "stderr_path": str(paths.stderr_path),
    }
    paths.submission_path.write_text(json.dumps(initial_payload, indent=2, sort_keys=True), encoding="utf-8")

    if not submit:
        result = SlurmSubmission(
            manifest_id=entry.manifest_id,
            run_id=identity.run_id,
            status="dry_run",
            sbatch_command=sbatch_command,
            slurm_script_path=str(rendered_path),
            output_dir=str(paths.run_dir),
            stdout_path=str(paths.stdout_path),
            stderr_path=str(paths.stderr_path),
            submission_time=submission_time,
            message="Rendered sbatch script but did not submit.",
        )
    else:
        try:
            completed = subprocess.run(
                ["sbatch", str(rendered_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            stdout = completed.stdout.strip()
            result = SlurmSubmission(
                manifest_id=entry.manifest_id,
                run_id=identity.run_id,
                status="submitted",
                sbatch_command=sbatch_command,
                slurm_script_path=str(rendered_path),
                output_dir=str(paths.run_dir),
                stdout_path=str(paths.stdout_path),
                stderr_path=str(paths.stderr_path),
                submission_time=submission_time,
                slurm_job_id=parse_sbatch_job_id(stdout),
                message=stdout,
            )
        except (OSError, subprocess.SubprocessError) as error:
            result = SlurmSubmission(
                manifest_id=entry.manifest_id,
                run_id=identity.run_id,
                status="submission_error",
                sbatch_command=sbatch_command,
                slurm_script_path=str(rendered_path),
                output_dir=str(paths.run_dir),
                stdout_path=str(paths.stdout_path),
                stderr_path=str(paths.stderr_path),
                submission_time=submission_time,
                message=str(error),
            )

    paths.submission_path.write_text(
        json.dumps(result.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    append_registry_record(
        registry_path,
        RegistryRecord(
            manifest_id=entry.manifest_id,
            run_id=result.run_id,
            submission_time=result.submission_time,
            slurm_job_id=result.slurm_job_id,
            slurm_script_path=result.slurm_script_path,
            status=result.status,
            output_dir=result.output_dir,
            manifest_path=str(manifest_path),
            experiment_name=entry.experiment_name,
            method_name=entry.method_name,
            model_name=entry.model_name,
            seed=entry.seed,
            message=result.message,
        ),
    )
    update_manifest_status(manifest_path, entry.manifest_id, result.status)
    return result


def filter_manifest_entries(
    manifest_file: ManifestFile,
    manifest_id: str | None = None,
    experiment_name: str | None = None,
    method_name: str | None = None,
    tag: str | None = None,
    statuses: set[str] | None = None,
) -> list[ManifestEntry]:
    selected: list[ManifestEntry] = []
    for entry in manifest_file.entries:
        if manifest_id and entry.manifest_id != manifest_id:
            continue
        if experiment_name and entry.experiment_name != experiment_name:
            continue
        if method_name and entry.method_name != method_name:
            continue
        if tag and tag not in entry.tags:
            continue
        if statuses and entry.status not in statuses:
            continue
        selected.append(entry)
    return selected

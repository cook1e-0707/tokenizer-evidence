from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from src.infrastructure.paths import current_timestamp, get_git_hash


SELECTED_DEPENDENCIES = (
    "PyYAML",
    "numpy",
    "torch",
    "pytest",
)

SLURM_ENV_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_CLUSTER_NAME",
    "SLURM_SUBMIT_DIR",
)


@dataclass(frozen=True)
class EnvironmentSummary:
    git_commit: str
    git_dirty: bool | None
    python_version: str
    python_executable: str
    platform: str
    hostname: str
    cwd: str
    repo_root: str
    dependency_versions: dict[str, str]
    slurm_env: dict[str, str]
    timestamp: str

    @property
    def slurm_job_id(self) -> str | None:
        return self.slurm_env.get("SLURM_JOB_ID")

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def _git_dirty_state(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in SELECTED_DEPENDENCIES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _collect_slurm_env() -> dict[str, str]:
    return {key: value for key in SLURM_ENV_KEYS if (value := os.environ.get(key))}


def collect_environment_summary(repo_root: Path) -> EnvironmentSummary:
    return EnvironmentSummary(
        git_commit=get_git_hash(repo_root),
        git_dirty=_git_dirty_state(repo_root),
        python_version=sys.version.split()[0],
        python_executable=sys.executable,
        platform=platform.platform(),
        hostname=platform.node(),
        cwd=str(Path.cwd()),
        repo_root=str(repo_root),
        dependency_versions=_dependency_versions(),
        slurm_env=_collect_slurm_env(),
        timestamp=current_timestamp(),
    )


def save_environment_summary(summary: EnvironmentSummary, path: Path) -> Path:
    path.write_text(json.dumps(summary.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def collect_environment(repo_root: Path) -> EnvironmentSummary:
    return collect_environment_summary(repo_root)


def write_environment_summary(summary: EnvironmentSummary, path: Path) -> Path:
    return save_environment_summary(summary, path)

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class ExperimentName(str, Enum):
    ALIGNMENT = "exp_alignment"
    BUCKET = "exp_bucket"
    RECOVERY = "exp_recovery"
    MAIN = "exp_main"


EXPERIMENT_NAME_SET = {item.value for item in ExperimentName}
RESULT_DIR_NAMES = ("raw", "processed", "tables", "figures")


@dataclass(frozen=True)
class RunIdentity:
    experiment_name: str
    method_name: str
    model_name: str
    seed: int
    git_commit: str
    timestamp: str
    run_id: str


@dataclass(frozen=True)
class ResultsPaths:
    output_root: Path
    experiment_dir: Path
    run_dir: Path
    resolved_config_path: Path
    environment_path: Path
    submission_path: Path
    metrics_path: Path
    run_log_path: Path
    jsonl_log_path: Path
    stdout_path: Path
    stderr_path: Path


def discover_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def sanitize_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return normalized.strip("-") or "unknown"


def current_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def get_git_hash(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    git_hash = result.stdout.strip()
    return git_hash or "nogit"


def ensure_result_tree(repo_root: Path) -> dict[str, Path]:
    result_root = repo_root / "results"
    paths: dict[str, Path] = {}
    for name in RESULT_DIR_NAMES:
        path = result_root / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths


def resolve_output_root(repo_root: Path, output_root: str | Path) -> Path:
    output_path = Path(output_root)
    if output_path.is_absolute():
        return output_path
    return repo_root / output_path


def make_run_id(
    experiment_name: str,
    method_name: str,
    model_name: str,
    seed: int,
    git_commit: str,
    timestamp: str,
) -> str:
    return (
        f"{sanitize_component(experiment_name)}__"
        f"{sanitize_component(method_name)}__"
        f"{sanitize_component(model_name)}__"
        f"s{seed}__{sanitize_component(git_commit)}__{timestamp}"
    )


def build_run_identity(
    repo_root: Path,
    experiment_name: str,
    model_name: str,
    seed: int,
    method_name: str = "unknown_method",
    timestamp: str | None = None,
    git_commit: str | None = None,
) -> RunIdentity:
    resolved_timestamp = timestamp or current_timestamp()
    resolved_git_commit = git_commit or get_git_hash(repo_root)
    run_id = make_run_id(
        experiment_name=experiment_name,
        method_name=method_name,
        model_name=model_name,
        seed=seed,
        git_commit=resolved_git_commit,
        timestamp=resolved_timestamp,
    )
    return RunIdentity(
        experiment_name=experiment_name,
        method_name=method_name,
        model_name=model_name,
        seed=seed,
        git_commit=resolved_git_commit,
        timestamp=resolved_timestamp,
        run_id=run_id,
    )


def make_run_dir(output_root: Path, experiment_name: str, run_id: str) -> Path:
    return output_root / sanitize_component(experiment_name) / sanitize_component(run_id)


def ensure_run_dir(run_dir: Path, force: bool = False) -> Path:
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise FileExistsError(f"Run directory already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_results_paths(
    repo_root: Path,
    output_root: str | Path,
    experiment_name: str,
    run_id: str,
) -> ResultsPaths:
    ensure_result_tree(repo_root)
    resolved_root = resolve_output_root(repo_root, output_root)
    experiment_dir = resolved_root / sanitize_component(experiment_name)
    run_dir = make_run_dir(resolved_root, experiment_name, run_id)
    return ResultsPaths(
        output_root=resolved_root,
        experiment_dir=experiment_dir,
        run_dir=run_dir,
        resolved_config_path=run_dir / "config.resolved.yaml",
        environment_path=run_dir / "environment.json",
        submission_path=run_dir / "submission.json",
        metrics_path=run_dir / "metrics.json",
        run_log_path=run_dir / "run.log",
        jsonl_log_path=run_dir / "run.jsonl",
        stdout_path=run_dir / "stdout.log",
        stderr_path=run_dir / "stderr.log",
    )


def create_run_directory(
    repo_root: Path,
    output_root: str | Path,
    identity: RunIdentity,
    force: bool = False,
) -> Path:
    paths = get_results_paths(
        repo_root=repo_root,
        output_root=output_root,
        experiment_name=identity.experiment_name,
        run_id=identity.run_id,
    )
    ensure_run_dir(paths.run_dir, force=force)
    return paths.run_dir

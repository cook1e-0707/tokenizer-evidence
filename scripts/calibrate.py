from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from src.evaluation.calibration import calibrate_far_threshold
from src.evaluation.far_eval import evaluate_far
from src.evaluation.report import CalibrationSummary
from src.infrastructure.config import load_experiment_config, save_resolved_config
from src.infrastructure.environment import collect_environment, write_environment_summary
from src.infrastructure.logging import log_startup, setup_logging
from src.infrastructure.paths import (
    RunIdentity,
    build_run_identity,
    discover_repo_root,
    ensure_run_dir,
    get_git_hash,
    get_results_paths,
)


DEFAULT_SCORES = [0.97, 0.91, 0.84, 0.15, 0.10, 0.03]
DEFAULT_LABELS = [True, True, True, False, False, False]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a threshold for a target FAR.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override.")
    parser.add_argument("--scores-file", help="Optional JSON file containing {'scores': [...], 'labels': [...]} .")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing run dir.")
    parser.add_argument("--jsonl-log", action="store_true", help="Enable JSONL machine logs.")
    return parser.parse_args()


def load_scores(path: Path | None) -> tuple[list[float], list[bool]]:
    if path is None:
        return list(DEFAULT_SCORES), list(DEFAULT_LABELS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["scores"]), [bool(value) for value in payload["labels"]]


def _resolve_run_paths(repo_root: Path, config: object, force: bool) -> tuple[RunIdentity, object]:
    if config.runtime.run_id and config.runtime.output_dir:
        identity = RunIdentity(
            experiment_name=config.experiment_name,
            method_name=config.method_name,
            model_name=config.model_name,
            seed=config.seed,
            git_commit=get_git_hash(repo_root),
            timestamp="from_runtime",
            run_id=config.runtime.run_id,
        )
        output_root = Path(config.runtime.output_dir).parent.parent
    else:
        identity = build_run_identity(
            repo_root,
            config.experiment_name,
            config.model_name,
            config.seed,
            method_name=config.method_name,
        )
        output_root = config.output_root

    paths = get_results_paths(repo_root, output_root, config.experiment_name, identity.run_id)
    if config.runtime.output_dir:
        paths = get_results_paths(
            repo_root,
            Path(config.runtime.output_dir).parent.parent,
            config.experiment_name,
            identity.run_id,
        )
    ensure_run_dir(paths.run_dir, force=force)
    return identity, paths


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    config = load_experiment_config(config_path, overrides=args.override)
    identity, paths = _resolve_run_paths(repo_root, config, force=args.force)

    save_resolved_config(config, paths.resolved_config_path)
    environment = collect_environment(repo_root)
    write_environment_summary(environment, paths.environment_path)
    logger = setup_logging(paths.run_dir, run_id=identity.run_id, enable_jsonl=args.jsonl_log)
    log_startup(
        logger,
        config_summary={"run": config.run, "eval": config.eval},
        environment_summary=environment,
    )

    scores_file = Path(args.scores_file).resolve() if args.scores_file else None
    scores, labels = load_scores(scores_file)
    calibration = calibrate_far_threshold(scores, labels, target_far=config.eval.target_far)
    far_summary = evaluate_far(scores, labels, threshold=calibration.threshold)
    threshold_candidates = tuple(sorted({float(score) for score in scores}, reverse=True))

    output = CalibrationSummary(
        run_id=identity.run_id,
        experiment_name=config.experiment_name,
        method_name=config.method_name,
        model_name=config.model_name,
        seed=config.seed,
        git_commit=identity.git_commit,
        timestamp=environment.timestamp,
        hostname=environment.hostname,
        slurm_job_id=environment.slurm_job_id,
        status="completed",
        target_far=config.eval.target_far,
        threshold=calibration.threshold,
        observed_far=calibration.observed_far,
        sample_count=calibration.sample_count,
        calibration_target="false_accept_rate",
        operating_point_name="threshold",
        threshold_candidates=threshold_candidates,
        selected_metric_name="observed_far",
        selected_metric_value=far_summary.false_accept_rate,
        notes="threshold selected from provided score sweep",
        run_dir=str(paths.run_dir),
    )
    canonical_path = paths.run_dir / "calibration_summary.json"
    output.save_json(canonical_path)
    logger.info("wrote calibration summary to %s", canonical_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

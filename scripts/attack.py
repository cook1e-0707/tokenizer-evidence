from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
from dataclasses import replace
from pathlib import Path

from src.core.parser import load_evidence_records, load_expected_symbols
from src.core.verifier import VerificationConfig, verify_records
from src.evaluation.report import AttackOutput
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
from src.infrastructure.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight attack smoke test.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing run dir.")
    parser.add_argument("--jsonl-log", action="store_true", help="Enable JSONL machine logs.")
    return parser.parse_args()


def apply_attack(mode: str, strength: float, records):
    if mode == "whitespace_scrub":
        return [
            replace(record, score=max(0.0, record.score - strength), symbol=record.symbol.replace(" ", ""))
            for record in records
        ]
    if mode == "truncate_tail":
        keep = max(1, len(records) - max(1, int(strength * len(records))))
        return list(records[:keep])
    return list(records)


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    config = load_experiment_config(config_path, overrides=args.override)
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
    ensure_run_dir(paths.run_dir, force=args.force)

    save_resolved_config(config, paths.resolved_config_path)
    environment = collect_environment(repo_root)
    write_environment_summary(environment, paths.environment_path)
    logger = setup_logging(paths.run_dir, run_id=identity.run_id, enable_jsonl=args.jsonl_log)
    log_startup(
        logger,
        config_summary={"run": config.run, "attack": config.attack, "eval": config.eval},
        environment_summary=environment,
    )
    set_global_seed(config.run.seed)

    fixture_path = Path(config.data.eval_path)
    if not fixture_path.is_absolute():
        fixture_path = repo_root / fixture_path
    records = load_evidence_records(fixture_path)
    expected = load_expected_symbols(fixture_path)

    verify_config = VerificationConfig(
        min_score=config.eval.min_score,
        max_candidates=config.eval.max_candidates,
        min_match_ratio=1.0,
        scan_windows=True,
    )
    before = verify_records(records, expected, verify_config)
    after_records = apply_attack(config.attack.mode, config.attack.strength, records)
    after = verify_records(after_records, expected, verify_config)

    output = AttackOutput(
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
        attack_name=config.attack.name,
        perturbation=config.attack.mode,
        sample_count=1,
        accepted_before=before.accepted,
        accepted_after=after.accepted,
        run_dir=str(paths.run_dir),
    )
    output.save_json(paths.run_dir / "attack_output.json")
    logger.info("wrote attack output to %s", paths.run_dir / "attack_output.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

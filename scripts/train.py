from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from src.evaluation.report import TrainRunSummary
from src.infrastructure.checkpointing import save_checkpoint_metadata
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
from src.training.dataset import load_training_examples
from src.training.hf_causal_lm import run_minimal_hf_causal_lm_training
from src.training.trainer import TrainingPlan, execute_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal single-run training experiment.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Explicit dotted override, e.g. run.seed=13",
    )
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing run dir.")
    parser.add_argument("--jsonl-log", action="store_true", help="Enable JSONL machine logs.")
    return parser.parse_args()


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
            repo_root=repo_root,
            experiment_name=config.experiment_name,
            method_name=config.method_name,
            model_name=config.model_name,
            seed=config.seed,
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
        config_summary={
            "run": config.run,
            "model": config.model,
            "data": config.data,
            "train": config.train,
        },
        environment_summary=environment,
    )
    seed_report = set_global_seed(config.run.seed)
    logger.info("seed report: %s", seed_report)

    data_path = Path(config.data.train_path)
    if not data_path.is_absolute():
        data_path = repo_root / data_path
    dataset = load_training_examples(data_path)

    checkpoint_path: str
    generated_text: str
    if config.model.family == "huggingface-causal-lm":
        training_result = run_minimal_hf_causal_lm_training(
            model_name_or_path=config.model.tokenizer_name or config.model.name,
            max_length=config.model.max_length,
            dataset=dataset,
            batch_size=config.train.batch_size,
            epochs=config.train.epochs,
            learning_rate=config.train.learning_rate,
            run_dir=paths.run_dir,
            require_cuda=config.runtime.resources.num_gpus > 0,
        )
        status = training_result.status
        steps = training_result.steps
        examples_seen = training_result.examples_seen
        final_loss = training_result.final_loss
        checkpoint_path = training_result.checkpoint_dir
        generated_text = training_result.generated_text
    else:
        plan = TrainingPlan(
            dataset_name=config.data.name,
            objective=config.train.objective,
            batch_size=config.train.batch_size,
            epochs=config.train.epochs,
            learning_rate=config.train.learning_rate,
        )
        outcome = execute_training(plan, dataset_size=len(dataset))
        status = outcome.status
        steps = outcome.steps
        examples_seen = outcome.examples_seen
        final_loss = outcome.final_loss
        checkpoint_path = str(save_checkpoint_metadata(paths.run_dir, "latest", outcome.to_dict()))
        generated_text = dataset[0].prompt if dataset else config.eval.payload_text

    generated_text_path = paths.run_dir / "generated_text.txt"
    generated_text_path.write_text(generated_text, encoding="utf-8")

    eval_input_payload = {
        "schema_name": "train_eval_input",
        "source_train_run_id": identity.run_id,
        "source_experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "payload_text": config.eval.payload_text,
        "checkpoint_path": checkpoint_path,
        "generated_text_path": str(generated_text_path),
    }
    eval_input_path = paths.run_dir / "eval_input.json"
    eval_input_path.write_text(
        json.dumps(eval_input_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    latest_eval_input_path = paths.experiment_dir / "latest_eval_input.json"
    latest_eval_input_path.parent.mkdir(parents=True, exist_ok=True)
    latest_eval_input_path.write_text(
        json.dumps(eval_input_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    save_checkpoint_metadata(
        paths.run_dir,
        "latest",
        {
            "status": status,
            "steps": steps,
            "examples_seen": examples_seen,
            "final_loss": final_loss,
            "checkpoint_path": checkpoint_path,
            "generated_text_path": str(generated_text_path),
            "eval_input_path": str(eval_input_path),
            "latest_eval_input_path": str(latest_eval_input_path),
        },
    )

    summary = TrainRunSummary(
        run_id=identity.run_id,
        experiment_name=config.experiment_name,
        method_name=config.method_name,
        model_name=config.model_name,
        seed=config.seed,
        git_commit=identity.git_commit,
        timestamp=environment.timestamp,
        hostname=environment.hostname,
        slurm_job_id=environment.slurm_job_id,
        status=status,
        objective=config.train.objective,
        dataset_name=config.data.name,
        dataset_size=len(dataset),
        steps=steps,
        final_loss=final_loss,
        run_dir=str(paths.run_dir),
    )
    summary.save_json(paths.run_dir / "train_summary.json")
    logger.info("wrote training summary to %s", paths.run_dir / "train_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

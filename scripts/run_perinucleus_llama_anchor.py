from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.run_perinucleus_official_smoke import (
    _apply_compatibility_patches,
    _apply_override,
    _extract_accuracy,
    _extract_fingerprint_path,
    _get,
    _last_config_hash,
    _read,
    _run,
)
from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official-code Perinucleus Llama anchor reproduction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _mean_prob(score: dict[str, Any] | None) -> float | None:
    if not isinstance(score, dict):
        return None
    value = score.get("mean_first_token_probability")
    return float(value) if value is not None else None


def _mean_logprob(score: dict[str, Any] | None) -> float | None:
    if not isinstance(score, dict):
        return None
    value = score.get("mean_sequence_logprob")
    return float(value) if value is not None else None


def _stage_pass(row: dict[str, Any]) -> bool:
    base_acc = row.get("base_fingerprint_accuracy")
    trained_acc = row.get("trained_fingerprint_accuracy")
    base_prob = row.get("base_mean_first_token_probability")
    trained_prob = row.get("trained_mean_first_token_probability")
    return bool(
        trained_acc is not None
        and base_acc is not None
        and float(trained_acc) > float(base_acc)
        and float(trained_acc) > 0.0
        and trained_prob is not None
        and base_prob is not None
        and float(trained_prob) > float(base_prob)
    )


def _write_outputs(
    repo_root: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    compute: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    result_doc = repo_root / str(_get(config, "repo_outputs.result_doc"))
    table_path = repo_root / str(_get(config, "repo_outputs.table"))
    summary_path = repo_root / str(_get(config, "repo_outputs.summary"))
    compute_path = repo_root / str(_get(config, "repo_outputs.compute"))
    for path in [result_doc, table_path, summary_path, compute_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    compute_path.write_text(json.dumps(compute, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = [
        "stage",
        "num_fingerprints",
        "base_fingerprint_accuracy",
        "trained_fingerprint_accuracy",
        "base_mean_first_token_probability",
        "trained_mean_first_token_probability",
        "base_mean_sequence_logprob",
        "trained_mean_sequence_logprob",
        "accuracy_improved",
        "probability_improved",
        "stage_pass",
        "utility_status",
        "config_hash",
        "fingerprints_file",
        "finetuned_model_path",
        "run_root",
    ]
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    stage_lines = [
        "| {stage} | {n} | {base_acc} | {trained_acc} | {base_prob} | {trained_prob} | {passed} | {utility} |".format(
            stage=row["stage"],
            n=row["num_fingerprints"],
            base_acc=row.get("base_fingerprint_accuracy"),
            trained_acc=row.get("trained_fingerprint_accuracy"),
            base_prob=row.get("base_mean_first_token_probability"),
            trained_prob=row.get("trained_mean_first_token_probability"),
            passed=row.get("stage_pass"),
            utility=row.get("utility_status"),
        )
        for row in rows
    ]
    result_doc.write_text(
        "\n".join(
            [
                "# Perinucleus Llama Anchor Result",
                "",
                f"Generated at: `{summary['generated_at']}`",
                f"Decision: `{summary['decision']}`",
                f"Model: `{summary['model']}`",
                f"Official commit: `{summary['official_commit_actual']}`",
                f"Training mode: `{summary['training_mode']}`",
                f"Scratch run root: `{summary['run_root']}`",
                "",
                "This is an official-code Llama anchor reproduction. It is not a Qwen matched-budget final matrix.",
                "",
                "## Stage Metrics",
                "",
                "| stage | fingerprints | base acc | trained acc | base target prob | trained target prob | pass | utility |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
                *stage_lines,
                "",
                "## Decision",
                "",
                summary["decision"],
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = _load_yaml(config_path)
    for override in args.override:
        _apply_override(config, override)

    anchor = dict(config["anchor"])
    scratch_root = Path(str(anchor["scratch_root"]))
    run_id = str(_get(config, "runtime.run_id") or f"manual_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    runtime_output_dir = _get(config, "runtime.output_dir")
    run_root = Path(str(runtime_output_dir)) if runtime_output_dir else scratch_root / "runs" / run_id
    logs_dir = run_root / "logs"
    generated_root = run_root / "generated"
    official_results_root = run_root / "official_results"
    scores_root = run_root / "scores"

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": str(config_path),
                    "run_root": str(run_root),
                    "model": anchor["model"],
                    "stages": anchor["stages"],
                    "training_mode": "lora" if anchor.get("use_lora", False) else "full_finetune",
                    "outputs": config["repo_outputs"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    for path in [logs_dir, generated_root, official_results_root, scores_root]:
        path.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env.setdefault("WANDB_MODE", "disabled")
    base_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    base_env.setdefault("HF_HOME", str(scratch_root / "hf_home"))
    base_env.setdefault("TRANSFORMERS_CACHE", str(scratch_root / "hf_home"))

    official_repo = repo_root / str(_get(config, "official_repo.local_path"))
    repo_url = str(_get(config, "official_repo.url"))
    expected_commit = str(_get(config, "official_repo.commit_hash"))

    stages: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    overall_rc = 0

    if not (official_repo / ".git").exists():
        official_repo.parent.mkdir(parents=True, exist_ok=True)
        stage = _run("clone_official_repo", ["git", "clone", repo_url, str(official_repo)], cwd=repo_root, env=base_env, log_dir=logs_dir)
        stages.append(stage)
        overall_rc = stage["returncode"]
    if overall_rc == 0:
        for name, cmd in [
            ("checkout_official_commit", ["git", "-C", str(official_repo), "checkout", expected_commit]),
            ("record_official_commit", ["git", "-C", str(official_repo), "rev-parse", "HEAD"]),
        ]:
            stage = _run(name, cmd, cwd=repo_root, env=base_env, log_dir=logs_dir)
            stages.append(stage)
            if stage["returncode"] != 0:
                overall_rc = stage["returncode"]
                break

    compatibility_patches: list[dict[str, Any]] = []
    if overall_rc == 0:
        patch_stages, compatibility_patches, patch_rc = _apply_compatibility_patches(
            config=config,
            official_repo=official_repo,
            logs_dir=logs_dir,
        )
        stages.extend(patch_stages)
        if patch_rc != 0:
            overall_rc = patch_rc

    model = str(anchor["model"])
    seed = int(anchor["seed"])
    use_chat_template = bool(anchor.get("use_chat_template", False))
    stopped_after_failure = False

    if overall_rc == 0:
        for stage_cfg in anchor["stages"]:
            stage_name = str(stage_cfg["name"])
            num_fingerprints = int(stage_cfg["num_fingerprints"])
            stage_generated_dir = generated_root / stage_name
            stage_score_dir = scores_root / stage_name
            stage_result_dir = official_results_root / stage_name
            for path in [stage_generated_dir, stage_score_dir, stage_result_dir]:
                path.mkdir(parents=True, exist_ok=True)

            fingerprint_path: Path | None = None
            generate_output = stage_generated_dir / "fingerprints.json"
            cmd = [
                "python3",
                "generate_finetuning_data.py",
                "--num_fingerprints",
                str(num_fingerprints),
                "--response_length",
                str(anchor["response_length"]),
                "--key_length",
                str(anchor["key_length"]),
                "--batch_size",
                str(anchor.get("generation_batch_size", 1)),
                "--seed",
                str(seed),
                "--key_response_strategy",
                "perinucleus",
                "--model_used_for_key_generation",
                model,
                "--perinucleus_model",
                model,
                "--nucleus_t",
                str(anchor.get("nucleus_t", 0.8)),
                "--nucleus_k",
                str(anchor.get("nucleus_k", 3)),
                "--output_file_path",
                str(generate_output),
            ]
            if use_chat_template:
                cmd.append("--use_chat_template")
            stage = _run(f"{stage_name}_generate_fingerprints", cmd, cwd=official_repo, env=base_env, log_dir=logs_dir)
            stages.append(stage)
            if stage["returncode"] != 0:
                overall_rc = stage["returncode"]
                stopped_after_failure = True
                break
            fingerprint_path = _extract_fingerprint_path(_read(Path(stage["stdout_path"])), stage_generated_dir)
            if fingerprint_path is None or not fingerprint_path.exists():
                overall_rc = 20
                stages.append(
                    {
                        "name": f"{stage_name}_locate_fingerprints",
                        "status": "failed",
                        "returncode": 20,
                        "seconds": 0.0,
                        "stdout_path": str(fingerprint_path or ""),
                        "stderr_path": "",
                        "command_path": "",
                    }
                )
                stopped_after_failure = True
                break
            stages.append(
                {
                    "name": f"{stage_name}_locate_fingerprints",
                    "status": "completed",
                    "returncode": 0,
                    "seconds": 0.0,
                    "stdout_path": str(fingerprint_path),
                    "stderr_path": "",
                    "command_path": "",
                }
            )

            base_accuracy = None
            base_score = None
            cmd = [
                "python3",
                "check_fingerprints.py",
                "--model_path",
                model,
                "--num_fingerprints",
                str(num_fingerprints),
                "--fingerprints_file_path",
                str(fingerprint_path),
                "--fingerprint_generation_strategy",
                "perinucleus",
                "--max_key_length",
                str(anchor["key_length"]),
                "--max_response_length",
                str(anchor["response_length"]),
                "--seed",
                str(seed),
                "--verbose_eval",
            ]
            stage = _run(f"{stage_name}_check_base_fingerprints", cmd, cwd=official_repo, env=base_env, log_dir=logs_dir)
            stages.append(stage)
            if stage["returncode"] != 0:
                overall_rc = stage["returncode"]
                stopped_after_failure = True
                break
            base_accuracy = _extract_accuracy(_read(Path(stage["stdout_path"])))

            base_score_path = stage_score_dir / "base_response_probability.json"
            cmd = [
                "python3",
                str(repo_root / "scripts" / "score_perinucleus_official_probs.py"),
                "--model-path",
                model,
                "--fingerprints-file",
                str(fingerprint_path),
                "--output",
                str(base_score_path),
                "--limit",
                str(num_fingerprints),
            ]
            if use_chat_template:
                cmd.append("--use-chat-template")
            stage = _run(f"{stage_name}_score_base_probabilities", cmd, cwd=repo_root, env=base_env, log_dir=logs_dir)
            stages.append(stage)
            if stage["returncode"] != 0:
                overall_rc = stage["returncode"]
                stopped_after_failure = True
                break
            base_score = json.loads(base_score_path.read_text(encoding="utf-8")) if base_score_path.exists() else None

            cmd = [
                "deepspeed",
                "--num_gpus",
                str(anchor.get("deepspeed_num_gpus", 1)),
                "--master_port",
                str(anchor.get("deepspeed_master_port", 29517)),
                "finetune_multigpu.py",
                "--model_path",
                model,
                "--model_family",
                str(anchor.get("model_family", "llama")),
                "--model_size",
                str(anchor.get("model_size", "8B")),
                "--num_fingerprints",
                str(num_fingerprints),
                "--max_key_length",
                str(anchor["key_length"]),
                "--max_response_length",
                str(anchor["response_length"]),
                "--num_train_epochs",
                str(stage_cfg.get("train_epochs", anchor["train_epochs"])),
                "--learning_rate",
                str(anchor["learning_rate"]),
                "--batch_size",
                str(anchor["batch_size"]),
                "--fingerprint_generation_strategy",
                "perinucleus",
                "--fingerprints_file_path",
                str(fingerprint_path),
                "--forgetting_regularizer_strength",
                str(anchor.get("forgetting_regularizer_strength", 0.0)),
                "--benign_proportion",
                str(anchor.get("benign_proportion", 0.0)),
                "--seed",
                str(seed),
                "--result_path",
                str(stage_result_dir) + "/",
            ]
            if use_chat_template:
                cmd.append("--use_chat_template")
            if bool(anchor.get("use_lora", False)):
                cmd.extend(["--use_lora", "--lora_rank", str(anchor["lora_rank"]), "--lora_alpha_ratio", str(anchor["lora_alpha_ratio"])])
            stage = _run(f"{stage_name}_finetune_insert_fingerprints", cmd, cwd=official_repo, env=base_env, log_dir=logs_dir)
            stages.append(stage)
            if stage["returncode"] != 0:
                overall_rc = stage["returncode"]
                stopped_after_failure = True
                break

            config_hash = _last_config_hash(official_repo)
            model_path = stage_result_dir / "saved_models" / str(config_hash) / "final_model" if config_hash else None
            if model_path is None or not model_path.exists():
                overall_rc = 30
                stages.append(
                    {
                        "name": f"{stage_name}_locate_finetuned_model",
                        "status": "failed",
                        "returncode": 30,
                        "seconds": 0.0,
                        "stdout_path": str(model_path or ""),
                        "stderr_path": "",
                        "command_path": "",
                    }
                )
                stopped_after_failure = True
                break
            stages.append(
                {
                    "name": f"{stage_name}_locate_finetuned_model",
                    "status": "completed",
                    "returncode": 0,
                    "seconds": 0.0,
                    "stdout_path": str(model_path),
                    "stderr_path": "",
                    "command_path": "",
                }
            )

            trained_accuracy = None
            trained_score = None
            cmd = [
                "python3",
                "check_fingerprints.py",
                "--model_path",
                str(model_path),
                "--num_fingerprints",
                str(num_fingerprints),
                "--fingerprints_file_path",
                str(fingerprint_path),
                "--fingerprint_generation_strategy",
                "perinucleus",
                "--max_key_length",
                str(anchor["key_length"]),
                "--max_response_length",
                str(anchor["response_length"]),
                "--seed",
                str(seed),
                "--verbose_eval",
            ]
            stage = _run(f"{stage_name}_check_finetuned_fingerprints", cmd, cwd=official_repo, env=base_env, log_dir=logs_dir)
            stages.append(stage)
            if stage["returncode"] != 0:
                overall_rc = stage["returncode"]
                stopped_after_failure = True
                break
            trained_accuracy = _extract_accuracy(_read(Path(stage["stdout_path"])))

            trained_score_path = stage_score_dir / "trained_response_probability.json"
            cmd = [
                "python3",
                str(repo_root / "scripts" / "score_perinucleus_official_probs.py"),
                "--model-path",
                str(model_path),
                "--fingerprints-file",
                str(fingerprint_path),
                "--output",
                str(trained_score_path),
                "--limit",
                str(num_fingerprints),
            ]
            if use_chat_template:
                cmd.append("--use-chat-template")
            stage = _run(f"{stage_name}_score_finetuned_probabilities", cmd, cwd=repo_root, env=base_env, log_dir=logs_dir)
            stages.append(stage)
            if stage["returncode"] != 0:
                overall_rc = stage["returncode"]
                stopped_after_failure = True
                break
            trained_score = json.loads(trained_score_path.read_text(encoding="utf-8")) if trained_score_path.exists() else None

            utility_status = "not_run"
            if bool(anchor.get("run_utility", True)):
                cmd = ["python3", "eval_utility.py", "--model_path", str(model_path), "--eval_batch_size", str(anchor.get("eval_batch_size", 4))]
                if bool(anchor.get("utility_tiny_benchmarks", True)):
                    cmd.append("--tinyBenchmarks")
                stage = _run(f"{stage_name}_eval_utility_sanity", cmd, cwd=official_repo, env=base_env, log_dir=logs_dir, allow_failure=True)
                stages.append(stage)
                if stage["returncode"] == 0:
                    utility_status = "completed"
                else:
                    stderr = _read(Path(stage["stderr_path"]))
                    utility_status = "pending_environment_dependency" if "tinyBenchmarks" in stderr or "ModuleNotFoundError" in stderr else "failed"

            row = {
                "stage": stage_name,
                "num_fingerprints": num_fingerprints,
                "base_fingerprint_accuracy": base_accuracy,
                "trained_fingerprint_accuracy": trained_accuracy,
                "base_mean_first_token_probability": _mean_prob(base_score),
                "trained_mean_first_token_probability": _mean_prob(trained_score),
                "base_mean_sequence_logprob": _mean_logprob(base_score),
                "trained_mean_sequence_logprob": _mean_logprob(trained_score),
                "accuracy_improved": trained_accuracy is not None and base_accuracy is not None and float(trained_accuracy) > float(base_accuracy),
                "probability_improved": _mean_prob(trained_score) is not None and _mean_prob(base_score) is not None and float(_mean_prob(trained_score)) > float(_mean_prob(base_score)),
                "utility_status": utility_status,
                "config_hash": config_hash,
                "fingerprints_file": str(fingerprint_path),
                "finetuned_model_path": str(model_path),
                "run_root": str(run_root),
            }
            row["stage_pass"] = _stage_pass(row)
            rows.append(row)
            if not row["stage_pass"]:
                stopped_after_failure = True
                break

    actual_commit = None
    commit_stdout = logs_dir / "record_official_commit.stdout.log"
    if commit_stdout.exists():
        lines = [line.strip() for line in commit_stdout.read_text(encoding="utf-8").splitlines() if line.strip()]
        actual_commit = lines[-1] if lines else None

    all_stages_completed = overall_rc == 0 and all(stage["returncode"] == 0 for stage in stages if not stage["name"].endswith("_eval_utility_sanity"))
    all_rows_pass = bool(rows) and all(bool(row["stage_pass"]) for row in rows)
    anchor_pass = bool(all_stages_completed and all_rows_pass and not stopped_after_failure)
    decision = (
        "LLAMA_ANCHOR_PASS: official-code Llama anchor passed; Qwen capacity sweep may be considered after review."
        if anchor_pass
        else "LLAMA_ANCHOR_NOT_PASSED: do not run Qwen capacity sweep or final matrix; inspect stage logs."
    )

    summary = {
        "schema_name": "baseline_perinucleus_llama_anchor_summary",
        "schema_version": 1,
        "generated_at": _utc_now(),
        "anchor_pass": anchor_pass,
        "decision": decision,
        "model": model,
        "model_family": anchor.get("model_family"),
        "seed": seed,
        "response_length": anchor["response_length"],
        "use_chat_template": use_chat_template,
        "training_mode": "lora_adaptation" if anchor.get("use_lora", False) else "full_finetune",
        "lora_rank": anchor.get("lora_rank"),
        "official_repo": str(official_repo),
        "official_commit_expected": expected_commit,
        "official_commit_actual": actual_commit,
        "official_license": _get(config, "official_repo.license"),
        "run_root": str(run_root),
        "compatibility_patches": compatibility_patches,
        "rows": rows,
        "stages": stages,
    }
    compute = {
        "schema_name": "baseline_perinucleus_llama_anchor_compute",
        "schema_version": 1,
        "generated_at": summary["generated_at"],
        "run_root": str(run_root),
        "stage_seconds": {stage["name"]: stage["seconds"] for stage in stages},
        "total_seconds": sum(float(stage["seconds"]) for stage in stages),
        "requested_resources": {
            "num_gpus": int(os.environ.get("SLURM_GPUS_ON_NODE", anchor.get("deepspeed_num_gpus", 1))),
            "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    _write_outputs(repo_root, config, summary, compute, rows)
    return 0 if anchor_pass else (overall_rc if overall_rc else 2)


if __name__ == "__main__":
    raise SystemExit(main())

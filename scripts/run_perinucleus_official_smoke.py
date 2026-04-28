from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official Perinucleus smoke pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _coerce(raw: str) -> Any:
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        return raw


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        return
    key, value = override.split("=", 1)
    parts = key.split(".")
    cursor: dict[str, Any] = config
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, dict):
            current = {}
            cursor[part] = current
        cursor = current
    cursor[parts[-1]] = _coerce(value)


def _get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _run(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_dir: Path,
    allow_failure: bool = False,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    command_path = log_dir / f"{name}.command.txt"
    command_path.write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=stdout, stderr=stderr, text=True)
        except OSError as error:
            stderr.write(f"{type(error).__name__}: {error}\n")
            elapsed = time.time() - started
            return {
                "name": name,
                "status": "failed",
                "returncode": 127,
                "seconds": elapsed,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "command_path": str(command_path),
            }
    elapsed = time.time() - started
    status = "completed" if proc.returncode == 0 else "failed"
    if proc.returncode != 0 and not allow_failure:
        return {
            "name": name,
            "status": status,
            "returncode": proc.returncode,
            "seconds": elapsed,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "command_path": str(command_path),
        }
    return {
        "name": name,
        "status": status,
        "returncode": proc.returncode,
        "seconds": elapsed,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command_path": str(command_path),
    }


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _last_config_hash(official_repo: Path) -> str | None:
    path = official_repo / "current_config_hash.txt"
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[-1] if lines else None


def _extract_fingerprint_path(generate_stdout: str, generated_dir: Path) -> Path | None:
    matches = re.findall(r"Wrote fingerprints to ([^,\n]+)", generate_stdout)
    if matches:
        candidate = Path(matches[-1].strip())
        return candidate if candidate.is_absolute() else generated_dir / candidate
    files = sorted(generated_dir.glob("fingerprints-perinucleus-*.json"))
    return files[-1] if files else None


def _extract_accuracy(stdout_text: str) -> float | None:
    matches = re.findall(r"Fingerprint accuracy:\s*([0-9.eE+-]+)", stdout_text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _write_outputs(
    *,
    repo_root: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    compute: dict[str, Any],
    table_row: dict[str, Any],
) -> None:
    result_doc = repo_root / str(_get(config, "repo_outputs.result_doc"))
    table_path = repo_root / str(_get(config, "repo_outputs.table"))
    summary_path = repo_root / str(_get(config, "repo_outputs.summary"))
    compute_path = repo_root / str(_get(config, "repo_outputs.compute"))
    for path in [result_doc, table_path, summary_path, compute_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    compute_path.write_text(json.dumps(compute, indent=2, sort_keys=True), encoding="utf-8")
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_row.keys()))
        writer.writeheader()
        writer.writerow(table_row)

    stage_lines = "\n".join(
        f"- `{stage['name']}`: `{stage['status']}` (`returncode={stage['returncode']}`, `{stage['seconds']:.1f}s`)"
        for stage in summary["stages"]
    )
    result_doc.write_text(
        "\n".join(
            [
                "# Official Perinucleus Smoke Result",
                "",
                f"Generated at: `{summary['generated_at']}`",
                "",
                f"Smoke pass: `{summary['smoke_pass']}`",
                f"Model: `{summary['model']}`",
                f"Official commit: `{summary['official_commit_actual']}`",
                f"Scratch run root: `{summary['run_root']}`",
                "",
                "## Stage Status",
                "",
                stage_lines,
                "",
                "## Gate Metrics",
                "",
                f"- Base fingerprint accuracy: `{summary.get('base_fingerprint_accuracy')}`",
                f"- Trained fingerprint accuracy: `{summary.get('trained_fingerprint_accuracy')}`",
                f"- Base mean first-token probability: `{summary.get('base_mean_first_token_probability')}`",
                f"- Trained mean first-token probability: `{summary.get('trained_mean_first_token_probability')}`",
                f"- Utility status: `{summary.get('utility_status')}`",
                f"- Chat template used: `{summary.get('use_chat_template')}`",
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

    smoke = dict(config["smoke"])
    scratch_root = Path(str(smoke["scratch_root"]))
    runtime_output_dir = _get(config, "runtime.output_dir")
    run_id = str(_get(config, "runtime.run_id", f"manual_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"))
    run_root = Path(str(runtime_output_dir)) if runtime_output_dir else scratch_root / "runs" / run_id
    logs_dir = run_root / "logs"
    generated_dir = run_root / "generated"
    train_result_dir = run_root / "official_results"
    score_dir = run_root / "scores"
    for path in [logs_dir, generated_dir, train_result_dir, score_dir]:
        path.mkdir(parents=True, exist_ok=True)

    official_repo = repo_root / str(_get(config, "official_repo.local_path"))
    repo_url = str(_get(config, "official_repo.url"))
    expected_commit = str(_get(config, "official_repo.commit_hash"))

    base_env = os.environ.copy()
    base_env.setdefault("WANDB_MODE", "disabled")
    base_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    base_env.setdefault("HF_HOME", str(scratch_root / "hf_home"))
    base_env.setdefault("TRANSFORMERS_CACHE", str(scratch_root / "hf_home"))

    stages: list[dict[str, Any]] = []
    overall_rc = 0

    if not (official_repo / ".git").exists():
        official_repo.parent.mkdir(parents=True, exist_ok=True)
        stage = _run("clone_official_repo", ["git", "clone", repo_url, str(official_repo)], cwd=repo_root, env=base_env, log_dir=logs_dir)
        stages.append(stage)
        if stage["returncode"] != 0:
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

    python_bin = "python3"
    deepspeed_bin = "deepspeed"
    if overall_rc == 0 and bool(smoke.get("create_venv", False)):
        venv_path = Path(str(smoke["venv_path"]))
        stage = _run("create_venv", ["python3", "-m", "venv", str(venv_path)], cwd=repo_root, env=base_env, log_dir=logs_dir)
        stages.append(stage)
        if stage["returncode"] != 0:
            overall_rc = stage["returncode"]
        python_bin = str(venv_path / "bin" / "python")
        deepspeed_bin = str(venv_path / "bin" / "deepspeed")
        if overall_rc == 0 and bool(smoke.get("install_requirements", False)):
            for name, cmd in [
                ("pip_upgrade", [python_bin, "-m", "pip", "install", "--upgrade", "pip"]),
                ("pip_install_requirements", [python_bin, "-m", "pip", "install", "-r", str(official_repo / "requirements.txt")]),
            ]:
                stage = _run(name, cmd, cwd=official_repo, env=base_env, log_dir=logs_dir)
                stages.append(stage)
                if stage["returncode"] != 0:
                    overall_rc = stage["returncode"]
                    break

    fingerprint_path: Path | None = None
    model = str(smoke["model"])
    seed = int(smoke["seed"])
    num_fingerprints = int(smoke["num_fingerprints"])
    key_length = int(smoke["key_length"])
    response_length = int(smoke["response_length"])
    use_chat_template = bool(smoke.get("use_chat_template", False))
    if overall_rc == 0:
        generate_output = generated_dir / "fingerprints.json"
        cmd = [
            python_bin,
            "generate_finetuning_data.py",
            "--num_fingerprints",
            str(num_fingerprints),
            "--response_length",
            str(response_length),
            "--key_length",
            str(key_length),
            "--batch_size",
            str(smoke.get("batch_size", 1)),
            "--seed",
            str(seed),
            "--key_response_strategy",
            str(smoke["key_response_strategy"]),
            "--model_used_for_key_generation",
            model,
            "--perinucleus_model",
            model,
            "--nucleus_t",
            str(smoke["nucleus_t"]),
            "--nucleus_k",
            str(smoke["nucleus_k"]),
            "--output_file_path",
            str(generate_output),
        ]
        if use_chat_template:
            cmd.append("--use_chat_template")
        stage = _run("generate_fingerprints", cmd, cwd=official_repo, env=base_env, log_dir=logs_dir)
        stages.append(stage)
        if stage["returncode"] != 0:
            overall_rc = stage["returncode"]
        else:
            fingerprint_path = _extract_fingerprint_path(_read(Path(stage["stdout_path"])), generated_dir)
            if fingerprint_path is None or not fingerprint_path.exists():
                overall_rc = 20
                stages.append(
                    {
                        "name": "locate_fingerprints",
                        "status": "failed",
                        "returncode": 20,
                        "seconds": 0.0,
                        "stdout_path": "",
                        "stderr_path": "",
                        "command_path": "",
                    }
                )
            else:
                stages.append(
                    {
                        "name": "locate_fingerprints",
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
    if overall_rc == 0 and fingerprint_path is not None:
        cmd = [
            python_bin,
            "check_fingerprints.py",
            "--model_path",
            model,
            "--num_fingerprints",
            str(num_fingerprints),
            "--fingerprints_file_path",
            str(fingerprint_path),
            "--fingerprint_generation_strategy",
            str(smoke["key_response_strategy"]),
            "--max_key_length",
            str(key_length),
            "--max_response_length",
            str(response_length),
            "--seed",
            str(seed),
            "--verbose_eval",
        ]
        stage = _run("check_base_fingerprints", cmd, cwd=official_repo, env=base_env, log_dir=logs_dir)
        stages.append(stage)
        if stage["returncode"] != 0:
            overall_rc = stage["returncode"]
        else:
            base_accuracy = _extract_accuracy(_read(Path(stage["stdout_path"])))

    if overall_rc == 0 and fingerprint_path is not None:
        base_score_path = score_dir / "base_response_probability.json"
        cmd = [
            python_bin,
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
        stage = _run("score_base_probabilities", cmd, cwd=repo_root, env=base_env, log_dir=logs_dir)
        stages.append(stage)
        if stage["returncode"] != 0:
            overall_rc = stage["returncode"]
        elif base_score_path.exists():
            base_score = json.loads(base_score_path.read_text(encoding="utf-8"))

    config_hash = None
    model_path = None
    if overall_rc == 0 and fingerprint_path is not None:
        cmd = [
            deepspeed_bin,
            "--num_gpus",
            str(smoke.get("deepspeed_num_gpus", 1)),
            "--master_port",
            str(smoke.get("deepspeed_master_port", 29517)),
            "finetune_multigpu.py",
            "--model_path",
            model,
            "--model_family",
            str(smoke.get("model_family", "qwen")),
            "--model_size",
            str(smoke.get("model_size", "7B")),
            "--num_fingerprints",
            str(num_fingerprints),
            "--max_key_length",
            str(key_length),
            "--max_response_length",
            str(response_length),
            "--num_train_epochs",
            str(smoke["train_epochs"]),
            "--learning_rate",
            str(smoke["learning_rate"]),
            "--batch_size",
            str(smoke["batch_size"]),
            "--fingerprint_generation_strategy",
            str(smoke["key_response_strategy"]),
            "--fingerprints_file_path",
            str(fingerprint_path),
            "--forgetting_regularizer_strength",
            str(smoke.get("forgetting_regularizer_strength", 0.0)),
            "--benign_proportion",
            str(smoke.get("benign_proportion", 0.0)),
            "--seed",
            str(seed),
            "--result_path",
            str(train_result_dir) + "/",
        ]
        if use_chat_template:
            cmd.append("--use_chat_template")
        if bool(smoke.get("use_lora", False)):
            cmd.extend(["--use_lora", "--lora_rank", str(smoke["lora_rank"]), "--lora_alpha_ratio", str(smoke["lora_alpha_ratio"])])
        stage = _run("finetune_insert_fingerprints", cmd, cwd=official_repo, env=base_env, log_dir=logs_dir)
        stages.append(stage)
        if stage["returncode"] != 0:
            overall_rc = stage["returncode"]
        else:
            config_hash = _last_config_hash(official_repo)
            if config_hash:
                model_path = train_result_dir / "saved_models" / config_hash / "final_model"
            if not model_path or not model_path.exists():
                overall_rc = 30
                stages.append(
                    {
                        "name": "locate_finetuned_model",
                        "status": "failed",
                        "returncode": 30,
                        "seconds": 0.0,
                        "stdout_path": str(model_path or ""),
                        "stderr_path": "",
                        "command_path": "",
                    }
                )
            else:
                stages.append(
                    {
                        "name": "locate_finetuned_model",
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
    if overall_rc == 0 and fingerprint_path is not None and model_path is not None:
        cmd = [
            python_bin,
            "check_fingerprints.py",
            "--model_path",
            str(model_path),
            "--num_fingerprints",
            str(num_fingerprints),
            "--fingerprints_file_path",
            str(fingerprint_path),
            "--fingerprint_generation_strategy",
            str(smoke["key_response_strategy"]),
            "--max_key_length",
            str(key_length),
            "--max_response_length",
            str(response_length),
            "--seed",
            str(seed),
            "--verbose_eval",
        ]
        stage = _run("check_finetuned_fingerprints", cmd, cwd=official_repo, env=base_env, log_dir=logs_dir)
        stages.append(stage)
        if stage["returncode"] != 0:
            overall_rc = stage["returncode"]
        else:
            trained_accuracy = _extract_accuracy(_read(Path(stage["stdout_path"])))

    if overall_rc == 0 and fingerprint_path is not None and model_path is not None:
        trained_score_path = score_dir / "trained_response_probability.json"
        cmd = [
            python_bin,
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
        stage = _run("score_finetuned_probabilities", cmd, cwd=repo_root, env=base_env, log_dir=logs_dir)
        stages.append(stage)
        if stage["returncode"] != 0:
            overall_rc = stage["returncode"]
        elif trained_score_path.exists():
            trained_score = json.loads(trained_score_path.read_text(encoding="utf-8"))

    utility_status = "not_run"
    if overall_rc == 0 and model_path is not None and bool(smoke.get("run_utility", True)):
        cmd = [python_bin, "eval_utility.py", "--model_path", str(model_path), "--eval_batch_size", str(smoke.get("eval_batch_size", 4))]
        if bool(smoke.get("utility_tiny_benchmarks", True)):
            cmd.append("--tinyBenchmarks")
        stage = _run("eval_utility_sanity", cmd, cwd=official_repo, env=base_env, log_dir=logs_dir)
        stages.append(stage)
        utility_status = "completed" if stage["returncode"] == 0 else "failed"
        if stage["returncode"] != 0:
            overall_rc = stage["returncode"]

    actual_commit = None
    commit_stdout = logs_dir / "record_official_commit.stdout.log"
    if commit_stdout.exists():
        actual_commit = commit_stdout.read_text(encoding="utf-8").strip().splitlines()[-1]

    base_prob = base_score.get("mean_first_token_probability") if isinstance(base_score, dict) else None
    trained_prob = trained_score.get("mean_first_token_probability") if isinstance(trained_score, dict) else None
    prob_improved = trained_prob is not None and base_prob is not None and float(trained_prob) > float(base_prob)
    acc_improved = trained_accuracy is not None and base_accuracy is not None and float(trained_accuracy) > float(base_accuracy)
    acc_above_random = trained_accuracy is not None and float(trained_accuracy) > 0.0
    stage_success = overall_rc == 0 and all(stage["returncode"] == 0 for stage in stages)
    smoke_pass = bool(stage_success and prob_improved and acc_improved and acc_above_random and utility_status == "completed")
    decision = (
        "SMOKE_PASS: anchor reproduction may proceed."
        if smoke_pass
        else "SMOKE_NOT_PASSED: do not launch anchor or final matrices; inspect failed stages and rerun smoke."
    )

    summary = {
        "schema_name": "baseline_perinucleus_official_smoke_summary",
        "schema_version": 1,
        "generated_at": _utc_now(),
        "smoke_pass": smoke_pass,
        "decision": decision,
        "model": model,
        "seed": seed,
        "num_fingerprints": num_fingerprints,
        "response_length": response_length,
        "use_chat_template": use_chat_template,
        "official_repo": str(official_repo),
        "official_commit_expected": expected_commit,
        "official_commit_actual": actual_commit,
        "official_license": _get(config, "official_repo.license"),
        "run_root": str(run_root),
        "fingerprints_file": str(fingerprint_path) if fingerprint_path else None,
        "finetuned_model_path": str(model_path) if model_path else None,
        "config_hash": config_hash,
        "base_fingerprint_accuracy": base_accuracy,
        "trained_fingerprint_accuracy": trained_accuracy,
        "base_mean_first_token_probability": base_prob,
        "trained_mean_first_token_probability": trained_prob,
        "probability_improved": prob_improved,
        "accuracy_improved": acc_improved,
        "accuracy_above_random_proxy": acc_above_random,
        "utility_status": utility_status,
        "stages": stages,
    }
    compute = {
        "schema_name": "baseline_perinucleus_official_smoke_compute",
        "schema_version": 1,
        "generated_at": summary["generated_at"],
        "run_root": str(run_root),
        "stage_seconds": {stage["name"]: stage["seconds"] for stage in stages},
        "total_seconds": sum(float(stage["seconds"]) for stage in stages),
        "requested_resources": {
            "num_gpus": int(os.environ.get("SLURM_GPUS_ON_NODE", smoke.get("deepspeed_num_gpus", 1))),
            "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    table_row = {
        "smoke_pass": str(smoke_pass),
        "model": model,
        "seed": seed,
        "num_fingerprints": num_fingerprints,
        "use_chat_template": str(use_chat_template),
        "base_fingerprint_accuracy": base_accuracy,
        "trained_fingerprint_accuracy": trained_accuracy,
        "base_mean_first_token_probability": base_prob,
        "trained_mean_first_token_probability": trained_prob,
        "utility_status": utility_status,
        "official_commit_actual": actual_commit,
        "run_root": str(run_root),
    }
    _write_outputs(repo_root=repo_root, config=config, summary=summary, compute=compute, table_row=table_row)
    return 0 if smoke_pass else (overall_rc if overall_rc else 2)


if __name__ == "__main__":
    raise SystemExit(main())

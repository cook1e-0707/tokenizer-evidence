from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.run_perinucleus_llama_anchor import (
    _mean_logprob,
    _mean_prob,
    _stage_pass,
    _write_outputs,
)
from scripts.run_perinucleus_official_smoke import (
    _apply_compatibility_patches,
    _extract_accuracy,
    _get,
    _load_yaml,
    _read,
    _run,
)
from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun utility sanity and backfill repo outputs for an existing Perinucleus Llama anchor run."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stage", action="append", default=None, help="Optional stage name to backfill; defaults to all stages.")
    parser.add_argument("--skip-utility-rerun", action="store_true", help="Only rebuild repo outputs from existing utility JSON files.")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _last_line(path: Path) -> str | None:
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return lines[-1] if lines else None


def _find_one(paths: list[Path], label: str) -> Path:
    found = [path for path in paths if path.exists()]
    if not found:
        raise FileNotFoundError(f"Could not find {label}")
    if len(found) > 1:
        found = sorted(found, key=lambda p: p.stat().st_mtime)
    return found[-1]


def _utility_json_exists(model_path: Path, tiny: bool) -> bool:
    suffix = "eval_results_tiny.json" if tiny else "eval_results.json"
    return (model_path.parent / suffix).exists()


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = _load_yaml(config_path)
    anchor = dict(config["anchor"])
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = repo_root / run_root
    logs_dir = run_root / "logs"
    official_results_root = run_root / "official_results"
    scores_root = run_root / "scores"
    generated_root = run_root / "generated"

    base_env = os.environ.copy()
    scratch_root = Path(str(anchor["scratch_root"]))
    base_env.setdefault("WANDB_MODE", "disabled")
    base_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    base_env.setdefault("HF_HOME", str(scratch_root / "hf_home"))
    base_env.setdefault("TRANSFORMERS_CACHE", str(scratch_root / "hf_home"))

    official_repo = repo_root / str(_get(config, "official_repo.local_path"))
    expected_commit = str(_get(config, "official_repo.commit_hash"))

    stages: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    overall_rc = 0

    for name, cmd in [
        ("utility_backfill_checkout_official_commit", ["git", "-C", str(official_repo), "checkout", expected_commit]),
        ("utility_backfill_record_official_commit", ["git", "-C", str(official_repo), "rev-parse", "HEAD"]),
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
        overall_rc = patch_rc

    selected = set(args.stage or [])
    for stage_cfg in anchor["stages"]:
        stage_name = str(stage_cfg["name"])
        if selected and stage_name not in selected:
            continue
        num_fingerprints = int(stage_cfg["num_fingerprints"])
        fingerprint_path = _find_one(
            sorted((generated_root / stage_name).glob("fingerprints-perinucleus-*.json")),
            f"{stage_name} Perinucleus fingerprints",
        )
        model_path = _find_one(
            sorted((official_results_root / stage_name / "saved_models").glob("*/final_model")),
            f"{stage_name} final model",
        )

        utility_status = "not_run"
        if bool(anchor.get("run_utility", True)):
            if args.skip_utility_rerun and _utility_json_exists(model_path, bool(anchor.get("utility_tiny_benchmarks", True))):
                utility_status = "completed"
            else:
                cmd = [
                    "python3",
                    "eval_utility.py",
                    "--model_path",
                    str(model_path),
                    "--eval_batch_size",
                    str(anchor.get("eval_batch_size", 4)),
                ]
                if bool(anchor.get("utility_tiny_benchmarks", True)):
                    cmd.append("--tinyBenchmarks")
                stage = _run(f"{stage_name}_eval_utility_sanity_backfill", cmd, cwd=official_repo, env=base_env, log_dir=logs_dir)
                stages.append(stage)
                utility_status = "completed" if stage["returncode"] == 0 else "failed"
                if stage["returncode"] != 0 and overall_rc == 0:
                    overall_rc = stage["returncode"]

        base_score_path = scores_root / stage_name / "base_response_probability.json"
        trained_score_path = scores_root / stage_name / "trained_response_probability.json"
        base_score = json.loads(base_score_path.read_text(encoding="utf-8"))
        trained_score = json.loads(trained_score_path.read_text(encoding="utf-8"))
        base_accuracy = _extract_accuracy(_read(logs_dir / f"{stage_name}_check_base_fingerprints.stdout.log"))
        trained_accuracy = _extract_accuracy(_read(logs_dir / f"{stage_name}_check_finetuned_fingerprints.stdout.log"))
        config_hash_match = re.search(r"/saved_models/([^/]+)/final_model$", str(model_path))
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
            "probability_improved": _mean_prob(trained_score) is not None
            and _mean_prob(base_score) is not None
            and float(_mean_prob(trained_score)) > float(_mean_prob(base_score)),
            "utility_status": utility_status,
            "config_hash": config_hash_match.group(1) if config_hash_match else "",
            "fingerprints_file": str(fingerprint_path),
            "finetuned_model_path": str(model_path),
            "run_root": str(run_root),
        }
        row["stage_pass"] = _stage_pass(row)
        rows.append(row)

    actual_commit = _last_line(logs_dir / "utility_backfill_record_official_commit.stdout.log")
    all_rows_pass = bool(rows) and all(bool(row["stage_pass"]) for row in rows)
    anchor_pass = bool(overall_rc == 0 and all_rows_pass)
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
        "model": str(anchor["model"]),
        "model_family": anchor.get("model_family"),
        "seed": int(anchor["seed"]),
        "response_length": anchor["response_length"],
        "use_chat_template": bool(anchor.get("use_chat_template", False)),
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
        "backfill_mode": "utility_only",
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
        "backfill_mode": "utility_only",
    }
    _write_outputs(repo_root, config, summary, compute, rows)
    return 0 if anchor_pass else (overall_rc if overall_rc else 2)


if __name__ == "__main__":
    raise SystemExit(main())

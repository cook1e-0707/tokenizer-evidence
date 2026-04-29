from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from run_perinucleus_official_overfit_gate import (
    _apply_override,
    _ensure_official_repo,
    _extract_fingerprint_path,
    _get,
    _load_model_dependencies,
    _resolve,
    _run_stage,
    _train_stage,
    _utc_now,
    discover_repo_root,
)

torch: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen Perinucleus capacity sweep after the overfit gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="Accepted for manifest compatibility.")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="Validate config without loading models or training.")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "arm_id",
        "regularization_label",
        "target_modules_label",
        "target_modules",
        "lora_rank",
        "lora_alpha_ratio",
        "num_fingerprints",
        "max_epochs",
        "epoch",
        "train_step_loss_mean",
        "train_ce_mean",
        "target_probability_mean",
        "target_probability_min",
        "base_target_probability_mean",
        "target_rank_mean",
        "target_rank_max",
        "exact_count",
        "exact_accuracy",
        "rank1_count",
        "rank1_accuracy",
        "base_vs_adapter_logit_delta_max",
        "lora_parameter_count",
        "lora_nonzero_norm_count",
        "lora_total_norm",
        "lora_max_norm",
        "adapter_path",
        "utility_status",
        "mismatch_examples",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["target_modules"] = json.dumps(out.get("target_modules", []), ensure_ascii=True)
            out["mismatch_examples"] = json.dumps(out.get("mismatch_examples", []), ensure_ascii=True, sort_keys=True)
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def _candidate_key(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    final = summary.get("final", {})
    exact = float(final.get("exact_accuracy") or 0.0)
    rank = float(final.get("target_rank_mean") or 1.0e9)
    prob = float(final.get("target_probability_mean") or 0.0)
    epochs = float(summary.get("epochs_run") or 1.0e9)
    return (exact, -rank, prob, -epochs)


def _arm_pass(summary: dict[str, Any]) -> bool:
    final = summary.get("final", {})
    exact = float(final.get("exact_accuracy") or 0.0)
    adapted_prob = float(final.get("target_probability_mean") or 0.0)
    base_prob = float(final.get("base_target_probability_mean") or 0.0)
    delta = float(final.get("base_vs_adapter_logit_delta_max") or 0.0)
    return exact > 0.0 and adapted_prob > base_prob and delta > 0.0


def _strong_candidate(summary: dict[str, Any]) -> bool:
    final = summary.get("final", {})
    exact = float(final.get("exact_accuracy") or 0.0)
    rank = float(final.get("target_rank_mean") or 1.0e9)
    return exact >= 0.5 and rank <= 2.0


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen Perinucleus Capacity Sweep",
        "",
        "This is a diagnostic capacity sweep after the official-code forensic replay, Qwen overfit gate, and Llama anchor. It is not a final comparison matrix.",
        "",
        "## Decision",
        "",
        f"`{summary['decision']}`",
        "",
        "## Sweep Arms",
        "",
        "| arm | fingerprints | target modules | rank | epochs | exact accuracy | mean rank | mean probability | utility |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm in summary["arms"]:
        final = arm.get("final", {})
        lines.append(
            "| {arm_id} | {n} | {mods} | {rank} | {epochs} | {exact} | {mean_rank} | {prob} | {utility} |".format(
                arm_id=arm["arm_id"],
                n=arm["num_fingerprints"],
                mods=arm["target_modules_label"],
                rank=arm["lora_rank"],
                epochs=arm["epochs_run"],
                exact=final.get("exact_accuracy"),
                mean_rank=final.get("target_rank_mean"),
                prob=final.get("target_probability_mean"),
                utility=arm.get("utility_status", "pending_candidate_utility"),
            )
        )
    recommended = summary.get("recommended_candidate")
    lines.extend(
        [
            "",
            "## Recommended Candidate",
            "",
            json.dumps(recommended, indent=2, sort_keys=True) if recommended else "No candidate met the diagnostic gate.",
            "",
            "## Fidelity Notes",
            "",
            "- Fingerprint generation uses the official Scalable Fingerprinting repository at the recorded commit.",
            "- Training uses the same adapted diagnostic LoRA loop that passed the single-fingerprint overfit gate.",
            "- This sweep intentionally freezes a small diagnostic arm list before running; it must not be expanded post hoc using final-matrix feedback.",
            "- Utility sanity is not treated as complete unless the selected candidate is evaluated separately and recorded before final-matrix use.",
            "",
            "## Output Files",
            "",
            f"- Table: `{summary['output_table']}`",
            f"- Summary: `{summary['output_summary']}`",
            f"- Compute: `{summary['output_compute']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_fingerprints(
    *,
    sweep: dict[str, Any],
    official_repo: Path,
    generated_dir: Path,
    logs_dir: Path,
    env: dict[str, str],
    seed: int,
    num_fingerprints: int,
) -> tuple[Path | None, dict[str, Any]]:
    arm_generated_dir = generated_dir / f"fingerprints_{num_fingerprints}"
    arm_generated_dir.mkdir(parents=True, exist_ok=True)
    output_file = arm_generated_dir / "fingerprints.json"
    cmd = [
        "python3",
        "generate_finetuning_data.py",
        "--num_fingerprints",
        str(num_fingerprints),
        "--response_length",
        str(sweep["response_length"]),
        "--key_length",
        str(sweep["key_length"]),
        "--batch_size",
        str(sweep.get("generation_batch_size", 1)),
        "--seed",
        str(seed),
        "--key_response_strategy",
        "perinucleus",
        "--model_used_for_key_generation",
        str(sweep["model"]),
        "--perinucleus_model",
        str(sweep["model"]),
        "--nucleus_t",
        str(sweep.get("nucleus_t", 0.8)),
        "--nucleus_k",
        str(sweep.get("nucleus_k", 3)),
        "--output_file_path",
        str(output_file),
    ]
    if bool(sweep.get("use_chat_template", True)):
        cmd.append("--use_chat_template")
    stage = _run_stage(f"generate_fingerprints_{num_fingerprints}", cmd, official_repo, env, logs_dir)
    if stage["returncode"] != 0:
        return None, stage
    return _extract_fingerprint_path(Path(stage["stdout_path"]), arm_generated_dir), stage


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = _resolve(repo_root, args.config)
    if config_path is None:
        raise ValueError("Missing config path.")
    config = _load_yaml(config_path)
    for override in args.override:
        _apply_override(config, override)
    sweep = dict(config["capacity_sweep"])
    scratch_root = Path(str(sweep["scratch_root"]))
    run_id = str(_get(config, "runtime.run_id") or f"manual_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    run_root = Path(str(_get(config, "runtime.output_dir"))) if _get(config, "runtime.output_dir") else scratch_root / "runs" / run_id
    logs_dir = run_root / "logs"
    generated_dir = run_root / "generated"
    arms_root = run_root / "arms"

    outputs = dict(config["repo_outputs"])
    output_doc = _resolve(repo_root, outputs["doc"])
    output_table = _resolve(repo_root, outputs["table"])
    output_summary = _resolve(repo_root, outputs["summary"])
    output_compute = _resolve(repo_root, outputs["compute"])
    if None in {output_doc, output_table, output_summary, output_compute}:
        raise ValueError("Could not resolve output paths.")

    arms = list(sweep["arms"])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": str(config_path),
                    "run_root": str(run_root),
                    "scratch_root": str(scratch_root),
                    "num_arms": len(arms),
                    "arms": arms,
                    "outputs": outputs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    for path in [logs_dir, generated_dir, arms_root]:
        path.mkdir(parents=True, exist_ok=True)

    _load_model_dependencies()
    global torch
    import torch as torch_module

    torch = torch_module
    seed = int(sweep.get("seed", 17))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Qwen capacity sweep requires a CUDA GPU.")

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HOME", str(scratch_root / "hf_home"))
    env.setdefault("TRANSFORMERS_CACHE", str(scratch_root / "hf_home"))

    started = time.time()
    setup_stages: list[dict[str, Any]] = []
    official_repo, repo_stages, setup_rc = _ensure_official_repo(config, repo_root, logs_dir, env)
    setup_stages.extend(repo_stages)
    if setup_rc != 0:
        raise RuntimeError(f"Official repo setup failed with rc={setup_rc}")

    fingerprints_by_n: dict[int, Path] = {}
    for num_fingerprints in sorted({int(arm["num_fingerprints"]) for arm in arms}):
        fingerprints_file, stage = _generate_fingerprints(
            sweep=sweep,
            official_repo=official_repo,
            generated_dir=generated_dir,
            logs_dir=logs_dir,
            env=env,
            seed=seed,
            num_fingerprints=num_fingerprints,
        )
        setup_stages.append(stage)
        if stage["returncode"] != 0 or fingerprints_file is None or not fingerprints_file.exists():
            raise RuntimeError(f"Fingerprint generation failed for n={num_fingerprints}; see {stage.get('stderr_path')}")
        fingerprints_by_n[num_fingerprints] = fingerprints_file

    rows: list[dict[str, Any]] = []
    arm_summaries: list[dict[str, Any]] = []
    for arm in arms:
        arm_id = str(arm["arm_id"])
        arm_root = arms_root / arm_id
        num_fingerprints = int(arm["num_fingerprints"])
        lora = {
            "rank": int(arm["lora_rank"]),
            "alpha_ratio": float(arm.get("lora_alpha_ratio", sweep.get("lora_alpha_ratio", 2.0))),
            "dropout": float(arm.get("lora_dropout", sweep.get("lora_dropout", 0.0))),
            "target_modules": list(arm["target_modules"]),
        }
        arm_cfg = {
            "model": sweep["model"],
            "key_length": int(sweep["key_length"]),
            "response_length": int(sweep["response_length"]),
            "batch_size": int(sweep.get("batch_size", 1)),
            "learning_rate": float(sweep.get("learning_rate", 5.0e-5)),
            "weight_decay": float(sweep.get("weight_decay", 0.0)),
            "early_stop_ce": float(sweep.get("early_stop_ce", 0.0)),
            "max_sequence_length": int(sweep.get("max_sequence_length", 64)),
            "stop_on_pass": False,
            "lora": lora,
        }
        stage_cfg = {"name": arm_id, "num_fingerprints": num_fingerprints, "max_epochs": int(arm["max_epochs"])}
        stage_rows, stage_summary = _train_stage(arm_cfg, stage_cfg, fingerprints_by_n[num_fingerprints], arm_root, device)
        utility_status = str(arm.get("utility_status", sweep.get("utility_status", "pending_candidate_utility")))
        for row in stage_rows:
            row.update(
                {
                    "arm_id": arm_id,
                    "regularization_label": str(arm.get("regularization_label", "diagnostic_off")),
                    "target_modules_label": str(arm["target_modules_label"]),
                    "target_modules": list(arm["target_modules"]),
                    "lora_rank": int(arm["lora_rank"]),
                    "lora_alpha_ratio": lora["alpha_ratio"],
                    "max_epochs": int(arm["max_epochs"]),
                    "adapter_path": stage_summary["adapter_path"],
                    "utility_status": utility_status,
                }
            )
        rows.extend(stage_rows)
        stage_summary.update(
            {
                "arm_id": arm_id,
                "regularization_label": str(arm.get("regularization_label", "diagnostic_off")),
                "target_modules_label": str(arm["target_modules_label"]),
                "target_modules": list(arm["target_modules"]),
                "lora_rank": int(arm["lora_rank"]),
                "lora_alpha_ratio": lora["alpha_ratio"],
                "candidate_pass": _arm_pass(stage_summary),
                "strong_candidate": _strong_candidate(stage_summary),
                "utility_status": utility_status,
            }
        )
        arm_summaries.append(stage_summary)

    candidates = [arm for arm in arm_summaries if arm["candidate_pass"]]
    strong_candidates = [arm for arm in arm_summaries if arm["strong_candidate"]]
    recommended = max(candidates, key=_candidate_key) if candidates else None
    if strong_candidates:
        decision = "QWEN_CAPACITY_SWEEP_CANDIDATE_FOUND: run utility sanity for the selected candidate before any final matrix."
    elif candidates:
        decision = "QWEN_CAPACITY_SWEEP_WEAK_CANDIDATE_FOUND: inspect failures and utility before any final matrix."
    else:
        decision = "QWEN_CAPACITY_SWEEP_BLOCKED: no arm improved exact fingerprints enough to justify final matrix."

    summary = {
        "schema_name": "baseline_perinucleus_qwen_capacity_sweep_summary",
        "schema_version": 1,
        "generated_at": _utc_now(),
        "decision": decision,
        "candidate_found": bool(candidates),
        "strong_candidate_found": bool(strong_candidates),
        "recommended_candidate": recommended,
        "run_root": str(run_root),
        "official_repo": str(official_repo),
        "official_commit": str(_get(config, "official_repo.commit_hash")),
        "model": str(sweep["model"]),
        "seed": seed,
        "arms": arm_summaries,
        "setup_stages": setup_stages,
        "preconditions": config.get("preconditions", {}),
        "output_doc": str(output_doc),
        "output_table": str(output_table),
        "output_summary": str(output_summary),
        "output_compute": str(output_compute),
    }
    compute = {
        "schema_name": "baseline_perinucleus_qwen_capacity_sweep_compute",
        "schema_version": 1,
        "generated_at": summary["generated_at"],
        "run_root": str(run_root),
        "seconds": time.time() - started,
        "device": str(device),
        "arm_seconds": [{arm["arm_id"]: arm.get("seconds")} for arm in arm_summaries],
    }
    _write_csv(output_table, rows)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    output_compute.parent.mkdir(parents=True, exist_ok=True)
    output_compute.write_text(json.dumps(compute, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_doc, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())

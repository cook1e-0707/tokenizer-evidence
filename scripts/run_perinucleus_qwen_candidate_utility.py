from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


TINY_DATASETS = {
    "tinyARC": {"tasks": ["tinyArc"], "metrics": ["acc_norm"]},
    "tinyHellaswag": {"tasks": ["tinyHellaswag"], "metrics": ["acc_norm"]},
    "tinyTruthfulQA": {"tasks": ["tinyTruthfulQA"], "metrics": ["acc"]},
    "tinyMMLU": {"tasks": ["tinyMMLU"], "metrics": ["acc_norm"]},
    "tinyWinogrande": {"tasks": ["tinyWinogrande"], "metrics": ["acc_norm"]},
    "tinyGSM8k": {"tasks": ["tinyGSM8k"], "metrics": ["exact_match,strict-match", "exact_match,flexible-extract"]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run utility sanity for selected Qwen Perinucleus capacity-sweep adapters.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="Accepted for manifest compatibility.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return current


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _metric_value(task_res: dict[str, Any], metric: str) -> float | None:
    for key in [metric, f"{metric},none"]:
        if key in task_res:
            return float(task_res[key])
    return None


def _summarize_results(results: dict[str, Any]) -> tuple[float | None, dict[str, float], list[str]]:
    task_results = results.get("results", results)
    values: list[float] = []
    flat: dict[str, float] = {}
    missing: list[str] = []
    for dataset, spec in TINY_DATASETS.items():
        for task in spec["tasks"]:
            task_res = task_results.get(task)
            if not isinstance(task_res, dict):
                missing.append(f"{task}:missing_task")
                continue
            for metric in spec["metrics"]:
                value = _metric_value(task_res, metric)
                if value is None:
                    missing.append(f"{task}:{metric}")
                    continue
                flat[f"{dataset}/{task}/{metric}"] = value
                values.append(value)
    return (sum(values) / len(values) if values else None, flat, missing)


def _model_args(*, base_model: str, adapter_path: str | None, local_files_only: bool, trust_remote_code: bool) -> str:
    args = [f"pretrained={base_model}"]
    if adapter_path:
        args.append(f"peft={adapter_path}")
    args.append(f"local_files_only={str(local_files_only)}")
    args.append(f"trust_remote_code={str(trust_remote_code)}")
    return ",".join(args)


def _run_eval(
    *,
    name: str,
    base_model: str,
    adapter_path: str | None,
    utility: dict[str, Any],
    raw_dir: Path,
) -> dict[str, Any]:
    import lm_eval
    import torch

    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = "failed"
    error = ""
    output_path = raw_dir / f"{name}_eval_results.json"
    total_accuracy = None
    task_metrics: dict[str, float] = {}
    missing_metrics: list[str] = []
    model_args = _model_args(
        base_model=base_model,
        adapter_path=adapter_path,
        local_files_only=bool(utility.get("local_files_only", True)),
        trust_remote_code=bool(utility.get("trust_remote_code", True)),
    )
    try:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=str(utility.get("tasks", "tinyBenchmarks")),
            batch_size=int(utility.get("eval_batch_size", 4)),
            apply_chat_template=bool(utility.get("apply_chat_template", True)),
        )
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")
        total_accuracy, task_metrics, missing_metrics = _summarize_results(results)
        status = "completed" if total_accuracy is not None else "completed_missing_metrics"
    except Exception as exc:  # noqa: BLE001 - recorded for diagnostics.
        error = f"{type(exc).__name__}: {exc}"
        output_path.write_text(json.dumps({"error": error}, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "name": name,
        "status": status,
        "seconds": time.time() - started,
        "base_model": base_model,
        "adapter_path": adapter_path or "",
        "model_args": model_args,
        "eval_results_path": str(output_path),
        "total_accuracy": total_accuracy,
        "task_metrics": task_metrics,
        "missing_metrics": missing_metrics,
        "error": error,
    }


def _candidate_rows(summary: dict[str, Any], selected_arms: list[str]) -> list[dict[str, Any]]:
    by_id = {str(arm["arm_id"]): arm for arm in summary.get("arms", [])}
    rows = []
    for arm_id in selected_arms:
        if arm_id not in by_id:
            raise KeyError(f"Selected arm not found in capacity summary: {arm_id}")
        arm = by_id[arm_id]
        final = dict(arm.get("final", {}))
        rows.append(
            {
                "arm_id": arm_id,
                "adapter_path": str(arm["adapter_path"]),
                "num_fingerprints": int(arm["num_fingerprints"]),
                "target_modules_label": str(arm["target_modules_label"]),
                "lora_rank": int(arm["lora_rank"]),
                "epochs_run": int(arm["epochs_run"]),
                "exact_accuracy": final.get("exact_accuracy"),
                "target_rank_mean": final.get("target_rank_mean"),
                "target_probability_mean": final.get("target_probability_mean"),
                "train_ce_mean": final.get("train_ce_mean"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "arm_id",
        "num_fingerprints",
        "target_modules_label",
        "lora_rank",
        "epochs_run",
        "exact_accuracy",
        "target_rank_mean",
        "target_probability_mean",
        "train_ce_mean",
        "utility_status",
        "total_accuracy",
        "base_total_accuracy",
        "absolute_drop",
        "relative_drop",
        "utility_pass",
        "adapter_path",
        "eval_results_path",
        "missing_metrics",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["missing_metrics"] = json.dumps(out.get("missing_metrics", []), ensure_ascii=True)
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def _write_doc(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen Perinucleus Candidate Utility Sanity",
        "",
        "This is a utility sanity check for selected Qwen Perinucleus capacity-sweep adapters. It does not retrain and does not authorize a final matrix by itself.",
        "",
        "## Decision",
        "",
        f"`{summary['decision']}`",
        "",
        "## Utility Results",
        "",
        "| kind | arm | exact | utility | base utility | abs drop | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {kind} | {arm} | {exact} | {utility} | {base} | {drop} | {passed} |".format(
                kind=row["kind"],
                arm=row.get("arm_id", ""),
                exact=row.get("exact_accuracy", ""),
                utility=row.get("total_accuracy"),
                base=row.get("base_total_accuracy"),
                drop=row.get("absolute_drop"),
                passed=row.get("utility_pass", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Selected Candidate",
            "",
            json.dumps(summary.get("selected_candidate"), indent=2, sort_keys=True),
            "",
            "## Notes",
            "",
            "- Adapter utility is evaluated as `pretrained=Qwen/Qwen2.5-7B-Instruct,peft=<adapter_path>`.",
            "- `apply_chat_template=true` is used for the instruct backbone.",
            "- The selected candidate must be frozen before any final matrix.",
            "",
            "## Output Files",
            "",
            f"- Table: `{summary['output_table']}`",
            f"- Summary: `{summary['output_summary']}`",
            f"- Compute: `{summary['output_compute']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = _resolve(repo_root, args.config)
    if config_path is None:
        raise ValueError("Missing config path.")
    config = _load_yaml(config_path)
    utility = dict(config["utility"])
    capacity_summary_path = _resolve(repo_root, config["capacity_sweep_summary"])
    if capacity_summary_path is None or not capacity_summary_path.exists():
        raise FileNotFoundError(f"Capacity sweep summary not found: {capacity_summary_path}")
    capacity_summary = json.loads(capacity_summary_path.read_text(encoding="utf-8"))
    candidates = _candidate_rows(capacity_summary, list(utility["selected_arms"]))

    scratch_root = Path(str(utility["scratch_root"]))
    run_id = str(config.get("runtime", {}).get("run_id") or f"manual_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    run_root = Path(str(config.get("runtime", {}).get("output_dir"))) if config.get("runtime", {}).get("output_dir") else scratch_root / "runs" / run_id
    raw_dir = run_root / "eval_results"

    outputs = dict(config["repo_outputs"])
    output_doc = _resolve(repo_root, outputs["doc"])
    output_table = _resolve(repo_root, outputs["table"])
    output_summary = _resolve(repo_root, outputs["summary"])
    output_compute = _resolve(repo_root, outputs["compute"])
    if None in {output_doc, output_table, output_summary, output_compute}:
        raise ValueError("Could not resolve output paths.")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": str(config_path),
                    "capacity_summary": str(capacity_summary_path),
                    "run_root": str(run_root),
                    "base_model": utility["base_model"],
                    "selected_arms": candidates,
                    "outputs": outputs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if "SLURM_JOB_ID" not in os.environ and not os.environ.get("ALLOW_NON_SLURM_GPU_RUN"):
        print("WARNING: running outside Slurm; set ALLOW_NON_SLURM_GPU_RUN=1 to silence this warning.")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(scratch_root / "hf_home"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(scratch_root / "hf_home"))

    started = time.time()
    base_model = str(utility["base_model"])
    evals: list[dict[str, Any]] = []
    base_eval = _run_eval(name="base", base_model=base_model, adapter_path=None, utility=utility, raw_dir=raw_dir)
    evals.append(base_eval)
    base_total = base_eval.get("total_accuracy")
    rows: list[dict[str, Any]] = [
        {
            "kind": "base",
            "arm_id": "base",
            "num_fingerprints": "",
            "target_modules_label": "",
            "lora_rank": "",
            "epochs_run": "",
            "exact_accuracy": "",
            "target_rank_mean": "",
            "target_probability_mean": "",
            "train_ce_mean": "",
            "utility_status": base_eval["status"],
            "total_accuracy": base_total,
            "base_total_accuracy": base_total,
            "absolute_drop": 0.0 if base_total is not None else None,
            "relative_drop": 0.0 if base_total else None,
            "utility_pass": base_eval["status"] == "completed",
            "adapter_path": "",
            "eval_results_path": base_eval["eval_results_path"],
            "missing_metrics": base_eval["missing_metrics"],
            "error": base_eval["error"],
        }
    ]
    max_drop = float(utility.get("max_absolute_drop", 0.05))
    for candidate in candidates:
        eval_row = _run_eval(
            name=str(candidate["arm_id"]),
            base_model=base_model,
            adapter_path=str(candidate["adapter_path"]),
            utility=utility,
            raw_dir=raw_dir,
        )
        evals.append(eval_row)
        total = eval_row.get("total_accuracy")
        abs_drop = (float(base_total) - float(total)) if base_total is not None and total is not None else None
        rel_drop = (abs_drop / float(base_total)) if abs_drop is not None and base_total not in {None, 0.0} else None
        utility_pass = bool(eval_row["status"] == "completed" and abs_drop is not None and abs_drop <= max_drop)
        rows.append(
            {
                "kind": "adapter",
                **candidate,
                "utility_status": eval_row["status"],
                "total_accuracy": total,
                "base_total_accuracy": base_total,
                "absolute_drop": abs_drop,
                "relative_drop": rel_drop,
                "utility_pass": utility_pass,
                "eval_results_path": eval_row["eval_results_path"],
                "missing_metrics": eval_row["missing_metrics"],
                "error": eval_row["error"],
            }
        )

    adapter_rows = [row for row in rows if row["kind"] == "adapter"]
    passing = [row for row in adapter_rows if row["utility_pass"]]
    selected = None
    if passing:
        selected = sorted(
            passing,
            key=lambda row: (
                float(row["absolute_drop"]),
                -float(row.get("exact_accuracy") or 0.0),
                int(row.get("num_fingerprints") or 0),
                -int(row.get("lora_rank") or 0),
            ),
        )[0]
        decision = "QWEN_CANDIDATE_UTILITY_PASS: freeze the selected candidate before final protocol runs."
    else:
        completed = [row for row in adapter_rows if row["utility_status"] == "completed"]
        selected = sorted(completed, key=lambda row: float(row["absolute_drop"]))[0] if completed else None
        decision = "QWEN_CANDIDATE_UTILITY_NOT_PASSED: do not run final matrix; inspect utility failures."

    summary = {
        "schema_name": "baseline_perinucleus_qwen_candidate_utility_summary",
        "schema_version": 1,
        "generated_at": _utc_now(),
        "decision": decision,
        "utility_pass": bool(passing),
        "base_model": base_model,
        "capacity_sweep_summary": str(capacity_summary_path),
        "capacity_sweep_decision": capacity_summary.get("decision"),
        "run_root": str(run_root),
        "tasks": str(utility.get("tasks", "tinyBenchmarks")),
        "apply_chat_template": bool(utility.get("apply_chat_template", True)),
        "max_absolute_drop": max_drop,
        "selected_candidate": selected,
        "rows": rows,
        "evals": evals,
        "output_doc": str(output_doc),
        "output_table": str(output_table),
        "output_summary": str(output_summary),
        "output_compute": str(output_compute),
    }
    compute = {
        "schema_name": "baseline_perinucleus_qwen_candidate_utility_compute",
        "schema_version": 1,
        "generated_at": summary["generated_at"],
        "run_root": str(run_root),
        "seconds": time.time() - started,
        "requested_resources": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "num_gpus": os.environ.get("SLURM_GPUS_ON_NODE"),
            "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "eval_seconds": {item["name"]: item["seconds"] for item in evals},
    }
    _write_csv(output_table, rows)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    output_compute.parent.mkdir(parents=True, exist_ok=True)
    output_compute.write_text(json.dumps(compute, indent=2, sort_keys=True), encoding="utf-8")
    _write_doc(output_doc, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())

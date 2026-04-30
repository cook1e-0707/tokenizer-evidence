from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import subprocess
import sys
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
    parser = argparse.ArgumentParser(description="Run matched TinyBench utility for ours final Qwen adapters.")
    parser.add_argument("--config", help="Path to configs/experiment/comparison/ours_tinybench_utility.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Accepted for workflow compatibility.")
    parser.add_argument("--eval-one-spec", help=argparse.SUPPRESS)
    parser.add_argument("--eval-one-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def discover_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return current


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _relocate_chimera_path(repo_root: Path, value: str | Path | None) -> Path | None:
    path = _resolve(repo_root, value)
    if path is None or path.exists():
        return path
    raw = str(value)
    markers = [
        "/home/guanjie.lin001/tokenizer-evidence/",
        "/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/",
    ]
    for marker in markers:
        if raw.startswith(marker):
            candidate = repo_root / raw[len(marker):]
            if candidate.exists():
                return candidate
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


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


def _run_eval_in_process(
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
            batch_size=int(utility.get("eval_batch_size", 1)),
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


def _run_eval(
    *,
    name: str,
    base_model: str,
    adapter_path: str | None,
    utility: dict[str, Any],
    raw_dir: Path,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    spec_path = raw_dir / f"{name}_eval_spec.json"
    report_path = raw_dir / f"{name}_eval_report.json"
    stdout_path = raw_dir / f"{name}_eval.stdout.log"
    stderr_path = raw_dir / f"{name}_eval.stderr.log"
    spec = {
        "name": name,
        "base_model": base_model,
        "adapter_path": adapter_path,
        "utility": utility,
        "raw_dir": str(raw_dir),
    }
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--eval-one-spec",
        str(spec_path),
        "--eval-one-output",
        str(report_path),
    ]
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, text=True, check=False)
    if report_path.exists():
        report = _load_json(report_path)
    else:
        report = {
            "name": name,
            "status": "failed",
            "seconds": time.time() - started,
            "base_model": base_model,
            "adapter_path": adapter_path or "",
            "model_args": _model_args(
                base_model=base_model,
                adapter_path=adapter_path,
                local_files_only=bool(utility.get("local_files_only", True)),
                trust_remote_code=bool(utility.get("trust_remote_code", True)),
            ),
            "eval_results_path": "",
            "total_accuracy": None,
            "task_metrics": {},
            "missing_metrics": [],
            "error": f"subprocess_returncode={proc.returncode}; see {stderr_path}",
        }
    report["subprocess_returncode"] = proc.returncode
    report["stdout_path"] = str(stdout_path)
    report["stderr_path"] = str(stderr_path)
    return report


def _final_cases(repo_root: Path, comparison_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    source = comparison_cfg.get("source_artifacts") or {}
    clean_table_path = _resolve(repo_root, source.get("clean_table"))
    if clean_table_path is None or not clean_table_path.exists():
        raise FileNotFoundError(f"Clean table not found: {clean_table_path}")
    final = (comparison_cfg.get("method_contract") or {}).get("final_positive_cases") or {}
    payloads = {str(item) for item in final.get("payloads", [])}
    seeds = {_as_int(item) for item in final.get("seeds", [])}
    rows = []
    for row in _read_csv(clean_table_path):
        payload = str(row.get("payload", ""))
        seed = _as_int(row.get("seed"))
        if payload not in payloads or seed not in seeds:
            continue
        train_summary_path = _relocate_chimera_path(repo_root, row.get("train_summary_path"))
        adapter_path = train_summary_path.parent / "checkpoints" / "hf_last" if train_summary_path else None
        rows.append(
            {
                "case_id": row.get("case_id", f"{payload}_s{seed}"),
                "payload": payload,
                "seed": seed,
                "source_stage": row.get("source_stage", ""),
                "train_summary_path": str(train_summary_path) if train_summary_path else "",
                "adapter_path": str(adapter_path) if adapter_path else "",
                "adapter_exists": bool(adapter_path and (adapter_path / "adapter_config.json").exists()),
                "clean_status": row.get("status", ""),
            }
        )
    rows.sort(key=lambda item: (str(item["payload"]), int(item["seed"])))
    return rows


def _mean_ci95(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean, mean
    # Conservative normal approximation over adapter-level utility scores.
    stderr = statistics.stdev(values) / (len(values) ** 0.5)
    margin = 1.96 * stderr
    return mean, mean - margin, mean + margin


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "case_id",
        "payload",
        "seed",
        "source_stage",
        "utility_status",
        "total_accuracy",
        "base_total_accuracy",
        "absolute_drop",
        "relative_drop",
        "utility_pass",
        "adapter_path",
        "adapter_exists",
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
        "# Ours TinyBench Utility",
        "",
        "This evaluates the final positive-case adapters for the compiled tokenizer-aligned method on the same tinyBenchmarks sanity used for the Qwen-adapted Perinucleus candidate.",
        "",
        "## Decision",
        "",
        f"`{summary['decision']}`",
        "",
        "## Aggregate",
        "",
        f"- Base total accuracy: `{summary.get('base_total_accuracy')}`",
        f"- Adapter mean total accuracy: `{summary.get('adapter_total_accuracy_mean')}`",
        f"- Adapter min total accuracy: `{summary.get('adapter_total_accuracy_min')}`",
        f"- Mean absolute drop: `{summary.get('absolute_drop_mean')}`",
        f"- Max absolute drop: `{summary.get('absolute_drop_max')}`",
        f"- Utility pass: `{summary.get('utility_pass')}`",
        "",
        "## Caveat",
        "",
        "This is a TinyBench sanity, not full OpenLLM utility. It should be used only as the matched low-cost utility comparison against the existing Perinucleus TinyBench sanity.",
        "",
        "## Output Files",
        "",
        f"- Table: `{summary['output_table']}`",
        f"- Summary: `{summary['output_summary']}`",
        f"- Compute: `{summary['output_compute']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.eval_one_spec:
        if not args.eval_one_output:
            raise ValueError("--eval-one-output is required with --eval-one-spec")
        spec = _load_json(Path(args.eval_one_spec))
        report = _run_eval_in_process(
            name=str(spec["name"]),
            base_model=str(spec["base_model"]),
            adapter_path=spec.get("adapter_path"),
            utility=dict(spec["utility"]),
            raw_dir=Path(str(spec["raw_dir"])),
        )
        Path(args.eval_one_output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "completed" else 2

    if not args.config:
        raise ValueError("--config is required")
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = _resolve(repo_root, args.config)
    if config_path is None:
        raise ValueError("Missing config path.")
    config = _load_yaml(config_path)
    comparison_path = _resolve(repo_root, config["comparison_config"])
    if comparison_path is None or not comparison_path.exists():
        raise FileNotFoundError(f"Comparison config not found: {comparison_path}")
    comparison_cfg = _load_yaml(comparison_path)
    utility = dict(config["utility"])
    cases = _final_cases(repo_root, comparison_cfg)

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
                    "comparison_config": str(comparison_path),
                    "run_root": str(run_root),
                    "base_model": utility["base_model"],
                    "case_count": len(cases),
                    "missing_adapters": [case for case in cases if not case["adapter_exists"]],
                    "cases": cases,
                    "outputs": outputs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    missing = [case for case in cases if not case["adapter_exists"]]
    if missing:
        raise FileNotFoundError(f"Missing final adapter directories: {missing[:3]} ... total={len(missing)}")
    if "SLURM_JOB_ID" not in os.environ and not os.environ.get("ALLOW_NON_SLURM_GPU_RUN"):
        print("WARNING: running outside Slurm; set ALLOW_NON_SLURM_GPU_RUN=1 to silence this warning.")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(scratch_root / "hf_home"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(scratch_root / "hf_home"))

    started = time.time()
    base_model = str(utility["base_model"])
    max_drop = float(utility.get("max_absolute_drop", 0.05))
    evals: list[dict[str, Any]] = []
    base_eval = _run_eval(name="base", base_model=base_model, adapter_path=None, utility=utility, raw_dir=raw_dir)
    evals.append(base_eval)
    base_total = base_eval.get("total_accuracy")
    rows: list[dict[str, Any]] = [
        {
            "kind": "base",
            "case_id": "base",
            "payload": "",
            "seed": "",
            "source_stage": "",
            "utility_status": base_eval["status"],
            "total_accuracy": base_total,
            "base_total_accuracy": base_total,
            "absolute_drop": 0.0 if base_total is not None else None,
            "relative_drop": 0.0 if base_total else None,
            "utility_pass": base_eval["status"] == "completed",
            "adapter_path": "",
            "adapter_exists": "",
            "eval_results_path": base_eval["eval_results_path"],
            "missing_metrics": base_eval["missing_metrics"],
            "error": base_eval["error"],
        }
    ]
    for case in cases:
        eval_row = _run_eval(
            name=str(case["case_id"]),
            base_model=base_model,
            adapter_path=str(case["adapter_path"]),
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
                **case,
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
    completed_adapter_rows = [row for row in adapter_rows if row["utility_status"] == "completed" and row["total_accuracy"] is not None]
    adapter_scores = [float(row["total_accuracy"]) for row in completed_adapter_rows]
    adapter_drops = [float(row["absolute_drop"]) for row in completed_adapter_rows if row["absolute_drop"] is not None]
    score_mean, score_ci_low, score_ci_high = _mean_ci95(adapter_scores)
    drop_mean, drop_ci_low, drop_ci_high = _mean_ci95(adapter_drops)
    utility_pass = bool(
        base_eval["status"] == "completed"
        and len(completed_adapter_rows) == len(adapter_rows)
        and adapter_drops
        and max(adapter_drops) <= max_drop
    )
    decision = (
        "OURS_TINYBENCH_UTILITY_PASS: matched TinyBench utility can be reused by matched comparison."
        if utility_pass
        else "OURS_TINYBENCH_UTILITY_NOT_PASSED: inspect failed adapters or utility drop before utility comparison claims."
    )
    summary = {
        "schema_name": "ours_tinybench_utility_summary",
        "schema_version": 1,
        "generated_at": _utc_now(),
        "status": "completed" if completed_adapter_rows else "failed",
        "decision": decision,
        "utility_pass": utility_pass,
        "base_model": base_model,
        "comparison_config": str(comparison_path),
        "run_root": str(run_root),
        "tasks": str(utility.get("tasks", "tinyBenchmarks")),
        "apply_chat_template": bool(utility.get("apply_chat_template", True)),
        "max_absolute_drop": max_drop,
        "case_count": len(cases),
        "completed_adapter_count": len(completed_adapter_rows),
        "base_total_accuracy": base_total,
        "adapter_total_accuracy_mean": score_mean,
        "adapter_total_accuracy_min": min(adapter_scores) if adapter_scores else None,
        "adapter_total_accuracy_max": max(adapter_scores) if adapter_scores else None,
        "adapter_total_accuracy_ci95_low": score_ci_low,
        "adapter_total_accuracy_ci95_high": score_ci_high,
        "absolute_drop_mean": drop_mean,
        "absolute_drop_max": max(adapter_drops) if adapter_drops else None,
        "absolute_drop_ci95_low": drop_ci_low,
        "absolute_drop_ci95_high": drop_ci_high,
        "rows": rows,
        "evals": evals,
        "output_doc": str(output_doc),
        "output_table": str(output_table),
        "output_summary": str(output_summary),
        "output_compute": str(output_compute),
    }
    compute = {
        "schema_name": "ours_tinybench_utility_compute",
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
    return 0 if utility_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

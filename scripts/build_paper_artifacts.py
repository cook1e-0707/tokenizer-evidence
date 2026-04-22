from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.report import AttackRunSummary, EvalRunSummary, TrainRunSummary, load_result_json
from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manuscript-facing tables from standing run artifacts.")
    parser.add_argument(
        "--standing-config",
        default="configs/reporting/qwen7b_standing_evidence_v1.yaml",
        help="YAML file enumerating explicit standing run inclusions.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/processed/paper_stats",
        help="Directory for JSON paper-stat artifacts.",
    )
    parser.add_argument(
        "--tables-dir",
        default="results/tables",
        help="Directory for CSV/TeX table artifacts.",
    )
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _find_one(case_root: Path, pattern: str) -> Path:
    matches = sorted(case_root.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one match for pattern={pattern!r} under {case_root}, found {len(matches)}"
        )
    return matches[0]


def _parse_time_limit_hours(raw: str) -> float:
    hours, minutes, seconds = raw.split(":")
    return int(hours) + int(minutes) / 60.0 + int(seconds) / 3600.0


def _load_runtime_record(config_path: Path) -> dict[str, Any]:
    payload = _load_yaml(config_path)
    runtime = dict(payload.get("runtime", {}))
    resources = dict(runtime.get("resources", {}))
    return {
        "partition": str(resources.get("partition", "")),
        "num_gpus": int(resources.get("num_gpus", 0) or 0),
        "cpus": int(resources.get("cpus", 0) or 0),
        "mem_gb": int(resources.get("mem_gb", 0) or 0),
        "time_limit": str(resources.get("time_limit", "00:00:00")),
        "time_limit_hours": _parse_time_limit_hours(str(resources.get("time_limit", "00:00:00"))),
        "variant_name": str(payload.get("run", {}).get("variant_name", "")),
    }


def _load_submission_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0, "ci95_half_width": 0.0}
    mean = float(sum(values) / len(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    sem = float(std / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_half_width": float(1.96 * sem) if len(values) > 1 else 0.0,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _latex_escape(raw: Any) -> str:
    return str(raw).replace("_", "\\_")


def _write_main_tex(path: Path, summary_row: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llllll}",
        "\\toprule",
        "Stage & Blocks & Payload scope & Seed scope & Successful runs & Outcome \\\\",
        "\\midrule",
        (
            f"{_latex_escape(summary_row['stage'])} & {summary_row['blocks']} & "
            f"{_latex_escape(summary_row['payload_scope'])} & {_latex_escape(summary_row['seed_scope'])} & "
            f"{summary_row['successful_runs']} & {_latex_escape(summary_row['outcome'])} \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Accepted clean compiled-path runs for Qwen/Qwen2.5-7B-Instruct. All listed runs also achieved verifier success and decoded the intended payload correctly.}",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_robustness_stage_tex(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llllll}",
        "\\toprule",
        "Stage & Payloads & Seeds & Attack families & Runs & Outcome \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['stage'])} & {_latex_escape(row['payload_scope'])} & {_latex_escape(row['seed_scope'])} & "
            f"{_latex_escape(row['attack_family_scope'])} & {row['runs']} & {_latex_escape(row['outcome'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Accepted robustness stages on the Qwen/Qwen2.5-7B-Instruct compiled-c3 path.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_robustness_family_tex(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Attack family & Coverage & accepted\\_before & accepted\\_after=true & accepted\\_after=false \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['attack_name'])} & {_latex_escape(row['coverage'])} & "
            f"{row['accepted_before_true']}/{row['runs']} & {row['accepted_after_true']}/{row['runs']} & "
            f"{row['accepted_after_false']}/{row['runs']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Observed attack-family behavior on accepted compiled-c3 baselines. Whitespace-only perturbations preserved acceptance, while truncation and delimiter destruction consistently broke acceptance.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _noncanonical_members_from_gate(gate_payload: dict[str, Any]) -> str:
    members: list[str] = []
    for slot in gate_payload.get("slot_diagnostics", []):
        if not isinstance(slot, dict):
            continue
        if bool(slot.get("is_bucket_correct")) and not bool(slot.get("is_slot_exact")):
            slot_type = str(slot.get("slot_type", "")).strip()
            observed_value = str(slot.get("observed_value", "")).strip()
            if slot_type and observed_value:
                members.append(f"{slot_type}={observed_value}")
    return "; ".join(members) if members else "--"


def _collect_theorem_t2_records(
    repo_root: Path,
    standing_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    theorem_config = standing_config.get("theorem_t2")
    if not isinstance(theorem_config, dict):
        return [], []
    rows: list[dict[str, Any]] = []
    inclusion_rows: list[dict[str, Any]] = []
    for case in theorem_config.get("cases", []):
        case_root = _resolve_path(repo_root, str(case["case_root"]))
        train_summary_path = _find_one(case_root, "train_summary.json")
        eval_summary_path = _find_one(case_root, "eval_summary.json")
        training_health_path = _find_one(case_root, "training_health.json")
        compiled_gate_path = _find_one(case_root, "compiled_gate_result.json")

        train_summary = load_result_json(train_summary_path)
        eval_summary = load_result_json(eval_summary_path)
        if not isinstance(train_summary, TrainRunSummary):
            raise TypeError(f"{train_summary_path} is not a train summary")
        if not isinstance(eval_summary, EvalRunSummary):
            raise TypeError(f"{eval_summary_path} is not an eval summary")
        training_health = json.loads(training_health_path.read_text(encoding="utf-8"))
        gate_payload = json.loads(compiled_gate_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "stage": str(theorem_config.get("stage", "T2-r1")),
                "objective": str(case["objective"]),
                "payload": str(theorem_config.get("payload", "")),
                "seed": int(theorem_config.get("seed", 0) or 0),
                "accepted": bool(eval_summary.accepted),
                "verifier_success": bool(eval_summary.verifier_success),
                "decoded_payload": str(eval_summary.decoded_payload),
                "field_valid_rate": float(gate_payload.get("field_valid_rate", 0.0) or 0.0),
                "bucket_correct_rate": float(gate_payload.get("bucket_correct_rate", 0.0) or 0.0),
                "slot_exact_rate": float(gate_payload.get("slot_exact_rate", 0.0) or 0.0),
                "chosen_slot_values": " / ".join(str(value) for value in gate_payload.get("parsed_slot_values", [])),
                "chosen_noncanonical_bucket_members": _noncanonical_members_from_gate(gate_payload),
                "objective_mode": str(training_health.get("objective_mode", "")),
                "final_loss": float(train_summary.final_loss),
                "git_commit": str(eval_summary.git_commit),
                "outcome": (
                    "exact-slot pass"
                    if bool(eval_summary.accepted)
                    else "bucket-correct, exact-slot fail"
                    if float(gate_payload.get("bucket_correct_rate", 0.0) or 0.0) == 1.0
                    and float(gate_payload.get("slot_exact_rate", 0.0) or 0.0) < 1.0
                    else "non-accepted"
                ),
            }
        )
        inclusion_rows.append(
            {
                "workstream": "T2",
                "stage": str(theorem_config.get("stage", "T2-r1")),
                "objective": str(case["objective"]),
                "payload": str(theorem_config.get("payload", "")),
                "seed": int(theorem_config.get("seed", 0) or 0),
                "case_root": str(case_root),
                "train_summary_path": str(train_summary_path),
                "eval_summary_path": str(eval_summary_path),
                "training_health_path": str(training_health_path),
                "compiled_gate_result_path": str(compiled_gate_path),
                "git_commit": str(eval_summary.git_commit),
            }
        )
    return rows, inclusion_rows


def _build_theorem_t2_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    git_commits = sorted({str(row["git_commit"]) for row in rows})
    return {
        "stage": str(rows[0]["stage"]),
        "payload": str(rows[0]["payload"]),
        "seed": int(rows[0]["seed"]),
        "git_commits": git_commits,
        "provenance_note": (
            "bucket_mass comes from commit 1603838, while fixed_representative and uniform_bucket come from "
            "commit f968ce4. This is intentional: f968ce4 only repaired bucket-key handling in the non-bucket-mass "
            "branches and does not change the bucket_mass objective path."
            if set(git_commits) == {"1603838", "f968ce4"}
            else "all theorem variants came from the same code commit"
        ),
        "variants": rows,
        "core_finding": {
            "fixed_representative": "passes exact-slot",
            "bucket_mass": "bucket-correct but fails exact-slot by choosing a non-canonical member",
            "uniform_bucket": "bucket-correct but fails exact-slot by choosing non-canonical members",
        },
    }


def _write_t2_main_tex(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Objective & Accepted & bucket\\_correct\\_rate & slot\\_exact\\_rate & Outcome \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['objective'])} & {_latex_escape(str(row['accepted']).lower())} & "
            f"{row['bucket_correct_rate']:.2f} & {row['slot_exact_rate']:.2f} & {_latex_escape(row['outcome'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Theorem~2 objective comparison on the controlled multi-member-bucket Qwen setting at payload U15. Fixed representative supervision preserves exact-slot fidelity, while bucket-mass and uniform-bucket supervision remain bucket-correct but fail the canonical exact-slot gate.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_t2_supplement_tex(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Objective & bucket\\_correct\\_rate & slot\\_exact\\_rate & Chosen non-canonical bucket members \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['objective'])} & {row['bucket_correct_rate']:.2f} & "
            f"{row['slot_exact_rate']:.2f} & {_latex_escape(row['chosen_noncanonical_bucket_members'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Supplementary bucket-level metrics for Theorem~2. The uniform-bucket objective remains bucket-correct while selecting non-canonical bucket members such as SECTION=review and TOPIC=climate.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _collect_main_records(repo_root: Path, standing_config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage = str(standing_config["main_clean"]["stage"])
    rows: list[dict[str, Any]] = []
    inclusion_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    for case in standing_config["main_clean"]["cases"]:
        case_root = _resolve_path(repo_root, str(case["case_root"]))
        train_summary_path = _find_one(case_root, "train_summary.json")
        eval_summary_path = _find_one(case_root, "eval_summary.json")
        training_health_path = _find_one(case_root, "training_health.json")
        train_config_path = _find_one(case_root, "runs/exp_train/*/config.resolved.yaml")
        eval_config_path = _find_one(case_root, "runs/exp_eval/*/config.resolved.yaml")
        train_submission_path = _find_one(case_root, "runs/exp_train/*/submission.json")
        eval_submission_path = _find_one(case_root, "runs/exp_eval/*/submission.json")

        train_summary = load_result_json(train_summary_path)
        eval_summary = load_result_json(eval_summary_path)
        if not isinstance(train_summary, TrainRunSummary):
            raise TypeError(f"{train_summary_path} is not a train summary")
        if not isinstance(eval_summary, EvalRunSummary):
            raise TypeError(f"{eval_summary_path} is not an eval summary")
        if not eval_summary.accepted or not bool(eval_summary.verifier_success):
            raise ValueError(f"Included main clean run is not accepted: {eval_summary_path}")
        training_health = json.loads(training_health_path.read_text(encoding="utf-8"))
        if training_health.get("first_nan_step") is not None:
            raise ValueError(f"Included main clean run has first_nan_step set: {training_health_path}")

        rows.append(
            {
                "stage": stage,
                "case_id": str(case["id"]),
                "payload": str(case["payload"]),
                "seed": int(case["seed"]),
                "accepted": bool(eval_summary.accepted),
                "verifier_success": bool(eval_summary.verifier_success),
                "decoded_payload": str(eval_summary.decoded_payload),
                "match_ratio": float(eval_summary.match_ratio),
                "final_loss": float(train_summary.final_loss),
                "train_steps": int(train_summary.steps),
                "train_run_id": train_summary.run_id,
                "eval_run_id": eval_summary.run_id,
            }
        )
        inclusion_rows.append(
            {
                "workstream": "S1",
                "stage": stage,
                "case_id": str(case["id"]),
                "payload": str(case["payload"]),
                "seed": int(case["seed"]),
                "case_root": str(case_root),
                "train_summary_path": str(train_summary_path),
                "eval_summary_path": str(eval_summary_path),
                "training_health_path": str(training_health_path),
                "train_run_id": train_summary.run_id,
                "eval_run_id": eval_summary.run_id,
            }
        )
        for run_kind, config_path, submission_path in (
            ("train", train_config_path, train_submission_path),
            ("eval", eval_config_path, eval_submission_path),
        ):
            runtime = _load_runtime_record(config_path)
            submission = _load_submission_record(submission_path)
            compute_rows.append(
                {
                    "workstream": "S1",
                    "stage": stage,
                    "run_kind": run_kind,
                    "case_id": str(case["id"]),
                    "payload": str(case["payload"]),
                    "seed": int(case["seed"]),
                    "partition": runtime["partition"],
                    "num_gpus": runtime["num_gpus"],
                    "cpus": runtime["cpus"],
                    "mem_gb": runtime["mem_gb"],
                    "time_limit": runtime["time_limit"],
                    "time_limit_hours": runtime["time_limit_hours"],
                    "requested_gpu_hours": float(runtime["num_gpus"]) * float(runtime["time_limit_hours"]),
                    "requested_cpu_hours": float(runtime["cpus"]) * float(runtime["time_limit_hours"]),
                    "slurm_job_id": str(submission.get("slurm_job_id", "")),
                    "variant_name": runtime["variant_name"],
                }
            )
    return rows, inclusion_rows, compute_rows


def _collect_robustness_records(
    repo_root: Path,
    standing_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inclusion_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    for case in standing_config["robustness"]["cases"]:
        case_root = _resolve_path(repo_root, str(case["case_root"]))
        attack_summary_path = _find_one(case_root, "attack_output.json")
        attack_config_path = _find_one(case_root, "config.resolved.yaml")
        attack_submission_path = _find_one(case_root, "submission.json")
        attack_metadata_path = _find_one(case_root, "attack_metadata.json")

        attack_summary = load_result_json(attack_summary_path)
        if not isinstance(attack_summary, AttackRunSummary):
            raise TypeError(f"{attack_summary_path} is not an attack summary")
        if not attack_summary.accepted_before:
            raise ValueError(f"Included robustness run does not start from accepted baseline: {attack_summary_path}")
        if attack_summary.status != "completed":
            raise ValueError(f"Included robustness run is not completed: {attack_summary_path}")
        attack_metadata = json.loads(attack_metadata_path.read_text(encoding="utf-8"))
        runtime = _load_runtime_record(attack_config_path)
        submission = _load_submission_record(attack_submission_path)

        rows.append(
            {
                "stage": str(case["stage"]),
                "case_id": str(case["id"]),
                "payload": str(case["payload"]),
                "seed": int(case["seed"]),
                "attack_name": str(case["attack_name"]),
                "accepted_before": bool(attack_summary.accepted_before),
                "accepted_after": bool(attack_summary.accepted_after),
                "status": str(attack_summary.status),
                "git_commit": str(attack_summary.git_commit),
                "run_id": attack_summary.run_id,
                "evidence_source": str(attack_metadata.get("evidence_source", "")),
            }
        )
        inclusion_rows.append(
            {
                "workstream": "S2",
                "stage": str(case["stage"]),
                "case_id": str(case["id"]),
                "payload": str(case["payload"]),
                "seed": int(case["seed"]),
                "attack_name": str(case["attack_name"]),
                "case_root": str(case_root),
                "attack_summary_path": str(attack_summary_path),
                "attack_metadata_path": str(attack_metadata_path),
                "run_id": attack_summary.run_id,
            }
        )
        compute_rows.append(
            {
                "workstream": "S2",
                "stage": str(case["stage"]),
                "run_kind": "attack",
                "case_id": str(case["id"]),
                "payload": str(case["payload"]),
                "seed": int(case["seed"]),
                "partition": runtime["partition"],
                "num_gpus": runtime["num_gpus"],
                "cpus": runtime["cpus"],
                "mem_gb": runtime["mem_gb"],
                "time_limit": runtime["time_limit"],
                "time_limit_hours": runtime["time_limit_hours"],
                "requested_gpu_hours": float(runtime["num_gpus"]) * float(runtime["time_limit_hours"]),
                "requested_cpu_hours": float(runtime["cpus"]) * float(runtime["time_limit_hours"]),
                "slurm_job_id": str(submission.get("slurm_job_id", "")),
                "variant_name": runtime["variant_name"],
                "attack_name": str(case["attack_name"]),
            }
        )
    return rows, inclusion_rows, compute_rows


def _build_main_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": "compiled-c3-r4",
        "blocks": 2,
        "payloads": sorted({str(row["payload"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "successful_runs": f"{sum(1 for row in rows if bool(row['accepted']))}/{len(rows)}",
        "outcome": "all accepted",
        "metrics": {
            "accepted_rate": _stats([1.0 if bool(row["accepted"]) else 0.0 for row in rows]),
            "verifier_success_rate": _stats([1.0 if bool(row["verifier_success"]) else 0.0 for row in rows]),
            "match_ratio": _stats([float(row["match_ratio"]) for row in rows]),
            "final_loss": _stats([float(row["final_loss"]) for row in rows]),
        },
    }


def _build_robustness_summary(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_stage_family: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stage = str(row["stage"])
        attack_name = str(row["attack_name"])
        by_stage_family.setdefault((stage, attack_name), []).append(row)
        by_family.setdefault(attack_name, []).append(row)

    stage_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for (stage, attack_name), group in sorted(by_stage_family.items()):
        stage_rows.append(
            {
                "stage": stage,
                "attack_name": attack_name,
                "runs": len(group),
                "accepted_before_true": sum(1 for row in group if bool(row["accepted_before"])),
                "accepted_after_true": sum(1 for row in group if bool(row["accepted_after"])),
                "accepted_after_false": sum(1 for row in group if not bool(row["accepted_after"])),
            }
        )
    stage_summary_rows = [
        {
            "stage": "batch3c",
            "payload_scope": "U00/U03/U12/U15",
            "seed_scope": "17",
            "attack_family_scope": "whitespace, truncate",
            "runs": sum(1 for row in rows if str(row["stage"]) == "batch3c"),
            "outcome": "seed expansion passed",
        },
        {
            "stage": "batch3d",
            "payload_scope": "U00/U03/U12/U15",
            "seed_scope": "17",
            "attack_family_scope": "delimiter",
            "runs": sum(1 for row in rows if str(row["stage"]) == "batch3d"),
            "outcome": "new family passed",
        },
    ]
    for attack_name, group in sorted(by_family.items()):
        family_rows.append(
            {
                "attack_name": attack_name,
                "runs": len(group),
                "accepted_before_rate": _stats([1.0 if bool(row["accepted_before"]) else 0.0 for row in group]),
                "accepted_after_rate": _stats([1.0 if bool(row["accepted_after"]) else 0.0 for row in group]),
                "stages": sorted({str(row["stage"]) for row in group}),
            }
        )
    summary = {
        "stages": stage_rows,
        "families": family_rows,
    }
    return summary, stage_summary_rows, [
        {
            "attack_name": row["attack_name"],
            "runs": row["runs"],
            "accepted_before_mean": row["accepted_before_rate"]["mean"],
            "accepted_before_std": row["accepted_before_rate"]["std"],
            "accepted_after_mean": row["accepted_after_rate"]["mean"],
            "accepted_after_std": row["accepted_after_rate"]["std"],
            "stages": ",".join(row["stages"]),
            "coverage": "Batch 3C" if row["attack_name"] in {"whitespace_scrub", "truncate_tail"} else "Batch 3D",
            "accepted_before_true": row["runs"],
            "accepted_after_true": int(round(row["accepted_after_rate"]["mean"] * row["runs"])),
            "accepted_after_false": row["runs"] - int(round(row["accepted_after_rate"]["mean"] * row["runs"])),
        }
        for row in family_rows
    ]


def _build_stat_rows(main_summary: dict[str, Any], robustness_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, metric_payload in main_summary["metrics"].items():
        rows.append(
            {
                "workstream": "S1/C1",
                "scope": "compiled-c3-r4",
                "metric": metric_name,
                "n": metric_payload["n"],
                "mean": metric_payload["mean"],
                "std": metric_payload["std"],
                "ci95_half_width": metric_payload["ci95_half_width"],
            }
        )
    for family in robustness_summary["families"]:
        accepted_before_rate = family["accepted_before_rate"]
        accepted_after_rate = family["accepted_after_rate"]
        rows.append(
            {
                "workstream": "S2/C1",
                "scope": family["attack_name"],
                "metric": "accepted_before_rate",
                "n": accepted_before_rate["n"],
                "mean": accepted_before_rate["mean"],
                "std": accepted_before_rate["std"],
                "ci95_half_width": accepted_before_rate["ci95_half_width"],
            }
        )
        rows.append(
            {
                "workstream": "S2/C1",
                "scope": family["attack_name"],
                "metric": "accepted_after_rate",
                "n": accepted_after_rate["n"],
                "mean": accepted_after_rate["mean"],
                "std": accepted_after_rate["std"],
                "ci95_half_width": accepted_after_rate["ci95_half_width"],
            }
        )
    return rows


def _build_compute_accounting(compute_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in compute_rows:
        grouped.setdefault((str(row["stage"]), str(row["run_kind"])), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (stage, run_kind), group in sorted(grouped.items()):
        summary_rows.append(
            {
                "stage": stage,
                "run_kind": run_kind,
                "runs": len(group),
                "partition": group[0]["partition"],
                "num_gpus": group[0]["num_gpus"],
                "cpus": group[0]["cpus"],
                "mem_gb": group[0]["mem_gb"],
                "time_limit": group[0]["time_limit"],
                "requested_gpu_hours": round(sum(float(item["requested_gpu_hours"]) for item in group), 4),
                "requested_cpu_hours": round(sum(float(item["requested_cpu_hours"]) for item in group), 4),
            }
        )
    return {"rows": summary_rows}, summary_rows


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    standing_config = _load_yaml(_resolve_path(repo_root, args.standing_config))
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = _resolve_path(repo_root, args.tables_dir)

    main_rows, main_inclusion_rows, main_compute_rows = _collect_main_records(repo_root, standing_config)
    robustness_rows, robustness_inclusion_rows, robustness_compute_rows = _collect_robustness_records(
        repo_root, standing_config
    )
    theorem_t2_rows, theorem_t2_inclusion_rows = _collect_theorem_t2_records(repo_root, standing_config)
    main_summary = _build_main_summary(main_rows)
    robustness_summary, robustness_stage_rows, robustness_family_rows = _build_robustness_summary(
        robustness_rows
    )
    theorem_t2_summary = _build_theorem_t2_summary(theorem_t2_rows)
    stat_rows = _build_stat_rows(main_summary, robustness_summary)
    compute_summary, compute_summary_rows = _build_compute_accounting(main_compute_rows + robustness_compute_rows)

    _write_json(output_dir / "main_clean_summary.json", main_summary)
    _write_json(output_dir / "robustness_summary.json", robustness_summary)
    if theorem_t2_summary:
        _write_json(output_dir / "t2_r1_summary.json", theorem_t2_summary)
    _write_json(
        output_dir / "run_inclusion_lists.json",
        {
            "main_clean": main_inclusion_rows,
            "robustness": robustness_inclusion_rows,
            "theorem_t2": theorem_t2_inclusion_rows,
        },
    )
    _write_json(output_dir / "compute_accounting.json", compute_summary)

    _write_csv(tables_dir / "compiled_c3_r4_main.csv", main_rows)
    _write_csv(tables_dir / "batch3cd_appendix.csv", robustness_stage_rows)
    _write_csv(tables_dir / "batch3_family_summary.csv", robustness_family_rows)
    if theorem_t2_rows:
        _write_csv(tables_dir / "t2_r1_objective_comparison.csv", theorem_t2_rows)
        _write_csv(
            tables_dir / "t2_r1_bucket_supplement.csv",
            [
                {
                    "objective": row["objective"],
                    "bucket_correct_rate": row["bucket_correct_rate"],
                    "slot_exact_rate": row["slot_exact_rate"],
                    "chosen_noncanonical_bucket_members": row["chosen_noncanonical_bucket_members"],
                }
                for row in theorem_t2_rows
            ],
        )
    _write_csv(tables_dir / "stat_summary.csv", stat_rows)
    _write_csv(tables_dir / "compute_accounting.csv", compute_summary_rows)
    _write_main_tex(
        tables_dir / "compiled_c3_r4_main.tex",
        {
            "stage": main_summary["stage"],
            "blocks": main_summary["blocks"],
            "payload_scope": "/".join(main_summary["payloads"]),
            "seed_scope": ",".join(str(seed) for seed in main_summary["seeds"]),
            "successful_runs": main_summary["successful_runs"],
            "outcome": main_summary["outcome"],
        },
    )
    _write_robustness_stage_tex(tables_dir / "batch3cd_appendix.tex", robustness_stage_rows)
    _write_robustness_family_tex(tables_dir / "batch3_family_summary.tex", robustness_family_rows)
    if theorem_t2_rows:
        _write_t2_main_tex(tables_dir / "t2_r1_objective_comparison.tex", theorem_t2_rows)
        _write_t2_supplement_tex(tables_dir / "t2_r1_bucket_supplement.tex", theorem_t2_rows)

    print(f"wrote paper stats to {output_dir}")
    print(f"wrote paper tables to {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

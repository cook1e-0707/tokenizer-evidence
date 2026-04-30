from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the selected Qwen Perinucleus candidate for final protocol use.")
    parser.add_argument(
        "--utility-summary",
        default="results/processed/paper_stats/baseline_perinucleus_qwen_candidate_utility_summary.json",
    )
    parser.add_argument(
        "--capacity-summary",
        default="results/processed/paper_stats/baseline_perinucleus_qwen_capacity_sweep_summary.json",
    )
    parser.add_argument("--llama-anchor-doc", default="docs/baseline_perinucleus_llama_anchor_result.md")
    parser.add_argument("--selected-arm", default="")
    parser.add_argument("--output-doc", default="docs/baseline_perinucleus_qwen_candidate_freeze.md")
    parser.add_argument(
        "--output-summary",
        default="results/processed/paper_stats/baseline_perinucleus_qwen_candidate_freeze_summary.json",
    )
    parser.add_argument("--output-table", default="results/tables/baseline_perinucleus_qwen_candidate_freeze.csv")
    parser.add_argument(
        "--output-config",
        default="configs/experiment/baselines/perinucleus_official/qwen_frozen_candidate__baseline_perinucleus_official.yaml",
    )
    return parser.parse_args()


def discover_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return current


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _candidate_by_arm(summary: dict[str, Any], arm_id: str) -> dict[str, Any]:
    for arm in summary.get("arms", []):
        if str(arm.get("arm_id")) == arm_id:
            return arm
    raise RuntimeError(f"Selected arm {arm_id!r} was not found in capacity summary.")


def _write_table(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "arm_id",
        "status",
        "base_model",
        "adapter_path",
        "fingerprints_file",
        "num_fingerprints",
        "key_length",
        "response_length",
        "max_sequence_length",
        "target_modules_label",
        "target_modules",
        "lora_rank",
        "lora_alpha_ratio",
        "epochs_run",
        "regularization_label",
        "exact_accuracy",
        "target_rank_mean",
        "target_probability_mean",
        "base_utility",
        "adapter_utility",
        "absolute_drop",
        "relative_drop",
        "utility_pass",
        "llama_anchor_pass",
        "capacity_gate_pass",
        "final_protocol_allowed",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        out = dict(row)
        out["target_modules"] = json.dumps(out.get("target_modules", []), ensure_ascii=True)
        writer.writerow({key: out.get(key, "") for key in fieldnames})


def _write_doc(path: Path, summary: dict[str, Any]) -> None:
    selected = summary["selected_candidate"]
    gates = summary["gates"]
    lines = [
        "# Qwen Perinucleus Candidate Freeze",
        "",
        f"Generated at: `{summary['generated_at']}`",
        f"Decision: `{summary['decision']}`",
        "",
        "This freezes the Qwen-adapted official Scalable Fingerprinting / Perinucleus candidate for downstream final-protocol use. It is a candidate freeze, not a final matrix result.",
        "",
        "## Frozen Candidate",
        "",
        f"- Arm: `{selected['arm_id']}`",
        f"- Base model: `{selected['base_model']}`",
        f"- Adapter path: `{selected['adapter_path']}`",
        f"- Fingerprints file: `{selected['fingerprints_file']}`",
        f"- Fingerprints: `{selected['num_fingerprints']}`",
        f"- Target modules: `{selected['target_modules_label']}` / `{selected['target_modules']}`",
        f"- LoRA rank: `{selected['lora_rank']}`",
        f"- Epochs run: `{selected['epochs_run']}`",
        f"- Regularization label: `{selected['regularization_label']}`",
        "",
        "## Gate Evidence",
        "",
        "| gate | pass | evidence |",
        "| --- | --- | --- |",
        f"| Llama anchor | {gates['llama_anchor_pass']} | `{summary['inputs']['llama_anchor_doc']}` |",
        f"| Qwen capacity sweep | {gates['capacity_gate_pass']} | exact={selected['exact_accuracy']}, rank_mean={selected['target_rank_mean']}, prob_mean={selected['target_probability_mean']} |",
        f"| Qwen utility sanity | {gates['utility_gate_pass']} | utility={selected['adapter_utility']}, base={selected['base_utility']}, drop={selected['absolute_drop']} |",
        "",
        "## Final-Protocol Constraints",
        "",
        "- Final Perinucleus Qwen runs must use this exact adapter path unless this freeze is superseded before any final launch.",
        "- Do not use final-matrix feedback to change LoRA rank, target modules, epochs, fingerprints, or selected adapter.",
        "- The baseline must be described as an adapted Qwen LoRA reproduction of official Scalable/Perinucleus, not as an unmodified full fine-tune.",
        "- Utility drops are signed as `base_total_accuracy - adapter_total_accuracy`; negative values mean the adapter scored higher than the base on this tinyBenchmarks sanity.",
        "",
        "## Source Artifacts",
        "",
        f"- Utility summary: `{summary['inputs']['utility_summary']}`",
        f"- Capacity summary: `{summary['inputs']['capacity_summary']}`",
        f"- Llama anchor doc: `{summary['inputs']['llama_anchor_doc']}`",
        f"- Freeze summary: `{summary['outputs']['summary']}`",
        f"- Freeze table: `{summary['outputs']['table']}`",
        f"- Freeze config: `{summary['outputs']['config']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in [":", "#", "{", "}", "[", "]", ",", "\n"]):
        return json.dumps(text)
    return text


def _write_config(path: Path, summary: dict[str, Any]) -> None:
    selected = summary["selected_candidate"]
    lines = [
        "version: 1",
        "schema_name: baseline_perinucleus_qwen_frozen_candidate",
        "status: frozen_for_final_protocol",
        f"generated_at: {_yaml_scalar(summary['generated_at'])}",
        f"decision: {_yaml_scalar(summary['decision'])}",
        "official_source:",
        "  repo_url: https://github.com/SewoongLab/scalable-fingerprinting-of-llms",
        f"  commit_hash: {_yaml_scalar(summary['official_commit'])}",
        "model:",
        f"  base: {_yaml_scalar(selected['base_model'])}",
        "  use_chat_template: true",
        "candidate:",
        f"  arm_id: {_yaml_scalar(selected['arm_id'])}",
        f"  adapter_path: {_yaml_scalar(selected['adapter_path'])}",
        f"  fingerprints_file: {_yaml_scalar(selected['fingerprints_file'])}",
        f"  num_fingerprints: {selected['num_fingerprints']}",
        f"  key_length: {selected['key_length']}",
        f"  response_length: {selected['response_length']}",
        f"  max_sequence_length: {selected['max_sequence_length']}",
        f"  target_modules_label: {_yaml_scalar(selected['target_modules_label'])}",
        "  target_modules:",
        *[f"    - {_yaml_scalar(module)}" for module in selected["target_modules"]],
        f"  lora_rank: {selected['lora_rank']}",
        f"  lora_alpha_ratio: {selected['lora_alpha_ratio']}",
        f"  epochs_run: {selected['epochs_run']}",
        f"  regularization_label: {_yaml_scalar(selected['regularization_label'])}",
        "fingerprint_fidelity:",
        f"  exact_accuracy: {selected['exact_accuracy']}",
        f"  target_rank_mean: {selected['target_rank_mean']}",
        f"  target_probability_mean: {selected['target_probability_mean']}",
        f"  train_ce_mean: {selected['train_ce_mean']}",
        "utility_sanity:",
        f"  base_total_accuracy: {selected['base_utility']}",
        f"  adapter_total_accuracy: {selected['adapter_utility']}",
        f"  signed_absolute_drop: {selected['absolute_drop']}",
        f"  relative_drop: {selected['relative_drop']}",
        f"  utility_pass: {_yaml_scalar(selected['utility_pass'])}",
        "source_artifacts:",
        f"  capacity_sweep_summary: {_yaml_scalar(summary['inputs']['capacity_summary'])}",
        f"  candidate_utility_summary: {_yaml_scalar(summary['inputs']['utility_summary'])}",
        f"  llama_anchor_doc: {_yaml_scalar(summary['inputs']['llama_anchor_doc'])}",
        "final_protocol_constraints:",
        "  - use_exact_adapter_path",
        "  - no_hyperparameter_changes_after_freeze",
        "  - no_final_matrix_feedback_for_candidate_selection",
        "  - label_as_qwen_lora_adaptation_of_official_perinucleus",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    utility_summary_path = _resolve(repo_root, args.utility_summary)
    capacity_summary_path = _resolve(repo_root, args.capacity_summary)
    llama_anchor_doc_path = _resolve(repo_root, args.llama_anchor_doc)
    output_doc_path = _resolve(repo_root, args.output_doc)
    output_summary_path = _resolve(repo_root, args.output_summary)
    output_table_path = _resolve(repo_root, args.output_table)
    output_config_path = _resolve(repo_root, args.output_config)

    utility_summary = _load_json(utility_summary_path)
    capacity_summary = _load_json(capacity_summary_path)
    llama_anchor_doc = llama_anchor_doc_path.read_text(encoding="utf-8")

    selected_utility = utility_summary.get("selected_candidate") or {}
    selected_arm = args.selected_arm or str(selected_utility.get("arm_id", ""))
    _require(selected_arm, "No selected arm found in candidate utility summary.")
    _require(bool(utility_summary.get("utility_pass")), "Candidate utility gate has not passed.")
    _require(bool(selected_utility.get("utility_pass")), "Selected candidate utility gate has not passed.")
    _require("LLAMA_ANCHOR_PASS" in llama_anchor_doc, "Llama anchor doc does not record LLAMA_ANCHOR_PASS.")
    _require(bool(capacity_summary.get("candidate_found")), "Qwen capacity sweep did not find a candidate.")
    _require(bool(capacity_summary.get("strong_candidate_found")), "Qwen capacity sweep did not find a strong candidate.")

    capacity_arm = _candidate_by_arm(capacity_summary, selected_arm)
    capacity_final = capacity_arm.get("final") or {}
    _require(float(capacity_final.get("exact_accuracy") or 0.0) >= 1.0, "Selected arm does not have exact_accuracy=1.0.")
    _require(float(capacity_final.get("target_rank_mean") or 99.0) <= 1.0, "Selected arm does not have target_rank_mean=1.0.")
    _require(str(capacity_arm.get("adapter_path")) == str(selected_utility.get("adapter_path")), "Adapter path mismatch.")

    selected = {
        "arm_id": selected_arm,
        "base_model": str(capacity_summary.get("model", "Qwen/Qwen2.5-7B-Instruct")),
        "adapter_path": str(selected_utility["adapter_path"]),
        "fingerprints_file": str(capacity_arm["fingerprints_file"]),
        "num_fingerprints": int(capacity_arm["num_fingerprints"]),
        "key_length": int(capacity_arm.get("key_length", 16) or 16),
        "response_length": 1,
        "max_sequence_length": int(capacity_arm.get("max_sequence_length", 64) or 64),
        "target_modules_label": str(capacity_arm["target_modules_label"]),
        "target_modules": list(capacity_arm.get("target_modules", [])),
        "lora_rank": int(capacity_arm["lora_rank"]),
        "lora_alpha_ratio": float(capacity_arm["lora_alpha_ratio"]),
        "epochs_run": int(capacity_arm["epochs_run"]),
        "regularization_label": str(capacity_arm.get("regularization_label", "")),
        "exact_accuracy": float(capacity_final["exact_accuracy"]),
        "target_rank_mean": float(capacity_final["target_rank_mean"]),
        "target_probability_mean": float(capacity_final["target_probability_mean"]),
        "train_ce_mean": float(capacity_final["train_ce_mean"]),
        "base_utility": float(selected_utility["base_total_accuracy"]),
        "adapter_utility": float(selected_utility["total_accuracy"]),
        "absolute_drop": float(selected_utility["absolute_drop"]),
        "relative_drop": float(selected_utility["relative_drop"]),
        "utility_pass": bool(selected_utility["utility_pass"]),
    }
    row = {
        **selected,
        "status": "frozen_for_final_protocol",
        "llama_anchor_pass": True,
        "capacity_gate_pass": True,
        "final_protocol_allowed": True,
    }
    summary = {
        "schema_name": "baseline_perinucleus_qwen_candidate_freeze_summary",
        "schema_version": 1,
        "generated_at": _utc_now(),
        "decision": "QWEN_PERINUCLEUS_CANDIDATE_FROZEN: final protocol may use only this frozen adapter/config unless superseded before final launch.",
        "official_commit": str(capacity_summary.get("official_commit", "")),
        "selected_candidate": selected,
        "gates": {
            "llama_anchor_pass": True,
            "capacity_gate_pass": True,
            "utility_gate_pass": True,
            "selected_candidate_frozen": True,
            "final_protocol_allowed": True,
        },
        "inputs": {
            "utility_summary": str(utility_summary_path.relative_to(repo_root)),
            "capacity_summary": str(capacity_summary_path.relative_to(repo_root)),
            "llama_anchor_doc": str(llama_anchor_doc_path.relative_to(repo_root)),
        },
        "outputs": {
            "doc": str(output_doc_path.relative_to(repo_root)),
            "summary": str(output_summary_path.relative_to(repo_root)),
            "table": str(output_table_path.relative_to(repo_root)),
            "config": str(output_config_path.relative_to(repo_root)),
        },
    }

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_table(output_table_path, row)
    _write_doc(output_doc_path, summary)
    _write_config(output_config_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

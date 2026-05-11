from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALLOWLIST = ROOT / "configs/natural_evidence_v2/run_allowlist.yaml"
DEFAULT_GATE_STATUS = ROOT / "results/natural_evidence_v2/status/gate_status.json"
DEFAULT_CURRENT_STATE = ROOT / "docs/natural_evidence_v2/CURRENT_STATE.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check natural_evidence_v2 allowlist safety against gate_status. "
            "This is a control-plane check only: it does not submit Slurm, run "
            "models, train, aggregate FAR, run Llama, or make paper claims."
        )
    )
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--gate-status", type=Path, default=DEFAULT_GATE_STATUS)
    parser.add_argument("--current-state", type=Path, default=DEFAULT_CURRENT_STATE)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--require-zero-enabled",
        action="store_true",
        help="Hard-fail if any allowlist entry is enabled.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def iter_entries(allowlist: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        section_entries = allowlist.get(section, [])
        if not isinstance(section_entries, list):
            raise ValueError(f"{section} must be a list")
        for index, raw_entry in enumerate(section_entries):
            if not isinstance(raw_entry, dict):
                raise ValueError(f"{section}[{index}] must be an object")
            entry = dict(raw_entry)
            entry["section"] = section
            entry["index"] = index
            entries.append(entry)
    return entries


def entry_text(entry: Mapping[str, Any]) -> str:
    return " ".join(
        str(entry.get(key, ""))
        for key in ("name", "claim_note", "command_pattern", "enable_condition")
    ).lower()


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def classify_entry(entry: Mapping[str, Any]) -> set[str]:
    text = entry_text(entry)
    labels: set[str] = set()
    if has_any(text, ("llama",)):
        labels.add("llama")
    if has_any(text, ("same-family", "same_family")):
        labels.add("same_family")
    if has_any(text, ("sanitizer",)):
        labels.add("sanitizer")
    if has_any(text, ("far", "full_far", "organic_far")):
        labels.add("far")
    if has_any(text, ("paper", "manuscript", "claim")):
        labels.add("paper_claim")
    if has_any(text, ("train", "training", "lora", "teacher-forced")):
        labels.add("training")
    if str(entry.get("name", "")) == "v2_r3_2_qwen_locked_scale_eval":
        labels.add("r3_2")
    return labels


def gate_flag_for_label(label: str) -> str | None:
    return {
        "llama": "llama_allowed",
        "same_family": "same_family_null_allowed",
        "sanitizer": "sanitizer_allowed",
        "far": "far_aggregation_allowed",
        "paper_claim": "paper_claim_allowed",
        "training": "training_allowed",
    }.get(label)


def main() -> int:
    args = parse_args()
    allowlist_path = resolve(args.allowlist)
    gate_status_path = resolve(args.gate_status)
    current_state_path = resolve(args.current_state)
    output_path = resolve(args.output_json)

    allowlist = read_yaml(allowlist_path)
    gate_status = read_json(gate_status_path)
    current_state_text = current_state_path.read_text(encoding="utf-8")
    entries = iter_entries(allowlist)

    enabled_entries = [
        {
            "name": str(entry.get("name", "")),
            "section": str(entry.get("section", "")),
            "command_pattern": str(entry.get("command_pattern", "")),
            "labels": sorted(classify_entry(entry)),
        }
        for entry in entries
        if bool(entry.get("enabled", False))
    ]

    forbidden_enabled_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not bool(entry.get("enabled", False)):
            continue
        labels = classify_entry(entry)
        for label in sorted(labels):
            gate_flag = gate_flag_for_label(label)
            if gate_flag and not bool(gate_status.get(gate_flag, False)):
                forbidden_enabled_entries.append(
                    {
                        "name": str(entry.get("name", "")),
                        "label": label,
                        "required_gate": gate_flag,
                        "gate_value": bool(gate_status.get(gate_flag, False)),
                    }
                )

    def disabled_for(predicate: str) -> bool:
        return not any(predicate in classify_entry(entry) and bool(entry.get("enabled", False)) for entry in entries)

    r3_2_entry = next(
        (entry for entry in entries if str(entry.get("name", "")) == "v2_r3_2_qwen_locked_scale_eval"),
        None,
    )
    unknown_enabled_entries = [
        entry
        for entry in enabled_entries
        if entry["name"] != "v2_r3_2_qwen_locked_scale_eval" and not entry["labels"]
    ]

    failures: list[str] = []
    if args.require_zero_enabled and enabled_entries:
        failures.append("enabled_entries_not_empty")
    if forbidden_enabled_entries:
        failures.append("forbidden_enabled_entries_present")
    if unknown_enabled_entries:
        failures.append("unknown_enabled_entries_present")
    if r3_2_entry is None:
        failures.append("r3_2_entry_missing")
    elif bool(r3_2_entry.get("enabled", False)):
        failures.append("r3_2_entry_enabled_during_decontamination")
    if "V2_R3" not in current_state_text:
        failures.append("current_state_missing_v2_r3_phase")

    summary = {
        "schema_name": "natural_evidence_v2_allowlist_safety_summary_v1",
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "allowlist_path": str(args.allowlist),
        "allowlist_sha256": sha256_file(allowlist_path),
        "gate_status_path": str(args.gate_status),
        "gate_status_sha256": sha256_file(gate_status_path),
        "current_state_path": str(args.current_state),
        "current_state_sha256": sha256_file(current_state_path),
        "enabled_entries": enabled_entries,
        "enabled_entry_count": len(enabled_entries),
        "forbidden_enabled_entries": forbidden_enabled_entries,
        "unknown_enabled_entries": unknown_enabled_entries,
        "llama_entries_disabled": disabled_for("llama"),
        "same_family_entries_disabled": disabled_for("same_family"),
        "sanitizer_entries_disabled": disabled_for("sanitizer"),
        "far_entries_disabled": disabled_for("far"),
        "paper_claim_entries_disabled": disabled_for("paper_claim"),
        "training_entries_disabled": disabled_for("training"),
        "r3_2_entry_disabled": r3_2_entry is not None and not bool(r3_2_entry.get("enabled", False)),
        "gate_values": {
            "training_allowed": bool(gate_status.get("training_allowed", False)),
            "llama_allowed": bool(gate_status.get("llama_allowed", False)),
            "same_family_null_allowed": bool(gate_status.get("same_family_null_allowed", False)),
            "sanitizer_allowed": bool(gate_status.get("sanitizer_allowed", False)),
            "far_aggregation_allowed": bool(gate_status.get("far_aggregation_allowed", False)),
            "paper_claim_allowed": bool(gate_status.get("paper_claim_allowed", False)),
        },
        "slurm_job_submitted": False,
        "generation_started": False,
        "training_started": False,
        "llama_started": False,
        "far_aggregation_started": False,
        "paper_claim_started": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output_json": str(output_path)}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

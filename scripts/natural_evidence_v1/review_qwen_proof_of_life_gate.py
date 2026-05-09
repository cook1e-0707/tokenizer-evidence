from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.natural_evidence_v1.common import read_yaml, resolve_repo_path, write_csv, write_json


SUMMARY_SCHEMA = "natural_evidence_qwen_proof_of_life_gate_review_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review whether Qwen variable-radix proof-of-life training may be "
            "prepared. This is a gate review only: no training, no E2E, no FAR."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--allowlist", default="configs/natural_evidence_v1/run_allowlist.yaml")
    parser.add_argument("--variable-arity-manifest", required=True)
    parser.add_argument("--full-density-summary", required=True)
    parser.add_argument("--pre-null-summary", required=True)
    parser.add_argument("--variable-radix-preflight-summary", required=True)
    parser.add_argument("--variable-radix-frame-policy-summary", default="")
    parser.add_argument("--protocol-commitment", default="docs/natural_evidence_v1/protocol_commitment.md")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-compatible-entries", type=int, default=2000)
    parser.add_argument("--min-effective-bits-per-response", type=float, default=0.8)
    parser.add_argument("--min-heldout-density-per-100-tokens", type=float, default=0.5)
    parser.add_argument("--min-configured-subset", type=int, default=500)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _bool_status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _gate(name: str, status: str, evidence: str, action: str) -> dict[str, str]:
    return {
        "gate": name,
        "status": status,
        "evidence": evidence,
        "action": action,
    }


def _allowlist_action(allowlist: Mapping[str, Any], action_name: str) -> dict[str, Any] | None:
    for section in ("allowed_cpu_actions", "allowed_gpu_actions"):
        actions = allowlist.get(section, [])
        if not isinstance(actions, list):
            continue
        for action in actions:
            if isinstance(action, dict) and str(action.get("name", "")) == action_name:
                return dict(action)
    return None


def _repo_relative_exists(path_text: str, root: Path) -> bool:
    return bool(path_text) and (root / path_text).exists()


def _contains_all(text: str, patterns: list[str]) -> bool:
    return all(pattern in text for pattern in patterns)


def run_gate_review(
    *,
    config_path: Path,
    allowlist_path: Path,
    variable_arity_manifest_path: Path,
    full_density_summary_path: Path,
    pre_null_summary_path: Path,
    variable_radix_preflight_summary_path: Path,
    variable_radix_frame_policy_summary_path: Path | None,
    protocol_commitment_path: Path,
    output_dir: Path,
    repo_root: Path,
    min_compatible_entries: int = 2000,
    min_effective_bits_per_response: float = 0.8,
    min_heldout_density_per_100_tokens: float = 0.5,
    min_configured_subset: int = 500,
) -> dict[str, Any]:
    output_paths = [
        output_dir / "qwen_proof_of_life_gate_review.json",
        output_dir / "qwen_proof_of_life_gate_review.csv",
        output_dir / "qwen_proof_of_life_blockers.md",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite proof-of-life gate review outputs: " + ", ".join(existing))

    config = read_yaml(config_path)
    allowlist = read_yaml(allowlist_path)
    variable_arity = _read_json(variable_arity_manifest_path)
    density = _read_json(full_density_summary_path)
    pre_null = _read_json(pre_null_summary_path)
    variable_radix = _read_json(variable_radix_preflight_summary_path)
    frame_policy_summary = (
        _read_json(variable_radix_frame_policy_summary_path)
        if variable_radix_frame_policy_summary_path and variable_radix_frame_policy_summary_path.exists()
        else {}
    )
    protocol_text = protocol_commitment_path.read_text(encoding="utf-8") if protocol_commitment_path.exists() else ""

    gates: list[dict[str, str]] = []
    accepted_entries = int(variable_arity.get("accepted_entries", 0) or 0)
    configured_subset = int(variable_arity.get("configured_subset_entries", 0) or 0)
    effective_bits = float(density.get("effective_bits_per_response", 0.0) or 0.0)
    heldout_density = float(density.get("eligible_positions_per_100_tokens", 0.0) or 0.0)
    density_gates = dict(density.get("gate_status", {}))

    gates.append(
        _gate(
            "variable_arity_compatible_entries",
            _bool_status(accepted_entries >= int(min_compatible_entries)),
            f"accepted_entries={accepted_entries}, threshold={min_compatible_entries}",
            "required before proof-of-life training",
        )
    )
    gates.append(
        _gate(
            "effective_bits_per_response",
            _bool_status(effective_bits >= float(min_effective_bits_per_response)),
            f"effective_bits_per_response={effective_bits}, threshold={min_effective_bits_per_response}",
            "required before proof-of-life training",
        )
    )
    gates.append(
        _gate(
            "heldout_density",
            _bool_status(
                heldout_density >= float(min_heldout_density_per_100_tokens)
                and density_gates.get("heldout_viability_density") == "PASS"
            ),
            f"eligible_positions_per_100_tokens={heldout_density}, heldout_viability_density={density_gates.get('heldout_viability_density')}",
            "required before proof-of-life training",
        )
    )
    gates.append(
        _gate(
            "high_quality_configured_subset",
            _bool_status(configured_subset >= int(min_configured_subset)),
            f"configured_subset_entries={configured_subset}, threshold={min_configured_subset}",
            "required before proof-of-life training",
        )
    )
    gates.append(
        _gate(
            "full_budget_pre_null",
            _bool_status(
                pre_null.get("pre_null_status") == "PASS_NO_BLOCKING_ACCEPTS_DIAGNOSTIC"
                and int(pre_null.get("blocking_accept_count", -1)) == 0
                and [int(value) for value in pre_null.get("query_budgets", [])] == [64, 128, 256, 512]
            ),
            f"pre_null_status={pre_null.get('pre_null_status')}, blocking_accept_count={pre_null.get('blocking_accept_count')}, query_budgets={pre_null.get('query_budgets')}",
            "diagnostic only; not full FAR",
        )
    )
    gates.append(
        _gate(
            "variable_radix_train_eval_verifier_preflight",
            _bool_status(variable_radix.get("overall_status") == "PASS_PREFLIGHT_NOT_TRAINING"),
            f"overall_status={variable_radix.get('overall_status')}, blocking_null_accept_count={variable_radix.get('blocking_null_accept_count')}",
            "dry-run contract only; production integration still requires review",
        )
    )
    frame_quality = dict(frame_policy_summary.get("quality_gates", {}))
    decode_policy = dict(frame_policy_summary.get("decode_policy", {}))
    gates.append(
        _gate(
            "variable_radix_frame_policy_dry_run",
            _bool_status(
                frame_policy_summary.get("status") == "PASS_DRY_RUN_NOT_TRAINING"
                and frame_quality.get("frame_repetition_positions") == "PASS"
                and frame_quality.get("trainer_dry_run_review") == "PASS"
                and frame_quality.get("synthetic_query_budget_decode") == "PASS"
            ),
            (
                f"status={frame_policy_summary.get('status', 'MISSING')}, "
                f"frame_repetition_positions={frame_quality.get('frame_repetition_positions')}, "
                f"trainer_dry_run_review={frame_quality.get('trainer_dry_run_review')}, "
                f"decode_policy={decode_policy.get('status')}"
            ),
            "required variable-radix repetition/decode policy dry-run before launch",
        )
    )
    gates.append(
        _gate(
            "protocol_commitment",
            _bool_status(
                protocol_commitment_path.exists()
                and _contains_all(
                    protocol_text,
                    ["Post-hoc key search is disallowed", "wrong-key", "wrong-payload", "transcript_commitment"],
                )
            ),
            f"path={protocol_commitment_path}, exists={protocol_commitment_path.exists()}",
            "must remain fixed before new transcripts",
        )
    )

    config_text = json.dumps(config, sort_keys=True)
    gates.append(
        _gate(
            "task_only_lora_null_plan",
            _bool_status("task_only_lora" in config_text),
            "task_only_lora present in config" if "task_only_lora" in config_text else "task_only_lora missing in config",
            "required null arm",
        )
    )
    gates.append(
        _gate(
            "wrong_key_wrong_payload_null_plan",
            _bool_status("wrong_key" in config_text and "wrong_payload" in config_text),
            "wrong_key and wrong_payload present in config" if "wrong_key" in config_text and "wrong_payload" in config_text else "wrong-key/wrong-payload plan missing",
            "required null arms",
        )
    )

    organic_status = str(density_gates.get("organic_density", "NEEDS_RESULTS"))
    gates.append(
        _gate(
            "organic_generated_output_density",
            "PASS" if organic_status == "PASS" else "NEEDS_RESULTS",
            f"organic_density={organic_status}",
            "generate/audit organic outputs before paper-ready proof-of-life claims",
        )
    )

    natural_e2e = _allowlist_action(allowlist, "qwen_natural_e2e_pilot")
    command_pattern = str(natural_e2e.get("command_pattern", "")) if natural_e2e else ""
    script_path = command_pattern.removeprefix("sbatch ").strip()
    gates.append(
        _gate(
            "qwen_natural_e2e_allowlist_command",
            _bool_status(natural_e2e is not None and _repo_relative_exists(script_path, repo_root)),
            f"action_present={natural_e2e is not None}, enabled={natural_e2e.get('enabled') if natural_e2e else ''}, script_path={script_path}, script_exists={_repo_relative_exists(script_path, repo_root)}",
            "must add/review variable-radix proof-of-life Slurm wrapper before launch",
        )
    )
    gates.append(
        _gate(
            "training_allowlist_disabled_until_approval",
            _bool_status(natural_e2e is not None and not bool(natural_e2e.get("enabled", False))),
            f"qwen_natural_e2e_pilot enabled={natural_e2e.get('enabled') if natural_e2e else ''}",
            "safety gate should remain disabled until explicit approval",
        )
    )

    trainer_text = (repo_root / "scripts/natural_evidence_v1/train_natural_bucket_lora.py").read_text(encoding="utf-8")
    evaluator_text = (repo_root / "scripts/natural_evidence_v1/evaluate_diagnostic_e2e.py").read_text(encoding="utf-8")
    compile_text = (repo_root / "scripts/natural_evidence_v1/compile_train_dataset.py").read_text(encoding="utf-8")
    variable_radix_integrated = (
        "decode_bytes_variable_radices" in evaluator_text
        and "compatible_bucket_ids" in compile_text
        and "variable_radix" in trainer_text
    )
    gates.append(
        _gate(
            "variable_radix_production_integration",
            _bool_status(variable_radix_integrated),
            "fixed-radix paths still dominate compile/eval/trainer" if not variable_radix_integrated else "variable-radix production paths detected",
            "implement/review production train/eval integration before launch",
        )
    )

    fail_count = sum(1 for gate in gates if gate["status"] == "FAIL")
    needs_count = sum(1 for gate in gates if gate["status"] == "NEEDS_RESULTS")
    ready = fail_count == 0 and needs_count == 0
    status = "READY_FOR_EXPLICIT_LAUNCH_REVIEW" if ready else "BLOCKED_NOT_READY_FOR_TRAINING"
    blockers = [gate for gate in gates if gate["status"] != "PASS"]
    summary = {
        "schema_name": SUMMARY_SCHEMA,
        "status": status,
        "ready_for_training_submission": False,
        "explicit_launch_approval_present": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "gate_count": len(gates),
        "pass_count": sum(1 for gate in gates if gate["status"] == "PASS"),
        "fail_count": fail_count,
        "needs_results_count": needs_count,
        "blocker_gates": [gate["gate"] for gate in blockers],
        "gates": gates,
        "next_allowed_action": (
            "Resolve blocker gates only. Do not submit Qwen proof-of-life training "
            "without explicit approval and a reviewed allowlisted variable-radix Slurm wrapper."
        ),
        "result_claim": "qwen_proof_of_life_gate_review_not_training_not_recovery",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "qwen_proof_of_life_gate_review.json", summary)
    write_csv(output_dir / "qwen_proof_of_life_gate_review.csv", gates, ["gate", "status", "evidence", "action"])
    blocker_lines = [
        "# Qwen proof-of-life gate blockers",
        "",
        f"Status: `{status}`",
        "",
    ]
    for gate in blockers:
        blocker_lines.append(f"- `{gate['gate']}`: {gate['status']} - {gate['evidence']}")
    blocker_lines.append("")
    blocker_lines.append("No training or E2E launch is authorized by this review.")
    (output_dir / "qwen_proof_of_life_blockers.md").write_text("\n".join(blocker_lines), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    run_gate_review(
        config_path=resolve_repo_path(args.config, root),
        allowlist_path=resolve_repo_path(args.allowlist, root),
        variable_arity_manifest_path=resolve_repo_path(args.variable_arity_manifest, root),
        full_density_summary_path=resolve_repo_path(args.full_density_summary, root),
        pre_null_summary_path=resolve_repo_path(args.pre_null_summary, root),
        variable_radix_preflight_summary_path=resolve_repo_path(args.variable_radix_preflight_summary, root),
        variable_radix_frame_policy_summary_path=(
            resolve_repo_path(args.variable_radix_frame_policy_summary, root)
            if args.variable_radix_frame_policy_summary
            else None
        ),
        protocol_commitment_path=resolve_repo_path(args.protocol_commitment, root),
        output_dir=resolve_repo_path(args.output_dir, root),
        repo_root=root,
        min_compatible_entries=int(args.min_compatible_entries),
        min_effective_bits_per_response=float(args.min_effective_bits_per_response),
        min_heldout_density_per_100_tokens=float(args.min_heldout_density_per_100_tokens),
        min_configured_subset=int(args.min_configured_subset),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

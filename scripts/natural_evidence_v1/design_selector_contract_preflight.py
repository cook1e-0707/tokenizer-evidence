from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.natural_evidence_v1.common import write_csv, write_json


SCHEMA_NAME = "natural_evidence_v1_selector_contract_training_target_preflight_v1"
CONTRACT_SCHEMA_NAME = "natural_evidence_v1_selector_precommit_contract_draft_v1"
FIELD_ROWS = [
    ("protocol_id", "required", "Fixed protocol identifier before generation."),
    ("selector_id", "required", "Fixed prefix-conditioned selector id before generation."),
    ("match_policy", "required", "Fixed event-match policy; no post-hoc policy choice."),
    ("reference_model", "required", "Reference model used for candidate scoring."),
    ("reference_tokenizer", "required", "Tokenizer used for prefix/event reconstruction."),
    ("bucket_policy_id", "required", "Bucket construction and candidate filtering policy."),
    ("audit_key_id", "required", "Key fixed before transcript generation."),
    ("payload_id", "required", "Payload fixed before transcript generation."),
    ("query_budget", "required", "Budget fixed before transcript generation."),
    ("prompt_split", "required", "Prompt split and sampling rule fixed before generation."),
    ("thresholds", "required", "Verifier thresholds fixed before seeing transcript."),
    ("decode_rule", "required", "Coordinate collection and decode rule fixed in advance."),
    ("allowed_trials", "required", "Allowed number of keys/payloads/policies/thresholds fixed."),
    ("multiple_testing_rule", "required", "Correction or lockbox rule fixed before evaluation."),
]
PLAN_FIELDS = [
    "gate",
    "status",
    "evidence",
    "required_artifact",
    "minimum_metric",
    "action",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only selector precommit contract and branch-aware/"
            "regenerated-suffix training-target preflight. This designs the next "
            "contract and gates from R1 analysis outputs; it does not train, "
            "generate, run E2E, or claim recovery/FAR."
        )
    )
    parser.add_argument("--r1-analysis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selector-id", default="prefix_conditioned_observed_text_v0")
    parser.add_argument("--protocol-id", default="natural_evidence_v1")
    parser.add_argument("--audit-key-id", default="K001")
    parser.add_argument("--reference-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--reference-tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--bucket-policy-id", default="expanded_actual_prefix_topk64_b4_minarity2_v1")
    parser.add_argument("--payload-ids", default="P0421,P1729")
    parser.add_argument("--query-budgets", default="64,128,256,512")
    parser.add_argument("--candidate-match-policies", default="exact_full,suffix_32,suffix_16,suffix_8")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _field_rows() -> list[dict[str, str]]:
    return [
        {
            "field": field,
            "requirement": requirement,
            "rationale": rationale,
        }
        for field, requirement, rationale in FIELD_ROWS
    ]


def _selector_blocked(r1_summary: Mapping[str, Any]) -> bool:
    decision = r1_summary.get("selector_contract_decision", {})
    if isinstance(decision, dict) and decision.get("training_allowed") is False:
        return True
    return int(r1_summary.get("positive_vs_raw_rows_total", 0) or 0) == 0 or int(
        r1_summary.get("positive_vs_task_only_rows_total", 0) or 0
    ) == 0


def _preflight_plan(r1_summary: Mapping[str, Any]) -> list[dict[str, str]]:
    no_raw_lift = int(r1_summary.get("positive_vs_raw_rows_total", 0) or 0)
    no_task_lift = int(r1_summary.get("positive_vs_task_only_rows_total", 0) or 0)
    return [
        {
            "gate": "selector_contract_fields",
            "status": "PASS_DESIGNED_NOT_ACTIVE",
            "evidence": "Required precommit fields enumerated in selector_contract_precommit_fields.csv.",
            "required_artifact": "selector_precommit_contract_draft.json",
            "minimum_metric": "all required fields present before any fresh transcript",
            "action": "Use this only as a draft; do not activate until branch-aware/training-target preflights pass.",
        },
        {
            "gate": "r1_protected_lift_over_raw",
            "status": "FAIL_BLOCKER",
            "evidence": f"positive protected-minus-raw slices={no_raw_lift}/64.",
            "required_artifact": "r1_selector_contract_pairwise_lift.csv",
            "minimum_metric": "positive protected lift over raw in most payload/seed/policy/budget slices",
            "action": "Do not use direct replay verifier or train from current target data.",
        },
        {
            "gate": "r1_protected_lift_over_task_only",
            "status": "FAIL_BLOCKER",
            "evidence": f"positive protected-minus-task-only slices={no_task_lift}/64.",
            "required_artifact": "r1_selector_contract_pairwise_lift.csv",
            "minimum_metric": "positive protected lift over task-only in most slices",
            "action": "Repair training target before any new Qwen run.",
        },
        {
            "gate": "branch_aware_compatibility",
            "status": "NEEDS_RESULTS",
            "evidence": "No branch-aware short-continuation scoring artifact exists for the locked selector draft.",
            "required_artifact": "branch_aware_compatibility_summary.json",
            "minimum_metric": "branch-aware pass rate reported by token class/radix with disagreement table",
            "action": "Prepare Slurm-scored diagnostic only after contract draft review; no training.",
        },
        {
            "gate": "regenerated_suffix_repair",
            "status": "NEEDS_RESULTS",
            "evidence": "No local-suffix repaired training manifest exists.",
            "required_artifact": "regenerated_suffix_repair_manifest.json",
            "minimum_metric": "repaired rows by payload/seed/token-class plus invalid-suffix reason table",
            "action": "Construct artifact-only examples or dry-run manifest before model training.",
        },
        {
            "gate": "teacher_forced_repaired_target_mass",
            "status": "NEEDS_RESULTS",
            "evidence": "Existing protected target-mass lift is too small; repaired objective not probed.",
            "required_artifact": "teacher_forced_repaired_target_mass_probe_summary.json",
            "minimum_metric": "protected-base >= +0.05 and protected-task-only >= +0.05 target candidate mass lift",
            "action": "Run only after repaired data/objective preflight exists.",
        },
        {
            "gate": "sparse_coordinate_code",
            "status": "SYNTHETIC_PREFLIGHT_NEEDED",
            "evidence": "R1 coordinates are sparse and null-heavy; complete-frame recovery remains unsuitable.",
            "required_artifact": "sparse_coordinate_code_synthetic_preflight.json",
            "minimum_metric": "decode from known sparse coordinates with locked null accounting",
            "action": "Keep as decoder-side preflight after coordinate survival shows owner-specific lift.",
        },
        {
            "gate": "fresh_lockbox_or_locked_replay",
            "status": "NEEDS_RESULTS",
            "evidence": "R1 was used to diagnose policy behavior; it cannot also select a success policy.",
            "required_artifact": "locked_selector_replay_or_lockbox_summary.json",
            "minimum_metric": "precommitted policy evaluated without post-hoc policy/key/threshold choice",
            "action": "Use after branch-aware/regenerated-suffix repair passes artifact-only gates.",
        },
    ]


def _contract_draft(
    *,
    args: argparse.Namespace,
    r1_summary: Mapping[str, Any],
    payload_ids: Sequence[str],
    query_budgets: Sequence[int],
    candidate_match_policies: Sequence[str],
) -> dict[str, Any]:
    blocked = _selector_blocked(r1_summary)
    return {
        "schema_name": CONTRACT_SCHEMA_NAME,
        "contract_status": "DRAFT_NOT_ACTIVE_BLOCKED_BY_R1_NO_PROTECTED_LIFT" if blocked else "DRAFT_NOT_ACTIVE",
        "protocol_id": args.protocol_id,
        "selector": {
            "selector_id": args.selector_id,
            "selector_mode": "prefix_conditioned_observed_text_event_scan",
            "event_order": "left_to_right_observed_transcript",
            "match_policy": "UNSELECTED_BLOCKED_BY_R1" if blocked else "MUST_BE_FIXED_BEFORE_GENERATION",
            "candidate_match_policies_for_preflight_only": list(candidate_match_policies),
            "post_hoc_policy_selection_allowed": False,
            "direct_replay_verifier_allowed": False,
        },
        "reference": {
            "model": args.reference_model,
            "tokenizer": args.reference_tokenizer,
        },
        "bucket_policy": {
            "bucket_policy_id": args.bucket_policy_id,
            "bucket_construction": "keyed_bucketization_over_reference_topk_candidates",
            "candidate_filtering": "natural_surface_and_compatibility_filtered",
            "bucket_bank_role": "natural_next_token_measurable_opportunity_catalog",
        },
        "commitment": {
            "audit_key_id": args.audit_key_id,
            "payload_ids_for_preflight": list(payload_ids),
            "query_budgets_for_preflight": [int(value) for value in query_budgets],
            "prompt_split": "MUST_BE_FIXED_BEFORE_GENERATION",
            "thresholds": "MUST_BE_FIXED_BEFORE_GENERATION",
            "decode_rule": "known_coordinate_collection_then_locked_decoder",
            "allowed_trials": {
                "keys": "MUST_BE_FIXED",
                "payloads": "MUST_BE_FIXED",
                "policies": "MUST_BE_FIXED",
                "thresholds": "MUST_BE_FIXED",
            },
            "multiple_testing_rule": "lockbox_or_familywise_correction_required",
        },
        "r1_blocker": {
            "r1_status": r1_summary.get("status", ""),
            "positive_vs_raw_rows_total": int(r1_summary.get("positive_vs_raw_rows_total", 0) or 0),
            "positive_vs_task_only_rows_total": int(r1_summary.get("positive_vs_task_only_rows_total", 0) or 0),
            "comparison_rows": int(r1_summary.get("comparison_rows", 0) or 0),
            "training_allowed": False,
            "e2e_rerun_allowed": False,
            "paper_claim_allowed": False,
        },
    }


def _write_markdown(
    *,
    path: Path,
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    plan_rows: Sequence[Mapping[str, str]],
) -> None:
    lines = [
        "# Selector precommit and training-target preflight",
        "",
        "This is an artifact-only design/preflight output. It does not train, generate, rerun E2E, claim payload recovery, or estimate FAR.",
        "",
        "## Status",
        "",
        f"`{summary['status']}`",
        "",
        "Direct replay verifier use remains blocked. R1 had no protected lift over raw or task-only in any of the 64 comparison slices.",
        "",
        "## Draft Contract",
        "",
        f"- selector_id: `{contract['selector']['selector_id']}`",
        f"- selector_mode: `{contract['selector']['selector_mode']}`",
        f"- match_policy: `{contract['selector']['match_policy']}`",
        f"- reference_model: `{contract['reference']['model']}`",
        f"- reference_tokenizer: `{contract['reference']['tokenizer']}`",
        f"- bucket_policy_id: `{contract['bucket_policy']['bucket_policy_id']}`",
        f"- direct_replay_verifier_allowed: `{contract['selector']['direct_replay_verifier_allowed']}`",
        "",
        "Any active version of this contract must fix policy, key, payload, query budget, prompt split, thresholds, and decode rule before generation.",
        "",
        "## Gate Plan",
        "",
        "| Gate | Status | Required artifact | Action |",
        "|---|---|---|---|",
    ]
    for row in plan_rows:
        lines.append(
            f"| {row['gate']} | {row['status']} | {row['required_artifact']} | {row['action']} |"
        )
    lines.extend(
        [
            "",
            "## Next Allowed Action",
            "",
            "Prepare artifact-only branch-aware compatibility and regenerated/local-suffix repair diagnostics under the draft contract. Any CPU/GPU work on Chimera must be submitted through Slurm. Training remains forbidden.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_preflight(
    *,
    args: argparse.Namespace,
    r1_analysis_dir: Path,
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "selector_contract_training_target_preflight_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing preflight summary: {summary_path}")
    r1_summary_path = r1_analysis_dir / "r1_selector_contract_repair_summary.json"
    if not r1_summary_path.is_file() or r1_summary_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing R1 selector-contract summary: {r1_summary_path}")
    r1_summary = _read_json(r1_summary_path)
    payload_ids = _parse_csv_list(args.payload_ids)
    query_budgets = _parse_int_list(args.query_budgets)
    candidate_match_policies = _parse_csv_list(args.candidate_match_policies)
    contract = _contract_draft(
        args=args,
        r1_summary=r1_summary,
        payload_ids=payload_ids,
        query_budgets=query_budgets,
        candidate_match_policies=candidate_match_policies,
    )
    plan_rows = _preflight_plan(r1_summary)
    failed_gates = [row["gate"] for row in plan_rows if row["status"].startswith("FAIL")]
    needs_results = [
        row["gate"]
        for row in plan_rows
        if row["status"] in {"NEEDS_RESULTS", "SYNTHETIC_PREFLIGHT_NEEDED"}
    ]
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_SELECTOR_PRECOMMIT_DRAFT_BRANCH_AWARE_PREFLIGHT_PLAN_READY",
        "claim_control": {
            "paper_claim_allowed": False,
            "training_started": False,
            "generation_started": False,
            "e2e_eval_started": False,
            "not_payload_recovery": True,
            "not_full_far": True,
            "result_claim": "selector_contract_preflight_not_training_not_payload_recovery_not_far",
        },
        "inputs": {
            "r1_selector_contract_summary": str(r1_summary_path),
            "r1_status": r1_summary.get("status", ""),
            "positive_vs_raw_rows_total": int(r1_summary.get("positive_vs_raw_rows_total", 0) or 0),
            "positive_vs_task_only_rows_total": int(r1_summary.get("positive_vs_task_only_rows_total", 0) or 0),
            "comparison_rows": int(r1_summary.get("comparison_rows", 0) or 0),
        },
        "selector_contract_status": contract["contract_status"],
        "failed_gates": failed_gates,
        "needs_results": needs_results,
        "training_allowed": False,
        "generation_allowed": False,
        "e2e_rerun_allowed": False,
        "next_allowed_action": (
            "Prepare artifact-only branch-aware compatibility and regenerated/local-suffix "
            "repair diagnostics under the draft selector contract; use Slurm for any Chimera CPU/GPU work."
        ),
        "forbidden_claims_remain": [
            "natural-output success",
            "payload recovery",
            "full FAR",
            "cross-family generality",
            "robustness",
            "sanitizer resistance",
            "superiority over Scalable/Perinucleus",
            "24,576 fingerprints",
        ],
    }
    write_json(output_dir / "selector_precommit_contract_draft.json", contract)
    write_csv(
        output_dir / "selector_contract_precommit_fields.csv",
        _field_rows(),
        ["field", "requirement", "rationale"],
    )
    write_csv(output_dir / "branch_aware_training_target_preflight_plan.csv", plan_rows, PLAN_FIELDS)
    write_json(summary_path, summary)
    _write_markdown(
        path=output_dir / "selector_contract_training_target_preflight.md",
        summary=summary,
        contract=contract,
        plan_rows=plan_rows,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_preflight(
        args=args,
        r1_analysis_dir=_resolve(args.r1_analysis_dir),
        output_dir=_resolve(args.output_dir),
        force=bool(args.force),
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(_resolve(args.output_dir))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

from scripts.natural_evidence_v2.build_wp3_detector_bank_scaffold import (  # noqa: E402
    BUCKET_POLICY_ID,
    DETECTOR_ID,
    FORBIDDEN_ACTIONS,
    PROMPT_SET_ID,
    SLOT_POLICY_ID,
    detect_candidates,
    read_jsonl,
    response_id_from_row,
)
from src.core.bucket_mapping import BucketLayout, FieldBucketSpec  # noqa: E402
from src.core.tokenizer_utils import audit_carriers, load_tokenizer  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml"
DEFAULT_SCAFFOLD_DIR = (
    ROOT / "results/natural_evidence_v2/status/wp3_detector_bank_scaffold_20260508_2153"
)
DEFAULT_CONTRACT = DEFAULT_SCAFFOLD_DIR / "wp3_detector_contract.json"
DEFAULT_BUCKET_BANK = DEFAULT_SCAFFOLD_DIR / "two_way_bucket_bank_scaffold.json"
DEFAULT_SCAFFOLD_SUMMARY = DEFAULT_SCAFFOLD_DIR / "wp3_detector_bank_scaffold_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit fixed natural_evidence_v2 WP3 artifacts for tokenizer stability, "
            "density accounting, and fixed two-way mass artifacts. This does not "
            "generate transcripts, train, run E2E, aggregate FAR, or make claims."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--detector-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--bucket-bank", type=Path, default=DEFAULT_BUCKET_BANK)
    parser.add_argument("--scaffold-summary", type=Path, default=DEFAULT_SCAFFOLD_SUMMARY)
    parser.add_argument("--responses-jsonl", type=Path)
    parser.add_argument("--mass-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-backend")
    parser.add_argument("--tokenizer-name")
    parser.add_argument("--max-response-rows", type=int, default=0)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top level must be a mapping: {path}")
    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML top level must be a mapping: {path}")
    return payload


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_new(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def build_bucket_layout(bucket_bank: Mapping[str, Any]) -> BucketLayout:
    fields: list[FieldBucketSpec] = []
    raw_banks = bucket_bank.get("candidate_banks", [])
    if not isinstance(raw_banks, Sequence):
        raise ValueError("bucket bank scaffold candidate_banks must be a sequence")
    for bank in raw_banks:
        if not isinstance(bank, Mapping):
            raise ValueError("candidate bank must be a mapping")
        buckets = bank.get("buckets", {})
        if not isinstance(buckets, Mapping):
            raise ValueError(f"{bank.get('candidate_bank_id')}: buckets must be a mapping")
        fields.append(
            FieldBucketSpec(
                field_name=str(bank["candidate_bank_id"]),
                field_type=str(bank.get("slot_type", "micro_slot")),
                buckets={
                    int(bucket_id): tuple(str(member) for member in members)
                    for bucket_id, members in buckets.items()
                },
                tags=(str(bank.get("anchor_kind", "")),),
            )
        )
    return BucketLayout(
        catalog_name=str(bucket_bank.get("bucket_policy_id", BUCKET_POLICY_ID)),
        notes="natural_evidence_v2 WP3 two-way bucket-bank scaffold",
        tags=("natural_evidence_v2", "wp3", "two_way_bucket_bank"),
        provenance={
            "schema_name": str(bucket_bank.get("schema_name", "")),
            "bucket_policy_id": str(bucket_bank.get("bucket_policy_id", "")),
        },
        fields=tuple(fields),
    )


def configured_tokenizer(config: Mapping[str, Any]) -> tuple[str, str]:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    return str(model.get("tokenizer_backend", "huggingface")), str(model.get("tokenizer", ""))


def tokenizer_audit_payload(
    *,
    layout: BucketLayout,
    config: Mapping[str, Any],
    tokenizer_backend: str | None,
    tokenizer_name: str | None,
) -> dict[str, Any]:
    configured_backend, configured_name = configured_tokenizer(config)
    backend = tokenizer_backend or configured_backend
    name = tokenizer_name if tokenizer_name is not None else configured_name
    configured_tokenizer_used = backend == configured_backend and name == configured_name
    try:
        tokenizer = load_tokenizer(backend, name)
    except Exception as error:  # pragma: no cover - exercised by local dependency state.
        return {
            "schema_name": "natural_evidence_v2_wp3_tokenizer_audit_v1",
            "status": "BLOCKED_TOKENIZER_BACKEND_UNAVAILABLE",
            "tokenizer_backend": backend,
            "tokenizer_name": name,
            "configured_tokenizer_backend": configured_backend,
            "configured_tokenizer_name": configured_name,
            "configured_tokenizer_used": configured_tokenizer_used,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "tokenizer_stability_status": "NOT_EVALUATED",
            "unstable_token_rate": None,
            "artifact_only": True,
            "model_calls_started": False,
            "training_started": False,
            "e2e_eval_started": False,
            "paper_claim_allowed": False,
        }

    result = audit_carriers(list(layout.all_carriers()), tokenizer=tokenizer, bucket_layout=layout)
    unstable_reasons = {
        "tokenizer_returned_no_tokens",
        "multi_token",
        "detokenization_mismatch",
        "duplicate_normalized_form",
        "token_collision",
    }
    unstable_count = sum(
        1 for item in result.diagnostics if any(reason in unstable_reasons for reason in item.reasons)
    )
    unstable_rate = 0.0 if result.num_total == 0 else unstable_count / result.num_total
    if not configured_tokenizer_used:
        status = "PASS_MOCK_OR_OVERRIDE_DRY_RUN_NOT_CONFIGURED_TOKENIZER"
        stability_status = "NOT_GATE_RESULT"
    elif result.is_alignment_safe:
        status = "PASS_CONFIGURED_TOKENIZER_ALIGNMENT_SAFE"
        stability_status = "PASS"
    else:
        status = "FAIL_CONFIGURED_TOKENIZER_ALIGNMENT_UNSAFE"
        stability_status = "FAIL"
    return {
        "schema_name": "natural_evidence_v2_wp3_tokenizer_audit_v1",
        "status": status,
        "tokenizer_backend": backend,
        "tokenizer_name": name,
        "configured_tokenizer_backend": configured_backend,
        "configured_tokenizer_name": configured_name,
        "configured_tokenizer_used": configured_tokenizer_used,
        "tokenizer_stability_status": stability_status,
        "unstable_token_count": unstable_count,
        "unstable_token_rate": unstable_rate,
        "audit_result": result.to_dict(),
        "artifact_only": True,
        "model_calls_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def median(values: Sequence[int]) -> float:
    return 0.0 if not values else float(statistics.median(values))


def density_payload(
    *,
    config: Mapping[str, Any],
    detector_contract: Mapping[str, Any],
    bucket_bank: Mapping[str, Any],
    responses_jsonl: Path | None,
    max_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if responses_jsonl is None:
        return (
            {
                "schema_name": "natural_evidence_v2_wp3_density_audit_v1",
                "status": "NEEDS_FIXED_RESPONSE_ARTIFACT",
                "density_gate_status": "NOT_EVALUATED",
                "responses_jsonl": None,
                "total_responses": 0,
                "candidate_micro_slot_rows": 0,
                "artifact_only": True,
                "model_calls_started": False,
                "training_started": False,
                "e2e_eval_started": False,
                "paper_claim_allowed": False,
            },
            [],
        )

    forbidden_terms = [str(item) for item in config.get("forbidden_surface_terms", [])]
    response_rows = read_jsonl(responses_jsonl, max_rows=max(0, int(max_rows)))
    response_sources = Counter(
        str(row.get("response_source", row.get("artifact_role", "unspecified")))
        for row in response_rows
    )
    template_preflight_only = bool(response_rows) and all(
        str(row.get("response_source", row.get("artifact_role", ""))).startswith(
            "template_density_preflight"
        )
        or str(row.get("artifact_role", "")).startswith("template_density_preflight")
        for row in response_rows
    )
    candidate_rows: list[dict[str, Any]] = []
    for index, row in enumerate(response_rows):
        candidate_rows.extend(
            detect_candidates(
                row=row,
                index=index,
                banks=bucket_bank["candidate_banks"],
                forbidden_terms=forbidden_terms,
                protocol_id=str(detector_contract.get("protocol_id", "")),
            )
        )

    counts_by_response = Counter(str(row["response_id"]) for row in candidate_rows)
    response_slot_counts = [
        counts_by_response.get(response_id_from_row(row, index), 0)
        for index, row in enumerate(response_rows)
    ]
    rejected = Counter(str(row["rejection_reason"]) for row in candidate_rows if row["rejection_reason"])
    slot_counts_by_family = Counter(str(row.get("family_id", "")) for row in candidate_rows)
    slot_counts_by_type = Counter(str(row.get("slot_type", "")) for row in candidate_rows)
    density_gate = config.get("density_gate", {})
    if not isinstance(density_gate, Mapping):
        density_gate = {}
    coverage = 0.0 if not response_rows else len(counts_by_response) / len(response_rows)
    average = 0.0 if not response_rows else len(candidate_rows) / len(response_rows)
    forbidden_surface_rate = 0.0 if not candidate_rows else rejected.get("public_forbidden_surface_term", 0) / len(candidate_rows)
    passes_density = (
        average >= float(density_gate.get("average_micro_slots_per_response_min", 16))
        and coverage >= float(density_gate.get("prompt_coverage_min", 0.80))
        and forbidden_surface_rate <= float(density_gate.get("forbidden_surface_rate_max", 0.0))
    )
    if template_preflight_only:
        status = (
            "PASS_TEMPLATE_FIXED_RESPONSE_DENSITY_PREFLIGHT_NOT_MODEL_GATE"
            if passes_density
            else "FAIL_TEMPLATE_FIXED_RESPONSE_DENSITY_PREFLIGHT"
        )
        density_gate_status = "TEMPLATE_PREFLIGHT_PASS" if passes_density else "TEMPLATE_PREFLIGHT_FAIL"
    else:
        status = "PASS_FIXED_RESPONSE_DENSITY_GATE" if passes_density else "FAIL_FIXED_RESPONSE_DENSITY_GATE"
        density_gate_status = "PASS" if passes_density else "FAIL"
    return (
        {
            "schema_name": "natural_evidence_v2_wp3_density_audit_v1",
            "status": status,
            "density_gate_status": density_gate_status,
            "responses_jsonl": str(responses_jsonl),
            "response_source_counts": dict(sorted(response_sources.items())),
            "template_preflight_only": template_preflight_only,
            "total_responses": len(response_rows),
            "responses_with_any_slot": len(counts_by_response),
            "prompt_coverage": coverage,
            "average_micro_slots_per_response": average,
            "median_micro_slots_per_response": median(response_slot_counts),
            "candidate_micro_slot_rows": len(candidate_rows),
            "slot_counts_by_family": dict(sorted(slot_counts_by_family.items())),
            "slot_counts_by_type": dict(sorted(slot_counts_by_type.items())),
            "rejected_candidate_counts_by_reason": dict(sorted(rejected.items())),
            "forbidden_surface_rate": forbidden_surface_rate,
            "artifact_only": True,
            "model_calls_started": False,
            "training_started": False,
            "e2e_eval_started": False,
            "paper_claim_allowed": False,
        },
        candidate_rows,
    )


def load_mass_payload(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("mass_rows"), list):
        return [dict(item) for item in payload["mass_rows"]]
    raise ValueError(f"Unsupported mass artifact format: {path}")


def fixed_mass_by_bank(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    by_bank: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        bank_id = str(row.get("candidate_bank_id", ""))
        if not bank_id:
            raise ValueError("mass rows must include candidate_bank_id")
        if isinstance(row.get("bucket_masses"), Mapping):
            for bucket_id, mass in row["bucket_masses"].items():
                by_bank[bank_id][str(bucket_id)] = float(mass)
        else:
            by_bank[bank_id][str(row["bucket_id"])] = float(row["mass"])
    return dict(by_bank)


def structural_mass_proxy(bucket_bank: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for bank in bucket_bank.get("candidate_banks", []):
        buckets = bank["buckets"]
        total = sum(len(members) for members in buckets.values())
        bucket_masses = {
            str(bucket_id): (0.0 if total == 0 else len(members) / total)
            for bucket_id, members in buckets.items()
        }
        positive_masses = [mass for mass in bucket_masses.values() if mass > 0]
        ratio = None if not positive_masses else max(positive_masses) / min(positive_masses)
        rows.append(
            {
                "candidate_bank_id": str(bank["candidate_bank_id"]),
                "bucket_count": len(buckets),
                "member_counts": {str(bucket_id): len(members) for bucket_id, members in buckets.items()},
                "uniform_catalog_bucket_mass_proxy": bucket_masses,
                "uniform_catalog_mass_ratio_proxy": ratio,
            }
        )
    ratios = [float(row["uniform_catalog_mass_ratio_proxy"]) for row in rows if row["uniform_catalog_mass_ratio_proxy"] is not None]
    return {
        "schema_name": "natural_evidence_v2_wp3_structural_mass_proxy_v1",
        "status": "STRUCTURAL_PROXY_ONLY_NOT_MODEL_MASS_GATE",
        "bank_count": len(rows),
        "max_uniform_catalog_mass_ratio_proxy": None if not ratios else max(ratios),
        "rows": rows,
    }


def mass_payload(
    *,
    config: Mapping[str, Any],
    bucket_bank: Mapping[str, Any],
    mass_json: Path | None,
) -> dict[str, Any]:
    bucket_policy = config.get("bucket_policy", {})
    if not isinstance(bucket_policy, Mapping):
        bucket_policy = {}
    min_mass_required = float(bucket_policy.get("min_bucket_mass", 0.005))
    max_ratio_required = float(bucket_policy.get("max_mass_ratio", 5.0))
    proxy = structural_mass_proxy(bucket_bank)
    if mass_json is None:
        return {
            "schema_name": "natural_evidence_v2_wp3_mass_audit_v1",
            "status": "NEEDS_FIXED_MODEL_MASS_ARTIFACT",
            "mass_gate_status": "NOT_EVALUATED",
            "mass_json": None,
            "min_bucket_mass_required": min_mass_required,
            "max_mass_ratio_required": max_ratio_required,
            "structural_mass_proxy": proxy,
            "artifact_only": True,
            "model_calls_started": False,
            "training_started": False,
            "e2e_eval_started": False,
            "paper_claim_allowed": False,
        }

    rows = load_mass_payload(mass_json)
    by_bank = fixed_mass_by_bank(rows)
    bank_results: list[dict[str, Any]] = []
    all_pass = True
    for bank in bucket_bank.get("candidate_banks", []):
        bank_id = str(bank["candidate_bank_id"])
        masses = by_bank.get(bank_id, {})
        positive_masses = [mass for mass in masses.values() if mass > 0]
        min_mass = None if not positive_masses else min(positive_masses)
        ratio = None if not positive_masses else max(positive_masses) / min(positive_masses)
        passed = (
            set(masses) == {"0", "1"}
            and min_mass is not None
            and min_mass >= min_mass_required
            and ratio is not None
            and ratio <= max_ratio_required
        )
        all_pass = all_pass and passed
        bank_results.append(
            {
                "candidate_bank_id": bank_id,
                "bucket_masses": masses,
                "min_bucket_mass": min_mass,
                "max_bucket_mass_ratio": ratio,
                "passed": passed,
            }
        )
    return {
        "schema_name": "natural_evidence_v2_wp3_mass_audit_v1",
        "status": "PASS_FIXED_MODEL_MASS_GATE" if all_pass else "FAIL_FIXED_MODEL_MASS_GATE",
        "mass_gate_status": "PASS" if all_pass else "FAIL",
        "mass_json": str(mass_json),
        "min_bucket_mass_required": min_mass_required,
        "max_mass_ratio_required": max_ratio_required,
        "bank_results": bank_results,
        "structural_mass_proxy": proxy,
        "artifact_only": True,
        "model_calls_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def readme_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP3 Fixed-Artifact Audit",
            "",
            "Artifact-only audit implementation output for the recorded WP3 detector and bucket bank.",
            "",
            f"status: `{summary['status']}`",
            f"tokenizer_stability_status: `{summary['tokenizer_stability_status']}`",
            f"density_gate_status: `{summary['density_gate_status']}`",
            f"mass_gate_status: `{summary['mass_gate_status']}`",
            "",
            "This artifact does not unlock WP4, training, E2E, FAR aggregation, or positive claims.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")

    config = load_yaml(resolve_path(args.config))
    detector_contract = load_json(resolve_path(args.detector_contract))
    bucket_bank = load_json(resolve_path(args.bucket_bank))
    scaffold_summary = load_json(resolve_path(args.scaffold_summary))
    layout = build_bucket_layout(bucket_bank)

    responses_jsonl = args.responses_jsonl
    if responses_jsonl is None and scaffold_summary.get("responses_jsonl"):
        responses_jsonl = Path(str(scaffold_summary["responses_jsonl"]))
    if responses_jsonl is not None:
        responses_jsonl = resolve_path(responses_jsonl)

    mass_json = resolve_path(args.mass_json) if args.mass_json is not None else None
    tokenizer = tokenizer_audit_payload(
        layout=layout,
        config=config,
        tokenizer_backend=args.tokenizer_backend,
        tokenizer_name=args.tokenizer_name,
    )
    density, candidate_rows = density_payload(
        config=config,
        detector_contract=detector_contract,
        bucket_bank=bucket_bank,
        responses_jsonl=responses_jsonl,
        max_rows=args.max_response_rows,
    )
    mass = mass_payload(config=config, bucket_bank=bucket_bank, mass_json=mass_json)

    configured_tokenizer_used = bool(tokenizer.get("configured_tokenizer_used"))
    gate_ready = (
        configured_tokenizer_used
        and tokenizer.get("tokenizer_stability_status") == "PASS"
        and density.get("density_gate_status") == "PASS"
        and mass.get("mass_gate_status") == "PASS"
    )
    status = (
        "PASS_WP3_FIXED_ARTIFACT_AUDITS_WP4_REVIEW_ALLOWED"
        if gate_ready
        else "WP3_FIXED_ARTIFACT_AUDIT_IMPLEMENTED_NOT_GATE_PASS"
    )
    summary = {
        "schema_name": "natural_evidence_v2_wp3_fixed_artifact_audit_summary_v1",
        "status": status,
        "action_scope": "artifact_only_wp3_tokenizer_density_mass_audit_implementation",
        "protocol_id": str(detector_contract.get("protocol_id", "")),
        "prompt_set_id": str(detector_contract.get("prompt_set_id", PROMPT_SET_ID)),
        "detector_id": str(detector_contract.get("detector_id", DETECTOR_ID)),
        "slot_policy_id": str(detector_contract.get("slot_policy_id", SLOT_POLICY_ID)),
        "bucket_policy_id": str(bucket_bank.get("bucket_policy_id", BUCKET_POLICY_ID)),
        "scaffold_summary": str(resolve_path(args.scaffold_summary)),
        "detector_contract_json": str(resolve_path(args.detector_contract)),
        "bucket_bank_scaffold_json": str(resolve_path(args.bucket_bank)),
        "responses_jsonl": None if responses_jsonl is None else str(responses_jsonl),
        "mass_json": None if mass_json is None else str(mass_json),
        "output_dir": str(output_dir),
        "tokenizer_audit_json": str(output_dir / "tokenizer_audit.json"),
        "density_audit_json": str(output_dir / "density_audit.json"),
        "mass_audit_json": str(output_dir / "mass_audit.json"),
        "candidate_micro_slots_jsonl": None
        if not candidate_rows
        else str(output_dir / "candidate_micro_slots.jsonl"),
        "candidate_bank_count": int(bucket_bank.get("candidate_bank_count", len(bucket_bank.get("candidate_banks", [])))),
        "candidate_surface_count": int(bucket_bank.get("candidate_surface_count", len(layout.all_carriers()))),
        "candidate_micro_slot_rows": len(candidate_rows),
        "tokenizer_stability_status": str(tokenizer.get("tokenizer_stability_status")),
        "density_gate_status": str(density.get("density_gate_status")),
        "mass_gate_status": str(mass.get("mass_gate_status")),
        "configured_tokenizer_used": configured_tokenizer_used,
        "precommit_status": "not_committed",
        "wp4_allowed": gate_ready,
        "gates_unlocked": ["WP4_REVIEW"] if gate_ready else [],
        "artifact_only": True,
        "model_calls_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "forbidden_actions_confirmed": list(FORBIDDEN_ACTIONS),
        "next_allowed_action": (
            "WP3 configured-tokenizer fixed-artifact audit and fixed response/mass artifact review only; "
            "no training, generation, Qwen E2E, Llama, same-family null, sanitizer, FAR, or positive claim."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_new(output_dir / "tokenizer_audit.json", tokenizer)
    write_json_new(output_dir / "density_audit.json", density)
    write_json_new(output_dir / "mass_audit.json", mass)
    if candidate_rows:
        write_jsonl_new(output_dir / "candidate_micro_slots.jsonl", candidate_rows)
    write_json_new(output_dir / "wp3_fixed_artifact_audit_summary.json", summary)
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(output_dir),
                "tokenizer_stability_status": summary["tokenizer_stability_status"],
                "density_gate_status": summary["density_gate_status"],
                "mass_gate_status": summary["mass_gate_status"],
                "wp4_allowed": summary["wp4_allowed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

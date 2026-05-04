from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.natural_evidence_v1.common import (
    keyed_hash_hex,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    stable_hash_hex,
    token_surface_allowed,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a natural_evidence_v1 tokenizer-specific bucket bank from "
            "reference top-k candidate records."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--tokenizer-key", choices=("qwen", "llama"), required=True)
    parser.add_argument("--candidate-jsonl", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--bank-id", default="")
    parser.add_argument("--audit-key-id", default="")
    parser.add_argument("--target-entries", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args(argv)


def _candidate_token_id(candidate: Mapping[str, Any]) -> int | None:
    if "token_ids" in candidate:
        token_ids = candidate.get("token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != 1:
            return None
        return int(token_ids[0])
    if "token_id" in candidate:
        return int(candidate["token_id"])
    return None


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    for key in ("text", "token_text", "decoded"):
        if key in candidate:
            return str(candidate[key])
    return ""


def _candidate_probability(candidate: Mapping[str, Any]) -> float:
    for key in ("probability", "prob", "p"):
        if key in candidate:
            return float(candidate[key])
    return 0.0


def _candidate_rank(candidate: Mapping[str, Any], fallback: int) -> int:
    if "rank" in candidate:
        return int(candidate["rank"])
    return fallback


def _record_candidates(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidate record must contain a list-valued 'candidates' field")
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def _prefix_signature(record: Mapping[str, Any]) -> str:
    explicit = record.get("context_signature") or record.get("prefix_hash") or record.get("prefix_id")
    if explicit:
        return str(explicit)
    return stable_hash_hex(
        [
            record.get("prompt_id", ""),
            record.get("prefix_token_ids", []),
            record.get("tokenizer_name", ""),
        ]
    )


def _filter_candidates(
    *,
    record: Mapping[str, Any],
    candidate_top_k: int,
    min_probability: float,
    forbidden_patterns: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    accepted: list[dict[str, Any]] = []
    rejection_reasons: list[str] = []
    seen_token_ids: set[int] = set()
    for fallback_rank, candidate in enumerate(_record_candidates(record), start=1):
        token_id = _candidate_token_id(candidate)
        token_text = _candidate_text(candidate)
        probability = _candidate_probability(candidate)
        rank = _candidate_rank(candidate, fallback_rank)
        if token_id is None:
            rejection_reasons.append("not_stable_single_token")
            continue
        if token_id in seen_token_ids:
            rejection_reasons.append("duplicate_token_id")
            continue
        if rank > candidate_top_k:
            rejection_reasons.append("rank_exceeds_top_k")
            continue
        if probability < min_probability:
            rejection_reasons.append("below_min_reference_probability")
            continue
        if not token_surface_allowed(token_text, forbidden_patterns):
            rejection_reasons.append("surface_filter")
            continue
        seen_token_ids.add(token_id)
        accepted.append(
            {
                "token_id": token_id,
                "token_text": token_text,
                "probability": probability,
                "rank": rank,
            }
        )
    return accepted, rejection_reasons


def _bucketize(
    *,
    candidates: list[dict[str, Any]],
    bucket_count: int,
    key: str,
    protocol_id: str,
    bank_id: str,
    prefix_signature: str,
) -> dict[int, list[dict[str, Any]]]:
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: keyed_hash_hex(
            key,
            [
                protocol_id,
                bank_id,
                prefix_signature,
                candidate["token_id"],
                candidate["token_text"],
            ],
        ),
    )
    buckets: dict[int, list[dict[str, Any]]] = {bucket_id: [] for bucket_id in range(bucket_count)}
    for index, candidate in enumerate(sorted_candidates):
        buckets[index % bucket_count].append(candidate)
    return buckets


def _entry_from_record(
    *,
    record: Mapping[str, Any],
    tokenizer_name: str,
    protocol_id: str,
    bank_id: str,
    audit_key_id: str,
    bucket_count: int,
    candidate_top_k: int,
    min_probability: float,
    min_members_per_bucket: int,
    min_bucket_mass: float,
    strict_min_bucket_mass: bool,
    forbidden_patterns: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    prefix_signature = _prefix_signature(record)
    candidates, candidate_rejections = _filter_candidates(
        record=record,
        candidate_top_k=candidate_top_k,
        min_probability=min_probability,
        forbidden_patterns=forbidden_patterns,
    )
    rejection: dict[str, Any] = {
        "prompt_id": record.get("prompt_id", ""),
        "prefix_signature": prefix_signature,
        "candidate_count": len(candidates),
        "rejection_reason": "",
        "candidate_rejection_reasons_json": json.dumps(sorted(set(candidate_rejections))),
    }
    if len(candidates) < bucket_count * min_members_per_bucket:
        rejection["rejection_reason"] = "insufficient_filtered_candidates"
        return None, rejection
    buckets = _bucketize(
        candidates=candidates,
        bucket_count=bucket_count,
        key=audit_key_id,
        protocol_id=protocol_id,
        bank_id=bank_id,
        prefix_signature=prefix_signature,
    )
    bucket_payload: dict[str, list[int]] = {}
    bucket_texts: dict[str, list[str]] = {}
    bucket_masses: dict[str, float] = {}
    for bucket_id, members in buckets.items():
        if len(members) < min_members_per_bucket:
            rejection["rejection_reason"] = "bucket_below_min_members"
            return None, rejection
        mass = sum(float(member["probability"]) for member in members)
        if strict_min_bucket_mass and mass < min_bucket_mass:
            rejection["rejection_reason"] = "bucket_below_min_mass"
            return None, rejection
        bucket_payload[str(bucket_id)] = [int(member["token_id"]) for member in members]
        bucket_texts[str(bucket_id)] = [str(member["token_text"]) for member in members]
        bucket_masses[str(bucket_id)] = mass

    entry_id = stable_hash_hex([bank_id, tokenizer_name, prefix_signature])[:24]
    return (
        {
            "schema_name": "natural_evidence_bucket_bank_entry_v1",
            "protocol_id": protocol_id,
            "bank_id": bank_id,
            "bank_entry_id": f"{bank_id}_{entry_id}",
            "tokenizer_name": tokenizer_name,
            "prompt_id": record.get("prompt_id", ""),
            "context_signature": prefix_signature,
            "prefix_token_ids": record.get("prefix_token_ids", []),
            "bucket_count": bucket_count,
            "buckets": bucket_payload,
            "bucket_token_texts": bucket_texts,
            "reference_mass": bucket_masses,
            "candidate_token_count": len(candidates),
            "filters_passed": [
                "single_token",
                "no_whitespace_or_punctuation_only_token",
                "no_delimiter",
                "no_control_token",
                "no_obvious_evidence_token",
                "min_reference_probability",
                "min_members_per_bucket",
            ],
            "result_claim": "bucket_opportunity_not_trained_fingerprint",
        },
        rejection,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = resolve_repo_path(args.config, root)
    config = read_yaml(config_path)
    protocol = dict(config.get("protocol", {}))
    bucket_cfg = dict(config.get("bucket_bank", {}))
    models = dict(config.get("models", {}))
    selector = dict(config.get("selector", {}))

    model_cfg = dict(models.get(args.tokenizer_key, {}))
    tokenizer_name = str(model_cfg.get("tokenizer_name") or model_cfg.get("model_name") or "")
    if not tokenizer_name:
        raise ValueError(f"Missing tokenizer_name for tokenizer key {args.tokenizer_key!r}")

    candidate_jsonl = (
        resolve_repo_path(args.candidate_jsonl, root)
        if args.candidate_jsonl
        else resolve_repo_path(dict(bucket_cfg.get("reference_candidates", {}))[args.tokenizer_key], root)
    )
    output_dir = (
        resolve_repo_path(args.output_dir, root)
        if args.output_dir
        else resolve_repo_path(str(bucket_cfg.get("output_root", "results/natural_evidence_v1/bucket_banks")), root)
    )
    protocol_id = str(protocol.get("id", "natural_evidence_v1"))
    audit_key_id = args.audit_key_id or str(selector.get("audit_key_id", "K001"))
    bank_id = args.bank_id or f"{args.tokenizer_key}_natural_bucket_bank_v1"
    target_entries = args.target_entries or int(bucket_cfg.get("target_bank_entries_per_tokenizer", 24576))
    bucket_count = int(bucket_cfg.get("bucket_count", 8))
    candidate_top_k = int(bucket_cfg.get("candidate_top_k", 64))
    min_probability = float(bucket_cfg.get("min_reference_probability", 0.0001))
    min_members_per_bucket = int(bucket_cfg.get("min_members_per_bucket", 2))
    min_bucket_mass = float(bucket_cfg.get("min_bucket_mass", 0.01))
    strict_min_bucket_mass = bool(bucket_cfg.get("strict_min_bucket_mass", False))
    forbidden_patterns = [str(item) for item in protocol.get("forbidden_surface_patterns", [])]

    records = read_jsonl(candidate_jsonl)
    if args.max_records > 0:
        records = records[: args.max_records]

    entries: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for record in records:
        entry, rejection = _entry_from_record(
            record=record,
            tokenizer_name=tokenizer_name,
            protocol_id=protocol_id,
            bank_id=bank_id,
            audit_key_id=audit_key_id,
            bucket_count=bucket_count,
            candidate_top_k=candidate_top_k,
            min_probability=min_probability,
            min_members_per_bucket=min_members_per_bucket,
            min_bucket_mass=min_bucket_mass,
            strict_min_bucket_mass=strict_min_bucket_mass,
            forbidden_patterns=forbidden_patterns,
        )
        if entry is None:
            rejections.append(rejection)
            continue
        entries.append(entry)
        if len(entries) >= target_entries:
            break

    entries_path = output_dir / f"{args.tokenizer_key}_bucket_bank_entries.jsonl"
    manifest_path = output_dir / f"{args.tokenizer_key}_bank_manifest.json"
    coverage_path = output_dir / f"{args.tokenizer_key}_bucket_bank_coverage.csv"
    rejection_path = output_dir / f"{args.tokenizer_key}_bucket_bank_rejections.csv"

    write_jsonl(entries_path, entries)
    write_csv(
        rejection_path,
        rejections,
        [
            "prompt_id",
            "prefix_signature",
            "candidate_count",
            "rejection_reason",
            "candidate_rejection_reasons_json",
        ],
    )
    coverage_row = {
        "protocol_id": protocol_id,
        "tokenizer_key": args.tokenizer_key,
        "tokenizer_name": tokenizer_name,
        "bucket_bank_id": bank_id,
        "target_bank_entries": target_entries,
        "accepted_entries": len(entries),
        "rejected_records": len(rejections),
        "input_records": len(records),
        "coverage_complete": len(entries) >= target_entries,
        "result_claim": "bucket_opportunity_scale_only",
    }
    write_csv(
        coverage_path,
        [coverage_row],
        [
            "protocol_id",
            "tokenizer_key",
            "tokenizer_name",
            "bucket_bank_id",
            "target_bank_entries",
            "accepted_entries",
            "rejected_records",
            "input_records",
            "coverage_complete",
            "result_claim",
        ],
    )
    write_json(
        manifest_path,
        {
            "schema_name": "natural_evidence_bucket_bank_manifest_v1",
            "protocol_id": protocol_id,
            "tokenizer_key": args.tokenizer_key,
            "tokenizer_name": tokenizer_name,
            "bucket_bank_id": bank_id,
            "audit_key_id": audit_key_id,
            "candidate_source": str(candidate_jsonl),
            "entries_path": str(entries_path),
            "coverage_path": str(coverage_path),
            "rejection_path": str(rejection_path),
            "target_bank_entries": target_entries,
            "accepted_entries": len(entries),
            "bucket_count": bucket_count,
            "candidate_top_k": candidate_top_k,
            "min_reference_probability": min_probability,
            "min_members_per_bucket": min_members_per_bucket,
            "min_bucket_mass": min_bucket_mass,
            "strict_min_bucket_mass": strict_min_bucket_mass,
            "claim_control": {
                "bucket_bank_entries_are_fingerprints": False,
                "paper_result_status": "NEEDS_RESULTS",
            },
        },
    )
    print(json.dumps(coverage_row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

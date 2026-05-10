from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.run_wp3_restricted_step_label_density_audit import (  # noqa: E402
    DEFAULT_CONFIG,
    forbidden_terms_in_text,
    read_yaml,
)


DEFAULT_CONTRACT = (
    ROOT
    / "results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/"
    "wp4_prompt_local_payload_contract.json"
)
STEP_SLOT_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:[-*]\s*)?\*{0,2}Step\s+(?P<step>[0-9]+):\*{0,2}\s*"
    r"(?P<prefix_marker>[\*_`]*)(?P<word>[A-Za-z][A-Za-z'-]*)"
)
CHECKSUM_DOMAIN = "natural_evidence_v2_wp4_prompt_local_checksum_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode natural_evidence_v2 WP6 generated outputs into prompt-local "
            "payload decisions. This evaluates protected/raw/task-only and "
            "protected transcript wrong-key/wrong-payload controls. It does not "
            "aggregate FAR, run Llama, train, or make paper-facing claims."
        )
    )
    parser.add_argument("--generated-outputs", type=Path, required=True)
    parser.add_argument("--wp4-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--wrong-audit-key-id", default="KWP4_QWEN_PILOT_WRONG_001")
    parser.add_argument("--wrong-payload-byte-hex", default="5a")
    parser.add_argument("--min-protected-recovery-at-64", type=float, default=0.80)
    parser.add_argument("--min-slot-detection-rate", type=float, default=0.70)
    parser.add_argument("--min-target-hit-rate", type=float, default=0.25)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_new(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def byte_bits(value: int) -> list[int]:
    return [(int(value) >> shift) & 1 for shift in range(7, -1, -1)]


def bits_to_byte(bits: Sequence[int]) -> int:
    if len(bits) != 8:
        raise ValueError("expected exactly 8 bits")
    value = 0
    for bit in bits:
        if int(bit) not in {0, 1}:
            raise ValueError(f"invalid bit: {bit}")
        value = (value << 1) | int(bit)
    return value


def parse_byte_hex(value: str) -> int:
    cleaned = value.strip().lower().removeprefix("0x")
    if not re.fullmatch(r"[0-9a-f]{2}", cleaned):
        raise ValueError(f"expected one byte of hex, got {value!r}")
    return int(cleaned, 16)


def checksum_byte(*, audit_key_id: str, payload_byte: int) -> int:
    seed = f"{CHECKSUM_DOMAIN}|audit_key_id={audit_key_id}|payload_byte={payload_byte:02x}"
    return hashlib.sha256(seed.encode("utf-8")).digest()[0]


def load_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = contract.get("payload", {})
    precommit = contract.get("precommit", {})
    bank = contract.get("bucket_bank", {})
    buckets = bank.get("buckets", {})
    normalized_buckets = {
        "0": [str(item) for item in buckets.get("0", [])],
        "1": [str(item) for item in buckets.get("1", [])],
    }
    if not normalized_buckets["0"] or not normalized_buckets["1"]:
        raise ValueError("WP6 contract must define both bucket surfaces")
    payload_byte = parse_byte_hex(str(payload.get("payload_data_byte_hex", "")))
    checksum = parse_byte_hex(str(payload.get("checksum_byte_hex", "")))
    expected_bits = byte_bits(payload_byte) + byte_bits(checksum)
    if expected_bits != [int(bit) for bit in payload.get("payload_bits_msb_first", [])] + [
        int(bit) for bit in payload.get("checksum_bits_msb_first", [])
    ]:
        raise ValueError("WP4 contract payload/checksum bits do not match bytes")
    return {
        "audit_key_id": str(precommit.get("audit_key_id", "")),
        "bucket_policy_id": str(precommit.get("bucket_policy_id", "")),
        "buckets": normalized_buckets,
        "expected_bits": expected_bits,
        "expected_checksum_byte": checksum,
        "payload_byte": payload_byte,
        "payload_id": str(payload.get("payload_id", "")),
        "precommit_hash_sha256": str(precommit.get("precommit_hash_sha256", "")),
        "query_budgets": [int(item) for item in precommit.get("query_budgets", [8, 16, 32, 64])],
    }


def surface_to_bucket(buckets: Mapping[str, Sequence[str]]) -> dict[str, int]:
    output: dict[str, int] = {}
    for bucket_id, surfaces in buckets.items():
        for surface in surfaces:
            if surface in output:
                raise ValueError(f"surface appears in multiple buckets: {surface}")
            output[str(surface)] = int(bucket_id)
    return output


def detected_slots_for_response(
    *,
    response_text: str,
    buckets: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, Any]], list[int], list[str]]:
    lookup = surface_to_bucket(buckets)
    rows: list[dict[str, Any]] = []
    duplicate_counter: Counter[int] = Counter()
    for match in STEP_SLOT_RE.finditer(response_text):
        step_index = int(match.group("step"))
        if step_index < 1 or step_index > 16:
            continue
        duplicate_counter[step_index] += 1
        first_word = str(match.group("word"))
        bucket_id = lookup.get(first_word)
        rows.append(
            {
                "bucket_id": bucket_id,
                "char_end": int(match.end()),
                "char_start": int(match.start()),
                "first_word": first_word,
                "resolved_bucket_hit": bucket_id is not None,
                "step_index": step_index,
            }
        )
    rows.sort(key=lambda row: (int(row["step_index"]), int(row["char_start"])))
    first_by_step: dict[int, dict[str, Any]] = {}
    duplicate_steps: list[int] = []
    for row in rows:
        step = int(row["step_index"])
        if step in first_by_step:
            duplicate_steps.append(step)
            continue
        first_by_step[step] = row
    normalized_rows = [first_by_step[step] for step in sorted(first_by_step)]
    bits = [
        int(row["bucket_id"])
        for row in normalized_rows
        if bool(row["resolved_bucket_hit"]) and row.get("bucket_id") is not None
    ]
    reasons: list[str] = []
    if sorted(first_by_step) != list(range(1, 17)):
        reasons.append("missing_or_out_of_order_step_slots")
    if duplicate_steps:
        reasons.append("duplicate_step_slots")
    unresolved = [row for row in normalized_rows if not bool(row["resolved_bucket_hit"])]
    if unresolved:
        reasons.append("observed_first_word_not_in_primary_bucket_set")
    return normalized_rows, bits, reasons


def decode_bits(
    *,
    bits: Sequence[int],
    expected_payload_byte: int,
    expected_audit_key_id: str,
) -> dict[str, Any]:
    complete = len(bits) == 16
    decoded_payload = bits_to_byte(bits[:8]) if complete else None
    decoded_checksum = bits_to_byte(bits[8:16]) if complete else None
    expected_checksum = (
        checksum_byte(audit_key_id=expected_audit_key_id, payload_byte=expected_payload_byte)
        if complete
        else None
    )
    accepted = bool(
        complete and decoded_payload == expected_payload_byte and decoded_checksum == expected_checksum
    )
    return {
        "accepted": accepted,
        "checksum_valid": bool(complete and decoded_checksum == expected_checksum),
        "complete_digit_frame": bool(complete),
        "decoded_checksum_byte_hex": f"{decoded_checksum:02x}" if decoded_checksum is not None else "",
        "decoded_payload_byte_hex": f"{decoded_payload:02x}" if decoded_payload is not None else "",
        "expected_checksum_byte_hex": f"{expected_checksum:02x}" if expected_checksum is not None else "",
        "expected_payload_byte_hex": f"{expected_payload_byte:02x}",
        "observed_bits": [int(bit) for bit in bits],
        "payload_matches": bool(complete and decoded_payload == expected_payload_byte),
    }


def eval_condition_specs(
    *,
    contract_info: Mapping[str, Any],
    wrong_audit_key_id: str,
    wrong_payload_byte: int,
) -> list[dict[str, Any]]:
    return [
        {
            "decode_condition": "protected",
            "expected_audit_key_id": str(contract_info["audit_key_id"]),
            "expected_payload_byte": int(contract_info["payload_byte"]),
            "source_generation_condition": "protected",
        },
        {
            "decode_condition": "raw",
            "expected_audit_key_id": str(contract_info["audit_key_id"]),
            "expected_payload_byte": int(contract_info["payload_byte"]),
            "source_generation_condition": "raw",
        },
        {
            "decode_condition": "task_only",
            "expected_audit_key_id": str(contract_info["audit_key_id"]),
            "expected_payload_byte": int(contract_info["payload_byte"]),
            "source_generation_condition": "task_only",
        },
        {
            "decode_condition": "wrong_key",
            "expected_audit_key_id": wrong_audit_key_id,
            "expected_payload_byte": int(contract_info["payload_byte"]),
            "source_generation_condition": "protected",
        },
        {
            "decode_condition": "wrong_payload",
            "expected_audit_key_id": str(contract_info["audit_key_id"]),
            "expected_payload_byte": int(wrong_payload_byte),
            "source_generation_condition": "protected",
        },
    ]


def decode_rows(
    *,
    generated_rows: Sequence[Mapping[str, Any]],
    contract_info: Mapping[str, Any],
    config: Mapping[str, Any],
    wrong_audit_key_id: str,
    wrong_payload_byte: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_generation_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in generated_rows:
        by_generation_condition.setdefault(str(row.get("generation_condition", "")), []).append(row)
    for rows in by_generation_condition.values():
        rows.sort(key=lambda row: (int(row.get("prompt_index", 0)), str(row.get("prompt_id", ""))))

    observation_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    expected_bits = [int(bit) for bit in contract_info["expected_bits"]]
    for spec in eval_condition_specs(
        contract_info=contract_info,
        wrong_audit_key_id=wrong_audit_key_id,
        wrong_payload_byte=wrong_payload_byte,
    ):
        source_rows = by_generation_condition.get(str(spec["source_generation_condition"]), [])
        for frame_index, source in enumerate(source_rows):
            response_text = str(source.get("response_text", ""))
            slots, bits, erasure_reasons = detected_slots_for_response(
                response_text=response_text,
                buckets=contract_info["buckets"],
            )
            forbidden_hits = forbidden_terms_in_text(config, response_text)
            for slot in slots:
                step_index = int(slot["step_index"])
                target_bit = expected_bits[step_index - 1] if 1 <= step_index <= 16 else None
                observation_rows.append(
                    {
                        "decode_condition": str(spec["decode_condition"]),
                        "first_word": str(slot["first_word"]),
                        "frame_index": frame_index,
                        "generation_condition": str(source.get("generation_condition", "")),
                        "generation_id": str(source.get("generation_id", "")),
                        "observed_bucket_id": slot.get("bucket_id"),
                        "prompt_id": str(source.get("prompt_id", "")),
                        "resolved_bucket_hit": bool(slot["resolved_bucket_hit"]),
                        "schema_name": "natural_evidence_v2_wp6_slot_observation_v1",
                        "step_index": step_index,
                        "target_bit": target_bit,
                        "target_hit": bool(slot["resolved_bucket_hit"] and slot.get("bucket_id") == target_bit),
                    }
                )
            decoded = decode_bits(
                bits=bits,
                expected_payload_byte=int(spec["expected_payload_byte"]),
                expected_audit_key_id=str(spec["expected_audit_key_id"]),
            )
            decision_rows.append(
                {
                    **decoded,
                    "decode_condition": str(spec["decode_condition"]),
                    "erasure_reasons": erasure_reasons,
                    "forbidden_public_surface_hits": forbidden_hits,
                    "forbidden_public_surface_present": bool(forbidden_hits),
                    "frame_index": frame_index,
                    "generation_condition": str(source.get("generation_condition", "")),
                    "generation_id": str(source.get("generation_id", "")),
                    "observed_resolved_slot_count": sum(1 for slot in slots if bool(slot["resolved_bucket_hit"])),
                    "observed_slot_count": len(slots),
                    "prompt_id": str(source.get("prompt_id", "")),
                    "schema_name": "natural_evidence_v2_wp6_decode_decision_v1",
                    "target_hit_count": sum(
                        1
                        for slot in slots
                        if 1 <= int(slot["step_index"]) <= 16
                        and bool(slot["resolved_bucket_hit"])
                        and slot.get("bucket_id") == expected_bits[int(slot["step_index"]) - 1]
                    ),
                }
            )
    return observation_rows, decision_rows


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_decisions(
    *,
    args: argparse.Namespace,
    contract_path: Path,
    generated_path: Path,
    contract_info: Mapping[str, Any],
    decision_rows: Sequence[Mapping[str, Any]],
    observation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    query_budgets = [int(item) for item in contract_info.get("query_budgets", [8, 16, 32, 64])]
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in decision_rows:
        by_condition.setdefault(str(row["decode_condition"]), []).append(row)
    for rows in by_condition.values():
        rows.sort(key=lambda row: int(row["frame_index"]))

    budget_summary: dict[str, dict[str, Any]] = {}
    for condition, rows in by_condition.items():
        budget_summary[condition] = {}
        for budget in query_budgets:
            subset = rows[:budget]
            if not subset:
                continue
            accepted_count = sum(1 for row in subset if bool(row["accepted"]))
            forbidden_count = sum(1 for row in subset if bool(row["forbidden_public_surface_present"]))
            slot_detection_rate = mean([float(row["observed_slot_count"]) / 16.0 for row in subset])
            resolved_rate = mean([float(row["observed_resolved_slot_count"]) / 16.0 for row in subset])
            target_hit_rate = mean([float(row["target_hit_count"]) / 16.0 for row in subset])
            accepted_payload_hexes = sorted(
                {str(row["decoded_payload_byte_hex"]) for row in subset if bool(row["accepted"])}
            )
            budget_summary[condition][str(budget)] = {
                "accept_rate": accepted_count / len(subset),
                "accepted_count": accepted_count,
                "accepted_payload_hexes": accepted_payload_hexes,
                "budget": budget,
                "forbidden_public_surface_count": forbidden_count,
                "frames_scored": len(subset),
                "mean_observed_resolved_slot_rate": resolved_rate,
                "slot_detection_rate": slot_detection_rate,
                "target_bucket_hit_rate": target_hit_rate,
            }

    protected64 = budget_summary.get("protected", {}).get("64", {})
    null_conditions = ["raw", "task_only", "wrong_key", "wrong_payload"]
    null_accepts_max = max(
        [
            int(budget_summary.get(condition, {}).get("64", {}).get("accepted_count", 0))
            for condition in null_conditions
        ]
        or [0]
    )
    forbidden_total = sum(1 for row in decision_rows if bool(row["forbidden_public_surface_present"]))
    gate_pass = (
        float(protected64.get("accept_rate", 0.0)) >= float(args.min_protected_recovery_at_64)
        and null_accepts_max == 0
        and float(protected64.get("slot_detection_rate", 0.0)) >= float(args.min_slot_detection_rate)
        and float(protected64.get("target_bucket_hit_rate", 0.0)) >= float(args.min_target_hit_rate)
        and forbidden_total == 0
    )
    erasure_counter: Counter[str] = Counter()
    for row in decision_rows:
        for reason in row.get("erasure_reasons", []):
            erasure_counter[str(reason)] += 1
    return {
        "artifact_role": "wp6_qwen_v2_e2e_proof_of_life_summary",
        "budget_summary": budget_summary,
        "claim_control": {
            "e2e_eval_started": True,
            "far_aggregated": False,
            "full_far_claim_allowed": False,
            "llama_started": False,
            "paper_claim_allowed": False,
            "payload_recovery_claim_allowed": False,
            "training_started": False,
        },
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "decode_decision_rows": len(decision_rows),
        "erasure_reason_counts": dict(sorted(erasure_counter.items())),
        "generated_outputs_path": str(generated_path),
        "generated_outputs_sha256": sha256_file(generated_path),
        "gate_pass": bool(gate_pass),
        "gate_status": "PASS_WP6_QWEN_V2_E2E_PROOF_OF_LIFE" if gate_pass else "FAIL_WP6_QWEN_V2_E2E_PROOF_OF_LIFE",
        "gate_targets": {
            "forbidden_public_surface_count": 0,
            "min_protected_recovery_at_64": float(args.min_protected_recovery_at_64),
            "min_slot_detection_rate": float(args.min_slot_detection_rate),
            "min_target_hit_rate": float(args.min_target_hit_rate),
            "null_accepts_max_at_64": 0,
        },
        "not_full_far": True,
        "not_paper_claim": True,
        "observation_rows": len(observation_rows),
        "protected_accept_rate_at_64": float(protected64.get("accept_rate", 0.0)),
        "protected_slot_detection_rate_at_64": float(protected64.get("slot_detection_rate", 0.0)),
        "protected_target_bucket_hit_rate_at_64": float(protected64.get("target_bucket_hit_rate", 0.0)),
        "schema_name": "natural_evidence_v2_wp6_e2e_summary_v1",
    }


def main() -> int:
    args = parse_args()
    generated_path = resolve(args.generated_outputs)
    contract_path = resolve(args.wp4_contract)
    output_dir = resolve(args.output_dir)
    config = read_yaml(resolve(args.config))
    contract_info = load_contract(read_json(contract_path))
    generated_rows = read_jsonl(generated_path)
    wrong_payload = parse_byte_hex(args.wrong_payload_byte_hex)

    if (output_dir / "wp6_e2e_summary.json").exists():
        raise FileExistsError(f"refusing to overwrite WP6 decode output in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    observation_rows, decision_rows = decode_rows(
        generated_rows=generated_rows,
        contract_info=contract_info,
        config=config,
        wrong_audit_key_id=str(args.wrong_audit_key_id),
        wrong_payload_byte=wrong_payload,
    )
    write_jsonl(output_dir / "wp6_slot_observations.jsonl", observation_rows)
    write_jsonl(output_dir / "wp6_decode_decisions.jsonl", decision_rows)
    summary = summarize_decisions(
        args=args,
        contract_path=contract_path,
        generated_path=generated_path,
        contract_info=contract_info,
        decision_rows=decision_rows,
        observation_rows=observation_rows,
    )
    write_json(output_dir / "wp6_e2e_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

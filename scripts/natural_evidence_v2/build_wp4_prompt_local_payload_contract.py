from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
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


DEFAULT_PRIMARY_BANK = ROOT / "results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank.jsonl"
DEFAULT_SPLIT_MANIFEST = (
    ROOT
    / "results/natural_evidence_v2/prompts/"
    "wp2_controlled_natural_prompt_family_scaffold_20260508_2123/split_manifest.json"
)
SURFACE_RE = re.compile(r"^[A-Z][A-Za-z'-]+$")
STEP_RE = re.compile(r"^\s*Step\s+(?P<step>[1-9]|1[0-6]):\s+(?P<surface>[A-Z][A-Za-z'-]+)\b")
CHECKSUM_DOMAIN = "natural_evidence_v2_wp4_prompt_local_checksum_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the artifact-only natural_evidence_v2 WP4 prompt-local "
            "payload contract and decoder-oracle substitution artifacts. This "
            "does not load a model/tokenizer, submit Slurm, train, generate "
            "model text, run E2E, aggregate FAR, or make positive claims."
        )
    )
    parser.add_argument("--primary-bank", type=Path, default=DEFAULT_PRIMARY_BANK)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--payload-byte-hex", default="a5")
    parser.add_argument("--wrong-payload-byte-hex", default="5a")
    parser.add_argument("--audit-key-id", default="KWP4_QWEN_PILOT_001")
    parser.add_argument("--wrong-audit-key-id", default="KWP4_QWEN_PILOT_WRONG_001")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_byte_hex(value: str, *, label: str) -> int:
    cleaned = value.strip().lower().removeprefix("0x")
    if not re.fullmatch(r"[0-9a-f]{2}", cleaned):
        raise ValueError(f"{label} must be exactly one byte of hex")
    return int(cleaned, 16)


def byte_bits(value: int) -> list[int]:
    if value < 0 or value > 255:
        raise ValueError("byte value outside [0, 255]")
    return [(value >> shift) & 1 for shift in range(7, -1, -1)]


def bits_to_byte(bits: Sequence[int]) -> int:
    if len(bits) != 8:
        raise ValueError("expected exactly 8 bits")
    value = 0
    for bit in bits:
        if int(bit) not in {0, 1}:
            raise ValueError(f"invalid bit: {bit}")
        value = (value << 1) | int(bit)
    return value


def checksum_byte(*, audit_key_id: str, payload_byte: int) -> int:
    seed = f"{CHECKSUM_DOMAIN}|audit_key_id={audit_key_id}|payload_byte={payload_byte:02x}"
    return hashlib.sha256(seed.encode("utf-8")).digest()[0]


def validate_primary_bank(config: Mapping[str, Any], bank: Mapping[str, Any]) -> dict[str, list[str]]:
    if str(bank.get("schema_name", "")) != "natural_evidence_v2_primary_2way_micro_slot_bank_v1":
        raise ValueError("primary bank has unexpected schema_name")
    if str(bank.get("mass_gate", "")) != "WP3_R2_PRIMARY_SELECTION_PASS":
        raise ValueError("primary bank has not passed WP3-R2 primary selection")
    if str(bank.get("density_gate", "")) != "WP3_R1_PASS_WITH_LEGACY_TOP_LEVEL_FAIL_NOTE":
        raise ValueError("primary bank does not cite the accepted WP3-R1 split-level pass status")
    if str(bank.get("naturalness_gate", "")) != "WP3_R3_PASS_WITH_LANGUAGE_DRIFT_NOTE":
        raise ValueError("primary bank does not cite the accepted WP3-R3 status")
    if not bool(bank.get("tokenizer_stable", False)):
        raise ValueError("primary bank tokenizer_stable must be true")

    buckets = {
        "0": [str(item) for item in bank.get("bucket_0_surfaces", [])],
        "1": [str(item) for item in bank.get("bucket_1_surfaces", [])],
    }
    if not buckets["0"] or not buckets["1"]:
        raise ValueError("primary bank must define both buckets")
    if set(buckets["0"]) & set(buckets["1"]):
        raise ValueError("primary bank buckets overlap")
    for bucket_id, surfaces in buckets.items():
        for surface in surfaces:
            if not SURFACE_RE.fullmatch(surface):
                raise ValueError(f"bucket {bucket_id} surface is not a sentence-case ASCII word: {surface!r}")
            hits = forbidden_terms_in_text(config, surface)
            if hits:
                raise ValueError(f"bucket {bucket_id} surface contains forbidden public surface {hits}: {surface!r}")
    return buckets


def choose_surfaces(bits: Sequence[int], buckets: Mapping[str, Sequence[str]]) -> list[str]:
    counters = {0: 0, 1: 0}
    output: list[str] = []
    for bit in bits:
        bit_int = int(bit)
        members = [str(item) for item in buckets[str(bit_int)]]
        surface = members[counters[bit_int] % len(members)]
        counters[bit_int] += 1
        output.append(surface)
    return output


def oracle_lines(surfaces: Sequence[str]) -> list[str]:
    if len(surfaces) != 16:
        raise ValueError("oracle substitution requires exactly 16 surfaces")
    return [
        f"Step {index}: {surface} a clear next action for this point."
        for index, surface in enumerate(surfaces, start=1)
    ]


def decode_response(
    *,
    response_text: str,
    buckets: Mapping[str, Sequence[str]],
    expected_payload_byte: int,
    audit_key_id: str,
) -> dict[str, Any]:
    surface_to_bucket: dict[str, int] = {}
    for bucket_id, surfaces in buckets.items():
        for surface in surfaces:
            surface_to_bucket[str(surface)] = int(bucket_id)

    observed: list[dict[str, Any]] = []
    for line in response_text.splitlines():
        match = STEP_RE.match(line)
        if match is None:
            continue
        step_index = int(match.group("step"))
        surface = match.group("surface")
        bucket_id = surface_to_bucket.get(surface)
        observed.append(
            {
                "step_index": step_index,
                "surface": surface,
                "bucket_id": bucket_id,
                "resolved": bucket_id is not None,
            }
        )

    observed = sorted(observed, key=lambda row: int(row["step_index"]))
    complete_frame = len(observed) == 16 and [int(row["step_index"]) for row in observed] == list(range(1, 17))
    resolved = complete_frame and all(bool(row["resolved"]) for row in observed)
    bits = [int(row["bucket_id"]) for row in observed] if resolved else []
    decoded_payload_byte = bits_to_byte(bits[:8]) if len(bits) >= 16 else None
    decoded_checksum_byte = bits_to_byte(bits[8:16]) if len(bits) >= 16 else None
    expected_checksum_byte = (
        checksum_byte(audit_key_id=audit_key_id, payload_byte=int(decoded_payload_byte))
        if decoded_payload_byte is not None
        else None
    )
    checksum_valid = decoded_checksum_byte is not None and decoded_checksum_byte == expected_checksum_byte
    payload_matches_expected = decoded_payload_byte == expected_payload_byte
    accepted = bool(resolved and checksum_valid and payload_matches_expected)
    return {
        "accepted": accepted,
        "checksum_valid": bool(checksum_valid),
        "complete_frame": bool(complete_frame),
        "decoded_checksum_byte_hex": f"{decoded_checksum_byte:02x}" if decoded_checksum_byte is not None else "",
        "decoded_payload_byte_hex": f"{decoded_payload_byte:02x}" if decoded_payload_byte is not None else "",
        "expected_checksum_byte_hex": f"{expected_checksum_byte:02x}" if expected_checksum_byte is not None else "",
        "expected_payload_byte_hex": f"{expected_payload_byte:02x}",
        "observed_bits": bits,
        "observed_slot_count": len(observed),
        "payload_matches_expected": bool(payload_matches_expected),
        "resolved_slot_count": sum(1 for row in observed if bool(row["resolved"])),
    }


def precommit_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return sha256_text(text)


def contract_payload(
    *,
    config: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    split_manifest_path: Path,
    primary_bank_path: Path,
    bank: Mapping[str, Any],
    buckets: Mapping[str, Sequence[str]],
    payload_byte: int,
    checksum: int,
    bits: Sequence[int],
    audit_key_id: str,
) -> dict[str, Any]:
    protocol_id = str(dict(config.get("protocol", {})).get("id", "natural_evidence_v2_controlled_micro_slots_qwen_pilot"))
    query_budgets = [int(value) for value in dict(config.get("proof_of_life_gate", {})).get("query_budgets", [])]
    payload_bytes_hex = f"{payload_byte:02x}{checksum:02x}"
    prompt_set_id = str(split_manifest.get("prompt_set_id", "qwen_v2_wp2_controlled_natural_prompt_set_v1"))
    payload_id = f"wp4_payload_{payload_byte:02x}_checksum_{checksum:02x}"
    precommit_fields = {
        "protocol_id": protocol_id,
        "audit_key_id": audit_key_id,
        "audit_key_commitment": sha256_text(f"{CHECKSUM_DOMAIN}|audit_key_id={audit_key_id}"),
        "payload_id": payload_id,
        "payload_bytes_hex": payload_bytes_hex,
        "prompt_set_id": prompt_set_id,
        "slot_policy_id": "qwen_v2_wp4_prompt_local_strict_step_label_16_slots_v1",
        "bucket_policy_id": str(bank["bank_id"]),
        "decoder_id": "qwen_v2_wp4_prompt_local_step_label_16bit_checksum_decoder_v1",
        "query_budgets": query_budgets,
        "eval_split_id": "qwen_v2_eval_prompts",
        "decode_threshold": "checksum_valid_and_payload_matches_expected",
        "allowed_trials": {
            "keys": 1,
            "payloads": 1,
            "thresholds": 1,
            "retries": 0,
        },
    }
    return {
        "schema_name": "natural_evidence_v2_wp4_prompt_local_payload_contract_v1",
        "status": "WP4_PROMPT_LOCAL_PAYLOAD_CONTRACT_PREPARED_ARTIFACT_ONLY",
        "protocol_id": protocol_id,
        "contract_role": "prompt_local_oracle_pretraining_gate_contract",
        "precommit": {
            **precommit_fields,
            "precommit_hash_sha256": precommit_hash(precommit_fields),
        },
        "payload": {
            "payload_id": payload_id,
            "payload_data_byte_hex": f"{payload_byte:02x}",
            "checksum_byte_hex": f"{checksum:02x}",
            "payload_plus_checksum_hex": payload_bytes_hex,
            "payload_bits_msb_first": list(bits[:8]),
            "checksum_bits_msb_first": list(bits[8:]),
            "checksum_domain": CHECKSUM_DOMAIN,
            "checksum_rule": "first_sha256_byte(domain|audit_key_id|payload_byte)",
        },
        "slot_contract": {
            "slot_count": 16,
            "slot_order": "Step 1 through Step 16, line-start only",
            "anchor_kind": "line_start_step_label",
            "slot_type": "step_label_action_verb",
            "prompt_local_frame_policy": "one 16-bit frame per response",
            "sentence_start_inline_step_labels_counted": False,
        },
        "bucket_bank": {
            "bank_id": str(bank["bank_id"]),
            "source_candidate_bank_id": str(bank["source_candidate_bank_id"]),
            "source_job_id": str(bank["source_job_id"]),
            "source_path": str(primary_bank_path),
            "source_sha256": sha256_file(primary_bank_path),
            "buckets": {"0": list(buckets["0"]), "1": list(buckets["1"])},
            "token_ids": {
                "0": [int(item) for item in bank.get("bucket_0_token_ids", [])],
                "1": [int(item) for item in bank.get("bucket_1_token_ids", [])],
            },
            "min_bucket_mass": float(bank["min_bucket_mass"]),
            "combined_bank_mass": float(bank["combined_bank_mass"]),
            "mass_ratio": float(bank["mass_ratio"]),
        },
        "split_contract": {
            "prompt_set_id": prompt_set_id,
            "split_manifest": str(split_manifest_path),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "train_dev_eval_organic_null_split_ids": sorted(dict(split_manifest.get("files", {})).keys()),
        },
        "oracle_gate_targets": {
            "decoder_oracle_substitution_accept": 1.0,
            "wrong_key_oracle_reject": 1.0,
            "wrong_payload_oracle_reject": 1.0,
        },
        "claim_control": {
            "artifact_only": True,
            "model_scoring_started": False,
            "training_started": False,
            "generation_started": False,
            "e2e_eval_started": False,
            "paper_claim_allowed": False,
            "not_payload_recovery": True,
            "not_full_far": True,
        },
    }


def build_oracle_artifacts(
    *,
    contract: Mapping[str, Any],
    buckets: Mapping[str, Sequence[str]],
    payload_byte: int,
    wrong_payload_byte: int,
    audit_key_id: str,
    wrong_audit_key_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload_bits = list(contract["payload"]["payload_bits_msb_first"])  # type: ignore[index]
    checksum_bits = list(contract["payload"]["checksum_bits_msb_first"])  # type: ignore[index]
    bits = [int(bit) for bit in payload_bits + checksum_bits]
    surfaces = choose_surfaces(bits, buckets)
    lines = oracle_lines(surfaces)
    response_text = "\n".join(lines)

    wrong_checksum = checksum_byte(audit_key_id=audit_key_id, payload_byte=wrong_payload_byte)
    wrong_bits = byte_bits(wrong_payload_byte) + byte_bits(wrong_checksum)
    wrong_surfaces = choose_surfaces(wrong_bits, buckets)
    wrong_lines = oracle_lines(wrong_surfaces)
    wrong_response_text = "\n".join(wrong_lines)

    slot_rows: list[dict[str, Any]] = []
    for index, (bit, surface, line) in enumerate(zip(bits, surfaces, lines), start=1):
        bit_role = "payload" if index <= 8 else "checksum"
        slot_rows.append(
            {
                "schema_name": "natural_evidence_v2_wp4_prompt_local_oracle_slot_v1",
                "slot_index": index,
                "step_label": f"Step {index}:",
                "bit_role": bit_role,
                "bit_index_within_role": (index - 1) if bit_role == "payload" else (index - 9),
                "target_bit": bit,
                "target_bucket_id": str(bit),
                "selected_surface": surface,
                "bucket_surfaces": list(buckets[str(bit)]),
                "oracle_line": line,
                "model_generation_started": False,
                "training_started": False,
                "e2e_eval_started": False,
                "paper_claim_allowed": False,
            }
        )

    response_rows = [
        {
            "schema_name": "natural_evidence_v2_wp4_prompt_local_oracle_response_v1",
            "oracle_response_id": "correct_payload_oracle_substitution",
            "payload_byte_hex": f"{payload_byte:02x}",
            "checksum_byte_hex": str(contract["payload"]["checksum_byte_hex"]),  # type: ignore[index]
            "response_text": response_text,
            "response_text_sha256": sha256_text(response_text),
            "model_generation_started": False,
        },
        {
            "schema_name": "natural_evidence_v2_wp4_prompt_local_oracle_response_v1",
            "oracle_response_id": "wrong_payload_oracle_substitution",
            "payload_byte_hex": f"{wrong_payload_byte:02x}",
            "checksum_byte_hex": f"{wrong_checksum:02x}",
            "response_text": wrong_response_text,
            "response_text_sha256": sha256_text(wrong_response_text),
            "model_generation_started": False,
        },
    ]

    case_specs = [
        {
            "oracle_case_id": "correct_key_correct_payload_accept",
            "oracle_response_id": "correct_payload_oracle_substitution",
            "response_text": response_text,
            "expected_payload_byte": payload_byte,
            "audit_key_id": audit_key_id,
            "expected_accepted": True,
            "oracle_family": "accept",
        },
        {
            "oracle_case_id": "wrong_key_reject",
            "oracle_response_id": "correct_payload_oracle_substitution",
            "response_text": response_text,
            "expected_payload_byte": payload_byte,
            "audit_key_id": wrong_audit_key_id,
            "expected_accepted": False,
            "oracle_family": "wrong_key",
        },
        {
            "oracle_case_id": "wrong_expected_payload_reject",
            "oracle_response_id": "correct_payload_oracle_substitution",
            "response_text": response_text,
            "expected_payload_byte": wrong_payload_byte,
            "audit_key_id": audit_key_id,
            "expected_accepted": False,
            "oracle_family": "wrong_payload",
        },
        {
            "oracle_case_id": "wrong_payload_substitution_reject",
            "oracle_response_id": "wrong_payload_oracle_substitution",
            "response_text": wrong_response_text,
            "expected_payload_byte": payload_byte,
            "audit_key_id": audit_key_id,
            "expected_accepted": False,
            "oracle_family": "wrong_payload",
        },
    ]

    decode_rows: list[dict[str, Any]] = []
    for spec in case_specs:
        decoded = decode_response(
            response_text=str(spec["response_text"]),
            buckets=buckets,
            expected_payload_byte=int(spec["expected_payload_byte"]),
            audit_key_id=str(spec["audit_key_id"]),
        )
        passed = bool(decoded["accepted"]) is bool(spec["expected_accepted"])
        decode_rows.append(
            {
                "schema_name": "natural_evidence_v2_wp4_prompt_local_oracle_decode_case_v1",
                "oracle_case_id": str(spec["oracle_case_id"]),
                "oracle_response_id": str(spec["oracle_response_id"]),
                "oracle_family": str(spec["oracle_family"]),
                "expected_accepted": bool(spec["expected_accepted"]),
                "actual_accepted": bool(decoded["accepted"]),
                "case_passed": passed,
                "audit_key_id": str(spec["audit_key_id"]),
                **decoded,
                "training_started": False,
                "generation_started": False,
                "e2e_eval_started": False,
                "paper_claim_allowed": False,
                "not_payload_recovery": True,
                "not_full_far": True,
            }
        )

    accept_rows = [row for row in decode_rows if row["oracle_family"] == "accept"]
    wrong_key_rows = [row for row in decode_rows if row["oracle_family"] == "wrong_key"]
    wrong_payload_rows = [row for row in decode_rows if row["oracle_family"] == "wrong_payload"]

    def rate(rows: Sequence[Mapping[str, Any]], predicate: str) -> float:
        if not rows:
            return 0.0
        return sum(1 for row in rows if bool(row[predicate])) / float(len(rows))

    wrong_key_reject_rate = sum(1 for row in wrong_key_rows if not bool(row["actual_accepted"])) / float(len(wrong_key_rows) or 1)
    wrong_payload_reject_rate = sum(1 for row in wrong_payload_rows if not bool(row["actual_accepted"])) / float(len(wrong_payload_rows) or 1)
    accept_rate = rate(accept_rows, "actual_accepted")
    all_cases_passed = all(bool(row["case_passed"]) for row in decode_rows)
    oracle_gate_passed = all_cases_passed and accept_rate == 1.0 and wrong_key_reject_rate == 1.0 and wrong_payload_reject_rate == 1.0
    summary = {
        "schema_name": "natural_evidence_v2_wp4_prompt_local_oracle_summary_v1",
        "status": (
            "PASS_WP4_PROMPT_LOCAL_ORACLE_SUBSTITUTION_ARTIFACT_ONLY"
            if oracle_gate_passed
            else "FAIL_WP4_PROMPT_LOCAL_ORACLE_SUBSTITUTION_ARTIFACT_ONLY"
        ),
        "contract_status": str(contract["status"]),
        "precommit_hash_sha256": str(contract["precommit"]["precommit_hash_sha256"]),  # type: ignore[index]
        "decoder_oracle_accept_rate": accept_rate,
        "wrong_key_oracle_reject_rate": wrong_key_reject_rate,
        "wrong_payload_oracle_reject_rate": wrong_payload_reject_rate,
        "decode_case_count": len(decode_rows),
        "oracle_slot_count": len(slot_rows),
        "all_decode_cases_passed": all_cases_passed,
        "oracle_gate_passed": oracle_gate_passed,
        "training_allowed": False,
        "model_scoring_started": False,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
        "next_allowed_action": (
            "Review WP4 prompt-local contract artifacts before any WP5 "
            "teacher-forced scoring plan. Training remains forbidden."
        ),
    }
    return slot_rows, response_rows, decode_rows, summary


def readme_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP4 Prompt-Local Payload Contract",
            "",
            "Artifact-only prompt-local payload contract and decoder-oracle substitution output.",
            "",
            f"status: `{summary['status']}`",
            f"decoder_oracle_accept_rate: `{summary['decoder_oracle_accept_rate']}`",
            f"wrong_key_oracle_reject_rate: `{summary['wrong_key_oracle_reject_rate']}`",
            f"wrong_payload_oracle_reject_rate: `{summary['wrong_payload_oracle_reject_rate']}`",
            "",
            "This is not training, generation, Qwen E2E, FAR aggregation, payload recovery, or a paper-facing positive claim.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    primary_bank_path = resolve_path(args.primary_bank)
    split_manifest_path = resolve_path(args.split_manifest)
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_yaml(config_path)
    split_manifest = read_json(split_manifest_path)
    bank_rows = read_jsonl(primary_bank_path)
    if len(bank_rows) != 1:
        raise ValueError(f"expected exactly one primary bank row, found {len(bank_rows)}")
    bank = bank_rows[0]
    buckets = validate_primary_bank(config, bank)
    payload_byte = parse_byte_hex(args.payload_byte_hex, label="payload-byte-hex")
    wrong_payload_byte = parse_byte_hex(args.wrong_payload_byte_hex, label="wrong-payload-byte-hex")
    if payload_byte == wrong_payload_byte:
        raise ValueError("wrong payload byte must differ from payload byte")
    checksum = checksum_byte(audit_key_id=str(args.audit_key_id), payload_byte=payload_byte)
    bits = byte_bits(payload_byte) + byte_bits(checksum)
    contract = contract_payload(
        config=config,
        split_manifest=split_manifest,
        split_manifest_path=split_manifest_path,
        primary_bank_path=primary_bank_path,
        bank=bank,
        buckets=buckets,
        payload_byte=payload_byte,
        checksum=checksum,
        bits=bits,
        audit_key_id=str(args.audit_key_id),
    )
    slot_rows, response_rows, decode_rows, summary = build_oracle_artifacts(
        contract=contract,
        buckets=buckets,
        payload_byte=payload_byte,
        wrong_payload_byte=wrong_payload_byte,
        audit_key_id=str(args.audit_key_id),
        wrong_audit_key_id=str(args.wrong_audit_key_id),
    )

    for response in response_rows:
        hits = forbidden_terms_in_text(config, str(response["response_text"]))
        if hits:
            raise ValueError(f"oracle response contains forbidden public surface {hits}: {response['oracle_response_id']}")

    write_json(output_dir / "wp4_prompt_local_payload_contract.json", contract)
    write_jsonl(output_dir / "wp4_prompt_local_oracle_slots.jsonl", slot_rows)
    write_jsonl(output_dir / "wp4_prompt_local_oracle_responses.jsonl", response_rows)
    write_jsonl(output_dir / "wp4_prompt_local_oracle_decode_cases.jsonl", decode_rows)
    write_json(output_dir / "wp4_prompt_local_oracle_summary.json", summary)
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(json.dumps({"status": summary["status"], "output_dir": str(output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()

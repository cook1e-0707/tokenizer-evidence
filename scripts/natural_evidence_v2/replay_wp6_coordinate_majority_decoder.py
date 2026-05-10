from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT_DIR = ROOT / "results/natural_evidence_v2/status/wp6_e2e_eval_852086"
DEFAULT_CONTRACT = (
    ROOT
    / "results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/"
    "wp4_prompt_local_payload_contract.json"
)
CHECKSUM_DOMAIN = "natural_evidence_v2_wp4_prompt_local_checksum_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-only replay for a WP6-R1 repeated-coordinate majority "
            "decoder. This does not generate, train, submit Slurm, aggregate "
            "FAR, or make paper-facing claims."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--wp4-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wrong-audit-key-id", default="KWP4_QWEN_PILOT_WRONG_001")
    parser.add_argument("--wrong-payload-byte-hex", default="5a")
    parser.add_argument("--query-budgets", default="8,16,32,64")
    parser.add_argument("--min-support-at-64", type=int, default=16)
    parser.add_argument("--min-majority-margin-at-64", type=int, default=3)
    parser.add_argument(
        "--precommitted-transcript",
        action="store_true",
        help=(
            "Mark this run as a replacement evaluation where the decoder "
            "contract was written before generation. Without this flag the "
            "summary is labeled as post-hoc artifact replay."
        ),
    )
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="Write only the repeated-coordinate decoder contract and exit before reading observations.",
    )
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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def byte_bits(value: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(7, -1, -1)]


def bits_to_byte(bits: Sequence[int]) -> int:
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


def parse_budgets(raw: str) -> list[int]:
    budgets = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not budgets or sorted(set(budgets)) != budgets:
        raise ValueError("--query-budgets must be sorted unique integers")
    return budgets


def contract_info(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = contract.get("payload", {})
    precommit = contract.get("precommit", {})
    payload_byte = parse_byte_hex(str(payload.get("payload_data_byte_hex", "")))
    checksum = parse_byte_hex(str(payload.get("checksum_byte_hex", "")))
    expected_bits = byte_bits(payload_byte) + byte_bits(checksum)
    return {
        "audit_key_id": str(precommit.get("audit_key_id", "")),
        "checksum_byte": checksum,
        "expected_bits": expected_bits,
        "payload_byte": payload_byte,
        "payload_hex": f"{payload_byte:02x}",
        "precommit_hash_sha256": str(precommit.get("precommit_hash_sha256", "")),
        "source_payload_id": str(payload.get("payload_id", "")),
    }


def decode_specs(info: Mapping[str, Any], *, wrong_audit_key_id: str, wrong_payload_byte: int) -> list[dict[str, Any]]:
    return [
        {
            "decode_condition": "protected",
            "expected_audit_key_id": str(info["audit_key_id"]),
            "expected_payload_byte": int(info["payload_byte"]),
            "source_condition": "protected",
        },
        {
            "decode_condition": "raw",
            "expected_audit_key_id": str(info["audit_key_id"]),
            "expected_payload_byte": int(info["payload_byte"]),
            "source_condition": "raw",
        },
        {
            "decode_condition": "task_only",
            "expected_audit_key_id": str(info["audit_key_id"]),
            "expected_payload_byte": int(info["payload_byte"]),
            "source_condition": "task_only",
        },
        {
            "decode_condition": "wrong_key",
            "expected_audit_key_id": wrong_audit_key_id,
            "expected_payload_byte": int(info["payload_byte"]),
            "source_condition": "protected",
        },
        {
            "decode_condition": "wrong_payload",
            "expected_audit_key_id": str(info["audit_key_id"]),
            "expected_payload_byte": int(wrong_payload_byte),
            "source_condition": "protected",
        },
    ]


def majority_decode(
    *,
    observations: Sequence[Mapping[str, Any]],
    source_condition: str,
    budget: int,
) -> tuple[list[int | None], list[dict[str, Any]]]:
    coord_rows: list[dict[str, Any]] = []
    bits: list[int | None] = []
    for step_index in range(1, 17):
        rows = [
            row
            for row in observations
            if str(row.get("generation_condition", "")) == source_condition
            and str(row.get("decode_condition", "")) == source_condition
            and int(row.get("frame_index", 0)) < budget
            and int(row.get("step_index", 0)) == step_index
            and bool(row.get("resolved_bucket_hit"))
        ]
        counts = Counter(int(row["observed_bucket_id"]) for row in rows)
        if counts:
            ordered = counts.most_common()
            majority_bit = int(ordered[0][0])
            runner_up = int(ordered[1][1]) if len(ordered) > 1 else 0
            margin = int(ordered[0][1]) - runner_up
        else:
            majority_bit = None
            margin = 0
        bits.append(majority_bit)
        coord_rows.append(
            {
                "budget": budget,
                "majority_bit": "" if majority_bit is None else majority_bit,
                "majority_margin": margin,
                "observed_bucket_0_count": int(counts.get(0, 0)),
                "observed_bucket_1_count": int(counts.get(1, 0)),
                "resolved_count": sum(counts.values()),
                "source_condition": source_condition,
                "step_index": step_index,
            }
        )
    return bits, coord_rows


def decision_from_bits(
    *,
    bits: Sequence[int | None],
    expected_audit_key_id: str,
    expected_payload_byte: int,
) -> dict[str, Any]:
    complete = len(bits) == 16 and all(bit in {0, 1} for bit in bits)
    decoded_payload = bits_to_byte([int(bit) for bit in bits[:8]]) if complete else None
    decoded_checksum = bits_to_byte([int(bit) for bit in bits[8:]]) if complete else None
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
        "complete_coordinate_codeword": bool(complete),
        "decoded_checksum_byte_hex": f"{decoded_checksum:02x}" if decoded_checksum is not None else "",
        "decoded_payload_byte_hex": f"{decoded_payload:02x}" if decoded_payload is not None else "",
        "expected_checksum_byte_hex": f"{expected_checksum:02x}" if expected_checksum is not None else "",
        "expected_payload_byte_hex": f"{expected_payload_byte:02x}",
        "majority_bits": ["" if bit is None else int(bit) for bit in bits],
        "majority_hex": (
            f"{decoded_payload:02x}{decoded_checksum:02x}"
            if decoded_payload is not None and decoded_checksum is not None
            else ""
        ),
        "payload_matches": bool(complete and decoded_payload == expected_payload_byte),
    }


def build_contract(
    *,
    args: argparse.Namespace,
    input_dir: Path,
    contract_path: Path,
    info: Mapping[str, Any],
    budgets: Sequence[int],
) -> dict[str, Any]:
    precommitted_transcript = bool(getattr(args, "precommitted_transcript", False))
    return {
        "artifact_only_replay": True,
        "claim_control": {
            "e2e_rerun_started": False,
            "far_aggregation_started": False,
            "paper_claim_allowed": False,
            "training_started": False,
        },
        "coordinate_policy": {
            "coordinate_count": 16,
            "coordinate_id": "strict_step_label_index_1_to_16",
            "evidence_unit": "resolved primary 2-way bucket hit at Step N first word",
            "erasure_policy": "ignore unresolved out-of-bank first words",
        },
        "decoder_policy": {
            "accept_rule": "majority codeword checksum_valid_and_payload_matches_expected",
            "decoder_id": "qwen_v2_wp6_r1_repeated_coordinate_majority_decoder_v1",
            "minimum_majority_margin_at_64": int(args.min_majority_margin_at_64),
            "minimum_support_at_64": int(args.min_support_at_64),
            "query_budgets": list(budgets),
        },
        "input_dir": str(input_dir),
        "schema_name": "natural_evidence_v2_wp6_r1_coordinate_majority_contract_v1",
        "source_wp4_contract_path": str(contract_path),
        "source_wp4_contract_sha256": sha256_file(contract_path),
        "source_wp4_precommit_hash_sha256": str(info.get("precommit_hash_sha256", "")),
        "transcript_precommitted_before_generation": precommitted_transcript,
        "transcript_provenance": (
            "precommitted_replacement_run" if precommitted_transcript else "post_hoc_artifact_replay"
        ),
        "precommit_note": (
            "This replacement WP6-R1 contract is intended to be written before generation."
            if precommitted_transcript
            else (
                "This artifact-only replay is post-hoc repair evidence and cannot "
                "retroactively precommit an already generated transcript."
            )
        ),
    }


def replay(
    *,
    args: argparse.Namespace,
    observations: Sequence[Mapping[str, Any]],
    info: Mapping[str, Any],
    budgets: Sequence[int],
    wrong_payload_byte: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    precommitted_transcript = bool(getattr(args, "precommitted_transcript", False))
    coord_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    for budget in budgets:
        cached: dict[str, tuple[list[int | None], list[dict[str, Any]]]] = {}
        for spec in decode_specs(info, wrong_audit_key_id=args.wrong_audit_key_id, wrong_payload_byte=wrong_payload_byte):
            source = str(spec["source_condition"])
            if source not in cached:
                cached[source] = majority_decode(
                    observations=observations,
                    source_condition=source,
                    budget=budget,
                )
                for row in cached[source][1]:
                    coord_rows.append(row)
            bits = cached[source][0]
            decision = decision_from_bits(
                bits=bits,
                expected_audit_key_id=str(spec["expected_audit_key_id"]),
                expected_payload_byte=int(spec["expected_payload_byte"]),
            )
            support = [int(row["resolved_count"]) for row in cached[source][1]]
            margins = [int(row["majority_margin"]) for row in cached[source][1]]
            decode_rows.append(
                {
                    **decision,
                    "budget": budget,
                    "decode_condition": str(spec["decode_condition"]),
                    "min_majority_margin": min(margins) if margins else 0,
                    "min_support": min(support) if support else 0,
                    "schema_name": "natural_evidence_v2_wp6_r1_coordinate_majority_decode_v1",
                    "source_condition": source,
                }
            )
    summary_by_budget: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in decode_rows:
        summary_by_budget[str(row["budget"])][str(row["decode_condition"])] = {
            "accepted": bool(row["accepted"]),
            "decoded_hex": str(row["majority_hex"]),
            "min_majority_margin": int(row["min_majority_margin"]),
            "min_support": int(row["min_support"]),
        }
    protected64 = summary_by_budget.get("64", {}).get("protected", {})
    null64 = [
        summary_by_budget.get("64", {}).get(condition, {}).get("accepted", False)
        for condition in ("raw", "task_only", "wrong_key", "wrong_payload")
    ]
    replay_gate_pass = (
        bool(protected64.get("accepted"))
        and not any(bool(item) for item in null64)
        and int(protected64.get("min_support", 0)) >= int(args.min_support_at_64)
        and int(protected64.get("min_majority_margin", 0)) >= int(args.min_majority_margin_at_64)
    )
    if precommitted_transcript:
        pass_status = "PASS_WP6_R1_COORDINATE_MAJORITY_E2E_REPLACEMENT_RUN"
        fail_status = "FAIL_WP6_R1_COORDINATE_MAJORITY_E2E_REPLACEMENT_RUN"
    else:
        pass_status = "PASS_WP6_R1_COORDINATE_MAJORITY_REPLAY_READY_FOR_REPLACEMENT_PREFLIGHT"
        fail_status = "FAIL_WP6_R1_COORDINATE_MAJORITY_REPLAY"

    summary = {
        "artifact_role": (
            "wp6_r1_coordinate_majority_e2e_replacement_summary"
            if precommitted_transcript
            else "wp6_r1_coordinate_majority_replay_summary"
        ),
        "claim_control": {
            "e2e_rerun_allowed": bool(replay_gate_pass and not precommitted_transcript),
            "far_aggregation_allowed": False,
            "paper_claim_allowed": False,
            "scale_rerun_allowed": False,
            "training_allowed": False,
        },
        "post_hoc_artifact_replay": not precommitted_transcript,
        "precommitted_transcript": precommitted_transcript,
        "replay_gate_pass": bool(replay_gate_pass),
        "replay_gate_status": pass_status if replay_gate_pass else fail_status,
        "replacement_run_gate_pass": bool(replay_gate_pass and precommitted_transcript),
        "transcript_provenance": (
            "precommitted_replacement_run" if precommitted_transcript else "post_hoc_artifact_replay"
        ),
        "summary_by_budget": dict(summary_by_budget),
    }
    return coord_rows, decode_rows, summary


def main() -> int:
    args = parse_args()
    input_dir = resolve(args.input_dir)
    contract_path = resolve(args.wp4_contract)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    budgets = parse_budgets(args.query_budgets)
    info = contract_info(read_json(contract_path))
    wrong_payload_byte = parse_byte_hex(args.wrong_payload_byte_hex)

    contract = build_contract(
        args=args,
        input_dir=input_dir,
        contract_path=contract_path,
        info=info,
        budgets=budgets,
    )
    if args.contract_only:
        write_json(output_dir / "wp6_r1_coordinate_majority_contract.json", contract)
        return 0

    observations = read_jsonl(input_dir / "wp6_slot_observations.jsonl")
    coord_rows, decode_rows, summary = replay(
        args=args,
        observations=observations,
        info=info,
        budgets=budgets,
        wrong_payload_byte=wrong_payload_byte,
    )

    write_json(output_dir / "wp6_r1_coordinate_majority_contract.json", contract)
    write_jsonl(output_dir / "wp6_r1_coordinate_majority_decode_rows.jsonl", decode_rows)
    write_csv(
        output_dir / "wp6_r1_coordinate_support_by_budget.csv",
        coord_rows,
        [
            "budget",
            "source_condition",
            "step_index",
            "observed_bucket_0_count",
            "observed_bucket_1_count",
            "resolved_count",
            "majority_bit",
            "majority_margin",
        ],
    )
    write_json(output_dir / "wp6_r1_coordinate_majority_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

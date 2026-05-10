from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMARY_BANK = ROOT / "results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank.jsonl"
DEFAULT_PROMPTS = (
    ROOT
    / "results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/"
    "restricted_step_label_strict_density_audit_prompts.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/natural_evidence_v2/contracts/wp4_prompt_local_contract_20260509_0610"

PROTOCOL_ID = "natural_evidence_v2_controlled_micro_slots_wp4_prompt_local_v1"
CORRECT_AUDIT_KEY_ID = "qwen_v2_wp4_audit_key_seeded_v1"
PAYLOAD_SPECS = {
    "P00": 0x42,
    "P01": 0xA7,
}
SEEDS = (17, 23)
QUERY_BUDGETS = (8, 16, 32, 64)
FRAME_SLOT_COUNT = 16


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def stable_hash_hex(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_byte(*parts: object) -> int:
    text = "||".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(text).digest()[0]


def checksum_byte(*, payload_id: str, payload_byte: int, audit_key_id: str) -> int:
    return hash_byte(PROTOCOL_ID, "checksum8", payload_id, payload_byte, audit_key_id)


def byte_to_bits(value: int) -> list[int]:
    if value < 0 or value > 255:
        raise ValueError("byte value out of range")
    return [(value >> shift) & 1 for shift in range(7, -1, -1)]


def bits_to_byte(bits: list[int]) -> int:
    if len(bits) != 8:
        raise ValueError("expected exactly 8 bits")
    value = 0
    for bit in bits:
        if bit not in {0, 1}:
            raise ValueError("bit out of range")
        value = (value << 1) | int(bit)
    return value


def key_mask_bit(*, audit_key_id: str, payload_id: str, seed: int, prompt_id: str, slot_index: int) -> int:
    return hash_byte(PROTOCOL_ID, "mask", audit_key_id, payload_id, seed, prompt_id, slot_index) & 1


def codeword_bits(*, payload_id: str, payload_byte: int, audit_key_id: str) -> list[int]:
    check = checksum_byte(payload_id=payload_id, payload_byte=payload_byte, audit_key_id=audit_key_id)
    return byte_to_bits(payload_byte) + byte_to_bits(check)


def decode_frame(
    *,
    observed_digits: list[int],
    payload_id: str,
    seed: int,
    prompt_id: str,
    audit_key_id: str,
    expected_payload_byte: int,
) -> dict[str, Any]:
    if len(observed_digits) != FRAME_SLOT_COUNT:
        return {
            "accepted": False,
            "decode_status": "insufficient_symbols",
            "decoded_payload_byte": None,
            "decoded_checksum_byte": None,
            "checksum_ok": False,
            "payload_ok": False,
        }
    unmasked_bits: list[int] = []
    for slot_index, digit in enumerate(observed_digits):
        if digit not in {0, 1}:
            return {
                "accepted": False,
                "decode_status": "invalid_digit",
                "decoded_payload_byte": None,
                "decoded_checksum_byte": None,
                "checksum_ok": False,
                "payload_ok": False,
            }
        mask = key_mask_bit(
            audit_key_id=audit_key_id,
            payload_id=payload_id,
            seed=seed,
            prompt_id=prompt_id,
            slot_index=slot_index,
        )
        unmasked_bits.append(int(digit) ^ mask)
    decoded_payload = bits_to_byte(unmasked_bits[:8])
    decoded_checksum = bits_to_byte(unmasked_bits[8:])
    expected_checksum = checksum_byte(
        payload_id=payload_id,
        payload_byte=expected_payload_byte,
        audit_key_id=audit_key_id,
    )
    checksum_ok = decoded_checksum == expected_checksum
    payload_ok = decoded_payload == expected_payload_byte
    accepted = checksum_ok and payload_ok
    return {
        "accepted": accepted,
        "decode_status": "decoded_frame_accept" if accepted else "decoded_frame_reject",
        "decoded_payload_byte": decoded_payload,
        "decoded_checksum_byte": decoded_checksum,
        "expected_checksum_byte": expected_checksum,
        "checksum_ok": checksum_ok,
        "payload_ok": payload_ok,
    }


def select_prompts(prompts: list[dict[str, Any]], *, payload_id: str, seed: int, count: int = 64) -> list[dict[str, Any]]:
    eval_prompts = [row for row in prompts if row.get("split") == "wp3_r1_eval"]
    if len(eval_prompts) < count:
        raise ValueError(f"need at least {count} eval prompts")
    span = len(eval_prompts) - count + 1
    start = int(hashlib.sha256(f"{payload_id}:{seed}".encode("utf-8")).hexdigest()[:8], 16) % span
    return eval_prompts[start : start + count]


def target_digits_for_prompt(
    *,
    payload_id: str,
    payload_byte: int,
    seed: int,
    prompt_id: str,
    audit_key_id: str,
) -> list[int]:
    bits = codeword_bits(payload_id=payload_id, payload_byte=payload_byte, audit_key_id=audit_key_id)
    digits: list[int] = []
    for slot_index, bit in enumerate(bits):
        mask = key_mask_bit(
            audit_key_id=audit_key_id,
            payload_id=payload_id,
            seed=seed,
            prompt_id=prompt_id,
            slot_index=slot_index,
        )
        digits.append(bit ^ mask)
    return digits


def build_contracts(primary_bank: dict[str, Any], prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for payload_id, payload_byte in PAYLOAD_SPECS.items():
        for seed in SEEDS:
            selected = select_prompts(prompts, payload_id=payload_id, seed=seed, count=max(QUERY_BUDGETS))
            frames: list[dict[str, Any]] = []
            for frame_index, prompt in enumerate(selected):
                prompt_id = str(prompt["prompt_id"])
                digits = target_digits_for_prompt(
                    payload_id=payload_id,
                    payload_byte=payload_byte,
                    seed=seed,
                    prompt_id=prompt_id,
                    audit_key_id=CORRECT_AUDIT_KEY_ID,
                )
                slots = []
                for slot_index, digit in enumerate(digits):
                    slots.append(
                        {
                            "slot_index": slot_index,
                            "step_label": f"Step {slot_index + 1}:",
                            "coordinate": f"{prompt_id}:step_{slot_index + 1}",
                            "target_digit": digit,
                            "target_bucket": str(digit),
                            "bucket_surfaces": primary_bank[f"bucket_{digit}_surfaces"],
                            "bucket_token_ids": primary_bank[f"bucket_{digit}_token_ids"],
                        }
                    )
                frames.append(
                    {
                        "frame_index": frame_index,
                        "prompt_id": prompt_id,
                        "prompt_text_sha256": prompt.get("prompt_text_sha256"),
                        "split": prompt.get("split"),
                        "variant_id": prompt.get("variant_id"),
                        "target_digits": digits,
                        "slots": slots,
                    }
                )
            checksum = checksum_byte(
                payload_id=payload_id,
                payload_byte=payload_byte,
                audit_key_id=CORRECT_AUDIT_KEY_ID,
            )
            contract_core = {
                "protocol_id": PROTOCOL_ID,
                "audit_key_id": CORRECT_AUDIT_KEY_ID,
                "payload_id": payload_id,
                "payload_byte": payload_byte,
                "seed": seed,
                "checksum_byte": checksum,
                "payload_bits": byte_to_bits(payload_byte),
                "checksum_bits": byte_to_bits(checksum),
                "frame_policy": "prompt_local",
                "slot_count": FRAME_SLOT_COUNT,
                "query_budgets": list(QUERY_BUDGETS),
                "primary_bank_id": primary_bank["bank_id"],
                "source_primary_bank": primary_bank,
                "frames": frames,
            }
            precommit_payload = {
                "protocol_id": PROTOCOL_ID,
                "audit_key_id": CORRECT_AUDIT_KEY_ID,
                "payload_id": payload_id,
                "payload_byte": payload_byte,
                "prompt_ids": [frame["prompt_id"] for frame in frames],
                "slot_policy": "line_start_step_label_16_action_verb_v1",
                "bucket_policy": primary_bank["bank_id"],
                "decoder": "prompt_local_hash8_checksum_keyed_mask_v1",
                "query_budgets": list(QUERY_BUDGETS),
                "eval_split": "wp3_r1_eval",
            }
            contract_core["precommit_hash"] = stable_hash_hex(precommit_payload)
            contract_core["schema_name"] = "natural_evidence_v2_wp4_prompt_local_contract_v1"
            contract_core["training_started"] = False
            contract_core["generation_started"] = False
            contract_core["e2e_eval_started"] = False
            contract_core["paper_claim_allowed"] = False
            contract_core["not_payload_recovery"] = True
            contract_core["not_full_far"] = True
            contracts.append(contract_core)
    return contracts


def choose_wrong_key_id(contracts: list[dict[str, Any]]) -> str:
    for index in range(64):
        wrong_key = f"qwen_v2_wp4_wrong_key_{index:02d}"
        any_accept = False
        for contract in contracts:
            for frame in contract["frames"]:
                result = decode_frame(
                    observed_digits=list(frame["target_digits"]),
                    payload_id=contract["payload_id"],
                    seed=int(contract["frames"][0]["frame_index"] * 0 + contract.get("seed", 0)),
                    prompt_id=frame["prompt_id"],
                    audit_key_id=wrong_key,
                    expected_payload_byte=int(contract["payload_byte"]),
                )
                if result["accepted"]:
                    any_accept = True
                    break
            if any_accept:
                break
        if not any_accept:
            return wrong_key
    raise ValueError("could not find deterministic wrong key with zero oracle accepts")


def run_oracle(contracts: list[dict[str, Any]], *, wrong_key_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_rows: list[dict[str, Any]] = []
    for contract in contracts:
        payload_id = str(contract["payload_id"])
        payload_byte = int(contract["payload_byte"])
        seed = int(contract["seed"])
        other_payload_id = next(pid for pid in PAYLOAD_SPECS if pid != payload_id)
        other_payload_byte = PAYLOAD_SPECS[other_payload_id]
        for budget in QUERY_BUDGETS:
            frames = contract["frames"][:budget]
            scenarios = [
                ("target_oracle", CORRECT_AUDIT_KEY_ID, payload_id, payload_byte),
                ("wrong_key_oracle", wrong_key_id, payload_id, payload_byte),
                ("wrong_payload_oracle", CORRECT_AUDIT_KEY_ID, other_payload_id, other_payload_byte),
            ]
            for scenario, audit_key_id, expected_payload_id, expected_payload_byte in scenarios:
                accepted_frames = 0
                first_status = ""
                for frame in frames:
                    result = decode_frame(
                        observed_digits=list(frame["target_digits"]),
                        payload_id=payload_id,
                        seed=seed,
                        prompt_id=frame["prompt_id"],
                        audit_key_id=audit_key_id,
                        expected_payload_byte=expected_payload_byte,
                    )
                    if not first_status:
                        first_status = str(result["decode_status"])
                    if result["accepted"]:
                        accepted_frames += 1
                accepted = accepted_frames > 0
                trace_rows.append(
                    {
                        "contract_id": contract["contract_id"],
                        "payload_id": payload_id,
                        "seed": seed,
                        "scenario": scenario,
                        "audit_key_id": audit_key_id,
                        "expected_payload_id": expected_payload_id,
                        "query_budget": budget,
                        "scheduled_frames": len(frames),
                        "accepted": accepted,
                        "accepted_frame_count": accepted_frames,
                        "decode_status": "decoded_frame_accept" if accepted else first_status or "decoded_frame_reject",
                        "result_claim": "wp4_decoder_oracle_not_payload_recovery_not_far",
                    }
                )
    target_rows = [row for row in trace_rows if row["scenario"] == "target_oracle"]
    wrong_key_rows = [row for row in trace_rows if row["scenario"] == "wrong_key_oracle"]
    wrong_payload_rows = [row for row in trace_rows if row["scenario"] == "wrong_payload_oracle"]
    summary = {
        "schema_name": "natural_evidence_v2_wp4_decoder_oracle_summary_v1",
        "status": (
            "PASS_WP4_PROMPT_LOCAL_DECODER_ORACLE"
            if all(row["accepted"] for row in target_rows)
            and not any(row["accepted"] for row in wrong_key_rows)
            and not any(row["accepted"] for row in wrong_payload_rows)
            else "FAIL_WP4_PROMPT_LOCAL_DECODER_ORACLE"
        ),
        "contract_count": len(contracts),
        "payload_ids": sorted(PAYLOAD_SPECS),
        "seeds": list(SEEDS),
        "query_budgets": list(QUERY_BUDGETS),
        "target_oracle_rows": len(target_rows),
        "target_oracle_accept_rows": sum(1 for row in target_rows if row["accepted"]),
        "target_oracle_accept_rate": sum(1 for row in target_rows if row["accepted"]) / len(target_rows),
        "wrong_key_oracle_rows": len(wrong_key_rows),
        "wrong_key_oracle_accept_rows": sum(1 for row in wrong_key_rows if row["accepted"]),
        "wrong_payload_oracle_rows": len(wrong_payload_rows),
        "wrong_payload_oracle_accept_rows": sum(1 for row in wrong_payload_rows if row["accepted"]),
        "wrong_key_id": wrong_key_id,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "not_payload_recovery": True,
        "not_full_far": True,
    }
    return trace_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v2 WP4 prompt-local payload contracts and decoder oracle.")
    parser.add_argument("--primary-bank", type=Path, default=DEFAULT_PRIMARY_BANK)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    primary_rows = read_jsonl(args.primary_bank)
    if len(primary_rows) != 1:
        raise ValueError("expected exactly one primary bank row")
    primary_bank = primary_rows[0]
    prompts = read_jsonl(args.prompts)
    contracts = build_contracts(primary_bank, prompts)
    # Assign contract ids after stable content is present.
    for contract in contracts:
        contract["contract_id"] = (
            f"qwen_v2_wp4_{contract['payload_id']}_seed{contract['seed']}_prompt_local_16slot"
        )
    wrong_key_id = choose_wrong_key_id(contracts)
    trace_rows, oracle_summary = run_oracle(contracts, wrong_key_id=wrong_key_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "qwen_v2_wp4_prompt_local_contracts.jsonl", contracts)
    for contract in contracts:
        write_json(args.output_dir / f"{contract['contract_id']}.json", contract)
    trace_csv = args.output_dir / "qwen_v2_wp4_decoder_oracle_trace.csv"
    with trace_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)
    oracle_summary.update(
        {
            "contracts_jsonl": str(args.output_dir / "qwen_v2_wp4_prompt_local_contracts.jsonl"),
            "decoder_oracle_trace_csv": str(trace_csv),
            "primary_bank_jsonl": str(args.primary_bank),
            "prompt_source_jsonl": str(args.prompts),
            "wp5_allowed": False,
            "next_allowed_action": (
                "Review WP4 decoder oracle. If it passes, prepare WP5 teacher-forced "
                "target-mass gate plan only; training still requires separate gate review."
            ),
        }
    )
    write_json(args.output_dir / "qwen_v2_wp4_decoder_oracle_summary.json", oracle_summary)
    manifest = {
        "schema_name": "natural_evidence_v2_wp4_prompt_local_manifest_v1",
        "status": oracle_summary["status"],
        "output_dir": str(args.output_dir),
        "primary_bank_jsonl": str(args.primary_bank),
        "prompt_source_jsonl": str(args.prompts),
        "contract_count": len(contracts),
        "trace_rows": len(trace_rows),
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }
    write_json(args.output_dir / "qwen_v2_wp4_contract_manifest.json", manifest)
    print(json.dumps(oracle_summary, indent=2, sort_keys=True))
    return 0 if oracle_summary["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())

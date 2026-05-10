from __future__ import annotations

from scripts.natural_evidence_v2.decode_wp6_payload import decode_rows, load_contract, summarize_decisions


def _contract() -> dict:
    return {
        "bucket_bank": {"buckets": {"0": ["Set", "Plan"], "1": ["Create", "Prepare"]}},
        "payload": {
            "checksum_bits_msb_first": [0, 1, 0, 1, 1, 1, 1, 0],
            "checksum_byte_hex": "5e",
            "payload_bits_msb_first": [1, 0, 1, 0, 0, 1, 0, 1],
            "payload_data_byte_hex": "a5",
            "payload_id": "wp4_payload_a5_checksum_5e",
        },
        "precommit": {
            "audit_key_id": "KWP4_QWEN_PILOT_001",
            "bucket_policy_id": "test_bank",
            "precommit_hash_sha256": "abc",
            "query_budgets": [64],
        },
        "schema_name": "natural_evidence_v2_wp4_prompt_local_payload_contract_v1",
    }


def _response_for_bits(bits: list[int]) -> str:
    surfaces = {0: "Set", 1: "Create"}
    return "\n".join(
        f"Step {index}: {surfaces[int(bit)]} a plain natural action."
        for index, bit in enumerate(bits, start=1)
    )


def test_wp6_decode_accepts_trained_payload_and_rejects_controls(tmp_path) -> None:
    contract_info = load_contract(_contract())
    target_bits = list(contract_info["expected_bits"])
    generated = [
        {
            "generation_condition": "protected",
            "generation_id": "g_protected",
            "prompt_id": "p0",
            "prompt_index": 0,
            "response_text": _response_for_bits(target_bits),
        },
        {
            "generation_condition": "raw",
            "generation_id": "g_raw",
            "prompt_id": "p0",
            "prompt_index": 0,
            "response_text": _response_for_bits([0] * 16),
        },
        {
            "generation_condition": "task_only",
            "generation_id": "g_task",
            "prompt_id": "p0",
            "prompt_index": 0,
            "response_text": _response_for_bits([0] * 16),
        },
    ]

    observation_rows, decision_rows = decode_rows(
        generated_rows=generated,
        contract_info=contract_info,
        config={"forbidden_surface_terms": ["FIELD=", "PAYLOAD", "CERT", "EVIDENCE", "OWNER"]},
        wrong_audit_key_id="KWP4_QWEN_PILOT_WRONG_001",
        wrong_payload_byte=0x5A,
    )

    by_condition = {row["decode_condition"]: row for row in decision_rows}
    assert by_condition["protected"]["accepted"] is True
    assert by_condition["raw"]["accepted"] is False
    assert by_condition["task_only"]["accepted"] is False
    assert by_condition["wrong_key"]["accepted"] is False
    assert by_condition["wrong_payload"]["accepted"] is False
    assert len(observation_rows) == 5 * 16


def test_wp6_summary_requires_protected_accept_and_zero_nulls(tmp_path) -> None:
    contract_info = load_contract(_contract())
    target_bits = list(contract_info["expected_bits"])
    generated = [
        {
            "generation_condition": "protected",
            "generation_id": "g_protected",
            "prompt_id": "p0",
            "prompt_index": 0,
            "response_text": _response_for_bits(target_bits),
        },
        {
            "generation_condition": "raw",
            "generation_id": "g_raw",
            "prompt_id": "p0",
            "prompt_index": 0,
            "response_text": _response_for_bits([0] * 16),
        },
        {
            "generation_condition": "task_only",
            "generation_id": "g_task",
            "prompt_id": "p0",
            "prompt_index": 0,
            "response_text": _response_for_bits([0] * 16),
        },
    ]
    observation_rows, decision_rows = decode_rows(
        generated_rows=generated,
        contract_info=contract_info,
        config={"forbidden_surface_terms": []},
        wrong_audit_key_id="KWP4_QWEN_PILOT_WRONG_001",
        wrong_payload_byte=0x5A,
    )

    class Args:
        min_protected_recovery_at_64 = 0.80
        min_slot_detection_rate = 0.70
        min_target_hit_rate = 0.25

    generated_path = tmp_path / "generated.jsonl"
    contract_path = tmp_path / "contract.json"
    generated_path.write_text("{}\n", encoding="utf-8")
    contract_path.write_text("{}\n", encoding="utf-8")
    summary = summarize_decisions(
        args=Args(),
        contract_path=contract_path,
        generated_path=generated_path,
        contract_info=contract_info,
        decision_rows=decision_rows,
        observation_rows=observation_rows,
    )

    assert summary["gate_status"] == "PASS_WP6_QWEN_V2_E2E_PROOF_OF_LIFE"
    assert summary["protected_accept_rate_at_64"] == 1.0

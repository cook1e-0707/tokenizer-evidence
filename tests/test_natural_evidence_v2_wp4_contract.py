from __future__ import annotations

from scripts.natural_evidence_v2.build_wp4_prompt_local_contract import (
    PAYLOAD_SPECS,
    QUERY_BUDGETS,
    SEEDS,
    build_contracts,
    choose_wrong_key_id,
    run_oracle,
)


def test_wp4_prompt_local_contract_oracle_accepts_targets_and_rejects_nulls() -> None:
    primary_bank = {
        "bank_id": "qwen_v2_wp3_r2_primary_set_plan_vs_create_prepare_v1",
        "bucket_0_surfaces": ["Set", "Plan"],
        "bucket_1_surfaces": ["Create", "Prepare"],
        "bucket_0_token_ids": [2573, 9680],
        "bucket_1_token_ids": [4230, 31166],
    }
    prompts = [
        {
            "prompt_id": f"eval_{index}",
            "prompt_text_sha256": f"sha_{index}",
            "split": "wp3_r1_eval",
            "variant_id": "strict_literal_16_step_lines",
        }
        for index in range(80)
    ]

    contracts = build_contracts(primary_bank, prompts)
    for contract in contracts:
        contract["contract_id"] = (
            f"qwen_v2_wp4_{contract['payload_id']}_seed{contract['seed']}_prompt_local_16slot"
        )

    assert len(contracts) == len(PAYLOAD_SPECS) * len(SEEDS)
    assert all(len(contract["frames"]) == max(QUERY_BUDGETS) for contract in contracts)
    assert all(len(frame["slots"]) == 16 for contract in contracts for frame in contract["frames"])

    wrong_key_id = choose_wrong_key_id(contracts)
    trace_rows, summary = run_oracle(contracts, wrong_key_id=wrong_key_id)

    assert summary["status"] == "PASS_WP4_PROMPT_LOCAL_DECODER_ORACLE"
    assert summary["target_oracle_accept_rate"] == 1.0
    assert summary["wrong_key_oracle_accept_rows"] == 0
    assert summary["wrong_payload_oracle_accept_rows"] == 0
    assert len(trace_rows) == len(contracts) * len(QUERY_BUDGETS) * 3
    assert all(row["result_claim"] == "wp4_decoder_oracle_not_payload_recovery_not_far" for row in trace_rows)

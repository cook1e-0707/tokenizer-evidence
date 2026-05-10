from __future__ import annotations

from scripts.natural_evidence_v2.decode_wp6_r1_scale_blocks import (
    DEFAULT_PROMPTS,
    build_prompt_plan,
    build_scale_contract,
    decode_scale_blocks,
    read_prompt_rows_with_file_index,
    summarize_scale,
)
from scripts.natural_evidence_v2.generate_wp6_e2e_outputs import (
    read_jsonl_with_file_index,
    select_prompts,
)
from scripts.natural_evidence_v2.replay_wp6_coordinate_majority_decoder import contract_info, replay


def _contract() -> dict:
    return {
        "payload": {
            "checksum_byte_hex": "5e",
            "payload_data_byte_hex": "a5",
            "payload_id": "wp4_payload_a5_checksum_5e",
        },
        "precommit": {
            "audit_key_id": "KWP4_QWEN_PILOT_001",
            "precommit_hash_sha256": "abc",
        },
    }


def _obs(condition: str, bits: list[int], frames: int = 4) -> list[dict]:
    rows = []
    for frame in range(frames):
        for step, bit in enumerate(bits, start=1):
            rows.append(
                {
                    "decode_condition": condition,
                    "frame_index": frame,
                    "generation_condition": condition,
                    "observed_bucket_id": bit,
                    "resolved_bucket_hit": True,
                    "step_index": step,
                }
            )
    return rows


def _obs_frames(condition: str, frame_bits: list[list[int]]) -> list[dict]:
    rows = []
    for frame, bits in enumerate(frame_bits):
        for step, bit in enumerate(bits, start=1):
            rows.append(
                {
                    "decode_condition": condition,
                    "frame_index": frame,
                    "generation_condition": condition,
                    "observed_bucket_id": bit,
                    "resolved_bucket_hit": True,
                    "step_index": step,
                }
            )
    return rows


def test_coordinate_majority_replay_accepts_protected_and_rejects_controls() -> None:
    info = contract_info(_contract())
    observations = []
    observations.extend(_obs("protected", list(info["expected_bits"])))
    observations.extend(_obs("raw", [0] * 16))
    observations.extend(_obs("task_only", [0] * 16))

    class Args:
        min_majority_margin_at_64 = 1
        min_support_at_64 = 1
        wrong_audit_key_id = "KWP4_QWEN_PILOT_WRONG_001"

    _coord_rows, decode_rows, summary = replay(
        args=Args(),
        observations=observations,
        info=info,
        budgets=[4, 64],
        wrong_payload_byte=0x5A,
    )

    rows64 = {row["decode_condition"]: row for row in decode_rows if row["budget"] == 64}
    assert rows64["protected"]["accepted"] is True
    assert rows64["protected"]["majority_hex"] == "a55e"
    assert rows64["raw"]["accepted"] is False
    assert rows64["task_only"]["accepted"] is False
    assert rows64["wrong_key"]["accepted"] is False
    assert rows64["wrong_payload"]["accepted"] is False
    assert summary["replay_gate_pass"] is True
    assert summary["post_hoc_artifact_replay"] is True
    assert "post_hoc_not_precommitted_for_852086" not in summary


def test_coordinate_majority_replay_marks_precommitted_replacement() -> None:
    info = contract_info(_contract())
    observations = _obs("protected", list(info["expected_bits"]))
    observations.extend(_obs("raw", [0] * 16))
    observations.extend(_obs("task_only", [0] * 16))

    class Args:
        min_majority_margin_at_64 = 1
        min_support_at_64 = 1
        precommitted_transcript = True
        wrong_audit_key_id = "KWP4_QWEN_PILOT_WRONG_001"

    _coord_rows, decode_rows, summary = replay(
        args=Args(),
        observations=observations,
        info=info,
        budgets=[64],
        wrong_payload_byte=0x5A,
    )

    rows64 = {row["decode_condition"]: row for row in decode_rows if row["budget"] == 64}
    assert rows64["protected"]["accepted"] is True
    assert rows64["raw"]["accepted"] is False
    assert summary["replacement_run_gate_pass"] is True
    assert summary["precommitted_transcript"] is True
    assert summary["post_hoc_artifact_replay"] is False
    assert summary["transcript_provenance"] == "precommitted_replacement_run"
    assert summary["replay_gate_status"] == "PASS_WP6_R1_COORDINATE_MAJORITY_E2E_REPLACEMENT_RUN"
    assert "post_hoc_not_precommitted_for_852086" not in summary


def test_scale_prompt_plan_locks_first_256_wp3_r1_eval_rows() -> None:
    plan = build_prompt_plan(
        read_prompt_rows_with_file_index(DEFAULT_PROMPTS),
        split="wp3_r1_eval",
        max_prompts=256,
        block_count=4,
        block_size=64,
        expected_file_row_start=512,
        expected_file_row_end=767,
        prompts_path=DEFAULT_PROMPTS,
    )

    assert plan["selected_prompt_file_row_start"] == 512
    assert plan["selected_prompt_file_row_end_inclusive"] == 767
    assert [block["prompt_file_row_start"] for block in plan["blocks"]] == [512, 576, 640, 704]
    assert [block["prompt_file_row_end_inclusive"] for block in plan["blocks"]] == [575, 639, 703, 767]


def test_scale_prompt_plan_locks_wp6_r2_option_b_rows() -> None:
    plan = build_prompt_plan(
        read_prompt_rows_with_file_index(DEFAULT_PROMPTS),
        split="wp3_r1_eval",
        max_prompts=512,
        block_count=8,
        block_size=64,
        expected_file_row_start=768,
        expected_file_row_end=1279,
        prompts_path=DEFAULT_PROMPTS,
    )

    assert plan["selected_prompt_file_row_start"] == 768
    assert plan["selected_prompt_file_row_end_inclusive"] == 1279
    assert plan["selected_prompt_jsonl_sha256"] == (
        "d3966ce5c43347df9c68dc6cd6118102fb0708484ddd53e9b08b7b42b1f12ddd"
    )
    assert [block["prompt_file_row_start"] for block in plan["blocks"]] == [
        768,
        832,
        896,
        960,
        1024,
        1088,
        1152,
        1216,
    ]
    assert [block["prompt_file_row_end_inclusive"] for block in plan["blocks"]] == [
        831,
        895,
        959,
        1023,
        1087,
        1151,
        1215,
        1279,
    ]
    assert plan["blocks"][0]["row_jsonl_sha256"] == (
        "67e869d94e33b659cc00ef0e10f50ddf7eb4e30e7a8e47bcaa156ec3227ff066"
    )


def test_generation_prompt_selection_accepts_explicit_file_row_window() -> None:
    prompts = select_prompts(
        read_jsonl_with_file_index(DEFAULT_PROMPTS),
        split="wp3_r1_eval",
        max_prompts=512,
        prompt_file_row_start=768,
        prompt_file_row_end=1279,
    )

    assert len(prompts) == 512
    assert prompts[0]["selected_prompt_file_row_index"] == 768
    assert prompts[-1]["selected_prompt_file_row_index"] == 1279


def test_scale_contract_can_record_wp6_r2_option_b_ids() -> None:
    info = contract_info(_contract())
    prompt_plan = build_prompt_plan(
        read_prompt_rows_with_file_index(DEFAULT_PROMPTS),
        split="wp3_r1_eval",
        max_prompts=512,
        block_count=8,
        block_size=64,
        expected_file_row_start=768,
        expected_file_row_end=1279,
        prompts_path=DEFAULT_PROMPTS,
    )

    class Args:
        artifact_role = "wp6_r2_option_b_robust_block_scale"
        block_count = 8
        block_size = 64
        decode_artifact_dir = "coordinate_majority_r2_option_b"
        decoder_id = "qwen_v2_wp6_r2_robust_block_coordinate_majority_decoder_v1"
        min_majority_margin_at_64 = 3
        min_protected_block_accepts_at_64 = 6
        min_support_at_64 = 16
        output_prefix = "wp6_r2_option_b"
        precommitted_transcript = True
        protocol_id = "natural_evidence_v2_wp6_r2_option_b_robust_block_scale"
        split = "wp3_r1_eval"
        wrong_audit_key_id = "KWP4_QWEN_PILOT_WRONG_001"
        wrong_payload_byte_hex = "5a"
        accept_rule = "per_block majority codeword checksum_valid_and_payload_matches_expected"

    contract = build_scale_contract(
        args=Args(),
        input_dir=DEFAULT_PROMPTS.parent,
        prompts_path=DEFAULT_PROMPTS,
        contract_path=DEFAULT_PROMPTS,
        wp4_contract={"precommit": {"bucket_policy_id": "test_bank"}},
        info=info,
        prompt_plan=prompt_plan,
        budgets=[8, 16, 32, 64],
    )

    assert contract["schema_name"] == "natural_evidence_v2_wp6_r2_option_b_contract_v1"
    assert contract["decoder_policy"]["minimum_protected_block_accepts_at_64"] == 6
    assert contract["precommit"]["precommit_material"]["protocol_id"] == (
        "natural_evidence_v2_wp6_r2_option_b_robust_block_scale"
    )
    assert "precommit/wp6_r2_option_b_contract.json" in contract["required_outputs"]


def test_scale_block_window_majority_accepts_three_of_four_blocks() -> None:
    info = contract_info(_contract())
    expected_bits = list(info["expected_bits"])
    wrong_bits = [0] * 16
    protected_frames: list[list[int]] = []
    raw_frames: list[list[int]] = []
    task_only_frames: list[list[int]] = []
    for block_index in range(4):
        protected_bits = expected_bits if block_index != 2 else wrong_bits
        for _ in range(4):
            protected_frames.append(protected_bits)
            raw_frames.append(wrong_bits)
            task_only_frames.append(wrong_bits)

    observations = []
    observations.extend(_obs_frames("protected", protected_frames))
    observations.extend(_obs_frames("raw", raw_frames))
    observations.extend(_obs_frames("task_only", task_only_frames))
    exact_decisions = []
    for condition in ("protected", "raw", "task_only"):
        for frame_index in range(16):
            exact_decisions.append(
                {
                    "decode_condition": condition,
                    "forbidden_public_surface_present": False,
                    "frame_index": frame_index,
                }
            )

    class Args:
        block_count = 4
        block_size = 4
        min_majority_margin_at_64 = 1
        min_protected_block_accepts_at_64 = 3
        min_support_at_64 = 1
        precommitted_transcript = True
        wrong_audit_key_id = "KWP4_QWEN_PILOT_WRONG_001"

    _coord_rows, decode_rows = decode_scale_blocks(
        args=Args(),
        observations=observations,
        info=info,
        budgets=[2, 4],
        wrong_payload_byte=0x5A,
    )
    summary = summarize_scale(
        args=Args(),
        budgets=[2, 4],
        decode_rows=decode_rows,
        exact_decision_rows=exact_decisions,
    )

    protected_budget4 = {
        row["block_id"]: row
        for row in decode_rows
        if row["budget"] == 4 and row["decode_condition"] == "protected"
    }
    assert protected_budget4["block_0"]["accepted"] is True
    assert protected_budget4["block_1"]["accepted"] is True
    assert protected_budget4["block_2"]["accepted"] is False
    assert protected_budget4["block_3"]["accepted"] is True
    assert summary["scale_gate_pass"] is True
    assert summary["protected_block_accept_count_at_controlling_budget"] == 3
    assert summary["null_accept_counts_at_controlling_budget"] == {
        "raw": 0,
        "task_only": 0,
        "wrong_key": 0,
        "wrong_payload": 0,
    }

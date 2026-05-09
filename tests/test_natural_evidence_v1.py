from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.natural_evidence_v1 import (
    analyze_qwen_on_policy_survival,
    analyze_branch_aware_score_interpretation,
    analyze_qwen_protected_lift,
    analyze_r1_selector_contract,
    audit_compatible_density,
    audit_actual_prefix_bucketization,
    audit_opportunity_bank,
    build_expanded_actual_prefix_bucketized_candidates,
    build_variable_arity_diagnostic,
    build_bucket_bank,
    combine_variable_arity_density_inputs,
    compile_train_dataset,
    audit_variable_arity_full_density,
    compatibility_aware_supply_audit,
    diagnose_qwen_natural_e2e_zero_symbols,
    diagnose_verifier_alignment,
    design_repaired_teacher_forced_target_mass_probe,
    design_selector_contract_preflight,
    dry_run_local_suffix_repair,
    dry_run_compatibility_filtered_bank,
    evaluate_diagnostic_e2e,
    evaluate_qwen_natural_e2e,
    export_balanced_branch_aware_examples,
    freeze_density_prompt_splits,
    generate_reference_outputs,
    hermes_notify,
    join_compatibility_probabilities,
    oracle_qwen_decoder_substitution,
    prepare_branch_aware_suffix_repair_diagnostics,
    prepare_actual_prefix_scoring_plan,
    raw_wrong_key_pre_null,
    review_diagnostic_e2e_wrapper,
    review_qwen_natural_e2e_eval_wrapper,
    review_qwen_proof_of_life_gate,
    review_invalid_suffix_records,
    salvage_actual_prefix_diagnostic,
    score_branch_aware_compatibility,
    oracle_qwen_schedule_simulation,
    replay_qwen_frame_completion,
    replay_prefix_conditioned_selector,
    probe_qwen_teacher_forced_bucket_mass,
    score_repaired_teacher_forced_target_mass_probe,
    summarize_qwen_natural_e2e_observations,
    summarize_actual_prefix_suffix_sensitivity,
    score_actual_prefix_suffix_compatibility,
    score_actual_prefix_reference_candidates,
    score_reference_candidates,
    sweep_balance_gate,
    train_natural_bucket_lora,
    validate_static,
    variable_arity_pre_null,
    variable_radix_train_eval_preflight,
    verify_observations,
)
from scripts.natural_evidence_v1.common import token_surface_allowed
from scripts.natural_evidence_v1.opportunity_policy import construct_buckets_from_topk_record
from src.core.payload_codec import BucketPayloadCodec


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_qwen_observation_erasure_summary_normalizes_recovery_provenance(tmp_path: Path) -> None:
    observations_path = tmp_path / "qwen_natural_e2e_bucket_observations.jsonl"
    decode_trace_path = tmp_path / "qwen_natural_e2e_decode_trace.csv"
    summary_path = tmp_path / "qwen_natural_e2e_eval_summary.json"
    progress_path = tmp_path / "qwen_natural_e2e_eval_progress.json"
    output_path = tmp_path / "observation_erasure_summary_846699.json"
    _write_jsonl(
        observations_path,
        [
            {
                "schema_name": "natural_evidence_qwen_natural_e2e_bucket_observation_v1",
                "model_condition": "raw",
                "observation_condition": "correct_key",
                "payload_id": "P0421",
                "seed": "",
                "query_index": 0,
                "prompt_id": "nat_prompt_000001",
                "position_index": 0,
                "token_index": 2,
                "frame_index": 0,
                "frame_digit_index": 0,
                "frame_digit_count": 2,
                "bucket_id": "",
                "digit": "",
                "radix": "",
                "erasure": True,
                "erasure_reason": "observed_token_not_in_variable_radix_bucket_set",
                "observed_token_id": 4248,
                "observed_token_text": " Check",
                "compatible_bucket_ids": ["0", "3"],
                "entry_key": "entry||raw||0",
                "bank_entry_id": "entry",
                "anchor_policy": "prompt_id_token_index_variable_radix",
            },
            {
                "schema_name": "natural_evidence_qwen_natural_e2e_bucket_observation_v1",
                "model_condition": "protected_trained",
                "observation_condition": "correct_key",
                "payload_id": "P0421",
                "seed": "17",
                "query_index": 1,
                "prompt_id": "nat_prompt_000002",
                "position_index": 1,
                "token_index": 3,
                "frame_index": 0,
                "frame_digit_index": 1,
                "frame_digit_count": 2,
                "bucket_id": "2",
                "digit": 2,
                "radix": 3,
                "erasure": False,
                "erasure_reason": "",
                "observed_token_id": 16138,
                "observed_token_text": " kit",
                "compatible_bucket_ids": ["0", "1", "2"],
                "entry_key": "entry||protected||1",
                "bank_entry_id": "entry",
                "anchor_policy": "prompt_id_token_index_variable_radix",
            },
        ],
    )
    with decode_trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_condition",
                "payload_id",
                "seed",
                "far_family",
                "query_budget",
                "accepted",
                "eligible_positions",
                "observed_symbols",
                "usable_symbols",
                "decode_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_condition": "raw",
                "payload_id": "P0421",
                "seed": "",
                "far_family": "raw_exact_model",
                "query_budget": "64",
                "accepted": "False",
                "eligible_positions": "1",
                "observed_symbols": "0",
                "usable_symbols": "0",
                "decode_status": "insufficient_symbols",
            }
        )
    summary_path.write_text(
        json.dumps(
            {
                "status": "EVAL_COMPLETE_QWEN_NATURAL_VARIABLE_RADIX_NOT_PAPER_CLAIM",
                "generated_output_count": 2,
                "observation_count": 2,
                "decode_row_count": 1,
                "protected_accept_count": 0,
                "null_accept_count": 0,
                "diagnostic_recovery_observed": False,
                "null_accept_observed": False,
                "paper_claim_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    progress_path.write_text(json.dumps({"status": "COMPLETE", "stage": "eval"}), encoding="utf-8")
    status = summarize_qwen_natural_e2e_observations.main(
        [
            "--observations-jsonl",
            str(observations_path),
            "--decode-trace-csv",
            str(decode_trace_path),
            "--summary-json",
            str(summary_path),
            "--progress-json",
            str(progress_path),
            "--output-json",
            str(output_path),
            "--source-job-id",
            "846699",
            "--source-remote-path",
            "/remote/qwen_natural_e2e_eval_846627_recovery/qwen_natural_e2e_bucket_observations.jsonl",
            "--path-explanation",
            "846699 recovery evaluation intentionally reused the 846627 recovery output directory name.",
            "--expected-observation-count",
            "2",
            "--expected-decode-row-count",
            "1",
        ]
    )
    assert status == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS_EXPLAINED_RECOVERY_DIR_NAME"
    assert payload["claim_control"]["paper_claim_allowed"] is False
    assert payload["observations"]["row_count"] == 2
    assert payload["observations"]["erasure_reason_rows"] == 1
    assert payload["observations"]["compatible_variable_radix_digit_rows"] == 1
    assert payload["decode_trace"]["row_count"] == 1
    assert payload["provenance_mismatches"] == []


def test_qwen_frame_completion_replay_maps_decode_rows_and_counts_frames(tmp_path: Path) -> None:
    observations_path = tmp_path / "qwen_natural_e2e_bucket_observations.jsonl"
    decode_trace_path = tmp_path / "qwen_natural_e2e_decode_trace.csv"
    output_dir = tmp_path / "frame_replay"
    _write_jsonl(
        observations_path,
        [
            {
                "model_condition": "raw",
                "payload_id": "P0421",
                "seed": "",
                "observation_condition": "correct_key",
                "query_index": 0,
                "prompt_id": "p0",
                "position_index": 0,
                "frame_index": 0,
                "frame_digit_index": 0,
                "frame_digit_count": 2,
                "digit": 1,
                "radix": 4,
                "bucket_id": "1",
                "erasure_reason": "",
            },
            {
                "model_condition": "raw",
                "payload_id": "P0421",
                "seed": "",
                "observation_condition": "correct_key",
                "query_index": 1,
                "prompt_id": "p1",
                "position_index": 0,
                "frame_index": 0,
                "frame_digit_index": 1,
                "frame_digit_count": 2,
                "digit": 0,
                "radix": 4,
                "bucket_id": "0",
                "erasure_reason": "",
            },
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "observation_condition": "correct_key",
                "query_index": 0,
                "prompt_id": "p0",
                "position_index": 0,
                "frame_index": 1,
                "frame_digit_index": 0,
                "frame_digit_count": 2,
                "digit": 2,
                "radix": 3,
                "bucket_id": "2",
                "erasure_reason": "",
            },
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "observation_condition": "correct_key",
                "query_index": 1,
                "prompt_id": "p1",
                "position_index": 0,
                "frame_index": 1,
                "frame_digit_index": 1,
                "frame_digit_count": 2,
                "digit": "",
                "radix": "",
                "bucket_id": "",
                "erasure_reason": "observed_token_not_in_variable_radix_bucket_set",
            },
        ],
    )
    with decode_trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_condition",
                "payload_id",
                "expected_payload_id",
                "seed",
                "far_family",
                "query_budget",
                "eligible_positions",
                "observed_symbols",
                "usable_symbols",
                "decode_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_condition": "raw",
                "payload_id": "P0421",
                "expected_payload_id": "P0421",
                "seed": "",
                "far_family": "raw_exact_model",
                "query_budget": "1",
                "eligible_positions": "1",
                "observed_symbols": "1",
                "usable_symbols": "0",
                "decode_status": "insufficient_symbols",
            }
        )
        writer.writerow(
            {
                "model_condition": "raw",
                "payload_id": "P0421",
                "expected_payload_id": "P0421",
                "seed": "",
                "far_family": "raw_exact_model",
                "query_budget": "2",
                "eligible_positions": "2",
                "observed_symbols": "2",
                "usable_symbols": "2",
                "decode_status": "decoded_frame_accept",
            }
        )
        writer.writerow(
            {
                "model_condition": "wrong_payload",
                "payload_id": "P0421",
                "expected_payload_id": "P1729",
                "seed": "17",
                "far_family": "wrong_payload",
                "query_budget": "2",
                "eligible_positions": "2",
                "observed_symbols": "1",
                "usable_symbols": "0",
                "decode_status": "insufficient_symbols",
            }
        )
    status = replay_qwen_frame_completion.main(
        [
            "--observations-jsonl",
            str(observations_path),
            "--decode-trace-csv",
            str(decode_trace_path),
            "--output-dir",
            str(output_dir),
            "--top-closest-frames",
            "2",
        ]
    )
    assert status == 0
    summary = json.loads(
        (output_dir / "qwen_846699_frame_completion_replay_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "COMPLETE_REPLAY_OBSERVED_COMPLETE_FRAMES_FOUND"
    assert summary["aggregate"]["decode_rows_with_observed_complete_frames"] == 1
    assert summary["aggregate"]["decode_rows_with_scheduled_complete_frames_no_erasure"] == 2
    rows = list(
        csv.DictReader(
            (output_dir / "qwen_846699_frame_completion_by_decode_row.csv").open(encoding="utf-8")
        )
    )
    raw_budget_2 = rows[1]
    assert raw_budget_2["observed_complete_frame_count"] == "1"
    wrong_payload = rows[2]
    assert wrong_payload["observation_model_condition"] == "protected_trained"
    assert wrong_payload["scheduled_complete_frame_count_no_erasure"] == "1"
    assert wrong_payload["observed_complete_frame_count"] == "0"


def test_qwen_oracle_schedule_simulation_detects_unsalvageable_observed_subset(tmp_path: Path) -> None:
    observations_path = tmp_path / "qwen_natural_e2e_bucket_observations.jsonl"
    decode_trace_path = tmp_path / "qwen_natural_e2e_decode_trace.csv"
    frame_summary_path = tmp_path / "qwen_846699_frame_completion_replay_summary.json"
    output_dir = tmp_path / "oracle"
    _write_jsonl(
        observations_path,
        [
            {
                "model_condition": "raw",
                "payload_id": "P0421",
                "seed": "",
                "observation_condition": "correct_key",
                "query_index": 0,
                "prompt_id": "p0",
                "position_index": 0,
                "frame_index": 0,
                "frame_digit_index": 0,
                "frame_digit_count": 2,
                "digit": 1,
                "radix": 4,
                "bucket_id": "1",
                "erasure_reason": "",
            },
            {
                "model_condition": "raw",
                "payload_id": "P0421",
                "seed": "",
                "observation_condition": "correct_key",
                "query_index": 1,
                "prompt_id": "p1",
                "position_index": 0,
                "frame_index": 0,
                "frame_digit_index": 1,
                "frame_digit_count": 2,
                "digit": "",
                "radix": "",
                "bucket_id": "",
                "erasure_reason": "observed_token_not_in_variable_radix_bucket_set",
            },
            {
                "model_condition": "raw",
                "payload_id": "P0421",
                "seed": "",
                "observation_condition": "correct_key",
                "query_index": 2,
                "prompt_id": "p2",
                "position_index": 0,
                "frame_index": 1,
                "frame_digit_index": 0,
                "frame_digit_count": 2,
                "digit": "",
                "radix": "",
                "bucket_id": "",
                "erasure_reason": "observed_token_not_in_variable_radix_bucket_set",
            },
            {
                "model_condition": "raw",
                "payload_id": "P0421",
                "seed": "",
                "observation_condition": "correct_key",
                "query_index": 3,
                "prompt_id": "p3",
                "position_index": 0,
                "frame_index": 1,
                "frame_digit_index": 1,
                "frame_digit_count": 2,
                "digit": "",
                "radix": "",
                "bucket_id": "",
                "erasure_reason": "observed_token_not_in_variable_radix_bucket_set",
            },
        ],
    )
    with decode_trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_condition",
                "payload_id",
                "expected_payload_id",
                "seed",
                "far_family",
                "query_budget",
                "eligible_positions",
                "observed_symbols",
                "usable_symbols",
                "decode_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_condition": "raw",
                "payload_id": "P0421",
                "expected_payload_id": "P0421",
                "seed": "",
                "far_family": "raw_exact_model",
                "query_budget": "2",
                "eligible_positions": "2",
                "observed_symbols": "1",
                "usable_symbols": "0",
                "decode_status": "insufficient_symbols",
            }
        )
        writer.writerow(
            {
                "model_condition": "raw",
                "payload_id": "P0421",
                "expected_payload_id": "P0421",
                "seed": "",
                "far_family": "raw_exact_model",
                "query_budget": "4",
                "eligible_positions": "4",
                "observed_symbols": "1",
                "usable_symbols": "0",
                "decode_status": "insufficient_symbols",
            }
        )
    frame_summary_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE_REPLAY_NO_OBSERVED_COMPLETE_FRAMES_SCHEDULE_CAN_COMPLETE_WITH_NO_ERASURE",
                "inputs": {
                    "observations_jsonl": {
                        "sha256": "",
                        "row_count": 4,
                    },
                    "decode_trace_csv": {
                        "sha256": "",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    status = oracle_qwen_schedule_simulation.main(
        [
            "--observations-jsonl",
            str(observations_path),
            "--decode-trace-csv",
            str(decode_trace_path),
            "--frame-replay-summary-json",
            str(frame_summary_path),
            "--output-dir",
            str(output_dir),
            "--query-budgets",
            "2,4",
            "--max-prompt-examples",
            "4",
            "--max-frame-bounds",
            "4",
        ]
    )
    assert status == 0
    summary = json.loads(
        (output_dir / "qwen_846699_oracle_schedule_simulation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "COMPLETE_ORACLE_SCHEDULE_NO_OBSERVED_SUBSET_CAN_COMPLETE_FRAME"
    assert summary["aggregate"]["decode_rows_with_greedy_scheduled_complete_frames_no_erasure"] == 2
    assert summary["aggregate"]["decode_rows_with_any_subset_observed_complete_frames"] == 0
    rows = list(
        csv.DictReader(
            (output_dir / "qwen_846699_oracle_schedule_by_decode_row.csv").open(encoding="utf-8")
        )
    )
    assert rows[0]["greedy_scheduled_complete_frames_no_erasure"] == "1"
    assert rows[0]["any_subset_observed_complete_frames_all_prompts"] == "0"


def test_qwen_on_policy_survival_slices_joined_observations(tmp_path: Path) -> None:
    observations_path = tmp_path / "qwen_natural_e2e_bucket_observations.jsonl"
    train_data_dir = tmp_path / "train_data"
    oracle_summary_path = tmp_path / "qwen_846699_oracle_schedule_simulation_summary.json"
    output_dir = tmp_path / "survival"
    _write_jsonl(
        train_data_dir / "P0421" / "variable_radix_train.jsonl",
        [
            {
                "payload_id": "P0421",
                "prompt_id": "p0",
                "prompt_split": "heldout",
                "example_role": "eval",
                "eligible_positions": [
                    {
                        "bank_entry_id": "entry0",
                        "entry_key": "entry0||src_prompt_0||heldout||raw",
                        "token_index": 4,
                        "frame_index": 7,
                        "frame_digit_index": 0,
                        "frame_digit_count": 2,
                        "payload_digit_index": 0,
                        "target_digit": 0,
                        "target_radix": 2,
                        "target_bucket": "0",
                        "compatible_bucket_ids": ["0", "1"],
                        "bucket_to_token_ids": {"0": [10], "1": [11]},
                        "target_bucket_token_ids": [10],
                        "candidate_token_ids": [10, 11],
                    }
                ],
            },
            {
                "payload_id": "P0421",
                "prompt_id": "p1",
                "prompt_split": "heldout",
                "example_role": "eval",
                "eligible_positions": [
                    {
                        "bank_entry_id": "entry1",
                        "entry_key": "entry1||src_prompt_1||heldout||protected_trained",
                        "token_index": 5,
                        "frame_index": 8,
                        "frame_digit_index": 1,
                        "frame_digit_count": 2,
                        "payload_digit_index": 1,
                        "target_digit": 1,
                        "target_radix": 2,
                        "target_bucket": "1",
                        "compatible_bucket_ids": ["0", "1"],
                        "bucket_to_token_ids": {"0": [20], "1": [21]},
                        "target_bucket_token_ids": [21],
                        "candidate_token_ids": [20, 21],
                    }
                ],
            },
        ],
    )
    _write_jsonl(
        observations_path,
        [
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "observation_condition": "correct_key",
                "prompt_id": "p0",
                "position_index": 0,
                "entry_key": "entry0||src_prompt_0||heldout||raw",
                "observed_token_id": 10,
                "observed_token_text": " alpha",
                "bucket_id": "0",
                "digit": 0,
                "radix": 2,
                "erasure_reason": "",
            },
            {
                "model_condition": "raw",
                "payload_id": "P0421",
                "seed": "",
                "observation_condition": "correct_key",
                "prompt_id": "p1",
                "position_index": 0,
                "entry_key": "entry1||src_prompt_1||heldout||protected_trained",
                "observed_token_id": 99,
                "observed_token_text": " outside",
                "bucket_id": "",
                "digit": "",
                "radix": "",
                "erasure_reason": "observed_token_not_in_variable_radix_bucket_set",
            },
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "observation_condition": "wrong_key",
                "prompt_id": "p0",
                "position_index": 0,
                "entry_key": "entry0||src_prompt_0||heldout||raw",
                "observed_token_id": 11,
                "observed_token_text": " beta",
                "bucket_id": "1",
                "digit": 1,
                "radix": 2,
                "erasure_reason": "",
            },
        ],
    )
    oracle_summary_path.write_text(
        json.dumps(
            {
                "schema_name": "natural_evidence_qwen_846699_oracle_schedule_simulation_v1",
                "status": "COMPLETE_ORACLE_SCHEDULE_NO_OBSERVED_SUBSET_CAN_COMPLETE_FRAME",
                "inputs": {"observations_jsonl": {"row_count": 3, "sha256": ""}},
            }
        ),
        encoding="utf-8",
    )
    status = analyze_qwen_on_policy_survival.main(
        [
            "--observations-jsonl",
            str(observations_path),
            "--train-data-dir",
            str(train_data_dir),
            "--oracle-summary-json",
            str(oracle_summary_path),
            "--output-dir",
            str(output_dir),
            "--payload-ids",
            "P0421",
            "--max-examples",
            "8",
        ]
    )
    assert status == 0
    summary = json.loads(
        (output_dir / "qwen_846699_on_policy_survival_summary.json").read_text(
            encoding="utf-8"
        )
    )
    aggregate = summary["aggregate"]
    assert summary["status"] == "COMPLETE_ON_POLICY_SURVIVAL_DIAGNOSTIC"
    assert aggregate["row_count"] == 3
    assert aggregate["compatible_hit_rows"] == 2
    assert aggregate["target_comparable_rows"] == 2
    assert aggregate["target_hit_rows"] == 1
    assert aggregate["bucket_miss_rows"] == 1
    assert aggregate["join_source_counts"] == {"payload_prompt_position": 3}
    assert summary["inputs"]["oracle_summary_json"]["provenance_mismatches"] == []
    by_slice = list(
        csv.DictReader(
            (output_dir / "qwen_846699_on_policy_survival_by_slice.csv").open(
                encoding="utf-8"
            )
        )
    )
    raw_slice = next(row for row in by_slice if row["slice_kind"] == "model_condition" and row["slice_value"] == "raw")
    assert raw_slice["bucket_miss_rows"] == "1"
    protected_slice = next(
        row for row in by_slice if row["slice_kind"] == "model_condition" and row["slice_value"] == "protected_trained"
    )
    assert protected_slice["compatible_hit_rows"] == "2"
    miss_examples = [
        json.loads(line)
        for line in (output_dir / "qwen_846699_on_policy_bucket_miss_examples.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert miss_examples[0]["observed_token_id"] == 99
    assert miss_examples[0]["observed_token_in_candidate_set"] is False


def test_qwen_protected_lift_compares_task_only_by_slice(tmp_path: Path) -> None:
    observations_path = tmp_path / "qwen_natural_e2e_bucket_observations.jsonl"
    train_data_dir = tmp_path / "train_data"
    survival_summary_path = tmp_path / "qwen_846699_on_policy_survival_summary.json"
    output_dir = tmp_path / "lift"
    _write_jsonl(
        train_data_dir / "P0421" / "variable_radix_train.jsonl",
        [
            {
                "payload_id": "P0421",
                "prompt_id": "p0",
                "prompt_split": "heldout",
                "example_role": "eval",
                "eligible_positions": [
                    {
                        "bank_entry_id": "entry0",
                        "entry_key": "entry0||src_prompt_0||heldout||raw",
                        "token_index": 4,
                        "frame_index": 7,
                        "frame_digit_index": 0,
                        "frame_digit_count": 2,
                        "payload_digit_index": 0,
                        "target_digit": 0,
                        "target_radix": 2,
                        "target_bucket": "0",
                        "compatible_bucket_ids": ["0", "1"],
                        "bucket_to_token_ids": {"0": [10], "1": [11]},
                        "target_bucket_token_ids": [10],
                        "candidate_token_ids": [10, 11],
                    }
                ],
            },
            {
                "payload_id": "P0421",
                "prompt_id": "p1",
                "prompt_split": "heldout",
                "example_role": "eval",
                "eligible_positions": [
                    {
                        "bank_entry_id": "entry1",
                        "entry_key": "entry1||src_prompt_1||heldout||raw",
                        "token_index": 5,
                        "frame_index": 8,
                        "frame_digit_index": 1,
                        "frame_digit_count": 2,
                        "payload_digit_index": 1,
                        "target_digit": 1,
                        "target_radix": 2,
                        "target_bucket": "1",
                        "compatible_bucket_ids": ["0", "1"],
                        "bucket_to_token_ids": {"0": [20], "1": [21]},
                        "target_bucket_token_ids": [21],
                        "candidate_token_ids": [20, 21],
                    }
                ],
            },
        ],
    )
    _write_jsonl(
        observations_path,
        [
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "observation_condition": "correct_key",
                "prompt_id": "p0",
                "position_index": 0,
                "entry_key": "entry0||src_prompt_0||heldout||raw",
                "observed_token_id": 10,
                "observed_token_text": " alpha",
                "bucket_id": "0",
                "digit": 0,
                "radix": 2,
                "erasure_reason": "",
            },
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "observation_condition": "correct_key",
                "prompt_id": "p1",
                "position_index": 0,
                "entry_key": "entry1||src_prompt_1||heldout||raw",
                "observed_token_id": 99,
                "observed_token_text": " outside",
                "bucket_id": "",
                "digit": "",
                "radix": "",
                "erasure_reason": "observed_token_not_in_variable_radix_bucket_set",
            },
            {
                "model_condition": "task_only_lora",
                "payload_id": "P0421",
                "seed": "17",
                "observation_condition": "correct_key",
                "prompt_id": "p0",
                "position_index": 0,
                "entry_key": "entry0||src_prompt_0||heldout||raw",
                "observed_token_id": 11,
                "observed_token_text": " beta",
                "bucket_id": "1",
                "digit": 1,
                "radix": 2,
                "erasure_reason": "",
            },
            {
                "model_condition": "task_only_lora",
                "payload_id": "P0421",
                "seed": "17",
                "observation_condition": "correct_key",
                "prompt_id": "p1",
                "position_index": 0,
                "entry_key": "entry1||src_prompt_1||heldout||raw",
                "observed_token_id": 99,
                "observed_token_text": " outside",
                "bucket_id": "",
                "digit": "",
                "radix": "",
                "erasure_reason": "observed_token_not_in_variable_radix_bucket_set",
            },
            {
                "model_condition": "wrong_key",
                "payload_id": "P0421",
                "seed": "17",
                "observation_condition": "K001_WRONG_0",
                "prompt_id": "p0",
                "position_index": 0,
                "entry_key": "entry0||src_prompt_0||heldout||raw",
                "observed_token_id": 10,
                "observed_token_text": " alpha",
                "bucket_id": "0",
                "digit": 0,
                "radix": 2,
                "erasure_reason": "",
            },
        ],
    )
    survival_summary_path.write_text(
        json.dumps(
            {
                "schema_name": "natural_evidence_qwen_846699_on_policy_survival_v1",
                "status": "COMPLETE_ON_POLICY_SURVIVAL_DIAGNOSTIC",
                "inputs": {"observations_jsonl": {"sha256": ""}},
            }
        ),
        encoding="utf-8",
    )
    status = analyze_qwen_protected_lift.main(
        [
            "--observations-jsonl",
            str(observations_path),
            "--train-data-dir",
            str(train_data_dir),
            "--survival-summary-json",
            str(survival_summary_path),
            "--output-dir",
            str(output_dir),
            "--payload-ids",
            "P0421",
            "--min-rows-per-arm",
            "1",
        ]
    )
    assert status == 0
    summary = json.loads(
        (output_dir / "qwen_846699_protected_vs_task_only_lift_summary.json").read_text(
            encoding="utf-8"
        )
    )
    aggregate = summary["aggregate"]
    assert summary["status"] == "COMPLETE_PROTECTED_VS_TASK_ONLY_LIFT_DIAGNOSTIC"
    assert summary["row_counts"]["included_by_model_condition"] == {
        "protected_trained": 2,
        "task_only_lora": 2,
    }
    assert summary["row_counts"]["skipped_non_lift_rows"] == 1
    assert aggregate["protected_rows"] == 2
    assert aggregate["task_only_rows"] == 2
    assert aggregate["protected_target_hit_rows"] == 1
    assert aggregate["task_only_target_hit_rows"] == 0
    assert aggregate["target_lift_direction"] == "protected_higher"
    rows = list(
        csv.DictReader(
            (output_dir / "qwen_846699_protected_vs_task_only_lift_by_slice.csv").open(
                encoding="utf-8"
            )
        )
    )
    slot_zero = next(
        row for row in rows if row["slice_kind"] == "prompt_slot" and row["slice_value"] == "0"
    )
    assert slot_zero["protected_target_hit_rows"] == "1"
    assert slot_zero["task_only_target_hit_rows"] == "0"
    extremes = [
        json.loads(line)
        for line in (output_dir / "qwen_846699_protected_vs_task_only_lift_extremes.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert extremes


def test_teacher_forced_bucket_probe_scores_target_bucket_mass() -> None:
    probe = probe_qwen_teacher_forced_bucket_mass.bucket_probe_from_token_logits(
        token_logits={
            10: 3.0,
            11: 1.0,
            20: 2.0,
            21: 0.0,
        },
        bucket_to_token_ids={
            "0": [10, 11],
            "1": [20, 21],
        },
        target_bucket="0",
    )
    assert probe["target_rank"] == 1
    assert probe["target_candidate_mass"] > probe["best_other_candidate_mass"]
    assert probe["target_margin"] > 0
    assert set(probe["bucket_masses"]) == {"0", "1"}


def test_validate_static_accepts_pilot_config(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    status = validate_static.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--summary",
            str(summary_path),
        ]
    )
    assert status == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert "protocol_precommitment" in payload["validated_components"]
    assert "task_only_and_near_null_controls" in payload["validated_components"]
    assert "automation_run_allowlist" in payload["validated_components"]
    assert "chimera_mail_notifications" in payload["validated_components"]
    config_text = (REPO_ROOT / "configs/natural_evidence_v1/pilot.yaml").read_text(
        encoding="utf-8"
    )
    assert "raw_entry_count_is_training_gate: false" in config_text
    assert "compatibility_adjusted_capacity:" in config_text
    assert "min1_compatible_entries_min: 1500" in config_text
    assert "diagnostic_high_risk_gate:" in config_text
    assert "paper_claim_allowed: false" in config_text
    assert "diagnostic_query_budgets:" in config_text
    assert "512" in config_text


def test_natural_evidence_slurm_scripts_enable_chimera_mail_notifications() -> None:
    slurm_scripts = sorted((REPO_ROOT / "scripts/natural_evidence_v1").rglob("*.sbatch"))
    assert slurm_scripts
    for script_path in slurm_scripts:
        script_text = script_path.read_text(encoding="utf-8")
        assert "#SBATCH --mail-type=ALL" in script_text, script_path
        assert "#SBATCH --mail-user=guanjie.lin001@umb.edu" in script_text, script_path


def test_build_bucket_bank_from_reference_candidates(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    candidates = [
        {
            "prompt_id": "p0",
            "prefix_token_ids": [1, 2, 3],
            "candidates": [
                {
                    "token_id": index + 10,
                    "text": f" token{index}",
                    "probability": 0.02,
                    "rank": index + 1,
                }
                for index in range(16)
            ],
        }
    ]
    _write_jsonl(candidate_path, candidates)
    output_dir = tmp_path / "bank"
    status = build_bucket_bank.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--tokenizer-key",
            "qwen",
            "--candidate-jsonl",
            str(candidate_path),
            "--output-dir",
            str(output_dir),
            "--target-entries",
            "1",
            "--bucket-count",
            "8",
        ]
    )
    assert status == 0
    entries = (output_dir / "qwen_bucket_bank_entries.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(entries) == 1
    entry = json.loads(entries[0])
    assert entry["bucket_count"] == 8
    assert entry["entry_role"] == "context_conditioned_measurable_opportunity"
    assert entry["fingerprint_claim"] is False
    assert "FIELD=" not in json.dumps(entry)


def test_strict_balance_gate_rejects_peaked_bucket_entries(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    candidates = [
        {
            "prompt_id": "p0",
            "prefix_token_ids": [1, 2, 3],
            "candidates": [
                {
                    "token_id": 10,
                    "text": " dominant",
                    "probability": 0.91,
                    "rank": 1,
                },
                *[
                    {
                        "token_id": index + 11,
                        "text": f" token{index}",
                        "probability": 0.001,
                        "rank": index + 2,
                    }
                    for index in range(15)
                ],
            ],
        }
    ]
    _write_jsonl(candidate_path, candidates)
    output_dir = tmp_path / "bank"
    status = build_bucket_bank.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--tokenizer-key",
            "qwen",
            "--candidate-jsonl",
            str(candidate_path),
            "--output-dir",
            str(output_dir),
            "--target-entries",
            "1",
            "--bucket-count",
            "4",
            "--strict-balance-gate",
            "--balance-min-bucket-mass",
            "0.005",
            "--max-bucket-mass-ratio",
            "5",
            "--min-bucket-entropy-fraction",
            "0.90",
        ]
    )
    assert status == 0
    assert (output_dir / "qwen_bucket_bank_entries.jsonl").read_text(encoding="utf-8") == ""
    rejections = list(csv.DictReader((output_dir / "qwen_bucket_bank_rejections.csv").open(encoding="utf-8")))
    assert rejections[0]["rejection_reason"].startswith("balance_gate_")
    manifest = json.loads((output_dir / "qwen_bank_manifest.json").read_text(encoding="utf-8"))
    assert manifest["strict_balance_gate"] is True
    assert manifest["claim_control"]["bucket_bank_entries_are_fingerprints"] is False


def test_balance_gate_sweep_reports_threshold_tradeoff(tmp_path: Path) -> None:
    entries_path = tmp_path / "entries.jsonl"
    rejections_path = tmp_path / "rejections.csv"
    output_csv = tmp_path / "sweep.csv"
    summary_json = tmp_path / "summary.json"
    _write_jsonl(
        entries_path,
        [
            {
                "prompt_id": "p0",
                "bucket_mass_summary": {
                    "min_bucket_mass": 0.006,
                    "bucket_mass_ratio": 4.0,
                    "bucket_entropy_fraction": 0.91,
                },
            }
        ],
    )
    with rejections_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_id",
                "rejection_reason",
                "min_bucket_mass",
                "bucket_mass_ratio",
                "bucket_entropy_fraction",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "prompt_id": "p1",
                "rejection_reason": "balance_gate_bucket_mass_ratio",
                "min_bucket_mass": 0.006,
                "bucket_mass_ratio": 9.0,
                "bucket_entropy_fraction": 0.86,
            }
        )
    status = sweep_balance_gate.main(
        [
            "--entries",
            str(entries_path),
            "--rejections",
            str(rejections_path),
            "--target-entries",
            "2",
            "--min-bucket-mass-values",
            "0.005",
            "--max-bucket-mass-ratio-values",
            "5,10",
            "--min-bucket-entropy-fraction-values",
            "0.85,0.90",
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
        ]
    )
    assert status == 0
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["any_threshold_reaches_target"] is True
    assert summary["result_claim"] == "threshold_coverage_diagnostic_not_fingerprint_count"
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert max(int(row["accepted_entries"]) for row in rows) == 2


def test_token_surface_filter_rejects_nonsemantic_surfaces() -> None:
    assert token_surface_allowed(" Start") is True
    assert token_surface_allowed(" ") is False
    assert token_surface_allowed("...") is False
    assert token_surface_allowed(" **") is False
    assert token_surface_allowed("<|eot_id|>") is False
    assert token_surface_allowed("_begin") is False
    assert token_surface_allowed(".Start") is False


def test_verifier_alignment_diagnosis_reports_strict_prefix_erasure(tmp_path: Path) -> None:
    generated_path = tmp_path / "generated.jsonl"
    observations_path = tmp_path / "observations.jsonl"
    bank_path = tmp_path / "bank.jsonl"
    output_dir = tmp_path / "alignment"
    _write_jsonl(
        generated_path,
        [
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "prompt_id": "p0",
                "query_index": 0,
                "prompt": "p q",
                "response_text": "a c bucket",
            }
        ],
    )
    _write_jsonl(
        observations_path,
        [
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "observation_condition": "correct_key",
                "prompt_id": "p0",
                "query_index": 0,
                "bank_entry_id": "entry0",
                "prefix_response_token_count": 2,
                "strict_prefix_match": False,
                "observed_token_id": 5,
                "observed_token_text": "bucket",
                "bucket_id": "",
                "erasure": True,
                "erasure_reason": "strict_prefix_mismatch",
            }
        ],
    )
    _write_jsonl(
        bank_path,
        [
            {
                "bank_entry_id": "entry0",
                "prefix_response_token_count": 2,
                "prefix_token_ids": [1, 2, 3, 99],
                "buckets": {"0": [5], "1": [6], "2": [7], "3": [8]},
            }
        ],
    )
    summary = diagnose_verifier_alignment.run_diagnosis(
        generated_outputs_path=generated_path,
        observations_path=observations_path,
        decode_trace_path=None,
        progress_path=None,
        bucket_bank_entries_path=bank_path,
        compatibility_jsonl_path=None,
        compatibility_by_entry_csv_path=None,
        bucket_count=4,
        tokenizer_name="__simple_whitespace_test__",
        output_dir=output_dir,
        max_examples=5,
    )
    assert summary["alignment_status"] == "FAIL_STRICT_PREFIX_ERASURE_DOMINATES"
    assert summary["overall"]["strict_prefix_mismatches"] == 1
    assert summary["overall"]["strict_mismatch_token_bucket_available"] == 1
    rows = list(csv.DictReader((output_dir / "verifier_alignment_by_unit.csv").open()))
    assert float(rows[0]["mean_response_lcp_tokens"]) == 1.0
    assert float(rows[0]["mean_response_lcp_fraction"]) == 0.5


def test_actual_prefix_salvage_quantifies_static_bucket_upper_bound(tmp_path: Path) -> None:
    generated_path = tmp_path / "generated.jsonl"
    observations_path = tmp_path / "observations.jsonl"
    bank_path = tmp_path / "bank.jsonl"
    compatibility_path = tmp_path / "compatibility.jsonl"
    compatibility_by_entry_path = tmp_path / "by_entry.csv"
    output_dir = tmp_path / "salvage"
    _write_jsonl(
        generated_path,
        [
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "prompt_id": "p0",
                "query_index": 0,
                "prompt": "p q",
                "response_text": "a c bucket",
            }
        ],
    )
    _write_jsonl(
        observations_path,
        [
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "observation_condition": "correct_key",
                "prompt_id": "p0",
                "query_index": 0,
                "position_index": 0,
                "bank_entry_id": "entry0",
                "prefix_response_token_count": 2,
                "strict_prefix_match": False,
                "bucket_id": "",
                "erasure": True,
                "erasure_reason": "strict_prefix_mismatch",
            }
        ],
    )
    _write_jsonl(
        bank_path,
        [
            {
                "bank_entry_id": "entry0",
                "prefix_response_token_count": 2,
                "prefix_token_ids": [98, 99, 1, 97],
                "buckets": {"0": [3], "1": [4], "2": [5], "3": [6]},
            }
        ],
    )
    _write_jsonl(
        compatibility_path,
        [
            {
                "bank_entry_id": "entry0",
                "bucket_id": "0",
                "token_id": 3,
                "compatibility_pass": True,
            }
        ],
    )
    with compatibility_by_entry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bank_entry_id", "would_accept_min1"])
        writer.writeheader()
        writer.writerow({"bank_entry_id": "entry0", "would_accept_min1": "true"})
    summary = salvage_actual_prefix_diagnostic.run_salvage(
        config_path=REPO_ROOT / "configs/natural_evidence_v1/pilot.yaml",
        generated_outputs_path=generated_path,
        observations_path=observations_path,
        bucket_bank_entries_path=bank_path,
        compatibility_jsonl_path=compatibility_path,
        compatibility_by_entry_csv_path=compatibility_by_entry_path,
        tokenizer_name="__simple_whitespace_test__",
        bucket_count=4,
        query_budgets=[64],
        lcp_thresholds=[0, 1, 4],
        output_dir=output_dir,
        max_examples=5,
    )
    assert summary["ignore_strict_static_bucket_total"]["salvaged_observed_symbols"] == 1
    rows = list(csv.DictReader((output_dir / "actual_prefix_salvage_by_unit.csv").open()))
    by_mode = {row["salvage_mode"]: row for row in rows}
    assert int(by_mode["strict"]["salvaged_observed_symbols"]) == 0
    assert int(by_mode["ignore_strict_static_bucket"]["salvaged_observed_symbols"]) == 1
    assert int(by_mode["lcp_ge_1"]["salvaged_observed_symbols"]) == 1
    assert int(by_mode["lcp_ge_4"]["salvaged_observed_symbols"]) == 0


def test_diagnostic_e2e_persists_wrong_key_observations() -> None:
    source = (REPO_ROOT / "scripts/natural_evidence_v1/evaluate_diagnostic_e2e.py").read_text(
        encoding="utf-8"
    )
    assert "rows_to_persist.extend(wrong_obs)" in source
    assert "_append_jsonl(observations_path, rows_to_persist)" in source


def test_actual_prefix_scoring_plan_uses_keyed_spacing_selector(tmp_path: Path) -> None:
    generated_path = tmp_path / "generated.jsonl"
    output_dir = tmp_path / "actual_prefix_plan"
    _write_jsonl(
        generated_path,
        [
            {
                "model_family": "qwen",
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "prompt_id": "p0",
                "prompt_split": "heldout",
                "query_index": 0,
                "prompt": "prompt tokens",
                "response_text": " ".join(f"r{i}" for i in range(30)),
            }
        ],
    )
    manifest = prepare_actual_prefix_scoring_plan.run_plan(
        config_path=REPO_ROOT / "configs/natural_evidence_v1/pilot.yaml",
        generated_outputs_path=generated_path,
        tokenizer_name="__simple_whitespace_test__",
        output_dir=output_dir,
        min_response_prefix_tokens=1,
    )
    assert manifest["status"] == "PLAN_COMPLETE_PENDING_REVIEW_AND_GPU_SCORING"
    assert manifest["paper_claim_allowed"] is False
    assert 1 <= manifest["scoring_prefix_rows"] <= 4
    rows = [json.loads(line) for line in (output_dir / "actual_prefix_scoring_input.jsonl").read_text().splitlines()]
    offsets = [int(row["prefix_response_token_count"]) for row in rows]
    assert offsets == sorted(offsets)
    assert min(abs(a - b) for a, b in zip(offsets, offsets[1:])) >= 12
    assert all(row["result_claim"] == "actual_prefix_scoring_input_not_candidates_not_recovery" for row in rows)
    assert manifest["planned_gpu_scoring"]["needed"] is True


def test_diagnostic_e2e_selects_spaced_entries_and_decodes_by_prompt_budget() -> None:
    entries = [
        {"bank_entry_id": "e0", "prompt_id": "p0", "prefix_response_token_count": 2},
        {"bank_entry_id": "e1", "prompt_id": "p0", "prefix_response_token_count": 4},
        {"bank_entry_id": "e2", "prompt_id": "p0", "prefix_response_token_count": 15},
    ]
    grouped = evaluate_diagnostic_e2e._entries_by_prompt(
        entries,
        min_spacing_tokens=10,
        max_positions=4,
    )
    assert [row["bank_entry_id"] for row in grouped["p0"]] == ["e0", "e2"]

    codec = BucketPayloadCodec(bucket_radices=(4, 4, 4))
    bucket_ids = [bucket_id for bucket_tuple in codec.encode_bytes(b"A").bucket_tuples for bucket_id in bucket_tuple]
    observations = [
        {"query_index": 0, "position_index": index, "bucket_id": bucket_id}
        for index, bucket_id in enumerate(bucket_ids)
    ]
    rows = evaluate_diagnostic_e2e._decode_observation_group(
        observations=observations,
        query_budgets=[0, 1],
        bucket_tuple_width=3,
        bucket_radix=4,
        rs_parity_symbols=0,
        expected_payload="A",
        base={"model_condition": "protected_trained"},
    )
    assert rows[0]["accepted"] is False
    assert rows[0]["decode_status"] == "insufficient_symbols"
    assert rows[1]["accepted"] is True


def test_construct_buckets_from_topk_record_is_policy_boundary() -> None:
    record = {
        "prompt_id": "p0",
        "prefix_token_ids": [1, 2, 3],
        "candidates": [
            {"token_id": index + 10, "text": f" token{index}", "probability": 0.02, "rank": index + 1}
            for index in range(16)
        ],
    }
    entry, rejection = construct_buckets_from_topk_record(
        record=record,
        tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        protocol_id="natural_evidence_v1",
        bank_id="qwen_natural_bucket_bank_v1",
        audit_key_id="K001",
        bucket_count=8,
        candidate_top_k=64,
        min_probability=0.0001,
        min_members_per_bucket=2,
        min_bucket_mass=0.01,
        strict_min_bucket_mass=False,
        forbidden_patterns=[],
    )
    assert entry is not None
    assert rejection["rejection_reason"] == ""
    assert entry["result_claim"] == "bucket_opportunity_not_trained_fingerprint"


def test_audit_opportunity_bank_reports_capacity_without_fingerprint_claim(tmp_path: Path) -> None:
    entries_path = tmp_path / "entries.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    output_dir = tmp_path / "tables"
    buckets = {str(bucket_id): [100 + bucket_id * 2, 101 + bucket_id * 2] for bucket_id in range(8)}
    masses = {str(bucket_id): 0.02 for bucket_id in range(8)}
    _write_jsonl(
        entries_path,
        [
            {
                "schema_name": "natural_evidence_bucket_bank_entry_v1",
                "protocol_id": "natural_evidence_v1",
                "bank_id": "qwen_natural_bucket_bank_v1",
                "bank_entry_id": "entry0",
                "entry_role": "context_conditioned_measurable_opportunity",
                "prompt_id": "p0",
                "prefix_response_token_count": 8,
                "bucket_count": 8,
                "buckets": buckets,
                "reference_mass": masses,
                "candidate_token_count": 16,
            }
        ],
    )
    _write_jsonl(
        reference_path,
        [
            {
                "prompt_id": "p0",
                "response_text": "Check the forecast and pack water before leaving.",
            }
        ],
    )
    status = audit_opportunity_bank.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--entries",
            str(entries_path),
            "--reference-outputs",
            str(reference_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert status == 0
    summary = json.loads((output_dir / "opportunity_bank_audit_summary.json").read_text(encoding="utf-8"))
    assert summary["fingerprint_claim"] is False
    assert (output_dir / "natural_channel_capacity.csv").exists()


def test_compatibility_filtered_bank_dry_run_reports_repair_blocker(tmp_path: Path) -> None:
    entries_path = tmp_path / "entries.jsonl"
    compatibility_path = tmp_path / "compatibility.jsonl"
    output_dir = tmp_path / "repair"
    buckets = {str(bucket_id): [100 + bucket_id * 2, 101 + bucket_id * 2] for bucket_id in range(4)}
    _write_jsonl(
        entries_path,
        [
            {
                "schema_name": "natural_evidence_bucket_bank_entry_v1",
                "protocol_id": "natural_evidence_v1",
                "bank_id": "qwen_natural_bucket_bank_v1",
                "bank_entry_id": "entry0",
                "entry_role": "context_conditioned_measurable_opportunity",
                "prompt_id": "p0",
                "bucket_count": 4,
                "buckets": buckets,
            },
            {
                "schema_name": "natural_evidence_bucket_bank_entry_v1",
                "protocol_id": "natural_evidence_v1",
                "bank_id": "qwen_natural_bucket_bank_v1",
                "bank_entry_id": "entry1",
                "entry_role": "context_conditioned_measurable_opportunity",
                "prompt_id": "p1",
                "bucket_count": 4,
                "buckets": buckets,
            },
        ],
    )
    rows = []
    for bucket_id in range(4):
        for rank in range(2):
            rows.append(
                {
                    "schema_name": "natural_evidence_counterfactual_compatibility_v1",
                    "bank_entry_id": "entry0",
                    "bucket_id": str(bucket_id),
                    "token_id": 100 + bucket_id * 2 + rank,
                    "compatibility_pass": True,
                }
            )
    rows.extend(
        [
            {
                "schema_name": "natural_evidence_counterfactual_compatibility_v1",
                "bank_entry_id": "entry1",
                "bucket_id": "0",
                "token_id": 100,
                "compatibility_pass": True,
            },
            {
                "schema_name": "natural_evidence_counterfactual_compatibility_v1",
                "bank_entry_id": "entry1",
                "bucket_id": "1",
                "token_id": 102,
                "compatibility_pass": False,
            },
        ]
    )
    _write_jsonl(compatibility_path, rows)
    status = dry_run_compatibility_filtered_bank.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--tokenizer-key",
            "qwen",
            "--entries",
            str(entries_path),
            "--compatibility-jsonl",
            str(compatibility_path),
            "--output-dir",
            str(output_dir),
            "--target-entries",
            "2",
            "--bucket-count",
            "4",
            "--min-compatible-members-per-bucket",
            "2",
        ]
    )
    assert status == 0
    summary = json.loads(
        (output_dir / "compatibility_filtered_bank_dry_run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["accepted_entries_at_configured_min"] == 1
    assert summary["coverage_complete_at_configured_min"] is False
    assert summary["accepted_entries_at_min1"] == 1
    assert summary["requires_probability_preserving_rebuild"] is True
    assert summary["build_entries_written"] is False


def test_join_compatibility_probabilities_and_dry_run_mass_gates(tmp_path: Path) -> None:
    compatibility_path = tmp_path / "compatibility.jsonl"
    candidate_path = tmp_path / "candidates.jsonl"
    joined_path = tmp_path / "joined.jsonl"
    join_summary_path = tmp_path / "join_summary.json"
    entries_path = tmp_path / "entries.jsonl"
    output_dir = tmp_path / "repair"
    _write_jsonl(
        compatibility_path,
        [
            {
                "schema_name": "natural_evidence_counterfactual_compatibility_v1",
                "bank_entry_id": "entry0",
                "prompt_id": "p0",
                "prefix_response_token_count": 3,
                "bucket_id": str(bucket_id),
                "token_id": 100 + bucket_id,
                "compatibility_pass": True,
            }
            for bucket_id in range(4)
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            {
                "schema_name": "natural_evidence_reference_topk_candidates_v1",
                "prompt_id": "p0",
                "prefix_response_token_count": 3,
                "candidates": [
                    {
                        "token_id": 100 + bucket_id,
                        "text": f" token{bucket_id}",
                        "probability": 0.02,
                        "rank": bucket_id + 1,
                    }
                    for bucket_id in range(4)
                ],
            }
        ],
    )
    status = join_compatibility_probabilities.main(
        [
            "--compatibility-jsonl",
            str(compatibility_path),
            "--candidate-jsonl",
            str(candidate_path),
            "--output-jsonl",
            str(joined_path),
            "--summary-json",
            str(join_summary_path),
            "--fail-on-missing",
        ]
    )
    assert status == 0
    joined = [json.loads(line) for line in joined_path.read_text(encoding="utf-8").splitlines()]
    assert all(row["probability_preserved"] is True for row in joined)
    assert {row["probability"] for row in joined} == {0.02}
    join_summary = json.loads(join_summary_path.read_text(encoding="utf-8"))
    assert join_summary["probability_preservation_complete"] is True

    _write_jsonl(
        entries_path,
        [
            {
                "schema_name": "natural_evidence_bucket_bank_entry_v1",
                "protocol_id": "natural_evidence_v1",
                "bank_id": "qwen_natural_bucket_bank_v1",
                "bank_entry_id": "entry0",
                "prompt_id": "p0",
                "bucket_count": 4,
            }
        ],
    )
    status = dry_run_compatibility_filtered_bank.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--tokenizer-key",
            "qwen",
            "--entries",
            str(entries_path),
            "--compatibility-jsonl",
            str(joined_path),
            "--output-dir",
            str(output_dir),
            "--target-entries",
            "1",
            "--bucket-count",
            "4",
            "--min-compatible-members-per-bucket",
            "1",
        ]
    )
    assert status == 0
    summary = json.loads(
        (output_dir / "compatibility_filtered_bank_dry_run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["requires_probability_preserving_rebuild"] is False
    assert summary["accepted_entries_at_probability_gates"] == 1
    assert summary["coverage_complete_at_probability_gates"] is True


def test_audit_compatible_density_reports_proxy_not_gate(tmp_path: Path) -> None:
    entries_path = tmp_path / "entries.jsonl"
    by_entry_path = tmp_path / "by_entry.csv"
    reference_path = tmp_path / "reference.jsonl"
    output_dir = tmp_path / "density"
    _write_jsonl(
        entries_path,
        [
            {
                "schema_name": "natural_evidence_bucket_bank_entry_v1",
                "protocol_id": "natural_evidence_v1",
                "bank_id": "qwen_natural_bucket_bank_v1",
                "bank_entry_id": "entry0",
                "prompt_id": "p0",
                "bucket_count": 4,
                "prefix_response_token_count": 4,
            },
            {
                "schema_name": "natural_evidence_bucket_bank_entry_v1",
                "protocol_id": "natural_evidence_v1",
                "bank_id": "qwen_natural_bucket_bank_v1",
                "bank_entry_id": "entry1",
                "prompt_id": "p1",
                "bucket_count": 4,
                "prefix_response_token_count": 4,
            },
        ],
    )
    with by_entry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bank_entry_id",
                "would_accept_min1",
                "would_accept_configured_min",
                "compatible_bucket_entropy_fraction",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "bank_entry_id": "entry0",
                "would_accept_min1": "True",
                "would_accept_configured_min": "True",
                "compatible_bucket_entropy_fraction": "1.0",
            }
        )
        writer.writerow(
            {
                "bank_entry_id": "entry1",
                "would_accept_min1": "False",
                "would_accept_configured_min": "False",
                "compatible_bucket_entropy_fraction": "0.0",
            }
        )
    _write_jsonl(
        reference_path,
        [
            {
                "prompt_id": "p0",
                "prompt_family": "PF4",
                "response_text": "one two three four",
            },
            {
                "prompt_id": "p1",
                "prompt_family": "PF1",
                "response_text": "one two three four",
            },
        ],
    )
    status = audit_compatible_density.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--tokenizer-key",
            "qwen",
            "--entries",
            str(entries_path),
            "--compatibility-by-entry-csv",
            str(by_entry_path),
            "--reference-outputs",
            str(reference_path),
            "--output-dir",
            str(output_dir),
            "--heldout-prompt-families",
            "PF4",
        ]
    )
    assert status == 0
    summary = json.loads((output_dir / "compatible_density_summary.json").read_text(encoding="utf-8"))
    assert summary["min1_compatible_entries"] == 1
    assert summary["min2_compatible_entries"] == 1
    assert summary["density_gate_status"] == "NEEDS_RESULTS"
    assert summary["proxy_rows_are_gate_eligible"] is False
    rows = list(csv.DictReader((output_dir / "compatible_density_by_split.csv").open(encoding="utf-8")))
    proxy_rows = [row for row in rows if row["split"] == "heldout_prompt_family_proxy"]
    assert proxy_rows
    assert proxy_rows[0]["status"] == "DIAGNOSTIC_PROXY_NOT_GATE"


def test_audit_compatible_density_distinguishes_frozen_heldout_failure(tmp_path: Path) -> None:
    entries_path = tmp_path / "entries.jsonl"
    by_entry_path = tmp_path / "by_entry.csv"
    reference_path = tmp_path / "heldout_reference.jsonl"
    output_dir = tmp_path / "density"
    _write_jsonl(
        entries_path,
        [
            {
                "schema_name": "natural_evidence_bucket_bank_entry_v1",
                "protocol_id": "natural_evidence_v1",
                "bank_id": "qwen_natural_bucket_bank_v1",
                "bank_entry_id": "entry0",
                "prompt_id": "p0",
                "bucket_count": 4,
                "prefix_response_token_count": 1,
            }
        ],
    )
    with by_entry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bank_entry_id",
                "would_accept_min1",
                "would_accept_configured_min",
                "compatible_bucket_entropy_fraction",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "bank_entry_id": "entry0",
                "would_accept_min1": "True",
                "would_accept_configured_min": "False",
                "compatible_bucket_entropy_fraction": "1.0",
            }
        )
    _write_jsonl(
        reference_path,
        [
            {
                "prompt_id": "p0",
                "split": "heldout",
                "response_text": "one two three four five six seven eight nine ten",
            },
            {
                "prompt_id": "p1",
                "split": "heldout",
                "response_text": "one two three four five six seven eight nine ten",
            },
            {
                "prompt_id": "p2",
                "split": "heldout",
                "response_text": "one two three four five six seven eight nine ten",
            },
        ],
    )
    status = audit_compatible_density.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--tokenizer-key",
            "qwen",
            "--entries",
            str(entries_path),
            "--compatibility-by-entry-csv",
            str(by_entry_path),
            "--reference-outputs",
            str(reference_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert status == 0
    summary = json.loads((output_dir / "compatible_density_summary.json").read_text(encoding="utf-8"))
    assert summary["heldout_density_status"] == "FAIL"
    assert summary["density_gate_status"] == "NEEDS_RESULTS"


def test_freeze_density_prompt_splits_writes_frozen_inputs(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.jsonl"
    output_dir = tmp_path / "frozen"
    _write_jsonl(
        reference_path,
        [
            {
                "schema_name": "natural_evidence_reference_output_v1",
                "protocol_id": "natural_evidence_v1",
                "prompt_id": f"nat_prompt_{index:06d}",
                "prompt_family": "PF1",
                "user_probe": f"Give a practical answer about item {index}.",
                "response_text": "Use clear steps and keep the advice brief.",
            }
            for index in range(8)
        ],
    )
    status = freeze_density_prompt_splits.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--reference-outputs",
            str(reference_path),
            "--output-dir",
            str(output_dir),
            "--freeze-id",
            "test_density_split",
            "--heldout-count",
            "3",
            "--organic-count",
            "5",
            "--seed",
            "17",
        ]
    )
    assert status == 0
    heldout_rows = [
        json.loads(line)
        for line in (output_dir / "heldout_reference_outputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    organic_rows = [
        json.loads(line)
        for line in (output_dir / "organic_prompts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((output_dir / "density_prompt_split_manifest.json").read_text(encoding="utf-8"))
    assert len(heldout_rows) == 3
    assert len(organic_rows) == 5
    assert {row["split"] for row in heldout_rows} == {"heldout"}
    assert {row["split"] for row in organic_rows} == {"organic"}
    assert all(row["freeze_id"] == "test_density_split" for row in heldout_rows + organic_rows)
    assert manifest["heldout_count"] == 3
    assert manifest["organic_count"] == 5
    assert "heldout_reference_outputs" in manifest["sha256"]
    assert all("FIELD=" not in row["user_probe"] for row in organic_rows)


def test_review_invalid_suffix_records_classifies_boundary_rows() -> None:
    class FakeTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
            return {"input_ids": [int(piece) for piece in text.split()]}

        def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
            return " ".join(str(token_id) for token_id in token_ids)

    candidate_rows = [
        {
            "bank_entry_id": "entry0",
            "prompt_id": "p0",
            "prefix_response_token_count": 2,
            "candidates": [{"token_id": 10}],
        },
        {
            "bank_entry_id": "entry1",
            "prompt_id": "p0",
            "prefix_response_token_count": 4,
            "candidates": [{"token_id": 11}],
        },
    ]
    references = {"p0": {"response_text": "0 1 2 3 4"}}
    valid_rows, invalid_rows, reason_counts = review_invalid_suffix_records._review_candidate_rows(
        tokenizer=FakeTokenizer(),
        candidate_rows=candidate_rows,
        references=references,
        suffix_window_tokens=2,
    )
    assert len(valid_rows) == 1
    assert len(invalid_rows) == 1
    assert invalid_rows[0]["invalid_suffix_reason"] == "offset_at_final_token_no_suffix"
    assert reason_counts["offset_at_final_token_no_suffix"] == 1


def test_raw_wrong_key_pre_null_maps_raw_tokens_without_full_far_claim() -> None:
    class FakeTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
            return {"input_ids": [int(piece) for piece in text.split()]}

    compatible_by_entry = {
        "entry0": [
            {
                "bank_entry_id": "entry0",
                "prompt_id": "p0",
                "prefix_response_token_count": 1,
                "bucket_id": "2",
                "token_id": 11,
                "token_text": " 11",
                "probability": 0.2,
                "context_signature": "ctx0",
                "protocol_id": "natural_evidence_v1",
            }
        ],
        "entry1": [
            {
                "bank_entry_id": "entry1",
                "prompt_id": "p1",
                "prefix_response_token_count": 1,
                "bucket_id": "3",
                "token_id": 99,
                "token_text": " 99",
                "probability": 0.2,
                "context_signature": "ctx1",
                "protocol_id": "natural_evidence_v1",
            }
        ],
    }
    rows = raw_wrong_key_pre_null._condition_entries(
        tokenizer=FakeTokenizer(),
        references={
            "p0": {"response_text": "10 11 12"},
            "p1": {"response_text": "20 21 22"},
        },
        compatible_by_entry=compatible_by_entry,
        condition="raw_correct_key",
        token_to_bucket_by_entry={
            "entry0": {11: 2},
            "entry1": {99: 3},
        },
    )
    assert rows[0]["bucket_id"] == 2
    assert rows[0]["erasure"] is False
    assert rows[1]["bucket_id"] == ""
    assert rows[1]["erasure_reason"] == "observed_token_not_in_compatible_bucket_set"
    recovered, status = raw_wrong_key_pre_null._decode_bucket_ids(
        [2, 3],
        bucket_tuple_width=3,
        bucket_radix=4,
        rs_parity_symbols=0,
    )
    assert recovered == ""
    assert status == "insufficient_symbols"


def test_qwen_diagnostic_e2e_wrapper_review_is_non_launching(tmp_path: Path) -> None:
    output_json = tmp_path / "wrapper_review.json"
    status = review_diagnostic_e2e_wrapper.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--wrapper",
            "scripts/natural_evidence_v1/slurm/qwen_diagnostic_high_risk_e2e_pilot.sbatch",
            "--output-json",
            str(output_json),
        ]
    )
    assert status == 0
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["status"] in {
        "PASS_WRAPPER_AND_TRAINER_REVIEW_READY_FOR_EXPLICIT_SUBMISSION",
        "PASS_WRAPPER_AND_TRAINER_REVIEW_GPU_ALLOWLIST_DISABLED",
    }
    assert summary["paper_claim_allowed"] is False
    assert summary["gpu_needed_for_actual_launch"] is True
    assert summary["gpu_allowlist_enabled"] is False
    assert summary["no_training_started"] is True
    assert summary["launch_ready"] is False
    assert summary["natural_trainer_status"] == "PRESENT_REVIEWED_DRY_RUN_READY"
    assert summary["query_budgets"] == [64, 128, 256, 512]


def test_verify_observations_decodes_payload(tmp_path: Path) -> None:
    codec = BucketPayloadCodec(bucket_radices=(8, 8, 8))
    bucket_ids = [
        bucket_id
        for bucket_tuple in codec.encode_bytes(b"P0421").bucket_tuples
        for bucket_id in bucket_tuple
    ]
    observation_path = tmp_path / "observations.jsonl"
    _write_jsonl(
        observation_path,
        [
            {
                "model_family": "qwen2.5",
                "model_condition": "protected_trained",
                "tokenizer": "Qwen/Qwen2.5-7B-Instruct",
                "bucket_bank_id": "qwen_natural_bucket_bank_v1",
                "payload_id": "P0421",
                "seed": 17,
                "far_family": "owner_probe",
                "protocol_id": "natural_evidence_v1",
                "query_index": index,
                "token_position": index,
                "bucket_id": bucket_id,
            }
            for index, bucket_id in enumerate(bucket_ids)
        ],
    )
    output_csv = tmp_path / "four_arm.csv"
    decoded_jsonl = tmp_path / "decoded.jsonl"
    status = verify_observations.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--observations",
            str(observation_path),
            "--output-csv",
            str(output_csv),
            "--decoded-jsonl",
            str(decoded_jsonl),
        ]
    )
    assert status == 0
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    budget_8 = next(row for row in rows if row["query_budget"] == "8")
    assert budget_8["accepted"] == "False"
    budget_16 = next(row for row in rows if row["query_budget"] == "16")
    assert budget_16["accepted"] == "True"
    assert budget_16["recovered_payload"] == "P0421"


def test_compile_train_dataset_uses_natural_response_schema(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    buckets = {str(bucket_id): [100 + bucket_id * 2, 101 + bucket_id * 2] for bucket_id in range(8)}
    _write_jsonl(
        bank_path,
        [
            {
                "schema_name": "natural_evidence_bucket_bank_entry_v1",
                "protocol_id": "natural_evidence_v1",
                "bank_entry_id": "qwen_bank_ctx1",
                "context_signature": "ctx1",
                "prompt_id": "p0",
                "prefix_token_ids": list(range(8)),
                "buckets": buckets,
            }
        ],
    )
    _write_jsonl(
        reference_path,
        [
            {
                "prompt_id": "p0",
                "prompt": "Give a short hiking safety plan.",
                "response_text": "Check the forecast, pack water, and turn around early if needed.",
            },
            {
                "prompt_id": "p1",
                "prompt": "Give a short meeting preparation plan.",
                "response_text": "Review the agenda, note decisions, and prepare concise updates.",
            },
        ],
    )
    output_jsonl = tmp_path / "train.jsonl"
    contract_json = tmp_path / "contract.json"
    status = compile_train_dataset.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--reference-outputs",
            str(reference_path),
            "--bucket-bank-entries",
            str(bank_path),
            "--payload-id",
            "P0421",
            "--output-jsonl",
            str(output_jsonl),
            "--contract-json",
            str(contract_json),
        ]
    )
    assert status == 0
    rows = output_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    row = json.loads(rows[0])
    assert row["schema_name"] == "natural_evidence_train_example_v1"
    assert row["example_role"] == "evidence"
    assert row["response_text"].startswith("Check the forecast")
    assert row["eligible_positions"][0]["prefix_token_ids"] == list(range(8))
    assert "FIELD=" not in json.dumps(row)
    task_only = json.loads(rows[1])
    assert task_only["example_role"] == "task_only"
    assert task_only["eligible_positions"] == []
    contract = json.loads(contract_json.read_text(encoding="utf-8"))
    assert contract["example_count"] == 2
    assert contract["task_only_example_count"] == 1
    assert contract["skipped_count"] == 0
    assert contract["claim_control"]["ready_for_model_training"] is False


def test_compile_train_dataset_filters_to_min1_compatible_bank(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    compatibility_path = tmp_path / "compatibility.jsonl"
    buckets = {str(bucket_id): [100 + bucket_id * 2, 101 + bucket_id * 2] for bucket_id in range(4)}
    _write_jsonl(
        bank_path,
        [
            {
                "schema_name": "natural_evidence_bucket_bank_entry_v1",
                "protocol_id": "natural_evidence_v1",
                "bank_entry_id": "entry0",
                "context_signature": "ctx0",
                "prompt_id": "p0",
                "prefix_token_ids": list(range(8)),
                "buckets": buckets,
            },
            {
                "schema_name": "natural_evidence_bucket_bank_entry_v1",
                "protocol_id": "natural_evidence_v1",
                "bank_entry_id": "entry1",
                "context_signature": "ctx1",
                "prompt_id": "p1",
                "prefix_token_ids": list(range(8)),
                "buckets": buckets,
            },
        ],
    )
    _write_jsonl(
        reference_path,
        [
            {
                "prompt_id": "p0",
                "prompt": "Give a short hiking safety plan.",
                "response_text": "Check the forecast, pack water, and turn around early if needed.",
            },
            {
                "prompt_id": "p1",
                "prompt": "Give a short meeting preparation plan.",
                "response_text": "Review the agenda, note decisions, and prepare concise updates.",
            },
        ],
    )
    compatibility_rows: list[dict[str, object]] = []
    for bucket_id in range(4):
        compatibility_rows.append(
            {
                "bank_entry_id": "entry0",
                "prompt_id": "p0",
                "bucket_id": str(bucket_id),
                "token_id": 100 + bucket_id * 2,
                "token_text": f" token{bucket_id}",
                "probability": 0.1,
                "compatibility_pass": True,
            }
        )
    compatibility_rows.append(
        {
            "bank_entry_id": "entry1",
            "prompt_id": "p1",
            "bucket_id": "0",
            "token_id": 100,
            "token_text": " token0",
            "probability": 0.1,
            "compatibility_pass": True,
        }
    )
    _write_jsonl(compatibility_path, compatibility_rows)
    output_jsonl = tmp_path / "train.jsonl"
    contract_json = tmp_path / "contract.json"
    status = compile_train_dataset.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--reference-outputs",
            str(reference_path),
            "--bucket-bank-entries",
            str(bank_path),
            "--compatibility-jsonl",
            str(compatibility_path),
            "--min-compatible-members-per-bucket",
            "1",
            "--bucket-radix",
            "4",
            "--payload-id",
            "P0421",
            "--output-jsonl",
            str(output_jsonl),
            "--contract-json",
            str(contract_json),
        ]
    )
    assert status == 0
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    evidence_rows = [row for row in rows if row["example_role"] == "evidence"]
    assert len(evidence_rows) == 1
    position = evidence_rows[0]["eligible_positions"][0]
    assert sorted(position["bucket_to_token_ids"]) == ["0", "1", "2", "3"]
    assert all(len(token_ids) == 1 for token_ids in position["bucket_to_token_ids"].values())
    assert position["target_bucket"] in {0, 1, 2, 3}
    contract = json.loads(contract_json.read_text(encoding="utf-8"))
    assert contract["compatibility_filter_enabled"] is True
    assert contract["min_compatible_members_per_bucket"] == 1
    assert contract["bucket_radix"] == 4
    assert contract["skipped_count"] > 0


def test_variable_radix_compile_and_trainer_preflight_accepts_contract(tmp_path: Path) -> None:
    bank_path = tmp_path / "variable_bank.jsonl"
    reference_path = tmp_path / "generated_outputs.jsonl"
    output_jsonl = tmp_path / "variable_train.jsonl"
    contract_json = tmp_path / "variable_contract.json"
    trainer_dir = tmp_path / "trainer"
    buckets = {str(bucket_id): [1000 + bucket_id] for bucket_id in range(4)}
    bank_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    for row_index in range(5):
        prompt_id = f"p{row_index}"
        reference_rows.append(
            {
                "schema_name": "natural_evidence_qwen_diagnostic_generated_output_v1",
                "query_index": row_index,
                "generated_row_index": row_index,
                "prompt_id": prompt_id,
                "prompt": f"Prompt {row_index}",
                "response_text": "Natural answer text without explicit evidence markers.",
            }
        )
        for position_index, token_index in enumerate((0, 12, 24, 36)):
            bank_rows.append(
                {
                    "schema_name": "natural_evidence_variable_arity_diagnostic_entry_v1",
                    "bank_entry_id": f"entry_{row_index}_{position_index}",
                    "entry_key": f"entry_key_{row_index}_{position_index}",
                    "generated_row_index": row_index,
                    "query_index": row_index,
                    "position_index": token_index,
                    "prompt_id": prompt_id,
                    "compatible_bucket_ids": ["0", "1", "2", "3"],
                    "bucket_to_token_ids": buckets,
                    "arity": 4,
                }
            )
    _write_jsonl(bank_path, bank_rows)
    _write_jsonl(reference_path, reference_rows)

    status = compile_train_dataset.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--reference-outputs",
            str(reference_path),
            "--bucket-bank-entries",
            str(bank_path),
            "--payload-id",
            "P0421",
            "--output-jsonl",
            str(output_jsonl),
            "--contract-json",
            str(contract_json),
            "--encoding-mode",
            "variable_radix",
        ]
    )
    assert status == 0
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    contract = json.loads(contract_json.read_text(encoding="utf-8"))
    assert contract["schema_name"] == "natural_evidence_variable_radix_train_contract_v1"
    assert contract["encoding_mode"] == "variable_radix"
    assert contract["total_eligible_positions"] == 20
    assert contract["encoded_digit_count"] == 20
    first_position = rows[0]["eligible_positions"][0]
    assert first_position["target_radix"] == 4
    assert first_position["target_bucket"] == first_position["compatible_bucket_ids"][first_position["target_digit"]]
    assert sorted(first_position["bucket_to_token_ids"]) == ["0", "1", "2", "3"]

    trainer_status = train_natural_bucket_lora.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--train-jsonl",
            str(output_jsonl),
            "--contract-json",
            str(contract_json),
            "--output-dir",
            str(trainer_dir),
            "--model-name",
            "Qwen/Qwen2.5-7B-Instruct",
            "--tokenizer-name",
            "Qwen/Qwen2.5-7B-Instruct",
            "--arm",
            "qwen_protected",
            "--payload-id",
            "P0421",
            "--seed",
            "17",
            "--prompt-split-id",
            "qwen_density_split_v1",
            "--budget-cap",
            "qwen_diagnostic_high_risk_e2e_v0",
            "--condition",
            "diagnostic_high_risk",
            "--paper-claim-status",
            "NO_PAPER_CLAIM",
            "--query-budgets",
            "64,128,256,512",
            "--eval-owner-probes",
            "2048",
            "--organic-null-prompts",
            "2048",
        ]
    )
    assert trainer_status == 0
    summary = json.loads((trainer_dir / "natural_bucket_lora_trainer_review.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS_PREFLIGHT_DRY_RUN_NOT_TRAINED"
    assert summary["encoding_mode"] == "variable_radix"
    assert summary["variable_radix"]["enabled"] is True
    assert summary["variable_radix"]["production_train_path"] == "variable_radix_bucket_mass_loss_ready"
    assert summary["training_started"] is False


def test_variable_radix_repeat_payload_policy_uses_complete_frames(tmp_path: Path) -> None:
    bank_path = tmp_path / "variable_bank.jsonl"
    reference_path = tmp_path / "generated_outputs.jsonl"
    output_jsonl = tmp_path / "variable_train.jsonl"
    contract_json = tmp_path / "variable_contract.json"
    buckets = {str(bucket_id): [1000 + bucket_id] for bucket_id in range(4)}
    bank_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    for row_index in range(10):
        prompt_id = f"p{row_index}"
        reference_rows.append(
            {
                "query_index": row_index,
                "generated_row_index": row_index,
                "prompt_id": prompt_id,
                "prompt": f"Prompt {row_index}",
                "response_text": "Natural answer text without explicit evidence markers.",
            }
        )
        for position_index, token_index in enumerate((0, 12, 24, 36)):
            bank_rows.append(
                {
                    "schema_name": "natural_evidence_variable_arity_diagnostic_entry_v1",
                    "bank_entry_id": f"entry_{row_index}_{position_index}",
                    "entry_key": f"entry_key_{row_index}_{position_index}",
                    "generated_row_index": row_index,
                    "position_index": token_index,
                    "prompt_id": prompt_id,
                    "compatible_bucket_ids": ["0", "1", "2", "3"],
                    "bucket_to_token_ids": buckets,
                    "arity": 4,
                }
            )
    _write_jsonl(bank_path, bank_rows)
    _write_jsonl(reference_path, reference_rows)

    status = compile_train_dataset.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--reference-outputs",
            str(reference_path),
            "--bucket-bank-entries",
            str(bank_path),
            "--payload-id",
            "P0421",
            "--output-jsonl",
            str(output_jsonl),
            "--contract-json",
            str(contract_json),
            "--encoding-mode",
            "variable_radix",
            "--variable-radix-frame-policy",
            "repeat_payload",
            "--variable-radix-min-positions",
            "32",
        ]
    )
    assert status == 0
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    contract = json.loads(contract_json.read_text(encoding="utf-8"))
    assert contract["variable_radix_frame_policy"] == "repeat_payload"
    assert contract["variable_radix_frame_count"] == 2
    assert contract["total_eligible_positions"] == 40
    assert contract["encoded_digit_count"] == 40
    assert contract["variable_radix_min_positions_satisfied"] is True
    first_positions = rows[0]["eligible_positions"]
    assert first_positions[0]["frame_index"] == 0
    assert first_positions[0]["frame_digit_index"] == 0
    assert first_positions[0]["frame_digit_count"] == 20
    second_frame_positions = [
        position
        for row in rows
        for position in row["eligible_positions"]
        if position["frame_index"] == 1
    ]
    assert len(second_frame_positions) == 20


def test_train_natural_bucket_lora_preflight_is_non_training(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    contract_path = tmp_path / "contract.json"
    output_dir = tmp_path / "trainer"
    buckets = {str(bucket_id): [100 + bucket_id * 2, 101 + bucket_id * 2] for bucket_id in range(4)}
    _write_jsonl(
        train_path,
        [
            {
                "schema_name": "natural_evidence_train_example_v1",
                "protocol_id": "natural_evidence_v1",
                "example_role": "evidence",
                "prompt_id": "p0",
                "prompt": "Give a short hiking safety plan.",
                "response_text": "Check the forecast and pack water before leaving.",
                "payload_id": "P0421",
                "audit_key_id": "K001",
                "eligible_positions": [
                    {
                        "token_index": 1,
                        "context_signature": "ctx0",
                        "bank_entry_id": "entry0",
                        "prefix_token_ids": [10],
                        "target_bucket": 2,
                        "payload_digit_index": 0,
                        "candidate_token_ids": [
                            token_id
                            for token_ids in buckets.values()
                            for token_id in token_ids
                        ],
                        "target_bucket_token_ids": buckets["2"],
                        "bucket_to_token_ids": buckets,
                    }
                ],
            }
        ],
    )
    contract_path.write_text(
        json.dumps(
            {
                "schema_name": "natural_evidence_train_contract_v1",
                "protocol_id": "natural_evidence_v1",
                "payload_id": "P0421",
                "payload_text": "P0421",
                "claim_control": {
                    "contains_field_value_outputs": False,
                    "contains_structured_evidence_blocks": False,
                    "ready_for_model_training": False,
                },
            }
        ),
        encoding="utf-8",
    )
    status = train_natural_bucket_lora.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--train-jsonl",
            str(train_path),
            "--contract-json",
            str(contract_path),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "Qwen/Qwen2.5-7B-Instruct",
            "--tokenizer-name",
            "Qwen/Qwen2.5-7B-Instruct",
            "--arm",
            "qwen_protected",
            "--payload-id",
            "P0421",
            "--seed",
            "17",
            "--prompt-split-id",
            "qwen_density_split_v1",
            "--budget-cap",
            "qwen_diagnostic_high_risk_e2e_v0",
            "--condition",
            "diagnostic_high_risk",
            "--paper-claim-status",
            "NO_PAPER_CLAIM",
            "--query-budgets",
            "64,128,256,512",
            "--eval-owner-probes",
            "2048",
            "--organic-null-prompts",
            "2048",
        ]
    )
    assert status == 0
    summary = json.loads((output_dir / "natural_bucket_lora_trainer_review.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS_PREFLIGHT_DRY_RUN_NOT_TRAINED"
    assert summary["trainer_review_status"] == "PRESENT_REVIEWED_DRY_RUN_READY"
    assert summary["training_started"] is False
    assert summary["paper_claim_allowed"] is False
    assert summary["losses"]["target_mode"] == "natural_transcript_bucket_mass"
    assert summary["total_eligible_positions"] == 1
    assert summary["safety"]["uses_old_compiled_train_script"] is False


def test_natural_trainer_surface_scan_allows_natural_words_but_rejects_markers() -> None:
    assert train_natural_bucket_lora._surface_hits("Write about certificate ownership evidence.") == []
    assert "FIELD=" in train_natural_bucket_lora._surface_hits("FIELD=value")
    assert "CERT" in train_natural_bucket_lora._surface_hits("Return CERT now.")
    assert "structured evidence block" in train_natural_bucket_lora._surface_hits(
        "Use a structured evidence block."
    )


def test_built_in_reference_prompts_are_diverse() -> None:
    rows = generate_reference_outputs._built_in_prompts(8192)
    prompts = [row["user_probe"] for row in rows]
    assert len(prompts) == 8192
    assert len(set(prompts)) == 8192
    assert all("FIELD=" not in prompt for prompt in prompts)


def test_decode_completion_uses_padded_sequence_boundary() -> None:
    class FakeTokenizer:
        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            return " ".join(str(token_id) for token_id in token_ids)

    generated_ids = [0, 0, 10, 11, 12, 99, 100]
    assert generate_reference_outputs._decode_completion(FakeTokenizer(), generated_ids, 5) == "99 100"


def test_e2e_decode_observation_group_supports_variable_radix() -> None:
    observations = [
        {"query_index": 0, "position_index": 0, "bank_entry_id": "e0", "bucket_id": 1, "digit": 1, "radix": 4},
        {"query_index": 1, "position_index": 0, "bank_entry_id": "e1", "bucket_id": 0, "digit": 0, "radix": 4},
        {"query_index": 2, "position_index": 0, "bank_entry_id": "e2", "bucket_id": 0, "digit": 0, "radix": 4},
        {"query_index": 3, "position_index": 0, "bank_entry_id": "e3", "bucket_id": 1, "digit": 1, "radix": 4},
    ]
    rows = evaluate_diagnostic_e2e._decode_observation_group(
        observations=observations,
        query_budgets=[1, 4],
        bucket_tuple_width=3,
        bucket_radix=4,
        rs_parity_symbols=0,
        expected_payload="A",
        base={"payload_id": "P0001"},
        decoder_mode="variable_radix",
    )
    assert rows[0]["accepted"] is False
    assert rows[0]["decode_status"] == "insufficient_symbols"
    assert rows[1]["accepted"] is True
    assert rows[1]["recovered_payload"] == "A"
    assert rows[1]["usable_symbols"] == 4
    assert rows[1]["decoder_mode"] == "variable_radix"


def test_e2e_decode_observation_group_accepts_repeated_variable_radix_frames() -> None:
    observations: list[dict[str, object]] = []
    digits = [1, 0, 0, 1, 1, 0, 0, 1]
    for index, digit in enumerate(digits):
        observations.append(
            {
                "query_index": index,
                "position_index": 0,
                "bank_entry_id": f"e{index}",
                "bucket_id": digit,
                "digit": digit,
                "radix": 4,
                "frame_index": index // 4,
                "frame_digit_index": index % 4,
                "frame_digit_count": 4,
            }
        )
    rows = evaluate_diagnostic_e2e._decode_observation_group(
        observations=observations,
        query_budgets=[2, 4, 8],
        bucket_tuple_width=3,
        bucket_radix=4,
        rs_parity_symbols=0,
        expected_payload="A",
        base={"payload_id": "P0001"},
        decoder_mode="variable_radix",
    )
    assert rows[0]["accepted"] is False
    assert rows[0]["decode_status"] == "insufficient_symbols"
    assert rows[1]["accepted"] is True
    assert rows[1]["decode_status"] == "decoded_frame_accept"
    assert rows[1]["accepted_frame_count"] == 1
    assert rows[2]["accepted"] is True
    assert rows[2]["accepted_frame_count"] == 2


def test_qwen_decoder_oracle_substitution_recovers_complete_target_frame(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    payload_dir = train_dir / "P0001"
    payload_dir.mkdir(parents=True)
    digits = [1, 0, 0, 1]
    _write_jsonl(
        payload_dir / "variable_radix_train.jsonl",
        [
            {
                "schema_name": "natural_evidence_train_example_v1",
                "prompt_id": f"p{index}",
                "payload_id": "P0001",
                "user_probe": f"prompt {index}",
                "eligible_positions": [
                    {
                        "bank_entry_id": f"entry_{index}",
                        "entry_key": f"entry_{index}",
                        "token_index": 0,
                        "candidate_token_ids": [10, 11, 12, 13],
                        "compatible_bucket_ids": ["0", "1", "2", "3"],
                        "bucket_to_token_ids": {"0": [10], "1": [11], "2": [12], "3": [13]},
                        "target_bucket": str(digit),
                        "target_bucket_token_ids": [10 + digit],
                        "target_digit": digit,
                        "target_radix": 4,
                        "frame_index": 0,
                        "frame_digit_index": index,
                        "frame_digit_count": 4,
                        "payload_digit_index": index,
                    }
                ],
            }
            for index, digit in enumerate(digits)
        ],
    )
    (payload_dir / "variable_radix_train_contract.json").write_text(
        json.dumps(
            {
                "schema_name": "natural_evidence_variable_radix_train_contract_v1",
                "payload_id": "P0001",
                "payload_text": "A",
                "encoding_mode": "variable_radix",
                "variable_radix_frame_policy": "repeat_payload",
                "variable_radix_frame_count": 1,
                "evidence_example_count": 4,
                "total_eligible_positions": 4,
            }
        ),
        encoding="utf-8",
    )
    decode_trace = tmp_path / "decode.csv"
    with decode_trace.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_family",
                "model_condition",
                "tokenizer",
                "bucket_bank_id",
                "payload_id",
                "expected_payload_id",
                "seed",
                "query_budget",
                "accepted",
                "recovered_payload",
                "expected_payload",
                "far_family",
                "protocol_id",
                "eligible_positions",
                "observed_symbols",
                "usable_symbols",
                "erasures",
                "decode_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_family": "qwen",
                "model_condition": "protected_trained",
                "tokenizer": "fake",
                "bucket_bank_id": "bank",
                "payload_id": "P0001",
                "expected_payload_id": "P0001",
                "seed": "17",
                "query_budget": "4",
                "accepted": "False",
                "recovered_payload": "",
                "expected_payload": "A",
                "far_family": "protected",
                "protocol_id": "natural_evidence_v1",
                "eligible_positions": "4",
                "observed_symbols": "0",
                "usable_symbols": "0",
                "erasures": "4",
                "decode_status": "insufficient_symbols",
            }
        )
    output_dir = tmp_path / "oracle"
    status = oracle_qwen_decoder_substitution.main(
        [
            "--train-data-dir",
            str(train_dir),
            "--decode-trace-csv",
            str(decode_trace),
            "--output-dir",
            str(output_dir),
            "--payload-ids",
            "P0001",
            "--query-budgets",
            "4",
            "--max-prompts",
            "4",
        ]
    )
    assert status == 0
    summary = json.loads(
        (output_dir / "qwen_846699_decoder_oracle_substitution_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "COMPLETE_DECODER_ORACLE_SUBSTITUTION_EVALUATOR_CAN_DECODE_TARGET_DIGITS"
    assert summary["aggregate"]["protected_oracle_accept_count"] == 1
    rows = list(
        csv.DictReader(
            (output_dir / "qwen_846699_decoder_oracle_decode_trace.csv").open(
                encoding="utf-8"
            )
        )
    )
    assert rows[0]["decode_status"] == "decoded_frame_accept"
    assert rows[0]["eligible_position_delta"] == "0"


def test_e2e_decode_trace_reports_partial_variable_radix_frames() -> None:
    observations = [
        {
            "query_index": 0,
            "position_index": 0,
            "bank_entry_id": "e0",
            "bucket_id": 1,
            "digit": 1,
            "radix": 4,
            "frame_index": 0,
            "frame_digit_index": 0,
            "frame_digit_count": 4,
        },
        {
            "query_index": 1,
            "position_index": 0,
            "bank_entry_id": "e1",
            "bucket_id": 0,
            "digit": 0,
            "radix": 4,
            "frame_index": 1,
            "frame_digit_index": 2,
            "frame_digit_count": 4,
        },
    ]
    rows = evaluate_diagnostic_e2e._decode_observation_group(
        observations=observations,
        query_budgets=[2],
        bucket_tuple_width=3,
        bucket_radix=4,
        rs_parity_symbols=0,
        expected_payload="A",
        base={"payload_id": "P0001"},
        decoder_mode="variable_radix",
    )
    row = rows[0]
    assert row["decode_status"] == "insufficient_symbols"
    assert row["observed_symbols"] == 2
    assert row["usable_symbols"] == 0
    assert row["complete_frame_count"] == 0
    assert row["incomplete_frame_count"] == 2
    assert row["partial_frame_symbol_count"] == 2
    assert row["max_partial_frame_symbols"] == 1


def test_qwen_decode_csv_persists_partial_frame_columns() -> None:
    diagnostic_source = (
        REPO_ROOT / "scripts/natural_evidence_v1/evaluate_diagnostic_e2e.py"
    ).read_text(encoding="utf-8")
    natural_source = (
        REPO_ROOT / "scripts/natural_evidence_v1/evaluate_qwen_natural_e2e.py"
    ).read_text(encoding="utf-8")
    for source in (diagnostic_source, natural_source):
        assert '"incomplete_frame_count"' in source
        assert '"partial_frame_symbol_count"' in source
        assert '"max_partial_frame_symbols"' in source


def test_reference_candidate_sharding_preserves_original_indices() -> None:
    rows = [{"prompt_id": f"p{index}"} for index in range(8)]
    shard = score_reference_candidates._sharded_rows(rows, shard_index=1, shard_count=3)
    assert [index for index, _ in shard] == [1, 4, 7]
    assert [row["prompt_id"] for _, row in shard] == ["p1", "p4", "p7"]


def test_actual_prefix_reference_candidate_record_preserves_frozen_prefix() -> None:
    class FakeTokenizer:
        def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
            return {10: " then", 11: " also", 99: " observed"}[token_ids[0]]

    row = {
        "protocol_id": "natural_evidence_v1",
        "selector_version": "keyed_actual_prefix_selector_v1",
        "model_family": "qwen",
        "model_condition": "qwen_protected",
        "payload_id": "P0421",
        "seed": "17",
        "prompt_id": "hp0",
        "prompt_split": "heldout",
        "query_index": 3,
        "generated_row_index": 12,
        "position_index": 1,
        "prefix_response_token_count": 9,
        "response_token_count": 32,
        "prefix_signature": "abc123",
        "prefix_token_ids": [101, 102, 103],
        "observed_token_id": 99,
        "observed_token_text": " observed",
    }
    record = score_actual_prefix_reference_candidates._candidate_record(
        row_index=7,
        row=row,
        protocol_id="natural_evidence_v1",
        tokenizer_key="qwen",
        tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        candidate_top_k=3,
        top_probabilities=[0.4, 0.3, 0.2],
        top_token_ids=[10, 99, 11],
        tokenizer=FakeTokenizer(),
    )
    assert record["schema_name"] == "natural_evidence_actual_prefix_reference_topk_candidates_v1"
    assert record["prefix_token_ids"] == [101, 102, 103]
    assert record["prefix_signature"] == "abc123"
    assert record["observed_token_in_topk"] is True
    assert record["observed_token_rank"] == 2
    assert record["observed_token_probability"] == 0.3
    assert record["result_claim"] == "actual_prefix_reference_scoring_not_bucketization_not_payload_recovery"


def test_actual_prefix_bucketization_audit_records_observed_bucket(tmp_path: Path) -> None:
    candidate_path = tmp_path / "actual_prefix_candidates.jsonl"
    output_dir = tmp_path / "actual_prefix_bucketization"
    candidates = [
        {
            "schema_name": "natural_evidence_actual_prefix_reference_topk_candidates_v1",
            "protocol_id": "natural_evidence_v1",
            "tokenizer_key": "qwen",
            "tokenizer_name": "Qwen/Qwen2.5-7B-Instruct",
            "prompt_id": "hp0",
            "prompt_split": "heldout",
            "model_condition": "protected_trained",
            "payload_id": "P0421",
            "seed": "17",
            "query_index": 0,
            "generated_row_index": 3,
            "position_index": 0,
            "prefix_response_token_count": 4,
            "response_token_count": 20,
            "prefix_signature": "sig0",
            "prefix_token_ids": [1, 2, 3],
            "observed_token_id": 10,
            "observed_token_text": " alpha",
            "observed_token_in_topk": True,
            "observed_token_rank": 1,
            "observed_token_probability": 0.4,
            "surface_allowed_candidate_count": 4,
            "candidates": [
                {"rank": 1, "token_id": 10, "text": " alpha", "probability": 0.4, "surface_allowed": True},
                {"rank": 2, "token_id": 11, "text": " beta", "probability": 0.3, "surface_allowed": True},
                {"rank": 3, "token_id": 12, "text": " gamma", "probability": 0.2, "surface_allowed": True},
                {"rank": 4, "token_id": 13, "text": " delta", "probability": 0.1, "surface_allowed": True},
            ],
        }
    ]
    _write_jsonl(candidate_path, candidates)

    status = audit_actual_prefix_bucketization.main(
        [
            "--config",
            str(REPO_ROOT / "configs/natural_evidence_v1/pilot.yaml"),
            "--tokenizer-key",
            "qwen",
            "--candidate-jsonl",
            str(candidate_path),
            "--output-dir",
            str(output_dir),
            "--bucket-count",
            "2",
            "--bank-id",
            "test_actual_prefix_bank",
        ]
    )

    assert status == 0
    summary = json.loads((output_dir / "actual_prefix_bucketization_summary.json").read_text(encoding="utf-8"))
    assert summary["accepted_entries"] == 1
    assert summary["observed_token_bucketized_rows"] == 1
    assert summary["result_claim"] == "actual_prefix_bucketization_audit_not_compatibility_not_payload_recovery"
    bucketized = [
        json.loads(line)
        for line in (output_dir / "qwen_actual_prefix_bucketized_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert bucketized[0]["observed_token_bucketized"] is True
    assert bucketized[0]["observed_token_bucket_id"] in {"0", "1"}
    assert all("bucket_id" in candidate for candidate in bucketized[0]["candidates"])


def test_build_expanded_actual_prefix_bucketized_candidates_keeps_relaxed_arity(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "actual_prefix_candidates.jsonl"
    output_dir = tmp_path / "expanded_bucketized"
    _write_jsonl(
        candidate_path,
        [
            {
                "schema_name": "natural_evidence_actual_prefix_reference_topk_candidates_v1",
                "protocol_id": "natural_evidence_v1",
                "tokenizer_key": "qwen",
                "tokenizer_name": "Qwen/Qwen2.5-7B-Instruct",
                "prefix_signature": "ctx0",
                "prompt_id": "hp0",
                "prompt_split": "heldout",
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "query_index": 0,
                "generated_row_index": 3,
                "position_index": 0,
                "prefix_response_token_count": 4,
                "response_token_count": 20,
                "prefix_token_ids": [1, 2, 3],
                "observed_token_id": 10,
                "observed_token_text": " alpha",
                "observed_token_in_topk": True,
                "candidates": [
                    {"rank": 1, "token_id": 10, "text": " alpha", "probability": 0.4},
                    {"rank": 2, "token_id": 11, "text": " beta", "probability": 0.3},
                ],
            }
        ],
    )

    status = build_expanded_actual_prefix_bucketized_candidates.main(
        [
            "--config",
            str(REPO_ROOT / "configs/natural_evidence_v1/pilot.yaml"),
            "--tokenizer-key",
            "qwen",
            "--candidate-jsonl",
            str(candidate_path),
            "--output-dir",
            str(output_dir),
            "--bucket-count",
            "4",
            "--candidate-top-k",
            "2",
            "--min-arity",
            "2",
        ]
    )

    assert status == 0
    manifest = json.loads(
        (output_dir / "expanded_actual_prefix_bucketized_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "COMPLETE_PENDING_SUFFIX_COMPATIBILITY_SCORING"
    assert manifest["accepted_rows"] == 1
    assert manifest["observed_token_bucketized_rows"] == 1
    assert manifest["arity_counts"] == {"2": 1}
    assert manifest["paper_claim_allowed"] is False
    row = json.loads(
        (output_dir / "expanded_actual_prefix_bucketized_candidates.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert row["arity"] == 2
    assert row["observed_token_bucketized"] is True
    assert row["result_claim"] == (
        "expanded_actual_prefix_bucketized_candidates_not_compatibility_not_payload_recovery"
    )
    assert {candidate["token_id"] for candidate in row["candidates"]} == {10, 11}
    assert all("bucket_id" in candidate for candidate in row["candidates"])
    assert "FIELD=" not in json.dumps(row)


def test_actual_prefix_suffix_compatibility_selects_candidates_by_bucket() -> None:
    candidates = [
        {"bucket_id": "0", "token_id": 10, "rank": 3, "probability": 0.1},
        {"bucket_id": "0", "token_id": 11, "rank": 1, "probability": 0.4},
        {"bucket_id": "0", "token_id": 12, "rank": 2, "probability": 0.2},
        {"bucket_id": "1", "token_id": 20, "rank": 1, "probability": 0.3},
        {"bucket_id": "1", "token_id": 21, "rank": 2, "probability": 0.25},
    ]
    selected = score_actual_prefix_suffix_compatibility._select_candidates_by_bucket(
        candidates,
        max_candidates_per_bucket=2,
    )

    assert [candidate["token_id"] for candidate in selected] == [11, 12, 20, 21]


def test_actual_prefix_suffix_sensitivity_summary_counts_rescues(tmp_path: Path) -> None:
    baseline_summary = tmp_path / "baseline_summary.json"
    baseline_by_entry = tmp_path / "baseline_by_entry.csv"
    cap_summary = tmp_path / "cap_summary.json"
    cap_by_entry = tmp_path / "cap_by_entry.csv"
    bucketized = tmp_path / "bucketized.jsonl"

    baseline_summary.write_text(
        json.dumps(
            {
                "max_candidates_per_bucket": 4,
                "processed_records": 2,
                "scored_candidate_count": 8,
                "compatible_candidate_count": 4,
                "compatibility_pass_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    cap_summary.write_text(
        json.dumps(
            {
                "max_candidates_per_bucket": 8,
                "processed_records": 2,
                "scored_candidate_count": 12,
                "compatible_candidate_count": 8,
                "compatibility_pass_rate": 2 / 3,
            }
        ),
        encoding="utf-8",
    )
    fieldnames = [
        "bank_entry_id",
        "would_accept_min1",
        "would_accept_configured_min",
        "would_accept_probability_gates",
        "rejection_reason",
        "scored_counts_by_bucket_json",
        "compatible_counts_by_bucket_json",
    ]
    with baseline_by_entry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "bank_entry_id": "e0",
                "would_accept_min1": "False",
                "would_accept_configured_min": "False",
                "would_accept_probability_gates": "False",
                "rejection_reason": "missing_compatible_bucket",
                "scored_counts_by_bucket_json": json.dumps({"0": 1, "1": 1}),
                "compatible_counts_by_bucket_json": json.dumps({"0": 1, "1": 0}),
            }
        )
        writer.writerow(
            {
                "bank_entry_id": "e1",
                "would_accept_min1": "True",
                "would_accept_configured_min": "False",
                "would_accept_probability_gates": "False",
                "rejection_reason": "below_configured_min_compatible_members",
                "scored_counts_by_bucket_json": json.dumps({"0": 1, "1": 1}),
                "compatible_counts_by_bucket_json": json.dumps({"0": 1, "1": 1}),
            }
        )
    with cap_by_entry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "bank_entry_id": "e0",
                "would_accept_min1": "True",
                "would_accept_configured_min": "False",
                "would_accept_probability_gates": "False",
                "rejection_reason": "below_configured_min_compatible_members",
                "scored_counts_by_bucket_json": json.dumps({"0": 2, "1": 2}),
                "compatible_counts_by_bucket_json": json.dumps({"0": 1, "1": 1}),
            }
        )
        writer.writerow(
            {
                "bank_entry_id": "e1",
                "would_accept_min1": "True",
                "would_accept_configured_min": "True",
                "would_accept_probability_gates": "True",
                "rejection_reason": "",
                "scored_counts_by_bucket_json": json.dumps({"0": 2, "1": 2}),
                "compatible_counts_by_bucket_json": json.dumps({"0": 2, "1": 2}),
            }
        )
    _write_jsonl(
        bucketized,
        [
            {
                "bank_entry_id": "e0",
                "candidates": [
                    {"bucket_id": "0"},
                    {"bucket_id": "0"},
                    {"bucket_id": "1"},
                    {"bucket_id": "1"},
                ],
            },
            {
                "bank_entry_id": "e1",
                "candidates": [
                    {"bucket_id": "0"},
                    {"bucket_id": "0"},
                    {"bucket_id": "1"},
                    {"bucket_id": "1"},
                ],
            },
        ],
    )

    summary, by_cap, _ = summarize_actual_prefix_suffix_sensitivity.summarize(
        baseline_summary_path=baseline_summary,
        baseline_by_entry_path=baseline_by_entry,
        comparisons=[(8, cap_summary, cap_by_entry)],
        bucketized_candidates_path=bucketized,
        generated_output_count=2,
        bucket_count=2,
    )

    assert summary["final_min1_rescued_vs_baseline"] == 1
    assert summary["final_configured_rescued_vs_baseline"] == 1
    assert summary["final_probability_rescued_vs_baseline"] == 1
    assert by_cap[-1]["diagnostic_min1_bits_per_response"] == 1.0


def test_actual_prefix_suffix_sensitivity_uses_actual_prefix_row_key(tmp_path: Path) -> None:
    baseline_summary = tmp_path / "baseline_summary.json"
    baseline_by_entry = tmp_path / "baseline_by_entry.csv"
    cap_summary = tmp_path / "cap_summary.json"
    cap_by_entry = tmp_path / "cap_by_entry.csv"
    bucketized = tmp_path / "bucketized.jsonl"

    baseline_summary.write_text(
        json.dumps(
            {
                "max_candidates_per_bucket": 4,
                "processed_records": 2,
                "scored_candidate_count": 4,
                "compatible_candidate_count": 2,
                "compatibility_pass_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    cap_summary.write_text(
        json.dumps(
            {
                "max_candidates_per_bucket": 16,
                "processed_records": 2,
                "scored_candidate_count": 8,
                "compatible_candidate_count": 4,
                "compatibility_pass_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    fieldnames = [
        "bank_entry_id",
        "prompt_id",
        "prompt_split",
        "model_condition",
        "payload_id",
        "seed",
        "query_index",
        "generated_row_index",
        "position_index",
        "would_accept_min1",
        "would_accept_configured_min",
        "would_accept_probability_gates",
        "rejection_reason",
        "scored_counts_by_bucket_json",
        "compatible_counts_by_bucket_json",
    ]
    base_row = {
        "bank_entry_id": "same-bank-id",
        "prompt_id": "p0",
        "prompt_split": "heldout",
        "model_condition": "raw",
        "payload_id": "",
        "seed": "",
        "query_index": "0",
        "position_index": "0",
        "would_accept_min1": "False",
        "would_accept_configured_min": "False",
        "would_accept_probability_gates": "False",
        "rejection_reason": "missing_compatible_bucket",
        "scored_counts_by_bucket_json": json.dumps({"0": 1, "1": 1}),
        "compatible_counts_by_bucket_json": json.dumps({"0": 1, "1": 0}),
    }
    with baseline_by_entry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for generated_row_index in ("10", "11"):
            writer.writerow({**base_row, "generated_row_index": generated_row_index})
    with cap_by_entry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for generated_row_index in ("10", "11"):
            writer.writerow(
                {
                    **base_row,
                    "generated_row_index": generated_row_index,
                    "would_accept_min1": "True",
                    "rejection_reason": "below_configured_min_compatible_members",
                    "compatible_counts_by_bucket_json": json.dumps({"0": 1, "1": 1}),
                }
            )
    _write_jsonl(
        bucketized,
        [
            {
                **base_row,
                "generated_row_index": generated_row_index,
                "candidates": [{"bucket_id": "0"}, {"bucket_id": "1"}],
            }
            for generated_row_index in ("10", "11")
        ],
    )

    summary, by_cap, by_entry = summarize_actual_prefix_suffix_sensitivity.summarize(
        baseline_summary_path=baseline_summary,
        baseline_by_entry_path=baseline_by_entry,
        comparisons=[(16, cap_summary, cap_by_entry)],
        bucketized_candidates_path=bucketized,
        generated_output_count=2,
        bucket_count=2,
    )

    assert summary["final_min1_compatible_entries"] == 2
    assert summary["final_min1_rescued_vs_baseline"] == 2
    assert by_cap[-1]["min1_compatible_entries"] == 2
    assert len(by_entry) == 4
    assert len({row["entry_key"] for row in by_entry}) == 2
    assert {row["bank_entry_id"] for row in by_entry} == {"same-bank-id"}


def test_variable_arity_diagnostic_counts_capacity_and_rejections(tmp_path: Path) -> None:
    by_entry = tmp_path / "by_entry.csv"
    bucketized = tmp_path / "bucketized.jsonl"
    output_dir = tmp_path / "variable_arity"
    fieldnames = [
        "bank_entry_id",
        "prompt_id",
        "prompt_split",
        "model_condition",
        "payload_id",
        "seed",
        "query_index",
        "generated_row_index",
        "position_index",
        "bucket_count",
        "compatible_counts_by_bucket_json",
        "compatible_probability_by_bucket_json",
    ]
    rows = [
        {
            "bank_entry_id": "e0",
            "prompt_id": "p0",
            "prompt_split": "heldout",
            "model_condition": "raw",
            "payload_id": "",
            "seed": "",
            "query_index": "0",
            "generated_row_index": "10",
            "position_index": "0",
            "bucket_count": "4",
            "compatible_counts_by_bucket_json": json.dumps({"0": 1, "1": 1, "2": 0, "3": 0}),
            "compatible_probability_by_bucket_json": json.dumps({"0": 0.05, "1": 0.05, "2": 0.0, "3": 0.0}),
        },
        {
            "bank_entry_id": "e1",
            "prompt_id": "p1",
            "prompt_split": "heldout",
            "model_condition": "raw",
            "payload_id": "",
            "seed": "",
            "query_index": "1",
            "generated_row_index": "11",
            "position_index": "0",
            "bucket_count": "4",
            "compatible_counts_by_bucket_json": json.dumps({"0": 1, "1": 0, "2": 0, "3": 0}),
            "compatible_probability_by_bucket_json": json.dumps({"0": 0.1, "1": 0.0, "2": 0.0, "3": 0.0}),
        },
        {
            "bank_entry_id": "e2",
            "prompt_id": "p2",
            "prompt_split": "train",
            "model_condition": "protected_trained",
            "payload_id": "P0421",
            "seed": "17",
            "query_index": "2",
            "generated_row_index": "12",
            "position_index": "0",
            "bucket_count": "4",
            "compatible_counts_by_bucket_json": json.dumps({"0": 2, "1": 2, "2": 2, "3": 0}),
            "compatible_probability_by_bucket_json": json.dumps({"0": 0.03, "1": 0.03, "2": 0.03, "3": 0.0}),
        },
    ]
    with by_entry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_jsonl(
        bucketized,
        [
            {
                **row,
                "observed_token_bucket_id": "0",
                "observed_token_bucketized": True,
                "observed_token_id": 100 + index,
                "response_token_count": 50,
                "candidates": [{"bucket_id": str(bucket_id)} for bucket_id in range(4)],
            }
            for index, row in enumerate(rows)
        ],
    )

    manifest = build_variable_arity_diagnostic.build_variable_arity_diagnostic(
        compatibility_by_entry_csv=by_entry,
        bucketized_candidates_jsonl=bucketized,
        output_dir=output_dir,
        generated_output_count=10,
        bucket_count=4,
        min_arity=2,
        configured_min_members=2,
    )

    assert manifest["accepted_entries"] == 2
    assert manifest["configured_subset_entries"] == 1
    assert manifest["arity_distribution"] == {"0": 0, "1": 1, "2": 1, "3": 1, "4": 0}
    assert math.isclose(manifest["effective_bits_per_response"], (1.0 + math.log2(3)) / 10)
    entries = [
        json.loads(line)
        for line in (output_dir / "variable_arity_bank_entries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["arity"] for entry in entries] == [2, 3]
    assert entries[0]["result_claim"] == "variable_arity_diagnostic_not_payload_recovery"
    rejections = list(csv.DictReader((output_dir / "variable_arity_rejections.csv").open()))
    assert rejections[0]["rejection_reason"] == "below_min_arity"
    density_rows = list(csv.DictReader((output_dir / "eligible_density_by_split.csv").open()))
    assert {row["denominator_scope"] for row in density_rows} == {"bucketized_unique_generated_rows_only"}


def test_variable_arity_full_density_uses_all_generated_outputs(tmp_path: Path) -> None:
    generated_outputs = tmp_path / "generated.jsonl"
    by_entry = tmp_path / "by_entry.csv"
    output_dir = tmp_path / "full_density"
    _write_jsonl(
        generated_outputs,
        [
            {
                "prompt_id": "p0",
                "prompt_split": "heldout",
                "model_condition": "raw",
                "payload_id": "",
                "seed": "",
                "query_index": 0,
                "response_text": "one two three four",
            },
            {
                "prompt_id": "p1",
                "prompt_split": "heldout",
                "model_condition": "raw",
                "payload_id": "",
                "seed": "",
                "query_index": 1,
                "response_text": "five six seven eight",
            },
        ],
    )
    fieldnames = [
        "bank_entry_id",
        "prompt_id",
        "prompt_split",
        "model_condition",
        "payload_id",
        "seed",
        "query_index",
        "generated_row_index",
        "position_index",
        "compatible_counts_by_bucket_json",
        "compatible_probability_by_bucket_json",
    ]
    with by_entry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "bank_entry_id": "e0",
                "prompt_id": "p0",
                "prompt_split": "heldout",
                "model_condition": "raw",
                "payload_id": "",
                "seed": "",
                "query_index": "0",
                "generated_row_index": "0",
                "position_index": "0",
                "compatible_counts_by_bucket_json": json.dumps({"0": 1, "1": 1, "2": 0, "3": 0}),
                "compatible_probability_by_bucket_json": json.dumps({"0": 0.1, "1": 0.1, "2": 0.0, "3": 0.0}),
            }
        )

    manifest = audit_variable_arity_full_density.audit_full_density(
        config_path=REPO_ROOT / "configs/natural_evidence_v1/pilot.yaml",
        tokenizer_key="qwen",
        generated_outputs_path=generated_outputs,
        compatibility_by_entry_csv=by_entry,
        output_dir=output_dir,
        bucket_count=4,
        min_arity=2,
        configured_min_members=1,
        min_bucket_mass=0.005,
        max_bucket_mass_ratio=5.0,
        min_entropy_fraction=0.90,
        token_count_mode="whitespace",
        tokenizer_name_override="",
    )

    assert manifest["generated_outputs_count"] == 2
    assert manifest["total_response_tokens"] == 8
    assert manifest["accepted_entries"] == 1
    assert manifest["rows_with_no_accepted_entries"] == 1
    rows = list(csv.DictReader((output_dir / "variable_arity_full_density_by_slice.csv").open()))
    all_row = next(row for row in rows if row["slice_kind"] == "all")
    assert all_row["denominator_scope"] == "all_generated_outputs_full_response_tokens"
    assert float(all_row["eligible_positions_per_100_tokens"]) == 12.5
    generated_rows = list(
        csv.DictReader((output_dir / "variable_arity_full_density_by_generated_row.csv").open())
    )
    assert [int(row["accepted_entry_count"]) for row in generated_rows] == [1, 0]
    assert manifest["result_claim"] == "full_denominator_variable_arity_density_not_payload_recovery"


def test_combine_variable_arity_density_inputs_offsets_organic_rows(tmp_path: Path) -> None:
    base_generated = tmp_path / "base_generated.jsonl"
    organic_generated = tmp_path / "organic_generated.jsonl"
    base_csv = tmp_path / "base.csv"
    organic_csv = tmp_path / "organic.csv"
    combined_generated = tmp_path / "combined.jsonl"
    combined_csv = tmp_path / "combined.csv"
    summary_json = tmp_path / "summary.json"
    _write_jsonl(
        base_generated,
        [
            {"prompt_id": "h0", "prompt_split": "heldout", "response_text": "one two"},
            {"prompt_id": "h1", "prompt_split": "heldout", "response_text": "three four"},
        ],
    )
    _write_jsonl(
        organic_generated,
        [{"prompt_id": "o0", "prompt_split": "organic", "response_text": "five six"}],
    )
    fieldnames = ["bank_entry_id", "prompt_id", "prompt_split", "generated_row_index", "position_index"]
    for path, prompt_id, split, generated_row_index in (
        (base_csv, "h0", "heldout", "0"),
        (organic_csv, "o0", "organic", "0"),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "bank_entry_id": f"entry_{prompt_id}",
                    "prompt_id": prompt_id,
                    "prompt_split": split,
                    "generated_row_index": generated_row_index,
                    "position_index": "0",
                }
            )

    summary = combine_variable_arity_density_inputs.combine_density_inputs(
        base_generated_outputs=base_generated,
        base_compatibility_by_entry_csv=base_csv,
        organic_generated_outputs=organic_generated,
        organic_compatibility_by_entry_csv=organic_csv,
        output_generated_outputs=combined_generated,
        output_compatibility_by_entry_csv=combined_csv,
        summary_json=summary_json,
    )

    assert summary["organic_generated_row_index_offset"] == 2
    generated_rows = [
        json.loads(line)
        for line in combined_generated.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert generated_rows[-1]["generated_row_index"] == 2
    combined_rows = list(csv.DictReader(combined_csv.open(encoding="utf-8")))
    assert combined_rows[-1]["prompt_split"] == "organic"
    assert combined_rows[-1]["generated_row_index"] == "2"
    assert json.loads(summary_json.read_text(encoding="utf-8"))["training_started"] is False


def test_variable_arity_pre_null_runs_mixed_radix_decode(tmp_path: Path) -> None:
    compatibility_jsonl = tmp_path / "compatibility.jsonl"
    by_entry = tmp_path / "by_entry.csv"
    output_dir = tmp_path / "pre_null"
    compatibility_rows: list[dict[str, object]] = []
    by_entry_rows: list[dict[str, object]] = []
    for index in range(24):
        entry_id = f"entry{index}"
        common = {
            "bank_entry_id": entry_id,
            "bank_id": "test_variable_bank",
            "context_signature": f"ctx{index}",
            "protocol_id": "natural_evidence_v1",
            "prompt_id": f"p{index}",
            "prompt_split": "heldout",
            "model_condition": "raw",
            "payload_id": "",
            "seed": "",
            "query_index": index,
            "generated_row_index": index,
            "position_index": 0,
            "observed_token_id": 1000 + index * 10,
            "observed_token_bucket_id": "0",
        }
        by_entry_rows.append(
            {
                **common,
                "bucket_count": 4,
                "compatible_counts_by_bucket_json": json.dumps({"0": 1, "1": 1, "2": 1, "3": 1}),
            }
        )
        for bucket_id in range(4):
            compatibility_rows.append(
                {
                    **common,
                    "bucket_id": str(bucket_id),
                    "token_id": 1000 + index * 10 + bucket_id,
                    "token_text": f" token{bucket_id}",
                    "probability": 0.1,
                    "rank": bucket_id + 1,
                    "compatibility_pass": True,
                }
            )
    _write_jsonl(compatibility_jsonl, compatibility_rows)
    with by_entry.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "bank_entry_id",
            "bank_id",
            "context_signature",
            "protocol_id",
            "prompt_id",
            "prompt_split",
            "model_condition",
            "payload_id",
            "seed",
            "query_index",
            "generated_row_index",
            "position_index",
            "bucket_count",
            "observed_token_id",
            "observed_token_bucket_id",
            "compatible_counts_by_bucket_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(by_entry_rows)

    summary = variable_arity_pre_null.run_pre_null(
        config_path=REPO_ROOT / "configs/natural_evidence_v1/pilot.yaml",
        tokenizer_key="qwen",
        compatibility_jsonl=compatibility_jsonl,
        compatibility_by_entry_csv=by_entry,
        output_dir=output_dir,
        bucket_count=4,
        min_arity=2,
        wrong_key_count=1,
        query_budgets_override="24",
        max_examples=5,
        max_records=0,
    )

    assert summary["entries_with_compatible_candidate_rows"] == 24
    assert summary["variable_radix_preflight_status"] == "PASS"
    assert summary["pre_null_status"] == "PASS_NO_BLOCKING_ACCEPTS_DIAGNOSTIC"
    decodes = list(csv.DictReader((output_dir / "variable_arity_pre_null_decodes.csv").open()))
    assert {row["null_family"] for row in decodes} >= {"raw", "wrong_key"}
    assert all(row["result_claim"] == "variable_arity_pre_null_diagnostic_not_full_far" for row in decodes)


def test_variable_radix_train_eval_preflight_builds_contract_and_null_trace(tmp_path: Path) -> None:
    bank_path = tmp_path / "variable_arity_bank_entries.jsonl"
    observations_path = tmp_path / "observations.jsonl"
    summary_path = tmp_path / "pre_null_summary.json"
    output_dir = tmp_path / "preflight"
    bank_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    for index in range(40):
        entry_key = f"entry{index}"
        common = {
            "entry_key": entry_key,
            "bank_entry_id": f"bank{index}",
            "prompt_id": f"p{index}",
            "prompt_split": "heldout",
            "generated_row_index": index,
            "position_index": 0,
        }
        bank_rows.append(
            {
                **common,
                "schema_name": "natural_evidence_variable_arity_diagnostic_entry_v1",
                "arity": 4,
                "compatible_bucket_ids": ["0", "1", "2", "3"],
            }
        )
        for condition, model_condition in (
            ("correct_key", "raw"),
            ("correct_key", "task_only_lora"),
            ("wrong_key_0", "protected_trained"),
        ):
            observation_rows.append(
                {
                    **common,
                    "condition": condition,
                    "model_condition": model_condition,
                    "source_payload_id": "P0421" if model_condition != "raw" else "",
                    "source_seed": 17 if model_condition != "raw" else "",
                    "query_index": index,
                    "digit": 0,
                    "radix": 4,
                    "arity": 4,
                    "bucket_id": "0",
                }
            )
    _write_jsonl(bank_path, bank_rows)
    _write_jsonl(observations_path, observation_rows)
    summary_path.write_text(
        json.dumps(
            {
                "schema_name": "natural_evidence_variable_arity_pre_null_summary_v1",
                "query_budgets": [20, 24],
                "pre_null_status": "PASS_NO_BLOCKING_ACCEPTS_DIAGNOSTIC",
            }
        ),
        encoding="utf-8",
    )

    summary = variable_radix_train_eval_preflight.run_preflight(
        config_path=REPO_ROOT / "configs/natural_evidence_v1/pilot.yaml",
        variable_arity_bank_entries=bank_path,
        pre_null_observations=observations_path,
        pre_null_summary=summary_path,
        output_dir=output_dir,
        payload_ids="P0421,P1729",
        query_budgets_override="20,24",
        max_assignments_per_payload=40,
        max_null_observations=0,
    )

    assert summary["overall_status"] == "PASS_PREFLIGHT_NOT_TRAINING"
    assert summary["train_contract_status"] == "PASS_DRY_RUN_NOT_TRAINING"
    assert summary["blocking_null_accept_count"] == 0
    assert summary["ready_for_training_submission"] is False
    contract = json.loads(
        (output_dir / "variable_radix_train_contract_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["schema_name"] == "natural_evidence_variable_radix_train_contract_preflight_v1"
    assert contract["claim_control"]["ready_for_model_training"] is False
    assignments = [
        json.loads(line)
        for line in (output_dir / "variable_radix_train_assignments_preflight.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert {row["payload_id"] for row in assignments} == {"P0421", "P1729"}
    decode_rows = list(
        csv.DictReader((output_dir / "variable_radix_eval_decode_preflight.csv").open())
    )
    protected_rows = [row for row in decode_rows if row["null_family"] == "synthetic_protected"]
    assert protected_rows
    assert all(row["accepted"] == "True" for row in protected_rows)
    assert all(row["result_claim"] != "payload_recovery" for row in decode_rows)


def test_qwen_proof_of_life_gate_review_blocks_on_organic_with_disabled_wrapper(tmp_path: Path) -> None:
    variable_arity = tmp_path / "variable_arity.json"
    density = tmp_path / "density.json"
    pre_null = tmp_path / "pre_null.json"
    variable_radix = tmp_path / "variable_radix.json"
    frame_policy = tmp_path / "frame_policy.json"
    protocol = tmp_path / "protocol_commitment.md"
    output_dir = tmp_path / "gate_review"
    variable_arity.write_text(
        json.dumps(
            {
                "accepted_entries": 23774,
                "configured_subset_entries": 892,
            }
        ),
        encoding="utf-8",
    )
    density.write_text(
        json.dumps(
            {
                "effective_bits_per_response": 2.2,
                "eligible_positions_per_100_tokens": 2.0,
                "gate_status": {
                    "heldout_viability_density": "PASS",
                    "organic_density": "NEEDS_ORGANIC_GENERATED_OUTPUTS",
                },
            }
        ),
        encoding="utf-8",
    )
    pre_null.write_text(
        json.dumps(
            {
                "pre_null_status": "PASS_NO_BLOCKING_ACCEPTS_DIAGNOSTIC",
                "blocking_accept_count": 0,
                "query_budgets": [64, 128, 256, 512],
            }
        ),
        encoding="utf-8",
    )
    variable_radix.write_text(
        json.dumps(
            {
                "overall_status": "PASS_PREFLIGHT_NOT_TRAINING",
                "blocking_null_accept_count": 0,
            }
        ),
        encoding="utf-8",
    )
    frame_policy.write_text(
        json.dumps(
            {
                "status": "PASS_DRY_RUN_NOT_TRAINING",
                "quality_gates": {
                    "frame_repetition_positions": "PASS",
                    "trainer_dry_run_review": "PASS",
                    "synthetic_query_budget_decode": "PASS",
                },
                "decode_policy": {"status": "PASS_SYNTHETIC_TARGET_TRACE"},
            }
        ),
        encoding="utf-8",
    )
    protocol.write_text(
        "transcript_commitment\nPost-hoc key search is disallowed\nwrong-key\nwrong-payload\n",
        encoding="utf-8",
    )

    summary = review_qwen_proof_of_life_gate.run_gate_review(
        config_path=REPO_ROOT / "configs/natural_evidence_v1/pilot.yaml",
        allowlist_path=REPO_ROOT / "configs/natural_evidence_v1/run_allowlist.yaml",
        variable_arity_manifest_path=variable_arity,
        full_density_summary_path=density,
        pre_null_summary_path=pre_null,
        variable_radix_preflight_summary_path=variable_radix,
        variable_radix_frame_policy_summary_path=frame_policy,
        protocol_commitment_path=protocol,
        output_dir=output_dir,
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "BLOCKED_NOT_READY_FOR_TRAINING"
    assert summary["ready_for_training_submission"] is False
    assert "organic_generated_output_density" in summary["blocker_gates"]
    assert "qwen_natural_e2e_allowlist_command" not in summary["blocker_gates"]
    assert "variable_radix_production_integration" not in summary["blocker_gates"]
    rows = list(csv.DictReader((output_dir / "qwen_proof_of_life_gate_review.csv").open()))
    assert any(row["gate"] == "full_budget_pre_null" and row["status"] == "PASS" for row in rows)
    assert any(row["gate"] == "qwen_natural_e2e_allowlist_command" and row["status"] == "PASS" for row in rows)
    assert any(row["gate"] == "training_allowlist_disabled_until_approval" and row["status"] == "PASS" for row in rows)
    assert any(row["gate"] == "variable_radix_production_integration" and row["status"] == "PASS" for row in rows)
    assert "No training" in (output_dir / "qwen_proof_of_life_blockers.md").read_text(
        encoding="utf-8"
    )


def test_compatibility_aware_supply_audit_reports_upper_bound(tmp_path: Path) -> None:
    candidate_jsonl = tmp_path / "topk.jsonl"
    output_dir = tmp_path / "supply"
    _write_jsonl(
        candidate_jsonl,
        [
            {
                "schema_name": "natural_evidence_actual_prefix_reference_topk_candidates_v1",
                "tokenizer_key": "qwen",
                "tokenizer_name": "Qwen/Qwen2.5-7B-Instruct",
                "prefix_signature": "ctx0",
                "prompt_id": "p0",
                "prompt_split": "heldout",
                "model_condition": "raw",
                "query_index": 0,
                "generated_row_index": 0,
                "position_index": 0,
                "prefix_response_token_count": 1,
                "response_token_count": 20,
                "prefix_token_ids": [1],
                "observed_token_id": 10,
                "candidates": [
                    {"rank": 1, "token_id": 10, "text": " alpha", "probability": 0.20},
                    {"rank": 2, "token_id": 11, "text": " beta", "probability": 0.10},
                    {"rank": 3, "token_id": 12, "text": " gamma", "probability": 0.05},
                    {"rank": 4, "token_id": 13, "text": " delta", "probability": 0.05},
                ],
            },
            {
                "schema_name": "natural_evidence_actual_prefix_reference_topk_candidates_v1",
                "tokenizer_key": "qwen",
                "tokenizer_name": "Qwen/Qwen2.5-7B-Instruct",
                "prefix_signature": "ctx1",
                "prompt_id": "p1",
                "prompt_split": "heldout",
                "model_condition": "raw",
                "query_index": 1,
                "generated_row_index": 1,
                "position_index": 0,
                "prefix_response_token_count": 1,
                "response_token_count": 30,
                "prefix_token_ids": [1],
                "observed_token_id": 20,
                "candidates": [
                    {"rank": 1, "token_id": 20, "text": " ok", "probability": 0.20},
                    {"rank": 2, "token_id": 21, "text": " **", "probability": 0.10},
                ],
            },
        ],
    )

    manifest = compatibility_aware_supply_audit.run_supply_audit(
        config_path=REPO_ROOT / "configs/natural_evidence_v1/pilot.yaml",
        tokenizer_key="qwen",
        candidate_jsonl_path=candidate_jsonl,
        output_dir=output_dir,
        generated_output_count=2,
        bucket_counts=[2, 4],
        candidate_top_k_override=4,
        min_arity=2,
        configured_min_members=2,
        min_bucket_mass=0.005,
        max_bucket_mass_ratio=5.0,
        min_entropy_fraction=0.50,
        max_records=0,
        write_position_csv=True,
    )

    assert manifest["input_records"] == 2
    assert manifest["best_effective_bits_per_response"] >= 0.5
    by_bucket = list(csv.DictReader((output_dir / "candidate_supply_by_bucket_count.csv").open()))
    bucket2 = next(row for row in by_bucket if row["bucket_count"] == "2")
    assert int(bucket2["accepted_positions"]) == 1
    rejections = list(csv.DictReader((output_dir / "candidate_supply_rejections.csv").open()))
    assert any(row["rejection_reason"] == "insufficient_filtered_candidates_for_min_arity" for row in rejections)
    positions = list(csv.DictReader((output_dir / "candidate_supply_by_position.csv").open()))
    assert {row["bucket_count"] for row in positions} == {"2", "4"}


def test_natural_protocol_docs_lock_security_boundaries() -> None:
    commitment = (REPO_ROOT / "docs/natural_evidence_v1/protocol_commitment.md").read_text(
        encoding="utf-8"
    )
    formal = (REPO_ROOT / "docs/natural_evidence_v1/formal_protocol.md").read_text(
        encoding="utf-8"
    )
    e2e = (REPO_ROOT / "docs/natural_evidence_v1/end_to_end_audit_plan.md").read_text(
        encoding="utf-8"
    )
    assert "Post-hoc key search is disallowed" in commitment
    assert "multiple keys" in commitment
    assert "family-wise false-accept accounting" in commitment
    assert "prefix-only" in formal
    assert "first-token measurable" in formal
    assert "Qwen task-only LoRA" in e2e
    assert "same-family" in e2e
    assert "oracle keyed sanitizer" in e2e


def test_automation_state_and_allowlist_are_conservative() -> None:
    allowlist = (REPO_ROOT / "configs/natural_evidence_v1/run_allowlist.yaml").read_text(
        encoding="utf-8"
    )
    state = (REPO_ROOT / "docs/natural_evidence_v1/AUTOMATION_STATE.md").read_text(
        encoding="utf-8"
    )
    gate_status = json.loads(
        (REPO_ROOT / "results/natural_evidence_v1/status/gate_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert "max_state_changing_actions_per_automation_run: 1" in allowlist
    assert "forbid_unlisted_gpu_jobs: true" in allowlist
    assert "require_chimera_mail_notifications: true" in allowlist
    assert "qwen_natural_e2e_pilot" in allowlist
    assert "qwen_diagnostic_high_risk_e2e_pilot" in allowlist
    assert "explicit_diagnostic_gate_passed_after_raw_wrong_key_pre_null_invalid_suffix_and_train_dataset_preflight" in allowlist
    assert "sweep_qwen_4way_balance_thresholds" in allowlist
    assert "qwen_candidate_supply_expansion" in allowlist
    assert "qwen_candidate_shard_merge_strict_bank" in allowlist
    assert "qwen_compatibility_filtered_bank_repair_dry_run" in allowlist
    assert "qwen_probability_preserving_compatibility_repair" in allowlist
    assert "qwen_min1_compatible_density_audit" in allowlist
    assert "qwen_frozen_density_split_and_heldout_density_audit" in allowlist
    assert "qwen_actual_prefix_reference_model_scoring" in allowlist
    assert "qwen_actual_prefix_scoring_plan_complete_pending_gpu_reference_scoring" in allowlist
    assert "qwen_actual_prefix_suffix_compatibility" in allowlist
    assert "qwen_actual_prefix_suffix_highcap_sensitivity" in allowlist
    assert "qwen_actual_prefix_compatibility_aware_supply_audit" in allowlist
    assert "qwen_expanded_variable_arity_organic_density" in allowlist
    assert "Hourly Slurm Mail Notifications" in state
    assert "844015" in state
    assert "guanjie.lin001@umb.edu" in state
    assert gate_status["gates"]["chimera_mail_notifications"] == "PASS_TESTED_JOB_844015"
    assert gate_status["gates"]["chimera_ip_pinned_ssh"] == "PASS_VERIFIED_20260506T150815Z"
    assert "Do not submit additional protected LoRA training" in state
    assert gate_status["gates"]["phase_a_outputs_complete"] in {
        "PASS",
        "PASS_LAST_KNOWN_REMOTE_UNVERIFIED_20260506_1403",
    }
    assert gate_status["gates"]["qwen_e2e_pilot"] in {
        "TODO_AFTER_RESULTS",
        "BLOCKED_BY_COUNTERFACTUAL_COMPATIBILITY_FAIL",
        "BLOCKED_BY_COMPATIBILITY_FILTERED_REPAIR_FAIL",
        "BLOCKED_BY_PROBABILITY_PRESERVING_REPAIR_FAIL",
        "BLOCKED_PENDING_DENSITY_AND_PRE_NULL",
        "BLOCKED_PENDING_FROZEN_DENSITY_AND_PRE_NULL",
        "BLOCKED_BY_FROZEN_HELDOUT_DENSITY_FAIL",
        "BLOCKED_BY_FROZEN_HELDOUT_DENSITY_FAIL_PAPER_READY_ONLY",
        "BLOCKED_PENDING_FULL_DENSITY_PRENULL_AND_VARIABLE_RADIX_E2E_DESIGN",
        "BLOCKED_PENDING_VARIABLE_ARITY_PRENULL_AND_VARIABLE_RADIX_E2E_DESIGN",
        "BLOCKED_PENDING_PROOF_OF_LIFE_GATE_REVIEW",
        "BLOCKED_BY_PROOF_OF_LIFE_GATE_REVIEW",
        "BLOCKED_BY_ORGANIC_DENSITY_JOB_846391",
        "BLOCKED_BY_VARIABLE_RADIX_REAL_ARTIFACT_DRY_RUN_LOW_TRAIN_SIGNAL",
    }
    assert gate_status["gates"]["llama_e2e_pilot"] in {
        "TODO_AFTER_RESULTS",
        "BLOCKED_BY_QWEN_PARTIAL_NEGATIVE_DIAGNOSTIC",
        "BLOCKED_BY_QWEN_VERIFIER_ALIGNMENT_FAILURE",
        "BLOCKED_BY_QWEN_STATIC_BUCKET_SALVAGE_FAIL",
        "BLOCKED_PENDING_QWEN_ACTUAL_PREFIX_REFERENCE_SCORING",
        "BLOCKED_UNTIL_QWEN_VARIABLE_ARITY_PROOF_OF_LIFE",
    }
    assert gate_status["next_allowed_action"] != "qwen_natural_e2e_pilot"
    assert gate_status["project_target_bank_entries_per_tokenizer"] == 24000
    assert gate_status["project_target_role"] == "raw_opportunity_scaling_placeholder_not_training_gate"
    assert (
        gate_status["compatibility_adjusted_capacity_targets"]["raw_entry_count_is_training_gate"]
        is False
    )
    assert gate_status["gates"]["qwen_compatibility_adjusted_bank_side_viability"] == (
        "PASS_MIN1_MIN2_PILOT_GATE"
    )
    assert "24000_fingerprints" in gate_status["forbidden_claims"]
    assert "24576_fingerprints" in gate_status["forbidden_claims"]


def test_qwen_candidate_expansion_slurm_is_scoped_to_qwen_4way() -> None:
    script = (
        REPO_ROOT / "scripts/natural_evidence_v1/slurm_qwen_candidate_supply_expansion.sbatch"
    ).read_text(encoding="utf-8")
    assert "--tokenizer-key qwen" in script
    assert "--bucket-count 4" in script
    assert "--strict-balance-gate" in script
    assert "score_counterfactual_compatibility" not in script
    assert "qwen_natural_e2e_pilot" not in script
    assert "--tokenizer-key llama" not in script


def test_qwen_candidate_expansion_sharded_slurm_only_scores_qwen_candidates() -> None:
    script = (
        REPO_ROOT / "scripts/natural_evidence_v1/slurm_qwen_candidate_supply_expansion_sharded.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2" in script
    assert "--gres=gpu:h200:1" in script
    assert "--tokenizer-key qwen" in script
    assert "--shard-index" in script
    assert "--shard-count" in script
    assert "--stream-output" in script
    assert "--progress-json" in script
    assert "build_bucket_bank.py" not in script
    assert "score_counterfactual_compatibility" not in script
    assert "qwen_natural_e2e_pilot" not in script
    assert "--tokenizer-key llama" not in script


def test_qwen_candidate_shard_merge_strict_bank_slurm_is_cpu_scoped() -> None:
    script = (
        REPO_ROOT
        / "scripts/natural_evidence_v1/slurm_qwen_candidate_shard_merge_strict_bank.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --job-name=nat-ev-qwen-bank" in script
    assert "--gres=" not in script
    assert 'qwen_candidate_shard_{index}_of_{shard_count}_progress.json' in script
    assert 'qwen_topk_candidates_shard_{index}_of_{shard_count}.jsonl' in script
    assert "PARTIAL_STRICT_BANK_OUTPUT_EXISTS_REFUSING_OVERWRITE" in script
    assert "--tokenizer-key \"$TOKENIZER_KEY\"" in script
    assert "--bucket-count \"$BUCKET_COUNT\"" in script
    assert "--strict-balance-gate" in script
    assert "score_reference_candidates.py" not in script
    assert "score_counterfactual_compatibility" not in script
    assert "qwen_natural_e2e_pilot" not in script
    assert "--tokenizer-key llama" not in script


def test_qwen_diagnostic_e2e_wrapper_requires_explicit_start_and_forbids_old_route() -> None:
    script = (
        REPO_ROOT / "scripts/natural_evidence_v1/slurm/qwen_diagnostic_high_risk_e2e_pilot.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:h200:1" in script
    assert "DIAGNOSTIC_HIGH_RISK" in script
    assert "NO_PAPER_CLAIM" in script
    assert 'ARMS="qwen_protected,qwen_raw,qwen_task_only_lora,wrong_key,wrong_payload"' in script
    assert 'QUERY_BUDGETS="64,128,256,512"' in script
    assert 'EVAL_OWNER_PROBES="2048"' in script
    assert 'ORGANIC_NULL_PROMPTS="2048"' in script
    assert 'DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"' in script
    assert 'START_DIAGNOSTIC_E2E="${START_DIAGNOSTIC_E2E:-0}"' in script
    assert "training_started\": false" in script
    assert "REQUIRED_TRAINING_DATASET_MISSING_OR_EMPTY" in script
    assert "train_natural_bucket_lora.py" in script
    assert "--start-training" in script
    assert "--require-cuda" in script
    assert "scripts/train.py" not in script
    assert "llama" not in script.lower()
    assert "8way" not in script.lower()


def test_qwen_natural_e2e_wrapper_requires_gate_review_and_defaults_to_dry_run() -> None:
    script = (
        REPO_ROOT / "scripts/natural_evidence_v1/slurm/qwen_natural_e2e_pilot.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=DGXA100" in script
    assert "#SBATCH --account=pi_yinxin.wan" in script
    assert "#SBATCH --qos=scavenger_unlim" in script
    assert "#SBATCH --gres=gpu:A100:1" in script
    assert "#SBATCH --mail-type=ALL" in script
    assert "#SBATCH --mail-user=guanjie.lin001@umb.edu" in script
    assert "QWEN_NATURAL_VARIABLE_RADIX_PROOF_OF_LIFE_PREFLIGHT" in script
    assert "NO_PAPER_CLAIM" in script
    assert 'ARMS="qwen_protected,qwen_raw,qwen_task_only_lora,wrong_key,wrong_payload"' in script
    assert 'TRAINING_ARMS="qwen_protected,qwen_task_only_lora"' in script
    assert 'QUERY_BUDGETS="${QUERY_BUDGETS:-64,128,256,512}"' in script
    assert 'EVAL_OWNER_PROBES="${EVAL_OWNER_PROBES:-2048}"' in script
    assert 'ORGANIC_NULL_PROMPTS="${ORGANIC_NULL_PROMPTS:-2048}"' in script
    assert 'DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"' in script
    assert 'START_QWEN_NATURAL_E2E="${START_QWEN_NATURAL_E2E:-0}"' in script
    assert "READY_FOR_EXPLICIT_LAUNCH_REVIEW" in script
    assert "PROOF_GATE_REVIEW_NOT_READY_FOR_LAUNCH" in script
    assert "variable_radix_frame_policy" in script
    assert "repeat_payload" in script
    assert "training_started\": false" in script
    assert "train_natural_bucket_lora.py" in script
    assert "--start-training" in script
    assert "--require-cuda" in script
    assert "QWEN_NATURAL_E2E_OUTPUT_EXISTS_REFUSING_OVERWRITE" in script
    assert "scripts/train.py" not in script
    assert "qwen_diagnostic_high_risk_e2e_pilot" not in script
    assert "llama" not in script.lower()
    assert "8way" not in script.lower()


def test_qwen_natural_five_arm_eval_wrapper_is_dgxa100_dry_run_only() -> None:
    script = (
        REPO_ROOT / "scripts/natural_evidence_v1/slurm/qwen_natural_e2e_eval.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --job-name=nat-ev-qwen-nat-eval" in script
    assert "#SBATCH --partition=DGXA100" in script
    assert "#SBATCH --account=pi_yinxin.wan" in script
    assert "#SBATCH --qos=scavenger_unlim" in script
    assert "#SBATCH --gres=gpu:A100:1" in script
    assert "#SBATCH --mail-type=ALL" in script
    assert "#SBATCH --mail-user=guanjie.lin001@umb.edu" in script
    assert "QWEN_NATURAL_VARIABLE_RADIX_FIVE_ARM_EVAL_PREFLIGHT" in script
    assert "NO_PAPER_CLAIM" in script
    assert 'ARMS="qwen_protected,qwen_raw,qwen_task_only_lora,wrong_key,wrong_payload"' in script
    assert 'TRAINED_ARMS="qwen_protected,qwen_task_only_lora"' in script
    assert 'QUERY_BUDGETS="${QUERY_BUDGETS:-64,128,256,512}"' in script
    assert 'EVAL_OWNER_PROBES="${EVAL_OWNER_PROBES:-2048}"' in script
    assert 'ORGANIC_NULL_PROMPTS="${ORGANIC_NULL_PROMPTS:-2048}"' in script
    assert 'DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"' in script
    assert 'START_QWEN_NATURAL_E2E_EVAL="${START_QWEN_NATURAL_E2E_EVAL:-0}"' in script
    assert "evaluate_qwen_natural_e2e.py" in script
    assert "--start-eval" in script
    assert "--require-cuda" in script
    assert "QWEN_NATURAL_E2E_EVAL_OUTPUT_EXISTS_REFUSING_OVERWRITE" in script
    assert "train_natural_bucket_lora.py" not in script
    assert "evaluate_diagnostic_e2e.py" not in script
    assert "scripts/train.py" not in script
    assert "llama" not in script.lower()
    assert "8way" not in script.lower()


def test_qwen_natural_eval_wrapper_review_passes_disabled_allowlist(tmp_path: Path) -> None:
    output_json = tmp_path / "eval_wrapper_review.json"
    status = review_qwen_natural_e2e_eval_wrapper.main(
        [
            "--output-json",
            str(output_json),
        ]
    )
    assert status == 0
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS_FIVE_ARM_EVAL_WRAPPER_DRY_RUN_REVIEW_NOT_EVAL"
    assert summary["training_started"] is False
    assert summary["e2e_eval_started"] is False
    assert summary["allowlist_enabled"] is False
    assert summary["query_budgets"] == [64, 128, 256, 512]


def test_qwen_natural_eval_imported_decoder_dependency_supports_variable_radix() -> None:
    assert evaluate_qwen_natural_e2e._decoder_dependency_errors() == []


def test_qwen_natural_zero_symbol_diagnosis_classifies_bucket_miss(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    _write_jsonl(
        eval_dir / "qwen_natural_e2e_bucket_observations.jsonl",
        [
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": 17,
                "observation_condition": "correct_key",
                "prompt_id": "p0",
                "query_index": 0,
                "position_index": 0,
                "observed_token_id": 99,
                "bucket_id": "",
                "digit": "",
                "radix": "",
                "compatible_bucket_ids": ["0", "1"],
                "frame_index": 0,
                "frame_digit_index": 0,
                "frame_digit_count": 2,
                "erasure_reason": "observed_token_not_in_variable_radix_bucket_set",
            },
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": 17,
                "observation_condition": "correct_key",
                "prompt_id": "p1",
                "query_index": 1,
                "position_index": 0,
                "observed_token_id": 100,
                "bucket_id": "",
                "digit": "",
                "radix": "",
                "compatible_bucket_ids": ["0", "1"],
                "frame_index": 0,
                "frame_digit_index": 1,
                "frame_digit_count": 2,
                "erasure_reason": "observed_token_not_in_variable_radix_bucket_set",
            },
        ],
    )
    with (eval_dir / "qwen_natural_e2e_decode_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_condition",
                "payload_id",
                "seed",
                "far_family",
                "query_budget",
                "accepted",
                "observed_symbols",
                "usable_symbols",
                "decode_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "far_family": "protected",
                "query_budget": "64",
                "accepted": "False",
                "observed_symbols": "0",
                "usable_symbols": "0",
                "decode_status": "insufficient_symbols",
            }
        )
    _write_jsonl(
        eval_dir / "qwen_natural_e2e_generated_outputs.jsonl",
        [{"response_text": "alpha beta"}],
    )
    (eval_dir / "qwen_natural_e2e_eval_summary.json").write_text(
        json.dumps({"status": "EVAL_COMPLETE", "observation_count": 2, "decode_row_count": 1}),
        encoding="utf-8",
    )

    output_json = tmp_path / "diagnosis.json"
    status = diagnose_qwen_natural_e2e_zero_symbols.main(
        ["--eval-dir", str(eval_dir), "--output-json", str(output_json)]
    )

    assert status == 0
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["failure_stage_classification"] == "observation_bucket_miss"
    assert summary["observations"]["bucket_hit_rows"] == 0
    assert summary["decode_trace"]["decode_status_counts"] == {"insufficient_symbols": 1}


def test_qwen_natural_zero_symbol_diagnosis_separates_partial_frame_hits(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    _write_jsonl(
        eval_dir / "qwen_natural_e2e_bucket_observations.jsonl",
        [
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": 17,
                "observation_condition": "correct_key",
                "prompt_id": "p0",
                "query_index": 0,
                "position_index": 0,
                "observed_token_id": 10,
                "bucket_id": "0",
                "digit": 0,
                "radix": 2,
                "compatible_bucket_ids": ["0", "1"],
                "frame_index": 0,
                "frame_digit_index": 0,
                "frame_digit_count": 2,
                "erasure_reason": "",
            },
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": 17,
                "observation_condition": "correct_key",
                "prompt_id": "p1",
                "query_index": 1,
                "position_index": 0,
                "observed_token_id": 99,
                "bucket_id": "",
                "digit": "",
                "radix": "",
                "compatible_bucket_ids": ["0", "1"],
                "frame_index": 0,
                "frame_digit_index": 1,
                "frame_digit_count": 2,
                "erasure_reason": "observed_token_not_in_variable_radix_bucket_set",
            },
        ],
    )
    with (eval_dir / "qwen_natural_e2e_decode_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_condition",
                "payload_id",
                "seed",
                "far_family",
                "query_budget",
                "accepted",
                "observed_symbols",
                "usable_symbols",
                "decode_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "far_family": "protected",
                "query_budget": "64",
                "accepted": "False",
                "observed_symbols": "1",
                "usable_symbols": "0",
                "decode_status": "insufficient_symbols",
            }
        )
    _write_jsonl(
        eval_dir / "qwen_natural_e2e_generated_outputs.jsonl",
        [{"response_text": "alpha beta"}],
    )
    (eval_dir / "qwen_natural_e2e_eval_summary.json").write_text(
        json.dumps({"status": "EVAL_COMPLETE", "observation_count": 2, "decode_row_count": 1}),
        encoding="utf-8",
    )

    output_json = tmp_path / "diagnosis.json"
    status = diagnose_qwen_natural_e2e_zero_symbols.main(
        ["--eval-dir", str(eval_dir), "--output-json", str(output_json)]
    )

    assert status == 0
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["failure_stage_classification"] == "decode_frame_assembly_incomplete"
    assert summary["observations"]["bucket_hit_rows"] == 1
    assert summary["frame_assembly"]["max_complete_frames_in_any_group"] == 0
    assert summary["frame_assembly"]["max_partial_symbol_hits_in_any_group"] == 1


def test_qwen_natural_eval_preflight_accepts_variable_radix_contracts(tmp_path: Path) -> None:
    train_dir = tmp_path / "train_data"
    output_dir = tmp_path / "eval_preflight"
    for payload_id in ("P0421", "P1729"):
        payload_dir = train_dir / payload_id
        payload_dir.mkdir(parents=True)
        contract = {
            "schema_name": "natural_evidence_variable_radix_train_contract_v1",
            "payload_id": payload_id,
            "encoding_mode": "variable_radix",
            "variable_radix_frame_policy": "repeat_payload",
            "variable_radix_min_positions_satisfied": True,
            "variable_radix_frame_count": 1,
            "evidence_example_count": 1,
            "total_eligible_positions": 1,
            "claim_control": {
                "contains_field_value_outputs": False,
                "contains_structured_evidence_blocks": False,
            },
        }
        (payload_dir / "variable_radix_train_contract.json").write_text(
            json.dumps(contract),
            encoding="utf-8",
        )
        _write_jsonl(
            payload_dir / "variable_radix_train.jsonl",
            [
                {
                    "schema_name": "natural_evidence_train_example_v1",
                    "prompt_id": "p0",
                    "prompt_split": "heldout",
                    "prompt": "User: say a word\nAssistant:",
                    "user_probe": "say a word",
                    "response_text": "alpha beta",
                    "eligible_positions": [
                        {
                            "bank_entry_id": "b0",
                            "entry_key": "e0",
                            "context_signature": "ctx0",
                            "candidate_token_ids": [10, 11],
                            "bucket_to_token_ids": {"0": [10], "1": [11]},
                            "compatible_bucket_ids": ["0", "1"],
                            "target_bucket": "0",
                            "target_bucket_token_ids": [10],
                            "target_digit": 0,
                            "target_radix": 2,
                            "token_index": 0,
                            "frame_index": 0,
                            "frame_digit_index": 0,
                            "frame_digit_count": 1,
                            "payload_digit_index": 0,
                        }
                    ],
                }
            ],
        )

    status = evaluate_qwen_natural_e2e.main(
        [
            "--config",
            "configs/natural_evidence_v1/pilot.yaml",
            "--train-data-dir",
            str(train_dir),
            "--run-root",
            str(tmp_path / "run"),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "Qwen/Qwen2.5-7B-Instruct",
            "--tokenizer-name",
            "Qwen/Qwen2.5-7B-Instruct",
            "--payload-ids",
            "P0421,P1729",
            "--seeds",
            "17,23",
            "--query-budgets",
            "64,128,256,512",
            "--eval-owner-probes",
            "2048",
            "--organic-null-prompts",
            "2048",
            "--prompt-split-id",
            "qwen_density_split_v1",
            "--budget-cap",
            "qwen_variable_radix_proof_of_life_v0_max_steps64",
        ]
    )
    assert status == 0
    preflight = json.loads((output_dir / "qwen_natural_e2e_eval_preflight.json").read_text(encoding="utf-8"))
    assert preflight["status"] == "PASS_DRY_RUN_READY_FOR_POST_TRAINING_EVAL"
    assert preflight["arms"] == [
        "qwen_protected",
        "qwen_raw",
        "qwen_task_only_lora",
        "wrong_key",
        "wrong_payload",
    ]
    assert preflight["eval_started"] is False
    assert preflight["training_started"] is False


def test_qwen_organic_density_wrapper_is_scoring_only_and_slurm_gated() -> None:
    script = (
        REPO_ROOT / "scripts/natural_evidence_v1/slurm/organic_variable_arity_density.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --job-name=nat-ev-qwen-orgdens" in script
    assert "#SBATCH --gres=gpu:A100:1" in script
    assert "#SBATCH --mail-type=ALL" in script
    assert "#SBATCH --mail-user=guanjie.lin001@umb.edu" in script
    assert 'START_ORGANIC_DENSITY="${START_ORGANIC_DENSITY:-0}"' in script
    assert 'DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"' in script
    assert "generate_reference_outputs.py" in script
    assert "prepare_actual_prefix_scoring_plan.py" in script
    assert "score_actual_prefix_reference_candidates.py" in script
    assert "build_expanded_actual_prefix_bucketized_candidates.py" in script
    assert "score_actual_prefix_suffix_compatibility.py" in script
    assert "combine_variable_arity_density_inputs.py" in script
    assert "audit_variable_arity_full_density.py" in script
    assert "ORGANIC_DENSITY_OUTPUT_EXISTS_REFUSING_OVERWRITE" in script
    assert "no training, no E2E, no payload recovery, no FAR" in script
    assert "train_natural_bucket_lora.py" not in script
    assert "--start-training" not in script
    assert "qwen_natural_e2e_pilot" not in script
    assert "llama" not in script.lower()
    assert "8way" not in script.lower()


def test_qwen_actual_prefix_reference_scoring_wrapper_is_scoring_only() -> None:
    script = (
        REPO_ROOT / "scripts/natural_evidence_v1/slurm/actual_prefix_reference_scoring.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --job-name=nat-ev-qwen-apscore" in script
    assert "#SBATCH --partition=DGXA100" in script
    assert "#SBATCH --account=pi_yinxin.wan" in script
    assert "#SBATCH --gres=gpu:A100:1" in script
    assert "#SBATCH --mail-type=ALL" in script
    assert "#SBATCH --mail-user=guanjie.lin001@umb.edu" in script
    assert 'TOKENIZER_KEY="${TOKENIZER_KEY:-qwen}"' in script
    assert 'CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-64}"' in script
    assert 'BUCKET_COUNT="${BUCKET_COUNT:-4}"' in script
    assert 'DRY_RUN_ONLY="${DRY_RUN_ONLY:-1}"' in script
    assert 'START_ACTUAL_PREFIX_SCORING="${START_ACTUAL_PREFIX_SCORING:-0}"' in script
    assert "score_actual_prefix_reference_candidates.py" in script
    assert "--input-jsonl \"$INPUT_JSONL\"" in script
    assert "--manifest-json \"$PLAN_MANIFEST\"" in script
    assert "--candidate-top-k \"$CANDIDATE_TOP_K\"" in script
    assert "--progress-json \"$PROGRESS_JSON\"" in script
    assert "--require-cuda" in script
    assert "ACTUAL_PREFIX_REFERENCE_SCORING_OUTPUT_EXISTS_REFUSING_OVERWRITE" in script
    assert "train_natural_bucket_lora.py" not in script
    assert "evaluate_diagnostic_e2e.py" not in script
    assert "qwen_diagnostic_high_risk_e2e_pilot" not in script
    assert "llama" not in script.lower()


def test_prefix_conditioned_selector_replay_matches_prefix_and_maps_bucket() -> None:
    exact = replay_prefix_conditioned_selector.find_prefix_conditioned_observed_token(
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        prefix_token_ids=[1, 2, 3],
        match_policy="exact_full",
    )
    assert exact["matched"] is True
    assert exact["observed_token_id"] == 4
    assert exact["response_token_index"] == 1

    suffix = replay_prefix_conditioned_selector.find_prefix_conditioned_observed_token(
        prompt_ids=[1, 2],
        response_ids=[8, 3, 4, 5],
        prefix_token_ids=[10, 11, 3, 4],
        match_policy="suffix_2",
    )
    assert suffix["matched"] is True
    assert suffix["observed_token_id"] == 5
    assert suffix["matched_prefix_token_count"] == 2

    classified = replay_prefix_conditioned_selector.classify_prefix_selector_event(
        match=suffix,
        token_to_bucket={5: "2"},
        compatible_bucket_ids=["1", "2"],
        target_bucket="2",
    )
    assert classified["prefix_matched"] is True
    assert classified["compatible_hit"] is True
    assert classified["target_hit"] is True
    assert classified["drift_reason"] == "target_hit"


def test_prefix_conditioned_selector_replay_is_artifact_only(tmp_path: Path) -> None:
    generated_path = tmp_path / "generated.jsonl"
    train_dir = tmp_path / "train"
    candidates_path = tmp_path / "expanded_actual_prefix_bucketized_candidates.jsonl"
    output_dir = tmp_path / "r1"
    _write_jsonl(
        generated_path,
        [
            {
                "schema_name": "natural_evidence_qwen_diagnostic_generated_output_v1",
                "model_condition": "protected_trained",
                "model_family": "qwen",
                "payload_id": "P0421",
                "seed": "17",
                "prompt_id": "p0",
                "prompt_split": "heldout",
                "query_index": 0,
                "prompt": "hello",
                "response_text": "alpha beta gamma",
            }
        ],
    )
    payload_dir = train_dir / "P0421"
    payload_dir.mkdir(parents=True)
    _write_jsonl(
        payload_dir / "variable_radix_train.jsonl",
        [
            {
                "schema_name": "natural_evidence_train_example_v1",
                "payload_id": "P0421",
                "prompt_id": "p0",
                "prompt_split": "heldout",
                "eligible_positions": [
                    {
                        "bank_entry_id": "b0",
                        "entry_key": "b0||p0||heldout||raw",
                        "compatible_bucket_ids": ["0", "1"],
                        "target_bucket": "1",
                        "target_bucket_token_ids": [3],
                        "candidate_token_ids": [3, 4],
                        "bucket_to_token_ids": {"0": [4], "1": [3]},
                        "token_index": 1,
                        "frame_index": 0,
                        "frame_digit_index": 0,
                        "frame_digit_count": 1,
                        "payload_digit_index": 0,
                        "target_digit": 1,
                        "target_radix": 2,
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        candidates_path,
        [
            {
                "schema_name": "natural_evidence_expanded_actual_prefix_bucketized_candidates_v1",
                "bank_entry_id": "b0",
                "prefix_token_ids": [1, 2],
                "prefix_response_token_count": 1,
                "candidates": [
                    {"bucket_id": "1", "token_id": 3, "text": "beta", "rank": 1, "probability": 0.6},
                    {"bucket_id": "0", "token_id": 4, "text": "gamma", "rank": 2, "probability": 0.4},
                ],
            }
        ],
    )

    status = replay_prefix_conditioned_selector.main(
        [
            "--generated-jsonl",
            str(generated_path),
            "--train-data-dir",
            str(train_dir),
            "--bucketized-candidates-jsonl",
            str(candidates_path),
            "--output-dir",
            str(output_dir),
            "--payload-ids",
            "P0421",
            "--tokenizer-name",
            "__simple_whitespace_test__",
            "--query-budgets",
            "1",
            "--match-policies",
            "exact_full",
        ]
    )
    assert status == 0
    summary = json.loads((output_dir / "prefix_conditioned_selector_replay_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE_PREFIX_CONDITIONED_SELECTOR_REPLAY_ARTIFACT_ONLY"
    assert summary["claim_control"]["paper_claim_allowed"] is False
    assert summary["claim_control"]["training_started"] is False
    assert summary["claim_control"]["not_payload_recovery"] is True
    aggregate = summary["aggregate_by_policy"][0]
    assert aggregate["scheduled_events"] == 1
    assert aggregate["prefix_matched_events"] == 1
    assert aggregate["compatible_hit_events"] == 1
    assert aggregate["target_hit_events"] == 1


def test_r1_selector_contract_analysis_blocks_when_protected_lacks_null_lift(tmp_path: Path) -> None:
    r1_dir = tmp_path / "r1"
    r1_dir.mkdir()
    (r1_dir / "prefix_conditioned_selector_replay_summary.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE_PREFIX_CONDITIONED_SELECTOR_REPLAY_ARTIFACT_ONLY",
                "claim_control": {
                    "paper_claim_allowed": False,
                    "not_payload_recovery": True,
                    "not_full_far": True,
                },
            }
        ),
        encoding="utf-8",
    )
    fields = [
        "model_condition",
        "payload_id",
        "seed",
        "expected_payload_id",
        "match_policy",
        "query_budget",
        "scheduled_events",
        "prefix_matched_events",
        "prefix_match_rate",
        "compatible_hit_events",
        "compatible_hit_rate",
        "target_comparable_events",
        "target_hit_events",
        "target_hit_rate",
        "scheduled_coordinate_count",
        "rediscovered_coordinate_count",
        "compatible_coordinate_count",
        "target_coordinate_count",
        "scheduled_frame_count",
        "rediscovered_frame_count",
        "compatible_frame_count",
        "target_frame_count",
        "max_target_slots_per_frame",
        "top_drift_reason",
    ]
    rows = [
        {
            "model_condition": "raw",
            "payload_id": "",
            "seed": "",
            "expected_payload_id": "P0421",
            "match_policy": "exact_full",
            "query_budget": "512",
            "prefix_match_rate": "0.9",
            "compatible_hit_rate": "0.8",
            "target_hit_rate": "0.4",
            "target_coordinate_count": "40",
            "max_target_slots_per_frame": "8",
        },
        {
            "model_condition": "protected_trained",
            "payload_id": "P0421",
            "seed": "17",
            "expected_payload_id": "P0421",
            "match_policy": "exact_full",
            "query_budget": "512",
            "prefix_match_rate": "0.2",
            "compatible_hit_rate": "0.1",
            "target_hit_rate": "0.05",
            "target_coordinate_count": "5",
            "max_target_slots_per_frame": "2",
        },
        {
            "model_condition": "task_only_lora",
            "payload_id": "P0421",
            "seed": "17",
            "expected_payload_id": "P0421",
            "match_policy": "exact_full",
            "query_budget": "512",
            "prefix_match_rate": "0.3",
            "compatible_hit_rate": "0.2",
            "target_hit_rate": "0.1",
            "target_coordinate_count": "10",
            "max_target_slots_per_frame": "3",
        },
    ]
    with (r1_dir / "prefix_conditioned_selector_replay_by_condition.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "analysis"
    status = analyze_r1_selector_contract.main(
        [
            "--r1-dir",
            str(r1_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert status == 0
    summary = json.loads((output_dir / "r1_selector_contract_repair_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE_R1_ANALYSIS_NO_PROTECTED_LIFT_OVER_NULLS"
    assert summary["claim_control"]["paper_claim_allowed"] is False
    assert summary["selector_contract_decision"]["training_allowed"] is False
    pairwise = list(csv.DictReader((output_dir / "r1_selector_contract_pairwise_lift.csv").open()))
    assert pairwise[0]["decision"] == "FAIL_BELOW_RAW_AND_TASK_ONLY"


def test_selector_contract_preflight_keeps_training_blocked_from_r1_no_lift(tmp_path: Path) -> None:
    r1_dir = tmp_path / "r1_analysis"
    r1_dir.mkdir()
    (r1_dir / "r1_selector_contract_repair_summary.json").write_text(
        json.dumps(
            {
                "schema_name": "natural_evidence_qwen_846699_r1_selector_contract_analysis_v1",
                "status": "COMPLETE_R1_ANALYSIS_NO_PROTECTED_LIFT_OVER_NULLS",
                "comparison_rows": 64,
                "positive_vs_raw_rows_total": 0,
                "positive_vs_task_only_rows_total": 0,
                "selector_contract_decision": {
                    "direct_replay_verifier_allowed": False,
                    "training_allowed": False,
                    "e2e_rerun_allowed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "preflight"

    status = design_selector_contract_preflight.main(
        [
            "--r1-analysis-dir",
            str(r1_dir),
            "--output-dir",
            str(output_dir),
            "--payload-ids",
            "P0421,P1729",
            "--query-budgets",
            "64,128,256,512",
        ]
    )

    assert status == 0
    summary = json.loads(
        (output_dir / "selector_contract_training_target_preflight_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "COMPLETE_SELECTOR_PRECOMMIT_DRAFT_BRANCH_AWARE_PREFLIGHT_PLAN_READY"
    assert summary["training_allowed"] is False
    assert summary["e2e_rerun_allowed"] is False
    assert "r1_protected_lift_over_raw" in summary["failed_gates"]
    assert "branch_aware_compatibility" in summary["needs_results"]
    contract = json.loads((output_dir / "selector_precommit_contract_draft.json").read_text(encoding="utf-8"))
    assert contract["contract_status"] == "DRAFT_NOT_ACTIVE_BLOCKED_BY_R1_NO_PROTECTED_LIFT"
    assert contract["selector"]["direct_replay_verifier_allowed"] is False
    plan_rows = list(csv.DictReader((output_dir / "branch_aware_training_target_preflight_plan.csv").open()))
    assert any(row["gate"] == "regenerated_suffix_repair" and row["status"] == "NEEDS_RESULTS" for row in plan_rows)


def test_branch_aware_suffix_repair_preparation_is_artifact_only(tmp_path: Path) -> None:
    selector_dir = tmp_path / "selector"
    selector_dir.mkdir()
    (selector_dir / "selector_contract_training_target_preflight_summary.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE_SELECTOR_PRECOMMIT_DRAFT_BRANCH_AWARE_PREFLIGHT_PLAN_READY",
                "selector_contract_status": "DRAFT_NOT_ACTIVE_BLOCKED_BY_R1_NO_PROTECTED_LIFT",
                "training_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    (selector_dir / "selector_precommit_contract_draft.json").write_text(
        json.dumps(
            {
                "schema_name": "natural_evidence_v1_selector_precommit_contract_draft_v1",
                "contract_status": "DRAFT_NOT_ACTIVE_BLOCKED_BY_R1_NO_PROTECTED_LIFT",
                "protocol_id": "natural_evidence_v1",
                "selector": {"selector_id": "prefix_conditioned_observed_text_v0"},
            }
        ),
        encoding="utf-8",
    )
    r1_dir = tmp_path / "r1"
    _write_jsonl(
        r1_dir / "prefix_conditioned_selector_replay_examples.jsonl",
        [
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "expected_payload_id": "P0421",
                "seed": "17",
                "prompt_id": "p0",
                "prompt_slot": 0,
                "query_index": 0,
                "match_policy": "suffix_8",
                "drift_reason": "compatible_non_target",
                "observed_token_id": 10,
                "observed_token_text": " boots",
                "observed_token_class": "word",
                "bucket_id": "0",
                "target_bucket": "1",
                "compatible_bucket_ids": ["0", "1"],
                "candidate_bucket_token_texts": {
                    "0": [{"token_id": 10, "token_text": " boots", "token_class": "word"}],
                    "1": [{"token_id": 11, "token_text": " shoes", "token_class": "word"}],
                },
            }
        ],
    )
    train_dir = tmp_path / "train"
    payload_dir = train_dir / "P0421"
    payload_dir.mkdir(parents=True)
    _write_jsonl(
        payload_dir / "variable_radix_train.jsonl",
        [
            {
                "prompt_id": "p0",
                "prompt": "User: checklist\nAssistant:",
                "user_probe": "checklist",
                "response_text": "Wear boots.",
                "eligible_positions": [
                    {
                        "token_index": 1,
                        "frame_index": 0,
                        "frame_digit_index": 0,
                        "target_bucket": "1",
                        "compatible_bucket_ids": ["0", "1"],
                        "target_bucket_token_ids": [11],
                        "candidate_token_ids": [10, 11],
                        "bank_entry_id": "b0",
                        "entry_key": "e0",
                    }
                ],
            }
        ],
    )
    output_dir = tmp_path / "prepared"
    status = prepare_branch_aware_suffix_repair_diagnostics.main(
        [
            "--selector-preflight-dir",
            str(selector_dir),
            "--r1-replay-dir",
            str(r1_dir),
            "--train-data-dir",
            str(train_dir),
            "--output-dir",
            str(output_dir),
            "--payload-ids",
            "P0421",
            "--max-plan-rows",
            "8",
        ]
    )

    assert status == 0
    summary = json.loads((output_dir / "branch_aware_compatibility_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE_BRANCH_AWARE_SUFFIX_REPAIR_INPUTS_PREPARED_NOT_SCORED"
    assert summary["claim_control"]["training_started"] is False
    assert summary["claim_control"]["model_scoring_started"] is False
    assert summary["needs_slurm_scoring"] is True
    assert summary["planned_branch_aware_rows"] == 1
    manifest = json.loads((output_dir / "regenerated_suffix_repair_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "REPAIR_INPUTS_READY_NOT_REGENERATED"
    plan_rows = [json.loads(line) for line in (output_dir / "branch_aware_compatibility_scoring_plan.jsonl").read_text(encoding="utf-8").splitlines()]
    assert plan_rows[0]["prompt"] == "User: checklist\nAssistant:"
    assert plan_rows[0]["generation_started"] is False


def test_local_suffix_repair_dry_run_is_artifact_only(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    (prepared_dir / "branch_aware_compatibility_summary.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE_BRANCH_AWARE_SUFFIX_REPAIR_INPUTS_PREPARED_NOT_SCORED",
                "claim_control": {
                    "training_started": False,
                    "generation_started": False,
                    "model_scoring_started": False,
                    "e2e_eval_started": False,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        prepared_dir / "branch_aware_compatibility_scoring_plan.jsonl",
        [
            {
                "row_id": "r0",
                "frame_index": 3,
                "frame_digit_index": 7,
                "generation_started": False,
                "training_started": False,
            }
        ],
    )
    _write_jsonl(
        prepared_dir / "regenerated_suffix_repair_examples.jsonl",
        [
            {
                "row_id": "r0",
                "model_condition": "protected_trained",
                "expected_payload_id": "P0421",
                "seed": "17",
                "prompt_id": "p0",
                "prompt_slot": 0,
                "match_policy": "suffix_8",
                "drift_reason": "compatible_non_target",
                "observed_token_text": " boots",
                "target_bucket": "1",
                "target_bucket_tokens": [
                    {"token_id": 11, "token_text": " shoes", "token_class": "word"}
                ],
                "original_response_text": "Wear boots on the trail.",
                "prompt": "User: checklist\nAssistant:",
                "user_probe": "checklist",
                "training_started": False,
                "generation_started": False,
                "e2e_eval_started": False,
                "paper_claim_allowed": False,
            }
        ],
    )
    output_dir = tmp_path / "dry_run"
    status = dry_run_local_suffix_repair.main(
        [
            "--repair-examples-jsonl",
            str(prepared_dir / "regenerated_suffix_repair_examples.jsonl"),
            "--branch-plan-jsonl",
            str(prepared_dir / "branch_aware_compatibility_scoring_plan.jsonl"),
            "--prepared-summary-json",
            str(prepared_dir / "branch_aware_compatibility_summary.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    summary = json.loads((output_dir / "local_suffix_repair_dry_run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE_LOCAL_SUFFIX_REPAIR_DRY_RUN_NOT_GENERATED"
    assert summary["claim_control"]["training_started"] is False
    assert summary["claim_control"]["generation_started"] is False
    assert summary["claim_control"]["model_scoring_started"] is False
    assert summary["repair_ready_rows"] == 1
    rows = [
        json.loads(line)
        for line in (output_dir / "local_suffix_repair_dry_run_rows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert rows[0]["dry_run_status"] == "REPAIR_DRY_RUN_TEXT_SUBSTITUTION_READY_NOT_REGENERATED"
    assert rows[0]["repaired_response_text"] == "Wear shoes on the trail."
    assert rows[0]["frame_index"] == 3
    assert rows[0]["paper_claim_allowed"] is False


def test_balanced_branch_aware_example_export_includes_protected_and_task_only(tmp_path: Path) -> None:
    generated_path = tmp_path / "generated.jsonl"
    train_dir = tmp_path / "train"
    payload_dir = train_dir / "P0421"
    payload_dir.mkdir(parents=True)
    candidates_path = tmp_path / "candidates.jsonl"
    output_dir = tmp_path / "balanced"
    _write_jsonl(
        generated_path,
        [
            {
                "model_condition": "raw",
                "payload_id": "",
                "seed": "",
                "expected_payload_id": "",
                "prompt_id": "p0",
                "prompt": "pfx",
                "response_text": "obs rest",
                "query_index": 0,
                "user_probe": "probe",
            },
            {
                "model_condition": "protected_trained",
                "payload_id": "P0421",
                "seed": "17",
                "prompt_id": "p0",
                "prompt": "pfx",
                "response_text": "obs rest",
                "query_index": 0,
                "user_probe": "probe",
            },
            {
                "model_condition": "task_only_lora",
                "payload_id": "P0421",
                "seed": "17",
                "prompt_id": "p0",
                "prompt": "pfx",
                "response_text": "obs rest",
                "query_index": 0,
                "user_probe": "probe",
            },
        ],
    )
    _write_jsonl(
        payload_dir / "variable_radix_train.jsonl",
        [
            {
                "prompt_id": "p0",
                "eligible_positions": [
                    {
                        "bank_entry_id": "b0",
                        "prompt_slot": 0,
                        "frame_index": 0,
                        "frame_digit_index": 0,
                        "target_bucket": "1",
                        "target_bucket_token_ids": [3],
                        "compatible_bucket_ids": ["0", "1"],
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        candidates_path,
        [
                {
                    "bank_entry_id": "b0",
                    "prefix_token_ids": [1],
                    "candidates": [
                    {"token_id": 2, "bucket_id": "0"},
                    {"token_id": 3, "bucket_id": "1"},
                ],
            }
        ],
    )

    status = export_balanced_branch_aware_examples.main(
        [
            "--generated-jsonl",
            str(generated_path),
            "--train-data-dir",
            str(train_dir),
            "--bucketized-candidates-jsonl",
            str(candidates_path),
            "--output-dir",
            str(output_dir),
            "--payload-ids",
            "P0421",
            "--tokenizer-name",
            "__simple_whitespace_test__",
            "--match-policies",
            "exact_full",
            "--model-conditions",
            "protected_trained,task_only_lora,raw",
            "--drift-reasons",
            "compatible_non_target",
            "--per-slice",
            "2",
            "--max-rows",
            "10",
        ]
    )

    assert status == 0
    summary = json.loads(
        (output_dir / "balanced_branch_aware_example_export_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "COMPLETE_BALANCED_BRANCH_AWARE_EXAMPLES_EXPORTED_NOT_SCORED"
    assert summary["claim_control"]["training_started"] is False
    assert summary["claim_control"]["model_scoring_started"] is False
    assert summary["condition_counts"]["protected_trained"] == 1
    assert summary["condition_counts"]["task_only_lora"] == 1
    assert summary["condition_counts"]["raw"] == 1
    examples = [
        json.loads(line)
        for line in (output_dir / "prefix_conditioned_selector_replay_examples.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {row["model_condition"] for row in examples} == {
        "protected_trained",
        "task_only_lora",
        "raw",
    }
    assert examples[0]["generated_response_text"] == "obs rest"
    assert all(row["paper_claim_allowed"] is False for row in examples)


def test_branch_aware_compatibility_pass_flags_are_diagnostic_only() -> None:
    flags = score_branch_aware_compatibility._score_pass_flags(
        response_delta=0.2,
        suffix_delta=0.8,
        suffix_token_count=5,
        max_response_delta_per_token=0.5,
        max_suffix_delta_per_token=1.0,
    )
    assert flags["response_naturalness_proxy_pass"] is True
    assert flags["suffix_preserving_proxy_pass"] is True
    assert flags["branch_aware_proxy_pass"] is True

    failed = score_branch_aware_compatibility._score_pass_flags(
        response_delta=0.2,
        suffix_delta=1.5,
        suffix_token_count=5,
        max_response_delta_per_token=0.5,
        max_suffix_delta_per_token=1.0,
    )
    assert failed["response_naturalness_proxy_pass"] is True
    assert failed["suffix_preserving_proxy_pass"] is False
    assert failed["branch_aware_proxy_pass"] is False


def test_branch_aware_scoring_pairs_use_repaired_suffix_context() -> None:
    pairs = score_branch_aware_compatibility.scoring_pairs_from_row(
        {
            "prompt": "Prompt:",
            "original_response_text": "Wear boots on the trail.",
            "repaired_response_text": "Wear shoes on the trail.",
            "prefix_before_observed": "Wear",
            "observed_match_text": " boots",
            "target_token_text": " shoes",
            "local_suffix_window_after_observed": " on the trail.",
        }
    )
    assert pairs["original_response"] == ("Prompt:", "Wear boots on the trail.")
    assert pairs["repaired_response"] == ("Prompt:", "Wear shoes on the trail.")
    assert pairs["observed_suffix_window"] == ("Prompt:Wear boots", " on the trail.")
    assert pairs["target_suffix_window"] == ("Prompt:Wear shoes", " on the trail.")


def test_branch_aware_score_interpretation_exports_primary_candidates(tmp_path: Path) -> None:
    score_rows = tmp_path / "score_rows.jsonl"
    score_summary = tmp_path / "score_summary.json"
    plan_rows = tmp_path / "plan.jsonl"
    dry_rows = tmp_path / "dry.jsonl"
    output_dir = tmp_path / "interpretation"
    base_row = {
        "row_id": "r1",
        "model_condition": "protected_trained",
        "payload_id": "P0421",
        "expected_payload_id": "P0421",
        "seed": "17",
        "prompt_id": "p1",
        "prompt_slot": 0,
        "match_policy": "suffix_8",
        "drift_reason": "compatible_non_target",
        "observed_token_class": "word",
        "target_bucket": "2",
        "observed_token_text": " packing",
        "target_token_text": " planning",
        "frame_index": 0,
        "frame_digit_index": 3,
        "token_index": 5,
    }
    _write_jsonl(
        score_rows,
        [
            {
                **base_row,
                "branch_aware_proxy_pass": True,
                "response_naturalness_proxy_pass": True,
                "suffix_preserving_proxy_pass": True,
                "response_delta_nll_per_token": 0.1,
                "suffix_delta_nll_per_token": 0.2,
            },
            {
                **base_row,
                "row_id": "r2",
                "observed_token_class": "punctuation",
                "branch_aware_proxy_pass": True,
                "response_naturalness_proxy_pass": True,
                "suffix_preserving_proxy_pass": True,
                "response_delta_nll_per_token": 0.1,
                "suffix_delta_nll_per_token": 0.2,
            },
            {
                **base_row,
                "row_id": "r3",
                "branch_aware_proxy_pass": False,
                "response_naturalness_proxy_pass": True,
                "suffix_preserving_proxy_pass": False,
                "response_delta_nll_per_token": 0.1,
                "suffix_delta_nll_per_token": 1.5,
            },
        ],
    )
    _write_jsonl(plan_rows, [base_row, {**base_row, "row_id": "r2"}, {**base_row, "row_id": "r3"}])
    _write_jsonl(
        dry_rows,
        [
            {**base_row, "repaired_response_text": "x"},
            {**base_row, "row_id": "r2", "repaired_response_text": "x"},
            {**base_row, "row_id": "r3", "repaired_response_text": "x"},
        ],
    )
    score_summary.write_text(
        json.dumps({"status": "COMPLETE_BRANCH_AWARE_COMPATIBILITY_MODEL_SCORED_PROXY_NOT_GENERATED"}),
        encoding="utf-8",
    )
    status = analyze_branch_aware_score_interpretation.main(
        [
            "--score-rows-jsonl",
            str(score_rows),
            "--score-summary-json",
            str(score_summary),
            "--scoring-plan-jsonl",
            str(plan_rows),
            "--dry-run-rows-jsonl",
            str(dry_rows),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert status == 0
    summary = json.loads((output_dir / "branch_aware_score_interpretation_summary.json").read_text(encoding="utf-8"))
    assert summary["primary_probe_candidate_rows"] == 1
    assert summary["secondary_candidate_rows"] == 1
    assert summary["rejected_rows"] == 1
    candidates = [
        json.loads(line)
        for line in (output_dir / "repaired_target_mass_probe_candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(candidates) == 1
    assert candidates[0]["candidate_tier"] == "PRIMARY_COMPATIBLE_NON_TARGET_LOW_DELTA"
    assert candidates[0]["training_started"] is False


def test_repaired_teacher_forced_target_mass_probe_design_expands_score_plan(tmp_path: Path) -> None:
    candidate_jsonl = tmp_path / "candidates.jsonl"
    examples_jsonl = tmp_path / "examples.jsonl"
    branch_summary_json = tmp_path / "branch_summary.json"
    output_dir = tmp_path / "design"
    base_candidate = {
        "candidate_id": "cand1",
        "source_row_id": "r1",
        "primary_probe_candidate": True,
        "candidate_tier": "PRIMARY_COMPATIBLE_NON_TARGET_LOW_DELTA",
        "branch_aware_proxy_pass": True,
        "model_condition": "protected_trained",
        "expected_payload_id": "P0421",
        "payload_id": "P0421",
        "seed": "17",
        "prompt_id": "p1",
        "prompt_slot": 0,
        "query_index": 1,
        "frame_index": 2,
        "frame_digit_index": 3,
        "match_policy": "exact_full",
        "drift_reason": "compatible_non_target",
        "observed_token_class": "word",
        "observed_token_text": " planning",
        "target_bucket": "1",
        "target_bucket_tokens": [{"token_id": 11, "token_text": " creating"}],
        "target_token_text": " creating",
        "compatible_bucket_ids": ["0", "1"],
        "prompt": "Prompt:",
        "prefix_before_observed": "When",
        "token_index": 0,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
    }
    raw_candidate = {
        **base_candidate,
        "candidate_id": "cand2",
        "source_row_id": "r2",
        "model_condition": "raw",
        "payload_id": "",
        "seed": "",
        "query_index": 2,
    }
    _write_jsonl(candidate_jsonl, [base_candidate, raw_candidate])
    _write_jsonl(
        examples_jsonl,
        [
            {
                **base_candidate,
                "candidate_bucket_token_texts": {
                    "0": [{"token_id": 10, "token_text": " planning"}],
                    "1": [{"token_id": 11, "token_text": " creating"}],
                },
                "target_bucket_token_ids": [11],
            },
            {
                **raw_candidate,
                "candidate_bucket_token_texts": {
                    "0": [{"token_id": 20, "token_text": " footwear"}],
                    "1": [{"token_id": 21, "token_text": " shoes"}],
                },
                "target_bucket_token_ids": [21],
            },
        ],
    )
    branch_summary_json.write_text(
        json.dumps({"status": "COMPLETE_BRANCH_AWARE_SCORE_INTERPRETATION_REPAIRED_TARGET_PREFLIGHT_NOT_TRAINING"}),
        encoding="utf-8",
    )
    status = design_repaired_teacher_forced_target_mass_probe.main(
        [
            "--candidate-jsonl",
            str(candidate_jsonl),
            "--branch-summary-json",
            str(branch_summary_json),
            "--balanced-examples-jsonl",
            str(examples_jsonl),
            "--output-dir",
            str(output_dir),
            "--checkpoint-root",
            "/ckpts",
        ]
    )
    assert status == 0
    summary = json.loads(
        (output_dir / "repaired_teacher_forced_target_mass_probe_design_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "COMPLETE_REPAIRED_TEACHER_FORCED_TARGET_MASS_PROBE_DESIGN_NOT_SCORED"
    assert summary["candidate_rows"] == 2
    assert summary["score_plan_rows"] == 8
    assert summary["training_started"] is False
    plan_rows = [
        json.loads(line)
        for line in (output_dir / "repaired_teacher_forced_target_mass_probe_scoring_plan.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {row["scoring_model_condition"] for row in plan_rows} == {
        "base",
        "protected_trained",
        "task_only_lora",
    }
    assert any(row["bucket_to_token_ids"] for row in plan_rows)
    assert all(row["model_scoring_started"] is False for row in plan_rows)


def test_repaired_teacher_forced_target_mass_score_stats() -> None:
    rows = [
        {
            "scoring_model_condition": "base",
            "scoring_payload_id": "P0421",
            "scoring_seed": "17",
            "source_model_condition": "protected_trained",
            "drift_reason": "compatible_non_target",
            "observed_token_class": "word",
            "prompt_id": "p1",
            "target_candidate_mass": 0.25,
            "non_target_compatible_candidate_mass": 0.50,
            "best_other_candidate_mass": 0.50,
            "target_margin": -0.25,
            "full_vocab_target_mass": 0.01,
            "target_rank": 2,
        },
        {
            "scoring_model_condition": "protected_trained",
            "scoring_payload_id": "P0421",
            "scoring_seed": "17",
            "source_model_condition": "protected_trained",
            "drift_reason": "compatible_non_target",
            "observed_token_class": "word",
            "prompt_id": "p1",
            "target_candidate_mass": 0.60,
            "non_target_compatible_candidate_mass": 0.20,
            "best_other_candidate_mass": 0.20,
            "target_margin": 0.40,
            "full_vocab_target_mass": 0.03,
            "target_rank": 1,
        },
    ]
    stats = score_repaired_teacher_forced_target_mass_probe._stats(rows)
    assert stats["scored_rows"] == 2
    assert math.isclose(stats["mean_target_candidate_mass"], 0.425)
    assert math.isclose(stats["target_rank1_rate"], 0.5)
    groups = score_repaired_teacher_forced_target_mass_probe._group_rows(rows)
    protected = [
        row
        for row in groups
        if row["group_kind"] == "scoring_model_condition"
        and row["group_value"] == "protected_trained"
    ][0]
    assert protected["scored_rows"] == 1
    assert math.isclose(protected["mean_target_margin"], 0.40)


def test_hermes_notify_dry_run_reports_required_channels(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_TG_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("HERMES_TG_CHAT_ID", "987654321")
    monkeypatch.setenv("HERMES_EMAIL_TO", "guanjie.lin001@umb.edu")
    monkeypatch.setenv("HERMES_SMTP_HOST", "smtp.example.invalid")
    payload = hermes_notify.notify(
        subject="Hermes dry run",
        body="phase: diagnostic",
        channels=["telegram", "email"],
        dry_run=True,
    )
    assert payload["schema_name"] == hermes_notify.SCHEMA_NAME
    assert payload["status"] == "DRY_RUN_COMPLETE"
    assert payload["configured_all_required_channels"] is True
    assert payload["sent_all_required_channels"] is False
    assert payload["channels"]["telegram"]["status"] == "DRY_RUN_NOT_SENT"
    assert payload["channels"]["email"]["status"] == "DRY_RUN_NOT_SENT"


def test_hermes_notify_missing_config_blocks_non_dry_run(monkeypatch) -> None:
    for name in (
        "HERMES_TG_BOT_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "TG_BOT_TOKEN",
        "HERMES_TG_CHAT_ID",
        "TELEGRAM_CHAT_ID",
        "TG_CHAT_ID",
        "HERMES_EMAIL_TO",
        "HERMES_NOTIFY_EMAIL_TO",
        "EMAIL_TO",
    ):
        monkeypatch.delenv(name, raising=False)
    payload = hermes_notify.notify(
        subject="Hermes notification",
        body="phase: diagnostic",
        channels=["telegram", "email"],
        dry_run=False,
    )
    assert payload["status"] == "NOT_CONFIGURED"
    assert payload["configured_all_required_channels"] is False
    assert payload["sent_all_required_channels"] is False
    assert payload["channels"]["telegram"]["status"] == "NOT_CONFIGURED"
    assert payload["channels"]["email"]["status"] == "NOT_CONFIGURED"


def test_hermes_notify_loads_hermes_env_aliases(monkeypatch, tmp_path: Path) -> None:
    for name in (
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_HOME_CHANNEL",
        "EMAIL_HOME_ADDRESS",
        "EMAIL_SMTP_HOST",
        "EMAIL_SMTP_PORT",
        "EMAIL_ADDRESS",
        "EMAIL_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=123456:test-token",
                "TELEGRAM_HOME_CHANNEL=987654321",
                "EMAIL_HOME_ADDRESS=owner@example.test",
                "EMAIL_SMTP_HOST=smtp.example.invalid",
                "EMAIL_SMTP_PORT=587",
                "EMAIL_ADDRESS=owner@example.test",
                "EMAIL_PASSWORD=secret",
            ]
        ),
        encoding="utf-8",
    )
    loaded = hermes_notify.load_env_file(env_file)
    assert "TELEGRAM_HOME_CHANNEL" in loaded["loaded_keys"]
    payload = hermes_notify.notify(
        subject="Hermes dry run",
        body="phase: diagnostic",
        channels=["telegram", "email"],
        dry_run=True,
    )
    assert payload["configured_all_required_channels"] is True
    assert payload["channels"]["telegram"]["status"] == "DRY_RUN_NOT_SENT"
    assert payload["channels"]["email"]["status"] == "DRY_RUN_NOT_SENT"
    assert payload["channels"]["email"]["smtp_host"] == "smtp.example.invalid"

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.natural_evidence_v1 import (
    audit_opportunity_bank,
    build_bucket_bank,
    compile_train_dataset,
    generate_reference_outputs,
    sweep_balance_gate,
    validate_static,
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
    assert "qwen_natural_e2e_pilot" in allowlist
    assert "enabled: false" in allowlist
    assert "sweep_qwen_4way_balance_thresholds" in allowlist
    assert "Do not start protected LoRA training" in state
    assert gate_status["gates"]["phase_a_outputs_complete"] == "PASS"
    assert gate_status["gates"]["qwen_e2e_pilot"] == "TODO_AFTER_RESULTS"
    assert gate_status["gates"]["llama_e2e_pilot"] == "TODO_AFTER_RESULTS"
    assert gate_status["next_allowed_action"] != "qwen_natural_e2e_pilot"
    assert "24576_fingerprints" in gate_status["forbidden_claims"]

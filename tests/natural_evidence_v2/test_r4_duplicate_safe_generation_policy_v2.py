from pathlib import Path

from scripts.natural_evidence_v2.validate_r4_first_token_event_duplicate_safe_generation_policy_v2 import (
    read_yaml,
    select_first_nonduplicate_attempt,
    validate_policy,
)


def test_duplicate_safe_generation_policy_v2_validates() -> None:
    config = read_yaml(Path("configs/natural_evidence_v2/r4_first_token_event_duplicate_safe_generation_v2.yaml"))
    summary = validate_policy(config)
    assert summary["status"] == "PASS_R4_FIRST_TOKEN_EVENT_DUPLICATE_SAFE_GENERATION_POLICY_V2"
    assert summary["slurm_allowed"] is False
    assert summary["generation_started"] is False


def test_retry_selection_is_blind_to_decode_success() -> None:
    attempts = [
        {"response_text_sha256": "dup", "decode_accept": True, "payload_match": True},
        {"response_text_sha256": "new", "decode_accept": False, "payload_match": False},
    ]
    selected = select_first_nonduplicate_attempt(attempts, {"dup"}, max_duplicate_retries=3)
    assert selected["status"] == "selected"
    assert selected["attempt_index"] == 1
    assert selected["response_text_sha256"] == "new"


def test_retry_selection_exhaustion_keeps_failed_row() -> None:
    attempts = [
        {"response_text_sha256": "dup", "decode_accept": False},
        {"response_text_sha256": "dup", "decode_accept": True},
    ]
    selected = select_first_nonduplicate_attempt(attempts, {"dup"}, max_duplicate_retries=3)
    assert selected["status"] == "duplicate_exhausted"
    assert selected["selection_reason"] == "duplicate_only_no_decode_or_payload_filter"


def test_same_policy_applies_to_all_arms() -> None:
    config = read_yaml(Path("configs/natural_evidence_v2/r4_first_token_event_duplicate_safe_generation_v2.yaml"))
    generation = config["generation"]
    assert generation["apply_same_policy_to_all_arms"] is True
    assert generation["retry_blind_to_decode_accept"] is True
    assert generation["retry_blind_to_payload_match"] is True

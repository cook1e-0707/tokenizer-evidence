from types import SimpleNamespace

import pytest

from scripts.natural_evidence_v2.generate_r4_after_868016_controller_outputs import (
    duplicate_policy_payload,
    hmac_seed,
    trace_bound_fields,
)
from scripts.natural_evidence_v2.verify_r4_first_token_event_trace_binding import verify_trace_binding


def test_hmac_seed_is_deterministic_and_attempt_specific() -> None:
    first = hmac_seed(
        public_run_salt="public-salt",
        condition="protected",
        shard_id="shard_00",
        prompt_id="prompt-001",
        attempt_index=0,
    )
    assert first == hmac_seed(
        public_run_salt="public-salt",
        condition="protected",
        shard_id="shard_00",
        prompt_id="prompt-001",
        attempt_index=0,
    )
    assert first != hmac_seed(
        public_run_salt="public-salt",
        condition="protected",
        shard_id="shard_00",
        prompt_id="prompt-001",
        attempt_index=1,
    )


def test_duplicate_policy_payload_requires_blind_retry(tmp_path) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "\n".join(
            [
                "generation:",
                "  retry_selection_rule: first_nonduplicate_exact_hash",
                "  retry_blind_to_decode_accept: true",
                "  retry_blind_to_payload_match: true",
                "  apply_same_policy_to_all_arms: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert duplicate_policy_payload(policy)["generation"]["retry_blind_to_decode_accept"] is True

    policy.write_text(
        "\n".join(
            [
                "generation:",
                "  retry_selection_rule: first_nonduplicate_exact_hash",
                "  retry_blind_to_decode_accept: false",
                "  retry_blind_to_payload_match: true",
                "  apply_same_policy_to_all_arms: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="blind to decode accept"):
        duplicate_policy_payload(policy)


def test_trace_bound_fields_produces_verifiable_binding_fields() -> None:
    args = SimpleNamespace(
        controller_bonus_nats=4.0,
        controller_penalty_nats=0.5,
        controller_max_target_mass=0.5,
        controller_max_kl_budget=0.5,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        surface_codebook_hash="surface-codebook-hash",
        score_rows="rows.jsonl",
        payload_id="a55e",
        key_id_not_secret_key="key-id",
        decoder_version_hash="decoder-hash",
    )
    row = {
        "coordinate_id": 7,
        "prompt_id": "prompt-007",
        "prompt_text": "Give one practical maintenance suggestion.",
    }
    bound = trace_bound_fields(
        args=args,
        condition="protected",
        row=row,
        response_text="Use a towel to dry the area.",
        output_token_ids=[101, 202, 303],
        first_generated_token_id=202,
        input_width=1,
        target_ids=[202],
        other_ids=[404],
        controller_enabled=True,
    )
    record = {"generation_id": "gen-007", **bound}
    result = verify_trace_binding(record)
    assert result["valid"] is True
    assert result["errors"] == []

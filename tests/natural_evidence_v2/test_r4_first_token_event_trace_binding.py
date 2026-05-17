from scripts.natural_evidence_v2.verify_r4_first_token_event_trace_binding import (
    compute_binding_hmac,
    event_merkle_root,
    sha256_json,
    sha256_text,
    verify_trace_binding,
)


def make_row() -> dict:
    token_ids = [10, 20, 30, 40]
    events = [
        {"position": 1, "token_id": 20, "coordinate_id": 3},
        {"position": 3, "token_id": 40, "coordinate_id": 7},
    ]
    row = {
        "generation_id": "g-test",
        "arm": "protected",
        "model_checkpoint_hash": "model-hash",
        "tokenizer_hash": "tok-hash",
        "controller_config_hash": "controller-hash",
        "surface_codebook_hash": "surface-codebook-hash",
        "prompt_hash": "prompt-hash",
        "output_text": "A useful ordinary answer.",
        "output_token_ids": token_ids,
        "event_trace_merkle_root": event_merkle_root(events),
        "selected_events": events,
        "selected_event_positions": [1, 3],
        "selected_token_ids": [20, 40],
        "coordinate_ids": [3, 7],
        "target_token_set_hashes": ["target-a", "target-b"],
        "wrong_key_token_set_hashes": ["wrong-key-a", "wrong-key-b"],
        "payload_id": "a55e",
        "key_id_not_secret_key": "key-01",
        "decoder_version_hash": "decoder-hash",
        "wrong_key_replay_accept": False,
        "wrong_payload_replay_accept": False,
    }
    row["output_text_sha256"] = sha256_text(row["output_text"])
    row["output_token_ids_sha256"] = sha256_json(token_ids)
    row["binding_hmac"] = compute_binding_hmac(row, "test-secret")
    return row


def test_trace_binding_valid_row() -> None:
    result = verify_trace_binding(make_row(), hmac_secret="test-secret")
    assert result["valid"] is True
    assert result["errors"] == []


def test_trace_binding_rejects_output_hash_mismatch() -> None:
    row = make_row()
    row["output_text_sha256"] = "bad"
    result = verify_trace_binding(row, hmac_secret="test-secret")
    assert result["valid"] is False
    assert "output_text_sha256 mismatch" in result["errors"]


def test_trace_binding_rejects_tokenizer_hash_missing() -> None:
    row = make_row()
    row["tokenizer_hash"] = ""
    result = verify_trace_binding(row, hmac_secret="test-secret")
    assert result["valid"] is False
    assert "missing required hash field: tokenizer_hash" in result["errors"]


def test_trace_binding_rejects_event_position_outside_tokens() -> None:
    row = make_row()
    row["selected_event_positions"] = [1, 99]
    row["binding_hmac"] = compute_binding_hmac(row, "test-secret")
    result = verify_trace_binding(row, hmac_secret="test-secret")
    assert result["valid"] is False
    assert "event position outside output token ids" in result["errors"]


def test_trace_binding_rejects_wrong_key_or_wrong_payload_accept() -> None:
    row = make_row()
    row["wrong_key_replay_accept"] = True
    row["wrong_payload_replay_accept"] = True
    row["binding_hmac"] = compute_binding_hmac(row, "test-secret")
    result = verify_trace_binding(row, hmac_secret="test-secret")
    assert result["valid"] is False
    assert "wrong-key replay accepted" in result["errors"]
    assert "wrong-payload replay accepted" in result["errors"]

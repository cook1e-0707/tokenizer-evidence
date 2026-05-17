from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_HASH_FIELDS = (
    "model_checkpoint_hash",
    "tokenizer_hash",
    "controller_config_hash",
    "surface_codebook_hash",
    "prompt_hash",
    "decoder_version_hash",
)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


def event_merkle_root(events: list[Mapping[str, Any]]) -> str:
    leaves = [sha256_json(event) for event in events]
    if not leaves:
        return sha256_text("")
    level = leaves
    while len(level) > 1:
        next_level: list[str] = []
        for idx in range(0, len(level), 2):
            right = level[idx + 1] if idx + 1 < len(level) else level[idx]
            next_level.append(sha256_text(level[idx] + right))
        level = next_level
    return level[0]


def binding_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "generation_id",
        "arm",
        "model_checkpoint_hash",
        "tokenizer_hash",
        "controller_config_hash",
        "surface_codebook_hash",
        "prompt_hash",
        "output_text_sha256",
        "output_token_ids_sha256",
        "event_trace_merkle_root",
        "selected_event_positions",
        "selected_token_ids",
        "coordinate_ids",
        "target_token_set_hashes",
        "wrong_key_token_set_hashes",
        "payload_id",
        "key_id_not_secret_key",
        "decoder_version_hash",
    ]
    return {key: row.get(key) for key in keys}


def compute_binding_hmac(row: Mapping[str, Any], secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), canonical_json(binding_payload(row)).encode("utf-8"), hashlib.sha256).hexdigest()


def verify_trace_binding(row: Mapping[str, Any], *, hmac_secret: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    output_text = str(row.get("output_text", row.get("response_text", "")))
    token_ids = row.get("output_token_ids", [])
    selected_positions = row.get("selected_event_positions", [])
    selected_token_ids = row.get("selected_token_ids", [])
    selected_events = row.get("selected_events", [])

    if not isinstance(token_ids, list):
        errors.append("output_token_ids must be a list")
        token_ids = []
    if not isinstance(selected_positions, list) or not isinstance(selected_token_ids, list):
        errors.append("selected_event_positions and selected_token_ids must be lists")
        selected_positions = []
        selected_token_ids = []
    if len(selected_positions) != len(selected_token_ids):
        errors.append("selected positions and token ids length mismatch")

    for field in REQUIRED_HASH_FIELDS:
        if not row.get(field):
            errors.append(f"missing required hash field: {field}")
    if row.get("key_id_not_secret_key") in ("", None):
        errors.append("missing key_id_not_secret_key")

    if row.get("output_text_sha256") != sha256_text(output_text):
        errors.append("output_text_sha256 mismatch")
    if row.get("output_token_ids_sha256") != sha256_json(token_ids):
        errors.append("output_token_ids_sha256 mismatch")

    for pos, token_id in zip(selected_positions, selected_token_ids):
        if not isinstance(pos, int):
            errors.append("event position is not integer")
            continue
        if pos < 0 or pos >= len(token_ids):
            errors.append("event position outside output token ids")
            continue
        if token_ids[pos] != token_id:
            errors.append("selected token id does not match output token id at event position")

    if isinstance(selected_events, list):
        expected_root = event_merkle_root([event for event in selected_events if isinstance(event, Mapping)])
        if row.get("event_trace_merkle_root") != expected_root:
            errors.append("event_trace_merkle_root mismatch")
    else:
        errors.append("selected_events must be a list")

    if row.get("wrong_key_replay_accept") is True:
        errors.append("wrong-key replay accepted")
    if row.get("wrong_payload_replay_accept") is True:
        errors.append("wrong-payload replay accepted")

    if hmac_secret is not None:
        expected = compute_binding_hmac(row, hmac_secret)
        if row.get("binding_hmac") != expected:
            errors.append("binding_hmac mismatch")

    return {
        "generation_id": row.get("generation_id", ""),
        "valid": not errors,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify R4 first-token event trace binding rows.")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hmac-secret", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    with args.jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    validations = [verify_trace_binding(row, hmac_secret=args.hmac_secret) for row in rows]
    invalid = [row for row in validations if not row["valid"]]
    summary = {
        "schema_name": "natural_evidence_v2_r4_first_token_event_trace_binding_validation_v1",
        "status": "PASS_R4_FIRST_TOKEN_EVENT_TRACE_BINDING" if not invalid else "FAIL_R4_FIRST_TOKEN_EVENT_TRACE_BINDING",
        "checked_rows": len(validations),
        "invalid_rows": len(invalid),
        "invalid_examples": invalid[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not invalid else 1


if __name__ == "__main__":
    raise SystemExit(main())

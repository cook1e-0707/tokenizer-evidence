from __future__ import annotations

import json
from itertools import combinations_with_replacement

from scripts.natural_evidence_v2.build_r4_positive_event_bank_precommit import (
    _DEV_AUDIT_KEY_MATERIAL,
    _DEV_WRONG_KEY_MATERIAL,
)
from scripts.natural_evidence_v2.decode_r4_positive_keyed_correlation import (
    DEFAULT_PRECOMMIT,
    _read_json,
    decode_generated_outputs,
)
from scripts.natural_evidence_v2.extract_r4_positive_phrase_events import load_surface_bank
from scripts.natural_evidence_v2.r4_keyed_correlation_decoder import decide_keyed_correlation


def _aligned_text() -> str:
    surface_bank = load_surface_bank(DEFAULT_PRECOMMIT / "surface_bank.json")
    codebook = _read_json(DEFAULT_PRECOMMIT / "codebook.json")
    mapping_rows = [
        json.loads(line)
        for line in (DEFAULT_PRECOMMIT / "coordinate_mapping.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row["surface_id"]: row for row in surface_bank}
    selected = []
    seen_coordinates = set()
    for row in mapping_rows:
        if row["polarity"] != 1 or row["coordinate_id"] in seen_coordinates:
            continue
        selected.append(by_id[row["surface_id"]]["surface_text"])
        seen_coordinates.add(row["coordinate_id"])
        if len(selected) >= 20:
            break
    assert len(selected) >= 20
    selected_surface_ids = [
        row["surface_id"]
        for row in mapping_rows
        if row["polarity"] == 1 and by_id[row["surface_id"]]["surface_text"] in selected
    ][:20]
    for repeat_indexes in combinations_with_replacement(range(len(selected_surface_ids)), 4):
        event_ids = selected_surface_ids + [selected_surface_ids[index] for index in repeat_indexes]
        decision = decide_keyed_correlation(
            [{"surface_id": surface_id, "weight": 1.0} for surface_id in event_ids],
            audit_key=_DEV_AUDIT_KEY_MATERIAL,
            payload_id=str(codebook["payload_id"]),
            wrong_audit_key=_DEV_WRONG_KEY_MATERIAL,
            wrong_payload_id=str(codebook["wrong_payload_id"]),
            coordinate_count=32,
            min_observed_events=24,
            min_distinct_coordinates=20,
            min_keyed_correlation_score=6.0,
            min_specificity_margin=3.0,
            max_wrong_score=1.5,
        )
        if decision.accept:
            selected.extend(by_id[selected_surface_ids[index]]["surface_text"] for index in repeat_indexes)
            break
    else:
        raise AssertionError("expected a static aligned fixture accepted by the precommitted decoder")
    return ". ".join(selected) + "."


def test_keyed_correlation_decode_accepts_aligned_protected_events() -> None:
    generated = [
        {
            "generation_condition": "protected",
            "generation_id": "g0",
            "prompt_id": "p0",
            "prompt_index": 0,
            "replicate_group_id": "shard_00",
            "response_text": _aligned_text(),
        }
    ]

    decode_rows, event_rows = decode_generated_outputs(
        generated_rows=generated,
        surface_bank=load_surface_bank(DEFAULT_PRECOMMIT / "surface_bank.json"),
        codebook=_read_json(DEFAULT_PRECOMMIT / "codebook.json"),
        decoder_spec=_read_json(DEFAULT_PRECOMMIT / "decoder_spec.json"),
        manifest=_read_json(DEFAULT_PRECOMMIT / "precommit_manifest.json"),
        audit_key=_DEV_AUDIT_KEY_MATERIAL,
        wrong_key=_DEV_WRONG_KEY_MATERIAL,
        prompts_per_block=64,
        scrub_mode="all",
        include_protected_controls=True,
    )

    by_arm = {row["arm"]: row for row in decode_rows}
    assert by_arm["protected"]["accept"] is True
    assert by_arm["protected"]["observed_events"] >= 24
    assert by_arm["protected"]["distinct_coordinates"] >= 20
    assert by_arm["wrong_key"]["accept"] is False
    assert by_arm["wrong_payload"]["accept"] is False
    assert len(event_rows) >= 24


def test_keyed_correlation_decode_rejects_raw_unaligned_support() -> None:
    generated = [
        {
            "generation_condition": "raw",
            "generation_id": "g0",
            "prompt_id": "p0",
            "prompt_index": 0,
            "replicate_group_id": "shard_00",
            "response_text": "Ask a focused question. Make a short plan. Use a calm tone.",
        }
    ]

    decode_rows, _ = decode_generated_outputs(
        generated_rows=generated,
        surface_bank=load_surface_bank(DEFAULT_PRECOMMIT / "surface_bank.json"),
        codebook=_read_json(DEFAULT_PRECOMMIT / "codebook.json"),
        decoder_spec=_read_json(DEFAULT_PRECOMMIT / "decoder_spec.json"),
        manifest=_read_json(DEFAULT_PRECOMMIT / "precommit_manifest.json"),
        audit_key=_DEV_AUDIT_KEY_MATERIAL,
        wrong_key=_DEV_WRONG_KEY_MATERIAL,
        prompts_per_block=64,
        scrub_mode="all",
        include_protected_controls=True,
    )

    assert decode_rows[0]["arm"] == "raw"
    assert decode_rows[0]["accept"] is False


def test_keyed_correlation_decode_requires_matching_key_commitment() -> None:
    generated = [
        {
            "generation_condition": "protected",
            "generation_id": "g0",
            "prompt_id": "p0",
            "prompt_index": 0,
            "replicate_group_id": "shard_00",
            "response_text": _aligned_text(),
        }
    ]

    try:
        decode_generated_outputs(
            generated_rows=generated,
            surface_bank=load_surface_bank(DEFAULT_PRECOMMIT / "surface_bank.json"),
            codebook=_read_json(DEFAULT_PRECOMMIT / "codebook.json"),
            decoder_spec=_read_json(DEFAULT_PRECOMMIT / "decoder_spec.json"),
            manifest=_read_json(DEFAULT_PRECOMMIT / "precommit_manifest.json"),
            audit_key="wrong material",
            wrong_key=_DEV_WRONG_KEY_MATERIAL,
            prompts_per_block=64,
            scrub_mode="all",
            include_protected_controls=True,
        )
    except ValueError as exc:
        assert "does not match precommit" in str(exc)
    else:
        raise AssertionError("expected key commitment mismatch")

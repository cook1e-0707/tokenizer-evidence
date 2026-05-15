from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.natural_evidence_v2.decode_r4_positive_support_window_correlation import decode_generated_outputs


PACKAGE_DIR = Path("results/natural_evidence_v2/precommit/r4_positive_selectivity_repair_package_20260514_2158")


def _write_generated(path: Path, response_text: str, condition: str = "protected") -> None:
    row = {
        "schema_name": "natural_evidence_v2_r4_generated_output_v1",
        "generation_condition": condition,
        "generation_id": f"{condition}_0",
        "prompt_id": "prompt_0",
        "prompt_index": 0,
        "replicate_group_id": "test",
        "response_text": response_text,
        "split": "dev",
    }
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def test_support_window_decode_accepts_toy_positive_and_rejects_controls(tmp_path: Path) -> None:
    fixture = json.loads((PACKAGE_DIR / "toy_positive_fixture.json").read_text(encoding="utf-8"))
    generated = tmp_path / "generated.jsonl"
    _write_generated(generated, fixture["response_text"])

    output_dir = tmp_path / "decode"
    summary = decode_generated_outputs(
        generated_outputs=generated,
        package_dir=PACKAGE_DIR,
        output_dir=output_dir,
        prompts_per_block=64,
        scrub_mode="all",
        include_protected_controls=True,
        allow_static_dev_keys=True,
    )

    assert summary["status"] == "PASS_R4_SUPPORT_WINDOW_DECODE_COMPLETED"
    assert summary["summary_by_arm"]["protected"]["accepts"] == 1
    assert summary["summary_by_arm"]["wrong_key"]["accepts"] == 0
    assert summary["summary_by_arm"]["wrong_payload"]["accepts"] == 0
    assert (output_dir / "decode_rows.jsonl").exists()
    assert (output_dir / "support_window_events.jsonl").exists()


def test_support_window_decode_requires_explicit_dev_key_ack(tmp_path: Path) -> None:
    fixture = json.loads((PACKAGE_DIR / "toy_positive_fixture.json").read_text(encoding="utf-8"))
    generated = tmp_path / "generated.jsonl"
    _write_generated(generated, fixture["response_text"])

    with pytest.raises(ValueError, match="allow-static-dev-keys"):
        decode_generated_outputs(
            generated_outputs=generated,
            package_dir=PACKAGE_DIR,
            output_dir=tmp_path / "decode",
            prompts_per_block=64,
            scrub_mode="all",
            include_protected_controls=True,
            allow_static_dev_keys=False,
        )

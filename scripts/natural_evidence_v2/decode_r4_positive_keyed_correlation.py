from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.build_r4_positive_event_bank_precommit import (  # noqa: E402
    _DEV_AUDIT_KEY_MATERIAL,
    _DEV_WRONG_KEY_MATERIAL,
)
from scripts.natural_evidence_v2.extract_r4_positive_phrase_events import (  # noqa: E402
    extract_phrase_events,
    load_surface_bank,
)
from scripts.natural_evidence_v2.r4_keyed_correlation_decoder import (  # noqa: E402
    decide_keyed_correlation,
)

DEFAULT_PRECOMMIT = ROOT / "results/natural_evidence_v2/precommit/r4_positive_event_bank_precommit_20260514_1605"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _material_from_args_or_env(args: argparse.Namespace) -> tuple[str, str]:
    audit_key = args.audit_key_material or os.environ.get(args.audit_key_env, "")
    wrong_key = args.wrong_audit_key_material or os.environ.get(args.wrong_audit_key_env, "")
    if args.allow_static_dev_keys:
        audit_key = audit_key or _DEV_AUDIT_KEY_MATERIAL
        wrong_key = wrong_key or _DEV_WRONG_KEY_MATERIAL
    if not audit_key or not wrong_key:
        raise ValueError("audit key material is required for decode; provide env vars or --allow-static-dev-keys")
    return audit_key, wrong_key


def _check_commitments(manifest: Mapping[str, Any], audit_key: str, wrong_key: str) -> None:
    audit_commitment = _sha256_text(audit_key)
    wrong_commitment = _sha256_text(wrong_key)
    if audit_commitment != manifest.get("audit_key_commitment"):
        raise ValueError("audit key material does not match precommit manifest commitment")
    if wrong_commitment != manifest.get("wrong_audit_key_commitment"):
        raise ValueError("wrong audit key material does not match precommit manifest commitment")


def block_id_for(row: Mapping[str, Any], prompts_per_block: int) -> str:
    shard = str(row.get("replicate_group_id", "local"))
    prompt_index = int(row.get("prompt_index", 0))
    return f"{shard}_block_{prompt_index // prompts_per_block:02d}"


def _decode_material_for_condition(
    *,
    condition: str,
    audit_key: str,
    wrong_key: str,
    payload_id: str,
    wrong_payload_id: str,
) -> tuple[str, str, str, str]:
    if condition == "wrong_key":
        return wrong_key, payload_id, audit_key, wrong_payload_id
    if condition == "wrong_payload":
        return audit_key, wrong_payload_id, wrong_key, payload_id
    return audit_key, payload_id, wrong_key, wrong_payload_id


def decode_block(
    *,
    block_id: str,
    condition: str,
    source_condition: str,
    rows: list[Mapping[str, Any]],
    surface_bank: list[Mapping[str, Any]],
    audit_key: str,
    wrong_key: str,
    payload_id: str,
    wrong_payload_id: str,
    decoder_spec: Mapping[str, Any],
    scrub_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    forbidden_hits: Counter[str] = Counter()
    for row in rows:
        text = str(row.get("response_text", ""))
        for literal in ("fingerprint", "watermark", "payload", "secret key", "decoder", "hidden signal", "evidence block", "bucket", "coordinate"):
            if literal in text.lower():
                forbidden_hits[literal] += 1
        extracted = extract_phrase_events(text, surface_bank, scrub_mode=scrub_mode)
        for event in extracted:
            events.append(
                {
                    **event,
                    "block_id": block_id,
                    "decode_condition": condition,
                    "source_condition": source_condition,
                    "generation_id": str(row.get("generation_id", "")),
                    "prompt_id": str(row.get("prompt_id", "")),
                }
            )
    accept_requires = decoder_spec["accept_requires"]
    required_before_accept = decoder_spec["required_before_accept"]
    test_key, test_payload, wrong_key_for_test, wrong_payload_for_test = _decode_material_for_condition(
        condition=condition,
        audit_key=audit_key,
        wrong_key=wrong_key,
        payload_id=payload_id,
        wrong_payload_id=wrong_payload_id,
    )
    decision = decide_keyed_correlation(
        events,
        audit_key=test_key,
        payload_id=test_payload,
        wrong_audit_key=wrong_key_for_test,
        wrong_payload_id=wrong_payload_for_test,
        coordinate_count=int(decoder_spec["coordinate_count"]),
        min_observed_events=int(accept_requires["min_observed_events"]),
        min_distinct_coordinates=int(accept_requires["min_distinct_coordinates"]),
        min_keyed_correlation_score=float(required_before_accept["protected_keyed_correlation_score_min"]),
        min_specificity_margin=float(required_before_accept["protected_minus_best_wrong_specificity_margin_min"]),
        max_wrong_score=float(required_before_accept["wrong_key_correlation_score_max"]),
    )
    accept = bool(decision.accept and not forbidden_hits)
    row = {
        "schema_name": "natural_evidence_v2_r4_positive_keyed_correlation_decode_row_v1",
        "block_id": block_id,
        "arm": condition,
        "source_condition": source_condition,
        "accept": accept,
        "keyed_correlation_score": decision.keyed_correlation_score,
        "wrong_key_correlation_score": decision.wrong_key_correlation_score,
        "wrong_payload_correlation_score": decision.wrong_payload_correlation_score,
        "specificity_margin": decision.specificity_margin,
        "observed_events": decision.observed_events,
        "distinct_coordinates": decision.distinct_coordinates,
        "format_scrub_mode": scrub_mode,
        "forbidden_public_surface_count": sum(forbidden_hits.values()),
        "forbidden_public_surfaces": dict(forbidden_hits),
        "generation_started": False,
        "paper_claim_allowed": False,
    }
    return row, events


def decode_generated_outputs(
    *,
    generated_rows: list[Mapping[str, Any]],
    surface_bank: list[Mapping[str, Any]],
    codebook: Mapping[str, Any],
    decoder_spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    audit_key: str,
    wrong_key: str,
    prompts_per_block: int,
    scrub_mode: str,
    include_protected_controls: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _check_commitments(manifest, audit_key, wrong_key)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in generated_rows:
        condition = str(row.get("generation_condition", "unknown"))
        grouped[(condition, block_id_for(row, prompts_per_block))].append(row)
    decode_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for (condition, block_id), rows in sorted(grouped.items()):
        decoded, events = decode_block(
            block_id=block_id,
            condition=condition,
            source_condition=condition,
            rows=rows,
            surface_bank=surface_bank,
            audit_key=audit_key,
            wrong_key=wrong_key,
            payload_id=str(codebook["payload_id"]),
            wrong_payload_id=str(codebook["wrong_payload_id"]),
            decoder_spec=decoder_spec,
            scrub_mode=scrub_mode,
        )
        decode_rows.append(decoded)
        event_rows.extend(events)
        if include_protected_controls and condition == "protected":
            for control in ("wrong_key", "wrong_payload"):
                control_decoded, control_events = decode_block(
                    block_id=block_id,
                    condition=control,
                    source_condition=condition,
                    rows=rows,
                    surface_bank=surface_bank,
                    audit_key=audit_key,
                    wrong_key=wrong_key,
                    payload_id=str(codebook["payload_id"]),
                    wrong_payload_id=str(codebook["wrong_payload_id"]),
                    decoder_spec=decoder_spec,
                    scrub_mode=scrub_mode,
                )
                decode_rows.append(control_decoded)
                event_rows.extend(control_events)
    return decode_rows, event_rows


def summarize_decode(
    *,
    decode_rows: list[Mapping[str, Any]],
    generated_outputs: Path,
    surface_bank: Path,
    codebook: Path,
    decoder_spec: Path,
    manifest: Path,
    scrub_mode: str,
) -> dict[str, Any]:
    by_arm: dict[str, Counter[str]] = defaultdict(Counter)
    for row in decode_rows:
        arm = str(row["arm"])
        by_arm[arm]["blocks"] += 1
        by_arm[arm]["accepts"] += int(bool(row["accept"]))
        by_arm[arm]["forbidden_public_surface_count"] += int(row["forbidden_public_surface_count"])
    return {
        "schema_name": "natural_evidence_v2_r4_positive_keyed_correlation_decode_summary_v1",
        "generated_outputs": str(generated_outputs),
        "surface_bank": str(surface_bank),
        "codebook": str(codebook),
        "decoder_spec": str(decoder_spec),
        "precommit_manifest": str(manifest),
        "format_scrub_mode": scrub_mode,
        "summary_by_arm": {arm: dict(counter) for arm, counter in sorted(by_arm.items())},
        "generation_started": False,
        "slurm_submitted": False,
        "paper_claim_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode R4 positive event-bank outputs with keyed correlation.")
    parser.add_argument("--generated-outputs", type=Path, required=True)
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_PRECOMMIT / "surface_bank.json")
    parser.add_argument("--codebook", type=Path, default=DEFAULT_PRECOMMIT / "codebook.json")
    parser.add_argument("--decoder-spec", type=Path, default=DEFAULT_PRECOMMIT / "decoder_spec.json")
    parser.add_argument("--precommit-manifest", type=Path, default=DEFAULT_PRECOMMIT / "precommit_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format-scrub", choices=["all", "none"], default="all")
    parser.add_argument("--prompts-per-block", type=int, default=64)
    parser.add_argument("--include-protected-controls", action="store_true")
    parser.add_argument("--audit-key-env", default="R4_POSITIVE_AUDIT_KEY_MATERIAL")
    parser.add_argument("--wrong-audit-key-env", default="R4_POSITIVE_WRONG_AUDIT_KEY_MATERIAL")
    parser.add_argument("--audit-key-material", default="")
    parser.add_argument("--wrong-audit-key-material", default="")
    parser.add_argument("--allow-static-dev-keys", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    generated_path = _resolve(args.generated_outputs)
    surface_path = _resolve(args.surface_bank)
    codebook_path = _resolve(args.codebook)
    decoder_path = _resolve(args.decoder_spec)
    manifest_path = _resolve(args.precommit_manifest)
    audit_key, wrong_key = _material_from_args_or_env(args)
    generated_rows = _read_jsonl(generated_path)
    surface_bank = load_surface_bank(surface_path)
    codebook = _read_json(codebook_path)
    decoder_spec = _read_json(decoder_path)
    manifest = _read_json(manifest_path)
    decode_rows, event_rows = decode_generated_outputs(
        generated_rows=generated_rows,
        surface_bank=surface_bank,
        codebook=codebook,
        decoder_spec=decoder_spec,
        manifest=manifest,
        audit_key=audit_key,
        wrong_key=wrong_key,
        prompts_per_block=int(args.prompts_per_block),
        scrub_mode=str(args.format_scrub),
        include_protected_controls=bool(args.include_protected_controls),
    )
    _write_jsonl(output_dir / "decode_rows.jsonl", decode_rows)
    _write_jsonl(output_dir / "extracted_events.jsonl", event_rows)
    _write_json(
        output_dir / "decode_summary.json",
        summarize_decode(
            decode_rows=decode_rows,
            generated_outputs=generated_path,
            surface_bank=surface_path,
            codebook=codebook_path,
            decoder_spec=decoder_path,
            manifest=manifest_path,
            scrub_mode=str(args.format_scrub),
        ),
    )
    _write_csv(
        output_dir / "per_block_decode.csv",
        decode_rows,
        [
            "block_id",
            "arm",
            "source_condition",
            "accept",
            "keyed_correlation_score",
            "wrong_key_correlation_score",
            "wrong_payload_correlation_score",
            "specificity_margin",
            "observed_events",
            "distinct_coordinates",
            "format_scrub_mode",
            "forbidden_public_surface_count",
        ],
    )
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "decode_rows": len(decode_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

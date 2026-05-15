from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.build_r4_positive_event_bank_precommit import (  # noqa: E402
    _DEV_AUDIT_KEY_MATERIAL,
    _DEV_WRONG_KEY_MATERIAL,
)
from scripts.natural_evidence_v2.extract_r4_positive_support_window_events import (  # noqa: E402
    extract_support_window_events,
    load_event_window_bank,
)
from scripts.natural_evidence_v2.r4_keyed_correlation_decoder import decide_keyed_correlation  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(payload)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def block_id_for(row: Mapping[str, Any], prompts_per_block: int) -> str:
    shard = str(row.get("replicate_group_id", "local"))
    prompt_index = int(row.get("prompt_index", 0))
    return f"{shard}_block_{prompt_index // prompts_per_block:02d}"


def _decision(
    events: list[Mapping[str, Any]],
    *,
    arm: str,
    contract: Mapping[str, Any],
    decoder_spec: Mapping[str, Any],
) -> Any:
    payload_id = str(contract["payload_id"])
    wrong_payload_id = "wrong_payload_a55e_control"
    audit_key = _DEV_AUDIT_KEY_MATERIAL
    wrong_key = _DEV_WRONG_KEY_MATERIAL
    if arm == "wrong_key":
        audit_key, wrong_key = wrong_key, audit_key
    if arm == "wrong_payload":
        payload_id, wrong_payload_id = wrong_payload_id, payload_id
    accept_requires = decoder_spec["accept_requires"]
    required = decoder_spec["required_before_accept"]
    return decide_keyed_correlation(
        events,
        audit_key=audit_key,
        payload_id=payload_id,
        wrong_audit_key=wrong_key,
        wrong_payload_id=wrong_payload_id,
        coordinate_count=int(contract["coordinate_count"]),
        min_observed_events=int(accept_requires["min_observed_events"]),
        min_distinct_coordinates=int(accept_requires["min_distinct_coordinates"]),
        min_keyed_correlation_score=float(required["protected_keyed_correlation_score_min"]),
        min_specificity_margin=float(required["protected_minus_best_wrong_specificity_margin_min"]),
        max_wrong_score=float(required["wrong_key_correlation_score_max"]),
    )


def decode_generated_outputs(
    *,
    generated_outputs: Path,
    package_dir: Path,
    output_dir: Path,
    prompts_per_block: int,
    scrub_mode: str,
    include_protected_controls: bool,
    allow_static_dev_keys: bool,
) -> dict[str, Any]:
    if not allow_static_dev_keys:
        raise ValueError("R4 support-window dev decoder requires --allow-static-dev-keys")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    rows = read_jsonl(generated_outputs)
    bank = load_event_window_bank(package_dir / "event_window_bank.json")
    contract = read_json(package_dir / "contract.json")
    decoder_spec = read_json(package_dir / "decoder_spec.json")
    max_events_per_segment = int(read_json(package_dir / "extractor_spec.json").get("max_events_per_segment", 3))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    extracted_rows: list[dict[str, Any]] = []
    for row in rows:
        condition = str(row.get("generation_condition", "unknown"))
        block_id = block_id_for(row, prompts_per_block)
        events = extract_support_window_events(
            str(row.get("response_text", "")),
            bank,
            scrub_mode=scrub_mode,
            max_events_per_segment=max_events_per_segment,
        )
        for event in events:
            payload = {
                **event,
                "block_id": block_id,
                "source_condition": condition,
                "generation_id": str(row.get("generation_id", "")),
                "prompt_id": str(row.get("prompt_id", "")),
            }
            grouped[(condition, block_id)].append(payload)
            extracted_rows.append(payload)

    decode_rows: list[dict[str, Any]] = []
    block_ids_by_condition: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        block_ids_by_condition[str(row.get("generation_condition", "unknown"))].add(block_id_for(row, prompts_per_block))
    for condition in ("protected", "raw", "task_only"):
        for block_id in sorted(block_ids_by_condition.get(condition, set())):
            events = grouped.get((condition, block_id), [])
            arms = [condition]
            if condition == "protected" and include_protected_controls:
                arms.extend(["wrong_key", "wrong_payload"])
            for arm in arms:
                decision = _decision(events, arm=arm, contract=contract, decoder_spec=decoder_spec)
                decode_rows.append(
                    {
                        "schema_name": "natural_evidence_v2_r4_support_window_decode_row_v1",
                        "block_id": block_id,
                        "arm": arm,
                        "source_condition": condition,
                        "accept": decision.accept,
                        "observed_events": decision.observed_events,
                        "distinct_coordinates": decision.distinct_coordinates,
                        "keyed_correlation_score": decision.keyed_correlation_score,
                        "wrong_key_correlation_score": decision.wrong_key_correlation_score,
                        "wrong_payload_correlation_score": decision.wrong_payload_correlation_score,
                        "specificity_margin": decision.specificity_margin,
                        "format_scrub_mode": scrub_mode,
                    }
                )

    summary_by_arm: dict[str, dict[str, Any]] = {}
    for row in decode_rows:
        arm = str(row["arm"])
        bucket = summary_by_arm.setdefault(
            arm,
            {
                "blocks": 0,
                "accepts": 0,
                "observed_events_total": 0,
                "distinct_coordinates_total": 0,
            },
        )
        bucket["blocks"] += 1
        bucket["accepts"] += 1 if row["accept"] else 0
        bucket["observed_events_total"] += int(row["observed_events"])
        bucket["distinct_coordinates_total"] += int(row["distinct_coordinates"])
    for bucket in summary_by_arm.values():
        blocks = max(1, int(bucket["blocks"]))
        bucket["observed_events_mean"] = bucket["observed_events_total"] / blocks
        bucket["distinct_coordinates_mean"] = bucket["distinct_coordinates_total"] / blocks

    summary = {
        "schema_name": "natural_evidence_v2_r4_support_window_decode_summary_v1",
        "status": "PASS_R4_SUPPORT_WINDOW_DECODE_COMPLETED",
        "generated_outputs": str(generated_outputs),
        "package_dir": str(package_dir),
        "generated_rows": len(rows),
        "extracted_events": len(extracted_rows),
        "format_scrub_mode": scrub_mode,
        "summary_by_arm": summary_by_arm,
        "allow_static_dev_keys": allow_static_dev_keys,
        "generation_started_by_decoder": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "decode_summary.json", summary)
    write_jsonl(output_dir / "support_window_events.jsonl", extracted_rows)
    write_jsonl(output_dir / "decode_rows.jsonl", decode_rows)
    write_csv(
        output_dir / "decode_rows.csv",
        decode_rows,
        [
            "block_id",
            "arm",
            "source_condition",
            "accept",
            "observed_events",
            "distinct_coordinates",
            "keyed_correlation_score",
            "wrong_key_correlation_score",
            "wrong_payload_correlation_score",
            "specificity_margin",
            "format_scrub_mode",
        ],
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode R4 support-window keyed correlation from generated outputs.")
    parser.add_argument("--generated-outputs", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts-per-block", type=int, default=64)
    parser.add_argument("--format-scrub", choices=["all", "none"], default="all")
    parser.add_argument("--include-protected-controls", action="store_true")
    parser.add_argument("--allow-static-dev-keys", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = decode_generated_outputs(
        generated_outputs=args.generated_outputs if args.generated_outputs.is_absolute() else ROOT / args.generated_outputs,
        package_dir=args.package_dir if args.package_dir.is_absolute() else ROOT / args.package_dir,
        output_dir=args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir,
        prompts_per_block=int(args.prompts_per_block),
        scrub_mode=str(args.format_scrub),
        include_protected_controls=bool(args.include_protected_controls),
        allow_static_dev_keys=bool(args.allow_static_dev_keys),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

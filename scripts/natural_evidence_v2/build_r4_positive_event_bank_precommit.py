from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_keyed_correlation_decoder import (  # noqa: E402
    map_surface_to_coordinate_and_polarity,
)
from scripts.natural_evidence_v2.validate_r4_positive_evidence_contract import (  # noqa: E402
    load_yaml,
    validate_contract,
)

DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_positive_event_bank_precommit.yaml"

# Static dev-only material for deterministic artifact construction. Only hashes and
# derived mappings are written; raw key material must never appear in output artifacts.
_DEV_AUDIT_KEY_MATERIAL = "r4_positive_event_bank_dev_mapping_key_static_v1"
_DEV_WRONG_KEY_MATERIAL = "r4_positive_event_bank_wrong_dev_mapping_key_static_v1"

SURFACE_FAMILIES: dict[str, list[str]] = {
    "clarify": [
        "ask a focused question",
        "confirm the main constraint",
        "check the important details",
        "review the current notes",
        "compare the available options",
        "identify the likely risk",
        "write down the open issue",
        "summarize the main tradeoff",
        "verify the shared goal",
        "collect the missing context",
        "name the first concern",
        "look for common mistakes",
    ],
    "plan": [
        "make a short plan",
        "choose a clear priority",
        "set a realistic target",
        "prepare a simple outline",
        "map the required handoff",
        "schedule a brief review",
        "define the next milestone",
        "keep the scope narrow",
        "split the work carefully",
        "reserve time for review",
        "start with the simplest case",
        "leave room for adjustment",
    ],
    "communicate": [
        "share a concise update",
        "use a calm tone",
        "explain the reason clearly",
        "invite a quick response",
        "keep the message brief",
        "confirm who owns the task",
        "mention the expected timing",
        "avoid unnecessary detail",
        "state the decision plainly",
        "ask for one concrete reply",
        "document the agreed change",
        "flag the remaining uncertainty",
    ],
    "quality": [
        "test the simple path",
        "review the final wording",
        "check for missing details",
        "compare against the goal",
        "look for edge cases",
        "verify the basic assumption",
        "inspect the recent change",
        "confirm the outcome",
        "measure the visible impact",
        "review the handoff notes",
        "trace the likely failure",
        "mark the unresolved risk",
    ],
    "safety": [
        "remove the obvious hazard",
        "keep the area clear",
        "pause before changing course",
        "check the warning signs",
        "confirm the safe option",
        "use the safer default",
        "keep a backup choice",
        "avoid rushing the decision",
        "watch for early trouble",
        "protect the most fragile part",
        "check that access is clear",
        "choose the lower risk option",
    ],
    "maintenance": [
        "clean the affected area",
        "tighten the loose part",
        "replace the worn item",
        "inspect the nearby surface",
        "record the maintenance date",
        "keep spare parts nearby",
        "test after the repair",
        "check the simple fix first",
        "listen for unusual noise",
        "remove the old residue",
        "confirm the item works",
        "store the tool safely",
    ],
    "troubleshooting": [
        "reproduce the issue once",
        "change one variable",
        "check the recent change",
        "restart the simple component",
        "compare with a known case",
        "save the error message",
        "isolate the failing part",
        "try the lowest risk fix",
        "restore the last working state",
        "note what changed recently",
        "check the connection first",
        "document the tested fix",
    ],
    "learning": [
        "practice the core skill",
        "review one example",
        "explain the idea aloud",
        "write a short summary",
        "ask for targeted feedback",
        "compare two approaches",
        "repeat the difficult part",
        "use a smaller example",
        "track one improvement",
        "focus on the weak area",
        "connect it to prior work",
        "review the result later",
    ],
}


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    data = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256_bytes(data)


def _write_text(path: Path, payload: str) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_sha(path: Path, digest: str, target_name: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.write_text(f"{digest}  {target_name}\n", encoding="utf-8")


def normalize_surface(text: str) -> str:
    lowered = text.lower()
    stripped = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


def _contains_forbidden_phrase(text: str, forbidden: Iterable[str]) -> str | None:
    normalized = normalize_surface(text)
    for phrase in forbidden:
        forbidden_norm = normalize_surface(str(phrase))
        if not forbidden_norm:
            continue
        if re.search(rf"(^|\s){re.escape(forbidden_norm)}($|\s)", normalized):
            return str(phrase)
    return None


def build_surface_bank(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy = config["surface_bank_policy"]
    forbidden = policy["forbidden_public_technical_literals"]
    source_rule_id = str(policy["source_rule_id"])
    normalization_rule = str(policy["normalization_rule"])
    event_type = str(policy["event_type"])

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for family, phrases in SURFACE_FAMILIES.items():
        for index, phrase in enumerate(phrases):
            canonical = normalize_surface(phrase)
            if canonical in seen:
                raise ValueError(f"duplicate canonical surface: {canonical}")
            seen.add(canonical)
            forbidden_match = _contains_forbidden_phrase(canonical, forbidden)
            if forbidden_match:
                raise ValueError(f"surface contains forbidden literal {forbidden_match!r}: {phrase}")
            rows.append(
                {
                    "surface_id": f"r4pe_{family}_{index:02d}",
                    "surface_text": phrase,
                    "canonical_phrase": canonical,
                    "surface_family": family,
                    "allowed_topic_domains": [
                        "practical_advice",
                        "task_explanation",
                        "planning_guidance",
                        "troubleshooting_guidance",
                    ],
                    "forbidden_contexts": [
                        "technical_watermark_discussion",
                        "protocol_or_verifier_description",
                        "fixed_label_or_numbered_slot_instruction",
                    ],
                    "normalization_rule": normalization_rule,
                    "source_rule_id": source_rule_id,
                    "naturalness_rationale": "Ordinary task-useful phrase that can appear in user-facing advice without structural labels.",
                    "event_type": event_type,
                    "structural_marker": False,
                    "technical_literal": False,
                }
            )
    return rows


def build_coordinate_mapping(config: Mapping[str, Any], surface_bank: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payload_id = str(config["payload_id"])
    coordinate_count = int(config["coordinate_count"])
    rows: list[dict[str, Any]] = []
    for surface in surface_bank:
        coordinate, polarity = map_surface_to_coordinate_and_polarity(
            audit_key=_DEV_AUDIT_KEY_MATERIAL,
            payload_id=payload_id,
            surface_id=str(surface["surface_id"]),
            coordinate_count=coordinate_count,
        )
        wrong_key_coordinate, wrong_key_polarity = map_surface_to_coordinate_and_polarity(
            audit_key=_DEV_WRONG_KEY_MATERIAL,
            payload_id=payload_id,
            surface_id=str(surface["surface_id"]),
            coordinate_count=coordinate_count,
        )
        wrong_payload_coordinate, wrong_payload_polarity = map_surface_to_coordinate_and_polarity(
            audit_key=_DEV_AUDIT_KEY_MATERIAL,
            payload_id=str(config["wrong_payload_id"]),
            surface_id=str(surface["surface_id"]),
            coordinate_count=coordinate_count,
        )
        rows.append(
            {
                "surface_id": surface["surface_id"],
                "surface_family": surface["surface_family"],
                "coordinate_id": coordinate,
                "polarity": polarity,
                "wrong_key_coordinate_id": wrong_key_coordinate,
                "wrong_key_polarity": wrong_key_polarity,
                "wrong_payload_coordinate_id": wrong_payload_coordinate,
                "wrong_payload_polarity": wrong_payload_polarity,
            }
        )
    return rows


def validate_package(
    *,
    config: Mapping[str, Any],
    contract_summary: Mapping[str, Any],
    surface_bank: list[Mapping[str, Any]],
    coordinate_mapping: list[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    policy = config["surface_bank_policy"]
    families = Counter(str(row["surface_family"]) for row in surface_bank)
    max_family_fraction = max(families.values()) / max(len(surface_bank), 1)
    coordinates = {int(row["coordinate_id"]) for row in coordinate_mapping}
    positive_events = [row for row in coordinate_mapping if int(row["polarity"]) == 1]
    forbidden = policy["forbidden_public_technical_literals"]

    if contract_summary["status"] != "PASS_R4_POSITIVE_EVIDENCE_CONTRACT_STATIC_VALIDATION_NO_COMPUTE":
        errors.append("contract static validation did not pass")
    if len(surface_bank) < int(policy["min_surface_count"]):
        errors.append("surface count below configured minimum")
    if len(families) < int(policy["min_distinct_surface_families"]):
        errors.append("distinct surface families below configured minimum")
    if max_family_fraction > float(policy["max_single_surface_family_fraction"]):
        errors.append("single surface family fraction exceeds cap")
    if len(coordinates) < int(policy["min_distinct_coordinates"]):
        errors.append("HMAC mapping distinct coordinate coverage below configured minimum")
    if len(positive_events) < int(policy["min_positive_polarity_events"]):
        errors.append("positive polarity event count below configured minimum")
    if len({row["surface_id"] for row in surface_bank}) != len(surface_bank):
        errors.append("surface_id values are not unique")
    if len({row["canonical_phrase"] for row in surface_bank}) != len(surface_bank):
        errors.append("canonical phrases are not unique")
    for row in surface_bank:
        forbidden_match = _contains_forbidden_phrase(str(row["canonical_phrase"]), forbidden)
        if forbidden_match:
            errors.append(f"forbidden literal in surface {row['surface_id']}: {forbidden_match}")
        if row.get("structural_marker") is not False:
            errors.append(f"structural marker surface is not allowed: {row['surface_id']}")
        if row.get("technical_literal") is not False:
            errors.append(f"technical literal surface is not allowed: {row['surface_id']}")

    return {
        "status": "PASS_R4_POSITIVE_EVENT_BANK_PRECOMMIT_PACKAGE" if not errors else "FAIL_R4_POSITIVE_EVENT_BANK_PRECOMMIT_PACKAGE",
        "errors": errors,
        "surface_count": len(surface_bank),
        "surface_family_count": len(families),
        "max_surface_family_fraction": max_family_fraction,
        "distinct_coordinate_count": len(coordinates),
        "positive_polarity_event_count": len(positive_events),
        "artifact_only": True,
        "generation_started": False,
        "training_started": False,
        "model_scoring_started": False,
        "slurm_started": False,
    }


def build_package(config_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("schema_name") != "natural_evidence_v2_r4_positive_event_bank_precommit_v1":
        raise ValueError("unexpected precommit config schema_name")
    contract_path = ROOT / str(config["contract_config"])
    contract = load_yaml(contract_path)
    contract_summary = validate_contract(contract)
    surface_bank = build_surface_bank(config)
    coordinate_mapping = build_coordinate_mapping(config, surface_bank)
    validation = validate_package(
        config=config,
        contract_summary=contract_summary,
        surface_bank=surface_bank,
        coordinate_mapping=coordinate_mapping,
    )
    if validation["status"].startswith("FAIL"):
        return validation

    output = output_dir or ROOT / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    decoder_spec = {
        "decoder_id": contract["decoder_policy"]["decoder_id"],
        "primary_scrub_mode": contract["decoder_policy"]["primary_scrub_mode"],
        "event_type": contract["carrier_event_policy"]["event_type"],
        "mapping": contract["key_payload_specificity_policy"]["mapping"],
        "coordinate_count": config["coordinate_count"],
        "accept_requires": contract["decoder_policy"]["accept_requires"],
        "required_before_accept": contract["key_payload_specificity_policy"]["required_before_accept"],
        "ignored_event_types": contract["carrier_event_policy"]["structural_features_excluded_from_votes"],
        "support_is_not_acceptance": True,
        "threshold_changes_after_generation_allowed": False,
    }
    codebook = {
        "contract_id": config["contract_id"],
        "event_bank_id": config["event_bank_id"],
        "payload_id": config["payload_id"],
        "wrong_payload_id": config["wrong_payload_id"],
        "payload_contract_scope": "same_contract_a55e_only_no_payload_diversity",
        "payload_checksum_sha256_12": hashlib.sha256(str(config["payload_id"]).encode("utf-8")).hexdigest()[:12],
        "coordinate_count": config["coordinate_count"],
        "mapping_key_policy": "audit_key_committed_before_generation_no_key_material_in_artifacts",
        "mapping_output": "surface_id_to_coordinate_and_polarity",
    }
    dev_gate = contract["pre_registered_dev_gate"]

    surface_sha = _write_json(output / "surface_bank.json", surface_bank)
    mapping_lines = "".join(json.dumps(row, sort_keys=True) + "\n" for row in coordinate_mapping)
    mapping_sha = _write_text(output / "coordinate_mapping.jsonl", mapping_lines)
    codebook_sha = _write_json(output / "codebook.json", codebook)
    decoder_sha = _write_json(output / "decoder_spec.json", decoder_spec)
    dev_gate_sha = _write_json(output / "dev_gate.json", dev_gate)
    contract_config_sha = _sha256_file(contract_path)

    for filename, digest in (
        ("surface_bank.json.sha256", surface_sha),
        ("coordinate_mapping.jsonl.sha256", mapping_sha),
        ("codebook.json.sha256", codebook_sha),
        ("decoder_spec.json.sha256", decoder_sha),
        ("dev_gate.json.sha256", dev_gate_sha),
    ):
        _write_sha(output / filename, digest, filename.removesuffix(".sha256"))

    audit_key_commitment = hashlib.sha256(_DEV_AUDIT_KEY_MATERIAL.encode("utf-8")).hexdigest()
    wrong_key_commitment = hashlib.sha256(_DEV_WRONG_KEY_MATERIAL.encode("utf-8")).hexdigest()
    precommit_material = {
        "protocol_id": config["contract_id"],
        "event_bank_id": config["event_bank_id"],
        "audit_key_id": config["audit_key_id"],
        "audit_key_commitment": audit_key_commitment,
        "wrong_audit_key_id": config["wrong_audit_key_id"],
        "wrong_audit_key_commitment": wrong_key_commitment,
        "payload_id": config["payload_id"],
        "wrong_payload_id": config["wrong_payload_id"],
        "contract_config_sha256": contract_config_sha,
        "surface_bank_sha256": surface_sha,
        "coordinate_mapping_sha256": mapping_sha,
        "codebook_sha256": codebook_sha,
        "decoder_spec_sha256": decoder_sha,
        "dev_gate_sha256": dev_gate_sha,
    }
    precommit_hash = _sha256_bytes(_json_bytes(precommit_material))
    manifest = {
        **precommit_material,
        "precommit_hash": precommit_hash,
        "artifact_only": True,
        "slurm_allowed": False,
        "generation_allowed": False,
        "training_allowed": False,
        "model_scoring_allowed": False,
        "paper_claim_allowed": False,
        "key_material_exposed": False,
        "surface_source_policy": config["surface_bank_policy"]["source_rule_id"],
    }
    manifest_sha = _write_json(output / "precommit_manifest.json", manifest)
    _write_sha(output / "precommit_manifest.json.sha256", manifest_sha, "precommit_manifest.json")

    summary = {
        **validation,
        "contract_id": config["contract_id"],
        "event_bank_id": config["event_bank_id"],
        "payload_id": config["payload_id"],
        "precommit_hash": precommit_hash,
        "output_dir": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output),
        "contract_config_sha256": contract_config_sha,
        "surface_bank_sha256": surface_sha,
        "coordinate_mapping_sha256": mapping_sha,
        "codebook_sha256": codebook_sha,
        "decoder_spec_sha256": decoder_sha,
        "dev_gate_sha256": dev_gate_sha,
        "precommit_manifest_sha256": manifest_sha,
        "key_material_exposed": False,
    }
    _write_json(output / "package_summary.json", summary)
    report = (
        "# R4 Positive Event-Bank Precommit Package\n\n"
        f"- status: `{summary['status']}`\n"
        f"- contract id: `{summary['contract_id']}`\n"
        f"- event bank id: `{summary['event_bank_id']}`\n"
        f"- payload id: `{summary['payload_id']}`\n"
        f"- surface count: `{summary['surface_count']}`\n"
        f"- surface families: `{summary['surface_family_count']}`\n"
        f"- max family fraction: `{summary['max_surface_family_fraction']:.3f}`\n"
        f"- distinct coordinates: `{summary['distinct_coordinate_count']}`\n"
        f"- positive polarity events: `{summary['positive_polarity_event_count']}`\n"
        f"- precommit hash: `{summary['precommit_hash']}`\n\n"
        "This package is artifact-only. It does not unlock Slurm, generation, "
        "training, tokenizer/model scoring, Llama, null/FAR, sanitizer, payload "
        "diversity, or paper-facing claims.\n"
    )
    _write_text(output / "package_report.md", report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the R4 positive event-bank precommit package.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_package(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

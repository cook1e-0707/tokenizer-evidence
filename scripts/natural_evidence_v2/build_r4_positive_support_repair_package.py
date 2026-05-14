from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.build_r4_positive_event_bank_precommit import (  # noqa: E402
    _DEV_AUDIT_KEY_MATERIAL,
    _DEV_WRONG_KEY_MATERIAL,
)
from scripts.natural_evidence_v2.extract_r4_positive_support_window_events import (  # noqa: E402
    extract_support_window_events,
)
from scripts.natural_evidence_v2.r4_keyed_correlation_decoder import (  # noqa: E402
    decide_keyed_correlation,
    map_surface_to_coordinate_and_polarity,
)

DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_positive_support_repair_package.yaml"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results/natural_evidence_v2/precommit/r4_positive_support_repair_package_20260514_2115"
)

EVENT_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "clarify": {
        "verbs": ["ask", "confirm", "check", "review", "compare", "identify"],
        "cues": ["question", "constraint", "detail", "option", "risk", "issue", "context", "goal"],
    },
    "plan": {
        "verbs": ["plan", "choose", "set", "prepare", "schedule", "define"],
        "cues": ["plan", "priority", "target", "outline", "handoff", "review", "milestone", "scope"],
    },
    "communicate": {
        "verbs": ["share", "explain", "invite", "mention", "document", "flag"],
        "cues": ["update", "tone", "response", "timing", "decision", "message", "change", "uncertainty"],
    },
    "quality": {
        "verbs": ["test", "inspect", "verify", "measure", "trace", "mark"],
        "cues": ["path", "wording", "goal", "assumption", "impact", "failure", "risk", "handoff"],
    },
    "safety": {
        "verbs": ["remove", "pause", "watch", "protect", "avoid", "choose"],
        "cues": ["hazard", "area", "option", "trouble", "part", "risk", "default", "access"],
    },
    "maintenance": {
        "verbs": ["clean", "tighten", "replace", "inspect", "store", "test"],
        "cues": ["area", "part", "item", "surface", "date", "repair", "tool", "noise"],
    },
    "troubleshooting": {
        "verbs": ["reproduce", "change", "restart", "isolate", "restore", "check"],
        "cues": ["issue", "variable", "component", "case", "message", "part", "fix", "connection"],
    },
    "learning": {
        "verbs": ["practice", "review", "explain", "write", "repeat", "track"],
        "cues": ["skill", "example", "idea", "summary", "feedback", "approach", "improvement", "area"],
    },
}


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    payload = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_sha(path: Path, digest: str, target_name: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.write_text(f"{digest}  {target_name}\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _contains_forbidden(text: str, forbidden: list[str]) -> str | None:
    normalized = text.lower()
    for term in forbidden:
        if term.lower() in normalized:
            return term
    return None


def build_event_window_bank(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy = config["event_window_policy"]
    forbidden = list(policy["forbidden_public_technical_literals"])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for family, taxonomy in EVENT_TAXONOMY.items():
        for verb_index, verb in enumerate(taxonomy["verbs"]):
            for cue_index, cue in enumerate(taxonomy["cues"]):
                surface_id = f"r4sr_{family}_{verb_index:02d}_{cue_index:02d}"
                canonical = f"{verb} <within_{policy['max_token_distance']}_tokens> {cue}"
                if surface_id in seen:
                    raise ValueError(f"duplicate support surface_id: {surface_id}")
                seen.add(surface_id)
                forbidden_match = _contains_forbidden(f"{verb} {cue} {canonical}", forbidden)
                if forbidden_match:
                    raise ValueError(f"support surface contains forbidden literal {forbidden_match!r}: {surface_id}")
                rows.append(
                    {
                        "surface_id": surface_id,
                        "surface_family": family,
                        "canonical_phrase": canonical,
                        "verb_lemmas": [verb],
                        "cue_lemmas": [cue],
                        "max_token_distance": int(policy["max_token_distance"]),
                        "weight": 1.0,
                        "event_type": "support_window_event",
                        "normalization_rule": policy["normalization_rule"],
                        "source_rule_id": policy["source_rule_id"],
                        "source_policy": "independent_static_taxonomy_not_859277_transcripts",
                        "naturalness_rationale": "Ordinary verb plus task cue window that can appear in useful advice without public structural labels.",
                        "structural_marker": False,
                        "technical_literal": False,
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
                    }
                )
    return rows


def _toy_positive_text(bank: list[Mapping[str, Any]], config: Mapping[str, Any]) -> str:
    payload_id = str(config["payload_id"])
    coordinate_count = int(config["coordinate_count"])
    selected: list[Mapping[str, Any]] = []
    seen_coordinates: set[int] = set()
    for row in bank:
        coordinate, polarity = map_surface_to_coordinate_and_polarity(
            audit_key=_DEV_AUDIT_KEY_MATERIAL,
            payload_id=payload_id,
            surface_id=str(row["surface_id"]),
            coordinate_count=coordinate_count,
        )
        if polarity != 1 or coordinate in seen_coordinates:
            continue
        selected.append(row)
        seen_coordinates.add(coordinate)
        if len(selected) >= 24:
            break
    if len(selected) < 24:
        raise ValueError("not enough positive-polarity support-window rows for toy fixture")
    sentences = []
    for row in selected:
        verb = row["verb_lemmas"][0]
        cue = row["cue_lemmas"][0]
        sentences.append(f"{verb.capitalize()} the current {cue} before continuing.")
    return " ".join(sentences)


def _decision_for_events(events: list[Mapping[str, Any]], config: Mapping[str, Any]) -> Any:
    decoder_spec = config["decoder_spec"]
    return decide_keyed_correlation(
        events,
        audit_key=_DEV_AUDIT_KEY_MATERIAL,
        payload_id=str(config["payload_id"]),
        wrong_audit_key=_DEV_WRONG_KEY_MATERIAL,
        wrong_payload_id=str(config["wrong_payload_id"]),
        coordinate_count=int(config["coordinate_count"]),
        min_observed_events=int(decoder_spec["accept_requires"]["min_observed_events"]),
        min_distinct_coordinates=int(decoder_spec["accept_requires"]["min_distinct_coordinates"]),
        min_keyed_correlation_score=float(
            decoder_spec["required_before_accept"]["protected_keyed_correlation_score_min"]
        ),
        min_specificity_margin=float(
            decoder_spec["required_before_accept"]["protected_minus_best_wrong_specificity_margin_min"]
        ),
        max_wrong_score=float(decoder_spec["required_before_accept"]["wrong_key_correlation_score_max"]),
    )


def build_package(config_path: Path = DEFAULT_CONFIG, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    config = _load_yaml(config_path)
    bank = build_event_window_bank(config)
    families = Counter(str(row["surface_family"]) for row in bank)
    max_family_fraction = max(families.values()) / len(bank)
    max_allowed = float(config["static_validation"]["max_surface_family_fraction"])
    if max_family_fraction > max_allowed:
        raise ValueError(f"surface family concentration too high: {max_family_fraction} > {max_allowed}")

    toy_text = _toy_positive_text(bank, config)
    toy_events = extract_support_window_events(
        toy_text,
        bank,
        scrub_mode="all",
        max_events_per_segment=int(config["event_window_policy"]["max_events_per_segment"]),
    )
    toy_decision = _decision_for_events(toy_events, config)
    if not toy_decision.accept:
        raise ValueError("toy positive fixture did not accept")

    wrong_key_decision = decide_keyed_correlation(
        toy_events,
        audit_key=_DEV_WRONG_KEY_MATERIAL,
        payload_id=str(config["payload_id"]),
        wrong_audit_key=_DEV_AUDIT_KEY_MATERIAL,
        wrong_payload_id=str(config["wrong_payload_id"]),
        coordinate_count=int(config["coordinate_count"]),
        min_observed_events=int(config["decoder_spec"]["accept_requires"]["min_observed_events"]),
        min_distinct_coordinates=int(config["decoder_spec"]["accept_requires"]["min_distinct_coordinates"]),
        min_keyed_correlation_score=float(
            config["decoder_spec"]["required_before_accept"]["protected_keyed_correlation_score_min"]
        ),
        min_specificity_margin=float(
            config["decoder_spec"]["required_before_accept"]["protected_minus_best_wrong_specificity_margin_min"]
        ),
        max_wrong_score=float(config["decoder_spec"]["required_before_accept"]["wrong_key_correlation_score_max"]),
    )
    wrong_payload_decision = decide_keyed_correlation(
        toy_events,
        audit_key=_DEV_AUDIT_KEY_MATERIAL,
        payload_id=str(config["wrong_payload_id"]),
        wrong_audit_key=_DEV_WRONG_KEY_MATERIAL,
        wrong_payload_id=str(config["payload_id"]),
        coordinate_count=int(config["coordinate_count"]),
        min_observed_events=int(config["decoder_spec"]["accept_requires"]["min_observed_events"]),
        min_distinct_coordinates=int(config["decoder_spec"]["accept_requires"]["min_distinct_coordinates"]),
        min_keyed_correlation_score=float(
            config["decoder_spec"]["required_before_accept"]["protected_keyed_correlation_score_min"]
        ),
        min_specificity_margin=float(
            config["decoder_spec"]["required_before_accept"]["protected_minus_best_wrong_specificity_margin_min"]
        ),
        max_wrong_score=float(config["decoder_spec"]["required_before_accept"]["wrong_key_correlation_score_max"]),
    )
    if wrong_key_decision.accept or wrong_payload_decision.accept:
        raise ValueError("wrong-key or wrong-payload toy fixture accepted")

    output_dir.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    contract = {
        "schema_name": "natural_evidence_v2_r4_positive_support_repair_contract_v1",
        "contract_id": config["contract_id"],
        "base_failed_job": config["base_failed_job"],
        "payload_id": config["payload_id"],
        "coordinate_count": config["coordinate_count"],
        "format_scrub_primary": "all",
        "post_hoc_phrase_mining_allowed": False,
        "unchanged_resubmission_allowed": False,
        "generation_allowed": False,
        "paper_claim_allowed": False,
    }
    artifacts: dict[str, Any] = {
        "contract.json": contract,
        "event_window_bank.json": bank,
        "extractor_spec.json": {
            "schema_name": "natural_evidence_v2_r4_support_window_extractor_spec_v1",
            "extractor": "extract_r4_positive_support_window_events.py",
            **config["event_window_policy"],
        },
        "decoder_spec.json": config["decoder_spec"],
        "source_policy.json": {
            "schema_name": "natural_evidence_v2_r4_support_repair_source_policy_v1",
            "source_rule_id": config["event_window_policy"]["source_rule_id"],
            "source_policy": "independent static taxonomy, not mined from 859277 transcripts",
            "859277_reuse_policy": "failure_taxonomy_only_no_surface_addition",
        },
        "toy_positive_fixture.json": {
            "schema_name": "natural_evidence_v2_r4_support_repair_toy_fixture_v1",
            "response_text": toy_text,
            "event_count": len(toy_events),
            "decision": toy_decision.__dict__,
            "wrong_key_decision": wrong_key_decision.__dict__,
            "wrong_payload_decision": wrong_payload_decision.__dict__,
        },
    }
    for name, payload in artifacts.items():
        digest = _write_json(output_dir / name, payload)
        hashes[name] = digest
        _write_sha(output_dir / f"{name}.sha256", digest, name)
    toy_events_digest = _write_jsonl(output_dir / "toy_positive_events.jsonl", toy_events)
    hashes["toy_positive_events.jsonl"] = toy_events_digest
    _write_sha(output_dir / "toy_positive_events.jsonl.sha256", toy_events_digest, "toy_positive_events.jsonl")

    summary = {
        "schema_name": "natural_evidence_v2_r4_positive_support_repair_package_summary_v1",
        "status": "PASS_SUPPORT_REPAIR_PACKAGE_STATIC_VALIDATION_NO_COMPUTE",
        "contract_id": config["contract_id"],
        "event_window_rows": len(bank),
        "surface_family_counts": dict(families),
        "max_surface_family_fraction": max_family_fraction,
        "toy_positive_accept": toy_decision.accept,
        "toy_positive_events": len(toy_events),
        "toy_positive_distinct_coordinates": toy_decision.distinct_coordinates,
        "wrong_key_accept": wrong_key_decision.accept,
        "wrong_payload_accept": wrong_payload_decision.accept,
        "post_hoc_phrase_mining_allowed": False,
        "slurm_allowed": False,
        "generation_allowed": False,
        "paper_claim_allowed": False,
        "artifact_hashes": hashes,
    }
    summary_digest = _write_json(output_dir / "package_summary.json", summary)
    _write_sha(output_dir / "package_summary.json.sha256", summary_digest, "package_summary.json")
    report = (
        "# R4 Positive Support-Repair Package\n\n"
        "Status: `PASS_SUPPORT_REPAIR_PACKAGE_STATIC_VALIDATION_NO_COMPUTE`\n\n"
        f"- contract id: `{config['contract_id']}`\n"
        f"- event-window rows: `{len(bank)}`\n"
        f"- max surface family fraction: `{max_family_fraction:.3f}`\n"
        f"- toy positive events: `{len(toy_events)}`\n"
        f"- toy positive accept: `{toy_decision.accept}`\n"
        f"- wrong-key accept: `{wrong_key_decision.accept}`\n"
        f"- wrong-payload accept: `{wrong_payload_decision.accept}`\n\n"
        "No Slurm, generation, model scoring, training, or paper claim is unlocked by this package.\n"
    )
    report_digest = _write_text(output_dir / "package_report.md", report)
    _write_sha(output_dir / "package_report.md.sha256", report_digest, "package_report.md")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the R4 support-repair artifact-only package.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_package(
        config_path=args.config if args.config.is_absolute() else ROOT / args.config,
        output_dir=args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

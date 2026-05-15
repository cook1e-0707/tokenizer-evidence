from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.extract_r4_positive_support_window_events import (  # noqa: E402
    extract_support_window_events,
    load_event_window_bank,
)

DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_positive_selectivity_prompt_policy.yaml"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
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


def forbidden_hits(text: str, forbidden_terms: Iterable[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in forbidden_terms if term.lower() in lowered]


def _policy_families(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    families = config["elicitation_policy"]["families"]
    if not isinstance(families, list) or not families:
        raise ValueError("elicitation_policy.families must be a non-empty list")
    return [dict(row) for row in families]


def elicited_prompt_text(source: Mapping[str, Any], family: Mapping[str, Any]) -> str:
    audience = str(source.get("audience", "the reader"))
    domain = str(source.get("domain", "the situation"))
    constraint = str(source.get("constraint", "keeping the advice practical"))
    angle = str(source.get("angle", "what matters most"))
    source_family = str(source.get("family", "practical_advice_short"))
    opening = {
        "practical_advice_short": "Give concise practical advice",
        "task_explanation_short": "Explain a practical approach",
        "maintenance_guidance": "Give brief maintenance guidance",
        "planning_guidance": "Give practical planning guidance",
        "troubleshooting_guidance": "Offer short troubleshooting guidance",
        "safety_or_quality_checklist_natural": "Provide a compact safety or quality check",
    }.get(source_family, "Give a useful practical answer")
    instruction = str(family["instruction"]).strip()
    return (
        f"{opening} for {audience} working on {domain}, with emphasis on {constraint}. "
        f"Focus on {angle}. {instruction} "
        "Write an ordinary useful answer in short prose or natural bullets. "
        "Avoid headings, numbered lists, repeated labels, and any technical marking terminology."
    )


def build_prompt_rows(
    source_rows: list[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    families = _policy_families(config)
    forbidden = list(config["forbidden_public_terms"])
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    for index, source in enumerate(source_rows):
        policy = families[index % len(families)]
        policy_family_id = str(policy["family_id"])
        text = elicited_prompt_text(source, policy)
        hits = forbidden_hits(text, forbidden)
        if hits:
            violations.append(
                {
                    "source_prompt_id": str(source.get("prompt_id", "")),
                    "policy_family_id": policy_family_id,
                    "hits": hits,
                    "prompt_text": text,
                }
            )
        row = dict(source)
        row.update(
            {
                "schema_name": "natural_evidence_v2_r4_positive_selectivity_prompt_policy_v1",
                "source_prompt_id": str(source.get("prompt_id", "")),
                "source_prompt_text_sha256": str(source.get("prompt_text_sha256", "")),
                "prompt_id": "r4_selectivity_policy_" + sha256_text(f"{index}:{policy_family_id}:{text}")[:20],
                "prompt_text": text,
                "prompt_text_sha256": sha256_text(text),
                "selectivity_prompt_policy_id": str(config["policy_id"]),
                "policy_family_id": policy_family_id,
                "source_policy": str(config["source_policy"]),
                "split": str(source.get("split", "dev")),
                "slurm_allowed": False,
                "generation_allowed": False,
                "model_scoring_allowed": False,
                "training_allowed": False,
                "paper_claim_allowed": False,
            }
        )
        rows.append(row)
        family_counts[policy_family_id] += 1
    return rows, violations, family_counts


def build_fixture_rows(config: Mapping[str, Any], event_bank: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family_event_counts: dict[str, int] = {}
    for family in _policy_families(config):
        family_id = str(family["family_id"])
        text = str(family["fixture"])
        events = extract_support_window_events(
            text,
            event_bank,
            scrub_mode="all",
            max_events_per_segment=8,
        )
        family_event_counts[family_id] = len(events)
        rows.append(
            {
                "schema_name": "natural_evidence_v2_r4_selectivity_prompt_policy_fixture_v1",
                "policy_family_id": family_id,
                "fixture_text": text,
                "fixture_text_sha256": sha256_text(text),
                "event_count": len(events),
                "event_surface_ids": [event["surface_id"] for event in events],
                "generation_allowed": False,
                "paper_claim_allowed": False,
            }
        )
    summary = {
        "family_event_counts": family_event_counts,
        "total_fixture_events": sum(family_event_counts.values()),
    }
    return rows, summary


def build_package(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_dir = ROOT / str(config["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    source_prompt_path = ROOT / str(config["source_prompt_bank"])
    selectivity_package_dir = ROOT / str(config["selectivity_package"])
    event_bank_path = selectivity_package_dir / "event_window_bank.json"
    event_bank = load_event_window_bank(event_bank_path)
    source_rows = read_jsonl(source_prompt_path)
    prompt_rows, violations, family_counts = build_prompt_rows(source_rows, config=config)
    fixture_rows, fixture_summary = build_fixture_rows(config, event_bank)

    duplicate_prompt_ids = len(prompt_rows) - len({str(row["prompt_id"]) for row in prompt_rows})
    max_family_fraction = max(family_counts.values()) / len(prompt_rows) if prompt_rows else 0.0
    validation = config["static_validation"]
    errors: list[str] = []
    if len(prompt_rows) < int(validation["min_prompt_count"]):
        errors.append("prompt_count_below_min")
    if duplicate_prompt_ids and not bool(validation["duplicate_prompt_ids_allowed"]):
        errors.append("duplicate_prompt_ids")
    if violations and not bool(validation["forbidden_prompt_violations_allowed"]):
        errors.append("forbidden_prompt_violations")
    if max_family_fraction > float(config["elicitation_policy"]["max_policy_family_fraction"]):
        errors.append("policy_family_fraction_too_high")
    min_family_events = int(validation["min_fixture_events_per_family"])
    weak_families = [
        family_id
        for family_id, count in fixture_summary["family_event_counts"].items()
        if count < min_family_events
    ]
    if weak_families:
        errors.append("fixture_family_event_count_below_min")
    if fixture_summary["total_fixture_events"] < int(validation["min_total_fixture_events"]):
        errors.append("total_fixture_events_below_min")
    status = "PASS_SELECTIVITY_PROMPT_POLICY_STATIC_VALIDATION_NO_COMPUTE" if not errors else "FAIL_SELECTIVITY_PROMPT_POLICY_STATIC_VALIDATION_NO_COMPUTE"

    output_dir.mkdir(parents=True, exist_ok=False)
    prompts_path = output_dir / "dev_prompts.jsonl"
    fixtures_path = output_dir / "expected_elicitation_fixtures.jsonl"
    write_jsonl(prompts_path, prompt_rows)
    write_jsonl(fixtures_path, fixture_rows)
    write_json(output_dir / "forbidden_prompt_violations.json", {"violations": violations})
    manifest = {
        "schema_name": "natural_evidence_v2_r4_selectivity_prompt_policy_manifest_v1",
        "status": status,
        "policy_id": config["policy_id"],
        "source_prompt_bank": str(source_prompt_path),
        "source_prompt_bank_sha256": sha256_file(source_prompt_path),
        "selectivity_package": str(selectivity_package_dir),
        "event_window_bank_sha256": sha256_file(event_bank_path),
        "prompt_count": len(prompt_rows),
        "prompt_bank_sha256": sha256_file(prompts_path),
        "fixture_bank_sha256": sha256_file(fixtures_path),
        "duplicate_prompt_id_count": duplicate_prompt_ids,
        "forbidden_prompt_violation_count": len(violations),
        "policy_family_counts": dict(family_counts),
        "max_policy_family_fraction": max_family_fraction,
        "fixture_summary": fixture_summary,
        "validation_errors": errors,
        "source_policy": config["source_policy"],
        "slurm_allowed": False,
        "generation_allowed": False,
        "model_scoring_allowed": False,
        "training_allowed": False,
        "paper_claim_allowed": False,
    }
    write_json(output_dir / "prompt_policy_manifest.json", manifest)
    report = [
        "# R4 Positive Selectivity Prompt Policy",
        "",
        f"Status: `{status}`",
        "",
        "Artifact-only prompt policy. No generation, scoring, training, or Slurm submission was started.",
        "",
        f"- prompts: `{len(prompt_rows)}`",
        f"- duplicate prompt ids: `{duplicate_prompt_ids}`",
        f"- forbidden violations: `{len(violations)}`",
        f"- max policy family fraction: `{max_family_fraction:.3f}`",
        f"- total fixture events: `{fixture_summary['total_fixture_events']}`",
        f"- prompt bank sha256: `{manifest['prompt_bank_sha256']}`",
    ]
    (output_dir / "prompt_policy_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if errors:
        raise ValueError(f"selectivity prompt policy validation failed: {errors}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the R4 selectivity prompt-policy package without compute.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_package(args.config if args.config.is_absolute() else ROOT / args.config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

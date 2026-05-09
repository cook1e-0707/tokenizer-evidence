from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/qwen_v2_micro_slot_pilot.yaml"

OWNER_SPLITS = ("train", "dev", "eval")
OWNER_FAMILY_IDS = (
    "F1_8_sentence_explanation",
    "F2_8_step_checklist",
    "F3_6_point_comparison",
    "F4_concise_advice_with_transitions",
)
NULL_FAMILY_IDS = (
    "N1_general_practical_answer",
    "N2_everyday_decision",
    "N3_brief_explanation",
    "N4_flexible_plan",
)
SPLIT_FILES = {
    "train": "qwen_v2_train_prompts.jsonl",
    "dev": "qwen_v2_dev_prompts.jsonl",
    "eval": "qwen_v2_eval_prompts.jsonl",
    "organic_null": "qwen_v2_organic_null_prompts.jsonl",
}

ACTIVITIES = (
    "keep a shared calendar useful",
    "plan a simple weekly meal routine",
    "reduce clutter on a desk",
    "choose a maintenance schedule for a bicycle",
    "write clearer meeting notes",
    "pack for a short work trip",
    "review a household budget",
    "prepare for a neighborhood event",
    "organize a small reading group",
    "set up a quiet study routine",
    "make a morning handoff smoother",
    "track recurring chores",
    "prepare a clear project update",
    "coordinate rides for a weekend event",
    "keep a small garden healthy",
    "plan a short class activity",
    "summarize feedback from a team",
    "choose supplies for a repair kit",
    "make a simple backup routine",
    "handle a busy errand list",
    "prepare a calm customer reply",
    "compare notes after a workshop",
    "make a shared kitchen easier to use",
    "plan a low-cost birthday gathering",
    "keep volunteer tasks organized",
    "make onboarding notes easier to follow",
    "prepare a tidy travel checklist",
    "choose a practical filing routine",
    "run a short planning meeting",
    "make hand-written notes easier to scan",
)

AUDIENCES = (
    "a new team",
    "a family",
    "a volunteer group",
    "a student club",
    "a small office",
    "a weekend class",
    "a remote team",
    "a neighborhood group",
    "a busy household",
    "a local committee",
)

CONSTRAINTS = (
    "saving time",
    "keeping costs low",
    "staying calm",
    "avoiding extra tools",
    "working with limited space",
    "keeping records simple",
    "sharing tasks fairly",
    "planning around a busy week",
)

ANGLES = (
    "daily habits",
    "common mistakes",
    "simple checks",
    "clear handoffs",
    "early planning",
    "steady follow-up",
)

COMPARISONS = (
    ("paper notes", "shared documents"),
    ("morning planning", "evening planning"),
    ("one long meeting", "two short meetings"),
    ("a checklist", "a calendar reminder"),
    ("packing early", "packing the night before"),
    ("written feedback", "spoken feedback"),
    ("a shared spreadsheet", "a simple notebook"),
    ("weekly review", "daily review"),
    ("group errands", "solo errands"),
    ("a quiet workspace", "a flexible workspace"),
    ("batch cooking", "daily cooking"),
    ("a fixed routine", "a rotating routine"),
    ("short notes", "detailed notes"),
    ("a phone reminder", "a wall calendar"),
    ("local storage", "cloud storage"),
    ("a single checklist", "separate lists"),
    ("a shared folder", "an email thread"),
    ("early cleanup", "cleanup after the event"),
    ("a simple repair", "calling a specialist"),
    ("scheduled breaks", "informal breaks"),
    ("a printed handout", "a spoken summary"),
    ("a group chat", "a weekly message"),
    ("a trial week", "a fixed plan"),
    ("a short agenda", "an open discussion"),
)

OWNER_DESCRIPTIONS = {
    "F1_8_sentence_explanation": "eight natural sentences",
    "F2_8_step_checklist": "eight concise natural checklist steps",
    "F3_6_point_comparison": "six natural comparison points",
    "F4_concise_advice_with_transitions": "concise advice with natural transitions",
}

OWNER_SLOT_INTENTS = {
    "F1_8_sentence_explanation": (
        "sentence_opener",
        "discourse_marker",
        "optional_hedge",
        "function_word_alternative",
    ),
    "F2_8_step_checklist": (
        "bullet_or_step_opener",
        "transition_word",
        "function_word_alternative",
    ),
    "F3_6_point_comparison": (
        "bullet_or_step_opener",
        "discourse_marker",
        "optional_hedge",
        "function_word_alternative",
    ),
    "F4_concise_advice_with_transitions": (
        "sentence_opener",
        "discourse_marker",
        "optional_hedge",
        "transition_word",
        "function_word_alternative",
    ),
}

RESPONSE_SHAPES = {
    "F1_8_sentence_explanation": "eight plain sentences",
    "F2_8_step_checklist": "eight one-sentence steps",
    "F3_6_point_comparison": "six short comparison points",
    "F4_concise_advice_with_transitions": "short practical advice with transitions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic natural_evidence_v2 WP2 prompt split artifacts "
            "and audit public prompt text for forbidden surface terms."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--use-config-split-sizes",
        action="store_true",
        help="Write the configured train/dev/eval/organic-null split counts.",
    )
    parser.add_argument("--rows-per-owner-family-per-split", type=int, default=4)
    parser.add_argument("--organic-null-rows", type=int, default=16)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML top level must be a mapping: {path}")
    return payload


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_new(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def distribute(total: int, ids: Sequence[str]) -> dict[str, int]:
    base = total // len(ids)
    remainder = total % len(ids)
    return {item_id: base + int(index < remainder) for index, item_id in enumerate(ids)}


def combo(index: int) -> dict[str, str]:
    activity_count = len(ACTIVITIES)
    audience_count = len(AUDIENCES)
    constraint_count = len(CONSTRAINTS)
    return {
        "activity": ACTIVITIES[index % activity_count],
        "audience": AUDIENCES[(index // activity_count) % audience_count],
        "constraint": CONSTRAINTS[(index // (activity_count * audience_count)) % constraint_count],
        "angle": ANGLES[(index // (activity_count * audience_count * constraint_count)) % len(ANGLES)],
    }


def comparison_combo(index: int) -> dict[str, str]:
    pair_count = len(COMPARISONS)
    audience_count = len(AUDIENCES)
    constraint_count = len(CONSTRAINTS)
    left, right = COMPARISONS[index % pair_count]
    return {
        "left": left,
        "right": right,
        "audience": AUDIENCES[(index // pair_count) % audience_count],
        "constraint": CONSTRAINTS[(index // (pair_count * audience_count)) % constraint_count],
        "angle": ANGLES[(index // (pair_count * audience_count * constraint_count)) % len(ANGLES)],
    }


def render_owner_prompt(family_id: str, index: int) -> str:
    if family_id == "F3_6_point_comparison":
        values = comparison_combo(index)
        return (
            f"Compare {values['left']} with {values['right']} for {values['audience']} in six short points. "
            f"Emphasize {values['angle']} while {values['constraint']}, and focus on practical tradeoffs."
        )

    values = combo(index)
    if family_id == "F1_8_sentence_explanation":
        return (
            f"Write eight natural sentences explaining how {values['audience']} can {values['activity']} "
            f"while {values['constraint']}. Emphasize {values['angle']}, and keep each sentence concrete and plain."
        )
    if family_id == "F2_8_step_checklist":
        return (
            f"Create an eight-step checklist for helping {values['audience']} {values['activity']} "
            f"while {values['constraint']}. Emphasize {values['angle']}, and use one short sentence per step."
        )
    if family_id == "F4_concise_advice_with_transitions":
        return (
            f"Give concise advice for {values['audience']} who need to {values['activity']} "
            f"while {values['constraint']}. Emphasize {values['angle']}, use smooth transitions, "
            "and keep the guidance practical."
        )
    raise ValueError(f"unknown owner family id: {family_id}")


def render_null_prompt(family_id: str, index: int) -> str:
    values = combo(index)
    if family_id == "N1_general_practical_answer":
        return (
            f"Please answer this everyday request for {values['audience']}: how can they {values['activity']} "
            f"while {values['constraint']}? Keep the response helpful and direct."
        )
    if family_id == "N2_everyday_decision":
        return (
            f"What should {values['audience']} consider when they need to {values['activity']} "
            f"while {values['constraint']}? Give a practical answer."
        )
    if family_id == "N3_brief_explanation":
        return (
            f"Explain why it can help {values['audience']} to {values['activity']} "
            f"while {values['constraint']}. Keep the answer clear and useful."
        )
    if family_id == "N4_flexible_plan":
        return (
            f"Suggest a practical plan for {values['audience']} to {values['activity']} "
            f"while {values['constraint']}. Keep it flexible and concise."
        )
    raise ValueError(f"unknown null family id: {family_id}")


def make_owner_row(
    *,
    protocol_id: str,
    split: str,
    prompt_id: str,
    family_id: str,
    family_index: int,
    prompt_text: str,
) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_prompt_row_v1",
        "protocol_id": protocol_id,
        "prompt_set_id": "qwen_v2_wp2_controlled_natural_prompt_set_v1",
        "prompt_id": prompt_id,
        "split": split,
        "family_id": family_id,
        "family_description": OWNER_DESCRIPTIONS[family_id],
        "family_index": family_index,
        "control_role": "controlled_probe",
        "prompt_text": prompt_text,
        "prompt_text_sha256": sha256_text(prompt_text),
        "intended_response_shape": RESPONSE_SHAPES[family_id],
        "intended_micro_slot_sources": list(OWNER_SLOT_INTENTS[family_id]),
        "precommit_status": "not_committed",
        "artifact_only": True,
        "model_calls_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }


def make_null_row(
    *,
    protocol_id: str,
    prompt_id: str,
    family_id: str,
    family_index: int,
    prompt_text: str,
) -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_prompt_row_v1",
        "protocol_id": protocol_id,
        "prompt_set_id": "qwen_v2_wp2_controlled_natural_prompt_set_v1",
        "prompt_id": prompt_id,
        "split": "organic_null",
        "family_id": family_id,
        "family_description": "disjoint organic-null prompt family",
        "family_index": family_index,
        "control_role": "organic_null",
        "prompt_text": prompt_text,
        "prompt_text_sha256": sha256_text(prompt_text),
        "intended_response_shape": "ordinary helpful answer without controlled shape",
        "intended_micro_slot_sources": [],
        "precommit_status": "not_committed",
        "artifact_only": True,
        "model_calls_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }


def build_rows(config: Mapping[str, Any], use_config_split_sizes: bool, rows_per_family: int, null_rows: int) -> dict[str, list[dict[str, Any]]]:
    protocol = config.get("protocol", {})
    if not isinstance(protocol, Mapping):
        raise ValueError("config.protocol must be a mapping")
    protocol_id = str(protocol.get("id", ""))
    if protocol_id == "":
        raise ValueError("config.protocol.id is required")

    split_sizes = config.get("split_sizes", {})
    if not isinstance(split_sizes, Mapping):
        raise ValueError("config.split_sizes must be a mapping")

    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in (*OWNER_SPLITS, "organic_null")}
    owner_cursor = {family_id: 0 for family_id in OWNER_FAMILY_IDS}
    for split in OWNER_SPLITS:
        if use_config_split_sizes:
            total = int(split_sizes.get(f"{split}_prompts", 0))
            counts = distribute(total, OWNER_FAMILY_IDS)
        else:
            counts = {family_id: int(rows_per_family) for family_id in OWNER_FAMILY_IDS}
        split_ordinal = 0
        for family_id in OWNER_FAMILY_IDS:
            for _ in range(counts[family_id]):
                family_index = owner_cursor[family_id]
                owner_cursor[family_id] += 1
                prompt_text = render_owner_prompt(family_id, family_index)
                rows_by_split[split].append(
                    make_owner_row(
                        protocol_id=protocol_id,
                        split=split,
                        prompt_id=f"qwen_v2_{split}_{split_ordinal:05d}",
                        family_id=family_id,
                        family_index=family_index,
                        prompt_text=prompt_text,
                    )
                )
                split_ordinal += 1

    null_total = int(split_sizes.get("organic_null_prompts", 0)) if use_config_split_sizes else int(null_rows)
    null_counts = distribute(null_total, NULL_FAMILY_IDS)
    null_ordinal = 0
    null_cursor = {family_id: 0 for family_id in NULL_FAMILY_IDS}
    for family_id in NULL_FAMILY_IDS:
        for _ in range(null_counts[family_id]):
            family_index = null_cursor[family_id]
            null_cursor[family_id] += 1
            prompt_text = render_null_prompt(family_id, family_index)
            rows_by_split["organic_null"].append(
                make_null_row(
                    protocol_id=protocol_id,
                    prompt_id=f"qwen_v2_organic_null_{null_ordinal:05d}",
                    family_id=family_id,
                    family_index=family_index,
                    prompt_text=prompt_text,
                )
            )
            null_ordinal += 1
    return rows_by_split


def forbidden_hits(text: str, forbidden_terms: Sequence[str]) -> list[str]:
    upper_text = text.upper()
    return [term for term in forbidden_terms if str(term).upper() in upper_text]


def audit_rows(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]], forbidden_terms: Sequence[str]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    prompt_texts: list[str] = []
    rows_by_family: Counter[str] = Counter()
    rows_by_split_count: Counter[str] = Counter()
    for split, rows in rows_by_split.items():
        for row in rows:
            prompt_text = str(row.get("prompt_text", ""))
            prompt_texts.append(prompt_text)
            rows_by_split_count[str(split)] += 1
            rows_by_family[str(row.get("family_id", ""))] += 1
            hits = forbidden_hits(prompt_text, forbidden_terms)
            if hits:
                violations.append(
                    {
                        "prompt_id": row.get("prompt_id", ""),
                        "split": split,
                        "family_id": row.get("family_id", ""),
                        "hits": hits,
                    }
                )

    text_counts = Counter(prompt_texts)
    duplicate_texts = [text for text, count in text_counts.items() if count > 1]
    total_rows = len(prompt_texts)
    empty_prompt_count = sum(1 for text in prompt_texts if not text.strip())
    status = "PASS_FORBIDDEN_SURFACE_AUDIT"
    if violations or duplicate_texts or empty_prompt_count:
        status = "FAIL_FORBIDDEN_SURFACE_AUDIT"
    return {
        "schema_name": "natural_evidence_v2_wp2_forbidden_surface_audit_v1",
        "status": status,
        "audit_scope": "public_prompt_text_only",
        "forbidden_terms": list(forbidden_terms),
        "total_rows": total_rows,
        "rows_by_split": dict(sorted(rows_by_split_count.items())),
        "rows_by_family": dict(sorted(rows_by_family.items())),
        "violation_count": len(violations),
        "forbidden_surface_rate": 0.0 if total_rows == 0 else len(violations) / total_rows,
        "violations": violations,
        "duplicate_prompt_text_count": len(duplicate_texts),
        "empty_prompt_count": empty_prompt_count,
        "model_calls_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }


def prompt_family_templates() -> dict[str, Any]:
    return {
        "schema_name": "natural_evidence_v2_wp2_prompt_family_templates_v1",
        "status": "TEMPLATE_SCAFFOLD_NO_MODEL_CALLS",
        "owner_family_ids": list(OWNER_FAMILY_IDS),
        "organic_null_family_ids": list(NULL_FAMILY_IDS),
        "owner_families": {
            family_id: {
                "description": OWNER_DESCRIPTIONS[family_id],
                "intended_response_shape": RESPONSE_SHAPES[family_id],
                "intended_micro_slot_sources": list(OWNER_SLOT_INTENTS[family_id]),
            }
            for family_id in OWNER_FAMILY_IDS
        },
        "organic_null_role": "disjoint natural prompts for accidental-acceptance checks only",
        "precommit_status": "not_committed",
        "artifact_only": True,
        "model_calls_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }


def split_manifest(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    use_config_split_sizes: bool,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    split_sizes = config.get("split_sizes", {})
    if not isinstance(split_sizes, Mapping):
        split_sizes = {}
    return {
        "schema_name": "natural_evidence_v2_wp2_prompt_split_manifest_v1",
        "status": "WP2_PROMPT_SPLITS_WRITTEN_AUDITED_NO_MODEL_CALLS",
        "protocol_id": str(dict(config.get("protocol", {})).get("id", "")),
        "prompt_set_id": "qwen_v2_wp2_controlled_natural_prompt_set_v1",
        "output_dir": str(output_dir),
        "use_config_split_sizes": bool(use_config_split_sizes),
        "target_split_sizes": {
            "train": int(split_sizes.get("train_prompts", 0)),
            "dev": int(split_sizes.get("dev_prompts", 0)),
            "eval": int(split_sizes.get("eval_prompts", 0)),
            "organic_null": int(split_sizes.get("organic_null_prompts", 0)),
        },
        "written_split_sizes": {split: len(rows) for split, rows in rows_by_split.items()},
        "files": {split: str(output_dir / filename) for split, filename in SPLIT_FILES.items()},
        "template_json": str(output_dir / "prompt_family_templates.json"),
        "forbidden_surface_audit_json": str(output_dir / "forbidden_surface_audit.json"),
        "forbidden_surface_status": audit.get("status", ""),
        "precommit_status": "not_committed",
        "artifact_only": True,
        "model_calls_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "next_gate": "WP3 micro-slot detector and 2-way policy design only; training and E2E remain forbidden",
    }


def readme_text(manifest: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP2 Prompt Family Scaffold",
            "",
            "Deterministic artifact-only prompt split files for the v2 controlled-natural route.",
            "",
            f"status: `{manifest['status']}`",
            f"prompt_set_id: `{manifest['prompt_set_id']}`",
            f"forbidden_surface_status: `{manifest['forbidden_surface_status']}`",
            "",
            "No model calls, training, E2E evaluation, FAR aggregation, or positive paper claim were started.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")

    config = load_yaml(args.config if args.config.is_absolute() else ROOT / args.config)
    forbidden_terms = [str(item) for item in config.get("forbidden_surface_terms", [])]
    if not forbidden_terms:
        raise ValueError("config.forbidden_surface_terms must not be empty")

    rows_by_split = build_rows(
        config=config,
        use_config_split_sizes=bool(args.use_config_split_sizes),
        rows_per_family=int(args.rows_per_owner_family_per_split),
        null_rows=int(args.organic_null_rows),
    )
    audit = audit_rows(rows_by_split, forbidden_terms)
    if audit["status"] != "PASS_FORBIDDEN_SURFACE_AUDIT":
        raise ValueError(f"prompt scaffold failed audit: {audit}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, filename in SPLIT_FILES.items():
        write_jsonl(output_dir / filename, rows_by_split[split])
    write_json(output_dir / "prompt_family_templates.json", prompt_family_templates())
    write_json(output_dir / "forbidden_surface_audit.json", audit)
    manifest = split_manifest(
        config=config,
        output_dir=output_dir,
        rows_by_split=rows_by_split,
        use_config_split_sizes=bool(args.use_config_split_sizes),
        audit=audit,
    )
    write_json(output_dir / "split_manifest.json", manifest)
    write_text_new(output_dir / "README.md", readme_text(manifest))

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(output_dir),
                "written_split_sizes": manifest["written_split_sizes"],
                "forbidden_surface_status": manifest["forbidden_surface_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.run_wp3_restricted_step_label_density_audit import (
    DEFAULT_CONFIG,
    DEFAULT_PLAN_DIR,
    forbidden_terms_in_text,
    read_yaml,
)


DEFAULT_AUDIT_DIR = (
    ROOT
    / "results/natural_evidence_v2/status/wp3_restricted_step_label_density_audit_850434"
)

STRICT_PROMPT_VARIANTS = (
    {
        "variant_id": "strict_literal_16_step_lines",
        "template": (
            "Return exactly sixteen lines for {topic}. "
            "Line 1 must begin with Step 1:, line 2 with Step 2:, and continue in order through Step 16:. "
            "Do not omit any Step label. Do not add headings, bullets, tables, or closing notes. "
            "After each label, write one natural sentence that begins with an action verb."
        ),
    },
    {
        "variant_id": "strict_no_heading_16_step_lines",
        "template": (
            "Write exactly 16 checklist lines for {topic}. "
            "Use only the labels Step 1: through Step 16:, one label per line and in numeric order. "
            "Do not include a title, introduction, bullets, table, or summary. "
            "Each line should be one ordinary sentence after the label."
        ),
    },
    {
        "variant_id": "strict_numbered_step_label_lines",
        "template": (
            "Create a normal practical checklist for {topic} with exactly sixteen labeled steps. "
            "Every line must start with the literal label Step N: for N from 1 to 16. "
            "Use no other labels or headings. "
            "Each step should be concise and should start with a clear action word after the colon."
        ),
    },
    {
        "variant_id": "strict_compact_step_label_lines",
        "template": (
            "For {topic}, provide exactly Step 1: through Step 16:. "
            "Put each Step label at the start of its own line, and include all sixteen labels. "
            "Avoid any title, preface, table, bullet marker, or final comment. "
            "Use natural checklist prose after each label."
        ),
    },
)

EXPANDED_BANK_CANDIDATES = (
    ("step_label_prepare_plan_determine_define_v1", ("Prepare", "Plan"), ("Determine", "Define")),
    ("step_label_ensure_confirm_check_review_v1", ("Ensure", "Confirm"), ("Check", "Review")),
    ("step_label_pack_bring_gather_collect_v1", ("Pack", "Bring"), ("Gather", "Collect")),
    ("step_label_select_decide_choose_make_v1", ("Select", "Decide"), ("Choose", "Make")),
    ("step_label_create_develop_establish_set_v1", ("Create", "Develop"), ("Establish", "Set")),
    ("step_label_arrange_schedule_organize_plan_v1", ("Arrange", "Schedule"), ("Organize", "Plan")),
    ("step_label_use_take_send_document_v1", ("Use", "Take"), ("Send", "Document")),
    ("step_label_identify_assess_research_review_v1", ("Identify", "Assess"), ("Research", "Review")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only repair plan after WP3 restricted Step-label "
            "density audit 850434. This does not call models, score logits, train, "
            "run E2E, compute FAR, or make claims."
        )
    )
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--source-plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=256)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL row must be an object: {path}")
                rows.append(payload)
    return rows


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_new(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_topics(source_plan_dir: Path) -> list[str]:
    rows = read_jsonl(source_plan_dir / "restricted_step_label_density_audit_prompts.jsonl")
    seen: set[str] = set()
    topics: list[str] = []
    for row in rows:
        topic = str(row.get("topic", "")).strip()
        if topic and topic not in seen:
            seen.add(topic)
            topics.append(topic)
    return topics


def build_strict_prompts(
    *,
    config: Mapping[str, Any],
    topics: list[str],
    max_prompts: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in STRICT_PROMPT_VARIANTS:
        for topic_index, topic in enumerate(topics):
            prompt_text = str(variant["template"]).format(topic=topic)
            hits = forbidden_terms_in_text(config, prompt_text)
            if hits:
                raise ValueError(f"strict repair prompt contains forbidden surface {hits}: {prompt_text}")
            payload = {"variant_id": variant["variant_id"], "topic_index": topic_index, "topic": topic}
            prompt_id = "qwen_v2_wp3_repair_density_" + sha256_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":"))
            )[:20]
            rows.append(
                {
                    "schema_name": "natural_evidence_v2_wp3_restricted_step_label_repair_prompt_v1",
                    "prompt_id": prompt_id,
                    "split": "wp3_density_repair_dev",
                    "family_id": "F2_16_step_checklist_step_label_repair",
                    "variant_id": str(variant["variant_id"]),
                    "topic": topic,
                    "topic_index": topic_index,
                    "prompt_text": prompt_text,
                    "prompt_text_sha256": sha256_text(prompt_text),
                    "expected_step_labels": [f"Step {index}:" for index in range(1, 17)],
                    "expected_structural_slots": 16,
                    "artifact_only_plan": True,
                    "model_generation_started": False,
                    "model_scoring_started": False,
                    "training_started": False,
                    "e2e_eval_started": False,
                    "paper_claim_allowed": False,
                }
            )
    return rows[: max(0, int(max_prompts))]


def observed_first_word_summary(slot_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    first_words = Counter(str(row.get("first_word", "")) for row in slot_rows)
    exact_hit_words = Counter(
        str(row.get("first_word", "")) for row in slot_rows if row.get("exact_candidate_hit")
    )
    no_hit_words = Counter(
        str(row.get("first_word", "")) for row in slot_rows if not row.get("exact_candidate_hit")
    )
    return {
        "slot_rows": len(slot_rows),
        "top_first_words": first_words.most_common(40),
        "top_exact_hit_words": exact_hit_words.most_common(40),
        "top_no_hit_words": no_hit_words.most_common(50),
    }


def expanded_bank_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bank_id, side0, side1 in EXPANDED_BANK_CANDIDATES:
        rows.append(
            {
                "schema_name": "natural_evidence_v2_wp3_step_label_expanded_bank_candidate_v1",
                "candidate_bank_id": bank_id,
                "slot_type": "step_label_action_verb",
                "bucket_count": 2,
                "bucket_0_surfaces": list(side0),
                "bucket_1_surfaces": list(side1),
                "source": "observed_base_qwen_step_openers_from_850434",
                "status": "CANDIDATE_REQUIRES_TOKENIZER_AND_CONTEXT_MASS_AUDIT",
                "training_started": False,
                "e2e_eval_started": False,
                "paper_claim_allowed": False,
            }
        )
    return rows


def readme_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP3 Restricted Step-Label Repair Plan",
            "",
            "Artifact-only repair plan after density audit 850434.",
            "",
            f"status: `{summary['status']}`",
            f"strict_prompt_count: `{summary['strict_prompt_count']}`",
            f"expanded_bank_candidate_count: `{summary['expanded_bank_candidate_count']}`",
            "",
            "This did not call a model, score logits, train, run E2E, aggregate FAR, or make a paper claim.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    audit_dir = resolve_path(args.audit_dir)
    source_plan_dir = resolve_path(args.source_plan_dir)
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    config = read_yaml(config_path)
    summary_850434 = read_json(audit_dir / "restricted_step_label_density_audit_summary.json")
    slot_rows = read_jsonl(audit_dir / "restricted_step_label_detected_slots.jsonl")
    topics = source_topics(source_plan_dir)
    strict_prompts = build_strict_prompts(
        config=config,
        topics=topics,
        max_prompts=max(0, int(args.max_prompts)),
    )
    bank_candidates = expanded_bank_candidates()
    first_word_summary = observed_first_word_summary(slot_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_name": "natural_evidence_v2_wp3_restricted_step_label_repair_plan_summary_v1",
        "status": "WP3_RESTRICTED_STEP_LABEL_REPAIR_PLAN_READY_ARTIFACT_ONLY",
        "source_audit_dir": str(audit_dir),
        "source_audit_status": str(summary_850434.get("status", "")),
        "source_complete_step_label_response_rate": summary_850434.get("complete_step_label_response_rate"),
        "source_mean_detected_structural_slots_per_response": summary_850434.get(
            "mean_detected_structural_slots_per_response"
        ),
        "source_raw_bank_surface_exact_hit_rate": summary_850434.get("raw_bank_surface_exact_hit_rate"),
        "strict_prompt_count": len(strict_prompts),
        "strict_prompt_variants": [item["variant_id"] for item in STRICT_PROMPT_VARIANTS],
        "expanded_bank_candidate_count": len(bank_candidates),
        "strict_prompts_jsonl": str(output_dir / "restricted_step_label_strict_repair_prompts.jsonl"),
        "expanded_bank_candidates_jsonl": str(output_dir / "restricted_step_label_expanded_bank_candidates.jsonl"),
        "first_word_summary_json": str(output_dir / "restricted_step_label_first_word_summary.json"),
        "model_generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "wp4_allowed": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Review this artifact-only repair plan. If approved, build or review a Slurm-only "
            "tokenizer/context-mass scoring plan for the expanded Step-label bank candidates. "
            "Do not start WP4 or training."
        ),
    }
    write_jsonl(output_dir / "restricted_step_label_strict_repair_prompts.jsonl", strict_prompts)
    write_jsonl(output_dir / "restricted_step_label_expanded_bank_candidates.jsonl", bank_candidates)
    write_json(output_dir / "restricted_step_label_first_word_summary.json", first_word_summary)
    write_json(output_dir / "restricted_step_label_repair_plan_summary.json", summary)
    write_text_new(output_dir / "README.md", readme_text(summary))
    print(json.dumps({"status": summary["status"], "strict_prompt_count": len(strict_prompts)}, sort_keys=True))


if __name__ == "__main__":
    main()

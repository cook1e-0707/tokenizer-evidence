from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.validate_r4_transfer_gap_repair_plan import DEFAULT_CONFIG, load_yaml, validate_plan

DEFAULT_SOURCE_PROMPTS = ROOT / "results/natural_evidence_v2/prompts/r4_cover_natural_prompt_bank_20260512_dev2048/dev_prompts.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results/natural_evidence_v2/prompts/r4_candidate_v3_transfer_gap_repaired_dev_prompts_20260514_0710"

DISALLOWED_PUBLIC_LITERALS = (
    "Step ",
    "exactly 16",
    "sixteen",
    "slot",
    "bucket",
    "fingerprint",
    "watermark",
    "payload",
    "secret key",
    "coordinate",
    "decoder",
    "hidden signal",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def forbidden_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [term for term in DISALLOWED_PUBLIC_LITERALS if term.lower() in lowered]


def repaired_prompt_text(source: Mapping[str, Any]) -> str:
    audience = str(source.get("audience", "the reader"))
    domain = str(source.get("domain", "the situation"))
    constraint = str(source.get("constraint", "keeping the advice practical"))
    angle = str(source.get("angle", "what matters most"))
    family = str(source.get("family", "practical_advice_short"))
    opening = {
        "practical_advice_short": "Give concise practical advice",
        "task_explanation_short": "Explain a practical approach",
        "maintenance_guidance": "Give brief maintenance guidance",
        "planning_guidance": "Give brief planning guidance",
        "troubleshooting_guidance": "Offer short troubleshooting guidance",
        "safety_or_quality_checklist_natural": "Provide a compact safety or quality check",
    }.get(family, "Give a useful practical answer")
    return (
        f"{opening} for {audience} working on {domain}, with emphasis on {constraint}. "
        f"Focus on {angle}. Write a useful ordinary answer in plain prose or short natural bullets. "
        "Include one or two concrete next actions as normal sentences when they fit the answer. "
        "Avoid headings, numbering, fixed labels, repeated lead-ins, and any wording that describes a hidden marking system."
    )


def build_prompt_bank(source_rows: list[Mapping[str, Any]], *, plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation = validate_plan(plan)
    if not str(validation["status"]).startswith("PASS"):
        raise ValueError(f"repair plan validation failed: {validation['errors']}")
    output_rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        text = repaired_prompt_text(source)
        hits = forbidden_hits(text)
        if hits:
            violations.append({"source_prompt_id": source.get("prompt_id"), "hits": hits, "prompt_text": text})
        row = dict(source)
        row.update(
            {
                "schema_name": "natural_evidence_v2_r4_transfer_gap_repaired_prompt_v1",
                "source_prompt_id": str(source.get("prompt_id", "")),
                "source_prompt_text_sha256": str(source.get("prompt_text_sha256", "")),
                "prompt_id": "r4_transfer_gap_repaired_" + sha256_text(f"{index}:{text}")[:20],
                "prompt_text": text,
                "prompt_text_sha256": sha256_text(text),
                "repair_policy": "transfer_gap_next_action_context_elicitation_v1",
                "split": str(source.get("split", "dev")),
                "generation_allowed": False,
                "training_allowed": False,
                "paper_claim_allowed": False,
            }
        )
        output_rows.append(row)
    return output_rows, violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R4 transfer-gap repaired dev prompts without generation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-prompts", type=Path, default=DEFAULT_SOURCE_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {args.output_dir}")
    source_rows = read_jsonl(args.source_prompts)
    plan = load_yaml(args.config)
    rows, violations = build_prompt_bank(source_rows, plan=plan)
    duplicate_prompt_ids = len(rows) - len({str(row["prompt_id"]) for row in rows})
    args.output_dir.mkdir(parents=True)
    prompts_path = args.output_dir / "dev_prompts.jsonl"
    write_jsonl(prompts_path, rows)
    summary = {
        "schema_name": "natural_evidence_v2_r4_transfer_gap_repaired_prompt_bank_summary_v1",
        "status": "PASS_R4_TRANSFER_GAP_REPAIRED_PROMPT_BANK_ARTIFACT_ONLY" if not violations and duplicate_prompt_ids == 0 else "FAIL_R4_TRANSFER_GAP_REPAIRED_PROMPT_BANK_ARTIFACT_ONLY",
        "source_prompts": str(args.source_prompts),
        "source_prompt_count": len(source_rows),
        "prompt_count": len(rows),
        "prompt_bank_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "forbidden_violation_count": len(violations),
        "duplicate_prompt_id_count": duplicate_prompt_ids,
        "generation_started": False,
        "training_started": False,
        "model_scoring_started": False,
    }
    write_json(args.output_dir / "prompt_bank_manifest.json", summary)
    write_json(args.output_dir / "forbidden_violations.json", {"violations": violations})
    report = [
        "# R4 Transfer-Gap Repaired Prompt Bank",
        "",
        "Artifact-only prompt bank. No generation, scoring, training, or Slurm submission was started.",
        "",
        f"- status: `{summary['status']}`",
        f"- prompts: `{summary['prompt_count']}`",
        f"- duplicate prompt ids: `{summary['duplicate_prompt_id_count']}`",
        f"- forbidden violations: `{summary['forbidden_violation_count']}`",
        f"- prompt bank sha256: `{summary['prompt_bank_sha256']}`",
    ]
    (args.output_dir / "prompt_bank_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

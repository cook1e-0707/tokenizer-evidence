from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/natural_evidence_v2/prompts/"
    "wp2_controlled_natural_prompt_family_scaffold_20260508_2123/qwen_v2_dev_prompts.jsonl"
)

FORBIDDEN_TERMS = (
    "FIELD=",
    "SECTION=",
    "TOPIC=",
    "PAYLOAD",
    "CERT",
    "EVIDENCE",
    "CARRIER",
    "OWNER",
    "fingerprint",
    "watermark",
    "bucket",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed template responses for natural_evidence_v2 WP3 density "
            "preflight. This is not model generation, training, E2E, FAR, or a "
            "paper-facing positive claim."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument(
        "--rows-per-family",
        type=int,
        default=64,
        help=(
            "Balanced row count per family. Set to 0 to use the first --max-rows "
            "rows in file order."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL row must be an object: {path}:{len(rows) + 1}")
                rows.append(payload)
    return rows


def select_prompt_rows(rows: list[dict[str, Any]], *, max_rows: int, rows_per_family: int) -> list[dict[str, Any]]:
    if rows_per_family <= 0:
        return rows[:max(0, max_rows)] if max_rows else rows
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        family_id = str(row.get("family_id", ""))
        if counts.get(family_id, 0) >= rows_per_family:
            continue
        selected.append(row)
        counts[family_id] = counts.get(family_id, 0) + 1
    if max_rows and len(selected) > max_rows:
        selected = selected[:max_rows]
    return selected


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def forbidden_hits(text: str) -> list[str]:
    upper_text = text.upper()
    return [term for term in FORBIDDEN_TERMS if term.upper() in upper_text]


def response_for_family(family_id: str) -> str:
    if family_id == "F2_8_step_checklist":
        return (
            "First, check the goal and list the basic tasks with the group. "
            "Second, choose a simple order, and make the first step easy. "
            "Third, review the supplies, or set a short backup plan. "
            "Then, check timing with each person and list any limits. "
            "Also, make one shared note, but keep it brief. "
            "Instead, set aside extra details, so the checklist stays useful. "
            "Finally, review the plan with the group and choose the next action. "
            "Now, finish by checking what changed and what to do again."
        )
    if family_id == "F3_6_point_comparison":
        return (
            "First, review the main tradeoff with the group, and list the cost. "
            "Second, choose the option that usually fits the schedule. "
            "Third, check which choice works with limited time or space. "
            "Also, compare who benefits, but avoid adding extra steps. "
            "However, set a clear fallback, so the group can adjust. "
            "Finally, make the decision now and review it again later."
        )
    if family_id == "F4_concise_advice_with_transitions":
        return (
            "First, check the goal with the group and list the smallest useful action. "
            "Also, choose one routine that usually works and set it up simply. "
            "Then, review progress with a short note, or make a quick adjustment. "
            "However, keep the plan flexible, so people can continue without stress. "
            "Instead, finish one clear task now and choose the next step. "
            "Finally, review what worked and make the process easier again."
        )
    return (
        "First, check the main goal and review the current routine with the group. "
        "Second, set a simple order for the work, and make the first step easy. "
        "Third, choose one habit to improve, but keep the plan normal. "
        "Also, list common problems, or review them with a short note. "
        "Then, make a small checklist with one task for each person. "
        "However, keep the routine flexible, so the group can adjust. "
        "Finally, choose a review time and set a clear next action. "
        "Now, finish by checking what worked and what to change again."
    )


def build_response_row(prompt_row: Mapping[str, Any]) -> dict[str, Any]:
    family_id = str(prompt_row.get("family_id", ""))
    response_text = response_for_family(family_id)
    hits = forbidden_hits(response_text)
    if hits:
        raise ValueError(f"template response contains forbidden terms {hits}: {family_id}")
    prompt_id = str(prompt_row.get("prompt_id", ""))
    return {
        "schema_name": "natural_evidence_v2_wp3_template_response_row_v1",
        "protocol_id": str(prompt_row.get("protocol_id", "")),
        "prompt_set_id": str(prompt_row.get("prompt_set_id", "")),
        "prompt_id": prompt_id,
        "split": str(prompt_row.get("split", "")),
        "family_id": family_id,
        "response_id": f"{prompt_id}_template_density_preflight",
        "response_source": "template_density_preflight_no_model_generation",
        "artifact_role": "template_density_preflight_not_model_output",
        "prompt_text_sha256": str(prompt_row.get("prompt_text_sha256", "")),
        "response_text": response_text,
        "response_text_sha256": sha256_text(response_text),
        "model_calls_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
    }


def main() -> None:
    args = parse_args()
    input_jsonl = resolve_path(args.input_jsonl)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    all_prompt_rows = read_jsonl(input_jsonl)
    prompt_rows = select_prompt_rows(
        all_prompt_rows,
        max_rows=max(0, int(args.max_rows)),
        rows_per_family=max(0, int(args.rows_per_family)),
    )
    response_rows = [build_response_row(row) for row in prompt_rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_jsonl = output_dir / "qwen_v2_wp3_template_density_responses.jsonl"
    write_jsonl(responses_jsonl, response_rows)
    family_counts: dict[str, int] = {}
    for row in response_rows:
        family_counts[row["family_id"]] = family_counts.get(row["family_id"], 0) + 1
    summary = {
        "schema_name": "natural_evidence_v2_wp3_template_response_summary_v1",
        "status": "WP3_TEMPLATE_FIXED_RESPONSE_DENSITY_PREFLIGHT_WRITTEN_NOT_MODEL_OUTPUT",
        "input_jsonl": str(input_jsonl),
        "responses_jsonl": str(responses_jsonl),
        "row_count": len(response_rows),
        "family_counts": dict(sorted(family_counts.items())),
        "response_source": "template_density_preflight_no_model_generation",
        "artifact_role": "template_density_preflight_not_model_output",
        "model_calls_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Run WP3 fixed-artifact density preflight through the Slurm audit wrapper; "
            "do not treat template density as model-output density."
        ),
    }
    write_json(output_dir / "template_response_summary.json", summary)
    write_text_new(
        output_dir / "README.md",
        "\n".join(
            [
                "# WP3 Template Fixed Responses",
                "",
                "Template-only response artifact for WP3 detector density preflight.",
                "These rows are not model generations and do not unlock WP4 or training.",
                "",
            ]
        ),
    )
    print(json.dumps({"status": summary["status"], "row_count": len(response_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()

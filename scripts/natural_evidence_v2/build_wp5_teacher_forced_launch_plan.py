from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.run_wp3_restricted_step_label_density_audit import (  # noqa: E402
    DEFAULT_CONFIG,
    forbidden_terms_in_text,
    read_yaml,
)


DEFAULT_PRIMARY_BANK = ROOT / "results/natural_evidence_v2/buckets/qwen_v2_primary_2way_bank.jsonl"
DEFAULT_WP4_CONTRACT = (
    ROOT
    / "results/natural_evidence_v2/status/wp4_prompt_local_payload_contract_20260509_0611/"
    "wp4_prompt_local_payload_contract.json"
)
DEFAULT_PROMPTS = (
    ROOT
    / "results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_plan_20260509_0355/"
    "restricted_step_label_strict_density_audit_prompts.jsonl"
)
DEFAULT_MODEL_OUTPUTS = (
    ROOT
    / "results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_audit_850885/"
    "restricted_step_label_model_outputs.jsonl"
)
DEFAULT_RESPONSE_AUDIT = (
    ROOT
    / "results/natural_evidence_v2/status/wp3_r1_strict_density_expansion_audit_850885/"
    "restricted_step_label_response_audit.jsonl"
)
STEP_LINE_RE = re.compile(
    r"^Step\s+(?P<step>[1-9]|1[0-6]):\s+(?P<first_word>[A-Z][A-Za-z'-]+)\b(?P<rest>.*)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build artifact-only natural_evidence_v2 WP5 teacher-forced "
            "target-mass training/scoring launch-plan artifacts. This script "
            "does not train, score with a model, submit Slurm, run E2E, "
            "aggregate FAR, or make positive claims."
        )
    )
    parser.add_argument("--primary-bank", type=Path, default=DEFAULT_PRIMARY_BANK)
    parser.add_argument("--wp4-contract", type=Path, default=DEFAULT_WP4_CONTRACT)
    parser.add_argument("--prompts-jsonl", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--model-outputs-jsonl", type=Path, default=DEFAULT_MODEL_OUTPUTS)
    parser.add_argument("--response-audit-jsonl", type=Path, default=DEFAULT_RESPONSE_AUDIT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="wp3_r1_dev")
    parser.add_argument("--max-responses", type=int, default=512)
    parser.add_argument("--trainer-path", type=Path, default=ROOT / "scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py")
    parser.add_argument("--scorer-path", type=Path, default=ROOT / "scripts/natural_evidence_v2/score_wp5_teacher_forced_bucket_mass.py")
    parser.add_argument("--slurm-path", type=Path, default=ROOT / "scripts/natural_evidence_v2/slurm/wp5_teacher_forced_train_and_score.sbatch")
    parser.add_argument("--allowlist", type=Path, default=ROOT / "configs/natural_evidence_v2/run_allowlist.yaml")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_primary_bank(config: Mapping[str, Any], path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"primary bank must contain exactly one row: {path}")
    bank = rows[0]
    if bank.get("schema_name") != "natural_evidence_v2_primary_2way_micro_slot_bank_v1":
        raise ValueError("primary bank schema mismatch")
    if bank.get("mass_gate") != "WP3_R2_PRIMARY_SELECTION_PASS":
        raise ValueError("primary bank did not pass WP3-R2 primary selection")
    if not bank.get("tokenizer_stable"):
        raise ValueError("primary bank tokenizer_stable is not true")
    buckets = {
        "0": [str(item) for item in bank.get("bucket_0_surfaces", [])],
        "1": [str(item) for item in bank.get("bucket_1_surfaces", [])],
    }
    if not buckets["0"] or not buckets["1"]:
        raise ValueError("primary bank must define both buckets")
    if set(buckets["0"]) & set(buckets["1"]):
        raise ValueError("primary bank buckets overlap")
    for bucket_id, surfaces in buckets.items():
        for surface in surfaces:
            if not re.fullmatch(r"[A-Z][A-Za-z'-]+", surface):
                raise ValueError(f"invalid bucket surface {surface!r} in bucket {bucket_id}")
            hits = forbidden_terms_in_text(config, surface)
            if hits:
                raise ValueError(f"forbidden surface hit {hits} in {surface!r}")
    bank["buckets"] = buckets
    return bank


def payload_bits_from_wp4(contract: Mapping[str, Any]) -> list[int]:
    payload = contract.get("payload", {})
    bits = list(payload.get("payload_bits_msb_first", [])) + list(payload.get("checksum_bits_msb_first", []))
    if len(bits) != 16:
        raise ValueError("WP4 contract must define 8 payload bits plus 8 checksum bits")
    normalized = [int(bit) for bit in bits]
    if any(bit not in {0, 1} for bit in normalized):
        raise ValueError("WP4 bits must be binary")
    return normalized


def parse_response_lines(response_text: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for raw_line in response_text.splitlines():
        match = STEP_LINE_RE.match(raw_line.strip())
        if match is None:
            continue
        step = int(match.group("step"))
        first_word = str(match.group("first_word"))
        rest = str(match.group("rest")).strip()
        parsed.append(
            {
                "step_index": step,
                "original_line": raw_line.strip(),
                "original_first_word": first_word,
                "original_rest": rest,
            }
        )
    parsed.sort(key=lambda row: int(row["step_index"]))
    return parsed


def normalize_action_phrase(first_word: str, rest: str) -> str:
    phrase = f"{first_word}{(' ' + rest) if rest else ''}".strip()
    changed = True
    while changed:
        changed = False
        if phrase.endswith((".", "!", "?")):
            phrase = phrase[:-1].rstrip()
            changed = True
        if phrase.endswith(('"', "'", "\u201d", "\u2019")) and len(phrase) >= 2 and phrase[-2] in ".!?":
            phrase = f"{phrase[:-2].rstrip()}{phrase[-1]}"
            changed = True
    return phrase[:1].lower() + phrase[1:]


def repaired_line(step_index: int, surface: str, first_word: str, rest: str) -> str:
    action = normalize_action_phrase(first_word, rest)
    bridge_by_surface = {
        "Set": "Set a routine to",
        "Plan": "Plan a routine to",
        "Create": "Create a simple way to",
        "Prepare": "Prepare a simple way to",
    }
    bridge = bridge_by_surface.get(surface, surface)
    return f"Step {step_index}: {bridge} {action}."


def choose_surface(bucket_surfaces: Sequence[str], *, response_index: int, step_index: int) -> str:
    if not bucket_surfaces:
        raise ValueError("cannot choose from an empty bucket")
    return str(bucket_surfaces[(response_index + step_index - 1) % len(bucket_surfaces)])


def build_prompt_index(prompt_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in prompt_rows:
        prompt_id = str(row.get("prompt_id", ""))
        if prompt_id:
            output[prompt_id] = dict(row)
    return output


def audit_by_response(response_audit_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in response_audit_rows:
        response_id = str(row.get("response_id", ""))
        if response_id:
            output[response_id] = dict(row)
    return output


def target_response_and_slots(
    *,
    response_row: Mapping[str, Any],
    prompt_row: Mapping[str, Any],
    payload_bits: Sequence[int],
    buckets: Mapping[str, Sequence[str]],
    response_index: int,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    parsed = parse_response_lines(str(response_row.get("response_text", "")))
    if len(parsed) != 16 or [int(row["step_index"]) for row in parsed] != list(range(1, 17)):
        raise ValueError("response does not contain exactly Step 1..Step 16 lines")

    target_lines: list[str] = []
    slot_targets: list[dict[str, Any]] = []
    warnings: list[str] = []
    prefix_lines: list[str] = []
    for line_info, bit in zip(parsed, payload_bits, strict=True):
        step_index = int(line_info["step_index"])
        target_bucket_id = str(int(bit))
        surface = choose_surface(buckets[target_bucket_id], response_index=response_index, step_index=step_index)
        line = repaired_line(
            step_index,
            surface,
            str(line_info["original_first_word"]),
            str(line_info["original_rest"]),
        )
        if line.endswith("..") or line.endswith(".\".") or line.endswith(".'."):
            warnings.append("target_line_has_terminal_punctuation_conflict")
        response_prefix = "\n".join(prefix_lines + [f"Step {step_index}: "])
        target_lines.append(line)
        slot_targets.append(
            {
                "assistant_prefix_before_surface": response_prefix,
                "bit_index": step_index - 1,
                "bit_role": "payload" if step_index <= 8 else "checksum",
                "bucket_0_surfaces": list(buckets["0"]),
                "bucket_1_surfaces": list(buckets["1"]),
                "dialog_prefix_text": (
                    str(prompt_row.get("prompt_text", ""))
                    + "\n\n"
                    + response_prefix
                ),
                "line_repair_policy": "surface_specific_bridge_to_original_action_phrase_v1",
                "non_target_bucket_id": "1" if target_bucket_id == "0" else "0",
                "original_first_word": str(line_info["original_first_word"]),
                "original_line": str(line_info["original_line"]),
                "prompt_text": str(prompt_row.get("prompt_text", "")),
                "prompt_id": str(response_row.get("prompt_id", "")),
                "response_id": str(response_row.get("response_id", "")),
                "schema_name": "natural_evidence_v2_wp5_teacher_forced_score_slot_v1",
                "slot_index": step_index,
                "target_bit": int(bit),
                "target_bucket_id": target_bucket_id,
                "target_line": line,
                "target_surface": surface,
            }
        )
        prefix_lines.append(line)
    return "\n".join(target_lines), slot_targets, warnings


def allowlist_has_wp5_training_entry(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return "v2_wp5_teacher_forced_train_and_score" in text and "wp5_teacher_forced_train_and_score.sbatch" in text


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output dir: {output_dir}")
    output_dir.mkdir(parents=True)

    config = read_yaml(resolve(args.config))
    primary_bank_path = resolve(args.primary_bank)
    wp4_contract_path = resolve(args.wp4_contract)
    prompts_path = resolve(args.prompts_jsonl)
    outputs_path = resolve(args.model_outputs_jsonl)
    response_audit_path = resolve(args.response_audit_jsonl)
    allowlist_path = resolve(args.allowlist)

    bank = load_primary_bank(config, primary_bank_path)
    wp4_contract = read_json(wp4_contract_path)
    payload_bits = payload_bits_from_wp4(wp4_contract)
    prompts_by_id = build_prompt_index(read_jsonl(prompts_path))
    response_audit = audit_by_response(read_jsonl(response_audit_path))

    selected_outputs: list[dict[str, Any]] = []
    skipped = Counter()
    for row in read_jsonl(outputs_path):
        if row.get("split") != args.split:
            skipped["split_mismatch"] += 1
            continue
        response_id = str(row.get("response_id", ""))
        audit = response_audit.get(response_id, {})
        if audit and int(audit.get("detected_structural_slots", 0)) != 16:
            skipped["not_16_slots_by_audit"] += 1
            continue
        try:
            parsed = parse_response_lines(str(row.get("response_text", "")))
        except ValueError:
            skipped["parse_error"] += 1
            continue
        if len(parsed) != 16 or [int(item["step_index"]) for item in parsed] != list(range(1, 17)):
            skipped["not_exact_step_1_to_16"] += 1
            continue
        if str(row.get("prompt_id", "")) not in prompts_by_id:
            skipped["missing_prompt"] += 1
            continue
        selected_outputs.append(dict(row))
        if len(selected_outputs) >= args.max_responses:
            break

    if not selected_outputs:
        raise ValueError("no complete responses selected for WP5 plan")

    protected_rows: list[dict[str, Any]] = []
    task_only_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    repair_warning_counts = Counter()

    for response_index, response_row in enumerate(selected_outputs):
        prompt_row = prompts_by_id[str(response_row["prompt_id"])]
        target_response, slot_targets, warnings = target_response_and_slots(
            response_row=response_row,
            prompt_row=prompt_row,
            payload_bits=payload_bits,
            buckets=bank["buckets"],
            response_index=response_index,
        )
        repair_warning_counts.update(warnings)
        protected_row = {
            "artifact_role": "wp5_teacher_forced_training_plan_not_launched",
            "e2e_eval_started": False,
            "family_id": response_row.get("family_id"),
            "model_generation_started": False,
            "model_scoring_started": False,
            "paper_claim_allowed": False,
            "prompt_id": response_row.get("prompt_id"),
            "prompt_text": prompt_row.get("prompt_text"),
            "response_id": response_row.get("response_id"),
            "schema_name": "natural_evidence_v2_wp5_teacher_forced_training_row_v1",
            "slot_count": 16,
            "slot_targets": slot_targets,
            "source_response_sha256": response_row.get("response_text_sha256"),
            "split": args.split,
            "target_response_sha256": sha256_text(target_response),
            "target_response_text": target_response,
            "training_arm": "qwen_protected_micro_slot_lora",
            "training_started": False,
        }
        task_only_response = str(response_row.get("response_text", ""))
        task_only_row = {
            "artifact_role": "wp5_task_only_training_plan_not_launched",
            "bucket_loss_enabled": False,
            "e2e_eval_started": False,
            "family_id": response_row.get("family_id"),
            "model_generation_started": False,
            "model_scoring_started": False,
            "paper_claim_allowed": False,
            "prompt_id": response_row.get("prompt_id"),
            "prompt_text": prompt_row.get("prompt_text"),
            "response_id": response_row.get("response_id"),
            "schema_name": "natural_evidence_v2_wp5_teacher_forced_training_row_v1",
            "slot_count": 16,
            "source_response_sha256": response_row.get("response_text_sha256"),
            "split": args.split,
            "target_response_sha256": sha256_text(task_only_response),
            "target_response_text": task_only_response,
            "training_arm": "qwen_task_only_lora",
            "training_started": False,
        }
        protected_rows.append(protected_row)
        task_only_rows.append(task_only_row)
        score_rows.extend(slot_targets)
        if len(preview_rows) < 16:
            preview_rows.append(
                {
                    "original_response_text": response_row.get("response_text"),
                    "prompt_id": response_row.get("prompt_id"),
                    "prompt_text": prompt_row.get("prompt_text"),
                    "schema_name": "natural_evidence_v2_wp5_training_preview_v1",
                    "target_response_text": target_response,
                }
            )

    trainer_exists = resolve(args.trainer_path).exists()
    scorer_exists = resolve(args.scorer_path).exists()
    slurm_exists = resolve(args.slurm_path).exists()
    allowlist_entry_exists = allowlist_has_wp5_training_entry(allowlist_path)
    launch_blockers: list[str] = []
    if not trainer_exists:
        launch_blockers.append("missing_v2_margin_trainer")
    if not scorer_exists:
        launch_blockers.append("missing_v2_teacher_forced_scorer")
    if not slurm_exists:
        launch_blockers.append("missing_v2_wp5_slurm_wrapper")
    if not allowlist_entry_exists:
        launch_blockers.append("missing_enabled_allowlist_entry")
    if repair_warning_counts:
        launch_blockers.append("target_repair_warnings_require_review")

    launch_gate_status = "PASS_READY_TO_SUBMIT_ONE_ALLOWLISTED_WP5_SLURM_JOB" if not launch_blockers else "FAIL_NOT_READY_TO_TRAIN"

    protected_path = output_dir / "wp5_protected_training_rows.jsonl"
    task_only_path = output_dir / "wp5_task_only_training_rows.jsonl"
    score_path = output_dir / "wp5_teacher_forced_score_rows.jsonl"
    preview_path = output_dir / "wp5_training_examples_preview.jsonl"
    summary_path = output_dir / "wp5_teacher_forced_launch_plan_summary.json"

    write_jsonl(protected_path, protected_rows)
    write_jsonl(task_only_path, task_only_rows)
    write_jsonl(score_path, score_rows)
    write_jsonl(preview_path, preview_rows)

    summary = {
        "allowlist_entry_exists": allowlist_entry_exists,
        "allowlist_path": str(allowlist_path),
        "artifact_role": "wp5_teacher_forced_launch_plan_not_training",
        "bucket_bank_id": bank.get("bank_id"),
        "bucket_surfaces": bank["buckets"],
        "e2e_eval_started": False,
        "input_hashes": {
            "model_outputs_sha256": sha256_file(outputs_path),
            "primary_bank_sha256": sha256_file(primary_bank_path),
            "prompts_sha256": sha256_file(prompts_path),
            "response_audit_sha256": sha256_file(response_audit_path),
            "wp4_contract_sha256": sha256_file(wp4_contract_path),
        },
        "launch_blockers": launch_blockers,
        "launch_gate_status": launch_gate_status,
        "model_generation_started": False,
        "model_scoring_started": False,
        "next_allowed_action": (
            "Implement/review the missing v2 WP5 margin trainer, teacher-forced scorer, "
            "Slurm wrapper, and allowlist entry; submit exactly one Slurm job only if "
            "launch_gate_status becomes PASS."
            if launch_blockers
            else "Submit exactly one allowlisted Chimera Slurm WP5 teacher-forced train-and-score job."
        ),
        "not_full_far": True,
        "not_payload_recovery": True,
        "output_files": {
            "protected_training_rows": str(protected_path),
            "score_rows": str(score_path),
            "summary": str(summary_path),
            "task_only_training_rows": str(task_only_path),
            "training_examples_preview": str(preview_path),
        },
        "paper_claim_allowed": False,
        "payload_bits_msb_first": payload_bits,
        "protected_training_row_count": len(protected_rows),
        "repair_warning_counts": dict(repair_warning_counts),
        "schema_name": "natural_evidence_v2_wp5_teacher_forced_launch_plan_summary_v1",
        "score_row_count": len(score_rows),
        "scorer_exists": scorer_exists,
        "selected_split": args.split,
        "selected_source_response_count": len(selected_outputs),
        "skipped_counts": dict(skipped),
        "slurm_wrapper_exists": slurm_exists,
        "task_only_training_row_count": len(task_only_rows),
        "teacher_forced_gate_targets": {
            "median_target_margin": ">0",
            "protected_target_bucket_mass_lift_vs_base": ">=+0.15",
            "protected_target_bucket_mass_lift_vs_task_only": ">=+0.10",
            "target_bucket_rank1_rate": ">=0.70",
            "task_only_target_bucket_mass_lift_vs_base": "not_significantly_positive",
        },
        "trainer_exists": trainer_exists,
        "training_started": False,
        "v1_trainer_approved_for_v2": False,
        "v1_trainer_approval_note": (
            "The natural_evidence_v1 trainer is not treated as approved for v2 because WP5 "
            "requires micro-slot margin loss, slot exact-token CE masking, and v2 plan schemas."
        ),
        "wp5_status": "PLAN_BUILT_ARTIFACT_ONLY",
    }
    write_json(summary_path, summary)
    write_text_new(
        output_dir / "README.md",
        "# WP5 Teacher-Forced Launch Plan\n\n"
        "Artifact-only natural_evidence_v2 WP5 launch plan. This directory contains "
        "planned protected/task-only training rows and teacher-forced scoring rows. "
        "It does not contain training results, E2E results, payload recovery, FAR, or "
        "paper-facing positive claims.\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

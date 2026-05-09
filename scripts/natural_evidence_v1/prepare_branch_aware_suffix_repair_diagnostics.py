from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.natural_evidence_v1.common import stable_hash_hex, write_csv, write_json, write_jsonl


SCHEMA_NAME = "natural_evidence_v1_branch_aware_suffix_repair_preparation_v1"
BRANCH_PLAN_SCHEMA = "natural_evidence_v1_branch_aware_scoring_plan_row_v1"
REPAIR_EXAMPLE_SCHEMA = "natural_evidence_v1_regenerated_suffix_repair_input_v1"
TOKEN_CLASS_FIELDS = [
    "token_class",
    "plan_rows",
    "compatible_non_target_rows",
    "observed_token_not_candidate_rows",
    "observed_bucket_not_compatible_rows",
    "prefix_miss_rows",
    "protected_rows",
    "task_only_rows",
    "raw_rows",
]
READINESS_FIELDS = [
    "gate",
    "status",
    "evidence",
    "next_action",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare artifact-only branch-aware compatibility and regenerated/"
            "local-suffix repair diagnostics. This builds scoring/repair inputs "
            "from existing R1 replay artifacts and train JSONL only. It does not "
            "load a model/tokenizer, generate text, train, run E2E, or claim FAR."
        )
    )
    parser.add_argument("--selector-preflight-dir", required=True)
    parser.add_argument("--r1-replay-dir", required=True)
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--payload-ids", default="P0421,P1729")
    parser.add_argument("--max-plan-rows", type=int, default=512)
    parser.add_argument("--max-repair-examples", type=int, default=240)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            yield payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value) == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _candidate_bucket_tokens(example: Mapping[str, Any], bucket_id: str) -> list[dict[str, Any]]:
    buckets = example.get("candidate_bucket_token_texts", {})
    if not isinstance(buckets, dict):
        return []
    rows = buckets.get(str(bucket_id), [])
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _load_train_examples(train_data_dir: Path, payload_ids: Sequence[str]) -> dict[tuple[str, str, int], dict[str, Any]]:
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for payload_id in payload_ids:
        train_path = train_data_dir / payload_id / "variable_radix_train.jsonl"
        if not train_path.is_file() or train_path.stat().st_size == 0:
            raise FileNotFoundError(f"missing train JSONL for {payload_id}: {train_path}")
        for row in _iter_jsonl(train_path):
            prompt_id = str(row.get("prompt_id", ""))
            positions = row.get("eligible_positions", [])
            if not isinstance(positions, list):
                continue
            for prompt_slot, position in enumerate(positions):
                if not isinstance(position, dict):
                    continue
                output[(payload_id, prompt_id, prompt_slot)] = {
                    "payload_id": payload_id,
                    "prompt_id": prompt_id,
                    "prompt_slot": prompt_slot,
                    "prompt": str(row.get("prompt", "")),
                    "user_probe": str(row.get("user_probe", "")),
                    "response_text": str(row.get("response_text", "")),
                    "prompt_split": str(row.get("prompt_split", "")),
                    "example_role": str(row.get("example_role", "")),
                    "token_index": _as_int(position.get("token_index", -1), -1),
                    "frame_index": _as_int(position.get("frame_index", -1), -1),
                    "frame_digit_index": _as_int(position.get("frame_digit_index", -1), -1),
                    "target_bucket": str(position.get("target_bucket", "")),
                    "compatible_bucket_ids": [str(value) for value in position.get("compatible_bucket_ids", [])],
                    "target_bucket_token_ids": [int(value) for value in position.get("target_bucket_token_ids", [])],
                    "candidate_token_ids": [int(value) for value in position.get("candidate_token_ids", [])],
                    "bank_entry_id": str(position.get("bank_entry_id", "")),
                    "entry_key": str(position.get("entry_key", "")),
                }
    return output


def _example_priority(row: Mapping[str, Any]) -> tuple[int, str, str, int]:
    condition = str(row.get("model_condition", ""))
    drift = str(row.get("drift_reason", ""))
    # Prefer protected/task-only failures over raw examples; keep raw for null context.
    condition_rank = {"protected_trained": 0, "task_only_lora": 1, "raw": 2}.get(condition, 3)
    drift_rank = {
        "compatible_non_target": 0,
        "observed_bucket_not_compatible": 1,
        "observed_token_not_candidate_set": 2,
        "suffix_prefix_not_found": 3,
        "exact_prefix_mismatch": 4,
    }.get(drift, 5)
    return (condition_rank, drift_rank, str(row.get("match_policy", "")), _as_int(row.get("query_index", 0)))


def _select_examples(examples: Sequence[Mapping[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int, str]] = set()
    for row in sorted(examples, key=_example_priority):
        key = (
            str(row.get("model_condition", "")),
            str(row.get("expected_payload_id", "")),
            str(row.get("prompt_id", "")),
            _as_int(row.get("prompt_slot", 0)),
            str(row.get("match_policy", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(dict(row))
        if len(selected) >= max_rows:
            break
    return selected


def _make_branch_plan_row(
    *,
    example: Mapping[str, Any],
    train_row: Mapping[str, Any] | None,
    selector_contract: Mapping[str, Any],
) -> dict[str, Any]:
    target_bucket = str(example.get("target_bucket", ""))
    observed_bucket = str(example.get("bucket_id", ""))
    prompt_text = str(example.get("prompt", "")) or ("" if train_row is None else str(train_row.get("prompt", "")))
    user_probe = str(example.get("user_probe", "")) or ("" if train_row is None else str(train_row.get("user_probe", "")))
    response_text = (
        str(example.get("generated_response_text", ""))
        or str(example.get("response_text", ""))
        or ("" if train_row is None else str(train_row.get("response_text", "")))
    )
    row_id = stable_hash_hex(
        [
            "branch_aware_plan",
            example.get("model_condition", ""),
            example.get("expected_payload_id", ""),
            example.get("prompt_id", ""),
            example.get("prompt_slot", ""),
            example.get("match_policy", ""),
            example.get("drift_reason", ""),
        ]
    )[:24]
    return {
        "schema_name": BRANCH_PLAN_SCHEMA,
        "row_id": row_id,
        "protocol_id": selector_contract.get("protocol_id", "natural_evidence_v1"),
        "selector_id": dict(selector_contract.get("selector", {})).get("selector_id", ""),
        "selector_contract_status": selector_contract.get("contract_status", ""),
        "scoring_status": "PLANNED_NOT_SCORED",
        "requires_slurm_for_chimera_cpu_gpu": True,
        "model_loading_started": False,
        "generation_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "model_condition": str(example.get("model_condition", "")),
        "payload_id": str(example.get("payload_id", "")),
        "expected_payload_id": str(example.get("expected_payload_id", "")),
        "seed": str(example.get("seed", "")),
        "prompt_id": str(example.get("prompt_id", "")),
        "prompt_slot": _as_int(example.get("prompt_slot", 0)),
        "query_index": _as_int(example.get("query_index", 0)),
        "match_policy": str(example.get("match_policy", "")),
        "drift_reason": str(example.get("drift_reason", "")),
        "observed_token_id": example.get("observed_token_id", ""),
        "observed_token_text": str(example.get("observed_token_text", "")),
        "observed_token_class": str(example.get("observed_token_class", "")),
        "observed_bucket": observed_bucket,
        "target_bucket": target_bucket,
        "target_bucket_tokens": _candidate_bucket_tokens(example, target_bucket),
        "observed_bucket_tokens": _candidate_bucket_tokens(example, observed_bucket),
        "compatible_bucket_ids": [str(value) for value in example.get("compatible_bucket_ids", [])],
        "prompt": prompt_text,
        "user_probe": user_probe,
        "original_response_text": response_text,
        "response_text_source": (
            "generated_response_text"
            if str(example.get("generated_response_text", ""))
            else "example_response_text"
            if str(example.get("response_text", ""))
            else "train_reference_response_text"
            if train_row is not None
            else "missing"
        ),
        "token_index": "" if train_row is None else train_row.get("token_index", ""),
        "frame_index": example.get("frame_index", ""),
        "frame_digit_index": example.get("frame_digit_index", ""),
        "branch_aware_tasks": [
            "score suffix-preserving compatibility for original suffix",
            "score prefix+candidate with short branch continuation",
            "compare local regenerated suffix naturalness/coherence",
        ],
        "result_claim": "branch_aware_scoring_plan_not_scored_not_payload_recovery_not_far",
        "paper_claim_allowed": False,
    }


def _make_repair_example(plan_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": REPAIR_EXAMPLE_SCHEMA,
        "row_id": plan_row.get("row_id", ""),
        "repair_status": "INPUT_READY_NOT_REGENERATED",
        "requires_slurm_for_chimera_cpu_gpu": True,
        "training_started": False,
        "generation_started": False,
        "e2e_eval_started": False,
        "prompt_id": plan_row.get("prompt_id", ""),
        "prompt_slot": plan_row.get("prompt_slot", ""),
        "expected_payload_id": plan_row.get("expected_payload_id", ""),
        "model_condition": plan_row.get("model_condition", ""),
        "seed": plan_row.get("seed", ""),
        "match_policy": plan_row.get("match_policy", ""),
        "drift_reason": plan_row.get("drift_reason", ""),
        "prompt": plan_row.get("prompt", ""),
        "user_probe": plan_row.get("user_probe", ""),
        "original_response_text": plan_row.get("original_response_text", ""),
        "token_index": plan_row.get("token_index", ""),
        "target_bucket": plan_row.get("target_bucket", ""),
        "target_bucket_tokens": plan_row.get("target_bucket_tokens", []),
        "observed_token_text": plan_row.get("observed_token_text", ""),
        "local_suffix_repair_tasks": [
            "construct prefix+target-token local continuation",
            "repair only the short suffix window after the evidence token",
            "preserve user-task semantics and natural style",
            "emit repaired response plus offset/provenance metadata",
        ],
        "result_claim": "regenerated_suffix_repair_input_not_generated_not_training",
        "paper_claim_allowed": False,
    }


def _token_class_rows(plan_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in plan_rows:
        token_class = str(row.get("observed_token_class", "")) or "unknown"
        counters[token_class]["plan_rows"] += 1
        drift = str(row.get("drift_reason", ""))
        if drift == "compatible_non_target":
            counters[token_class]["compatible_non_target_rows"] += 1
        elif drift == "observed_token_not_candidate_set":
            counters[token_class]["observed_token_not_candidate_rows"] += 1
        elif drift == "observed_bucket_not_compatible":
            counters[token_class]["observed_bucket_not_compatible_rows"] += 1
        elif "prefix" in drift:
            counters[token_class]["prefix_miss_rows"] += 1
        condition = str(row.get("model_condition", ""))
        if condition == "protected_trained":
            counters[token_class]["protected_rows"] += 1
        elif condition == "task_only_lora":
            counters[token_class]["task_only_rows"] += 1
        elif condition == "raw":
            counters[token_class]["raw_rows"] += 1
    rows: list[dict[str, Any]] = []
    for token_class, counter in sorted(counters.items()):
        rows.append(
            {
                "token_class": token_class,
                "plan_rows": counter["plan_rows"],
                "compatible_non_target_rows": counter["compatible_non_target_rows"],
                "observed_token_not_candidate_rows": counter["observed_token_not_candidate_rows"],
                "observed_bucket_not_compatible_rows": counter["observed_bucket_not_compatible_rows"],
                "prefix_miss_rows": counter["prefix_miss_rows"],
                "protected_rows": counter["protected_rows"],
                "task_only_rows": counter["task_only_rows"],
                "raw_rows": counter["raw_rows"],
            }
        )
    return rows


def _readiness_rows(summary: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "gate": "branch_aware_scoring_inputs",
            "status": "READY_FOR_SLURM_SCORING_NOT_SCORED",
            "evidence": f"planned rows={summary['planned_branch_aware_rows']}",
            "next_action": "Implement/run Slurm-scored branch-aware compatibility command; no training.",
        },
        {
            "gate": "regenerated_suffix_repair_inputs",
            "status": "READY_FOR_REPAIR_DRY_RUN_NOT_GENERATED",
            "evidence": f"repair examples={summary['regenerated_suffix_repair_example_rows']}",
            "next_action": "Construct local-suffix repair dry-run artifact before training.",
        },
        {
            "gate": "selector_contract_active",
            "status": "BLOCKED_DRAFT_ONLY",
            "evidence": str(summary["selector_contract_status"]),
            "next_action": "Keep selector inactive until branch-aware, suffix repair, target-mass, and lockbox gates pass.",
        },
        {
            "gate": "training_or_e2e",
            "status": "BLOCKED",
            "evidence": "training_allowed=false; e2e_eval_started=false",
            "next_action": "No training or E2E rerun.",
        },
    ]


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Branch-Aware Compatibility And Local-Suffix Repair Preparation",
        "",
        "This is an artifact-only preparation step. It builds diagnostic inputs only; it does not score branch-aware compatibility, regenerate suffixes, train, generate, rerun E2E, claim payload recovery, or estimate FAR.",
        "",
        "## Status",
        "",
        f"`{summary['status']}`",
        "",
        "## Outputs",
        "",
        "- `branch_aware_compatibility_summary.json`",
        "- `branch_aware_compatibility_by_token_class.csv`",
        "- `branch_aware_compatibility_scoring_plan.jsonl`",
        "- `regenerated_suffix_repair_manifest.json`",
        "- `regenerated_suffix_repair_examples.jsonl`",
        "- `branch_aware_suffix_repair_readiness.csv`",
        "",
        "## Counts",
        "",
        f"- planned branch-aware rows: `{summary['planned_branch_aware_rows']}`",
        f"- regenerated/local-suffix repair examples: `{summary['regenerated_suffix_repair_example_rows']}`",
        f"- source R1 examples: `{summary['source_r1_example_rows']}`",
        f"- train metadata matches: `{summary['train_metadata_matched_rows']}`",
        "",
        "## Next Allowed Action",
        "",
        "Run a Slurm-scored branch-aware compatibility diagnostic or construct a local-suffix repair dry-run from these inputs. Training remains forbidden.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_preparation(
    *,
    selector_preflight_dir: Path,
    r1_replay_dir: Path,
    train_data_dir: Path,
    output_dir: Path,
    payload_ids: Sequence[str],
    max_plan_rows: int,
    max_repair_examples: int,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "branch_aware_compatibility_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing branch-aware preparation: {summary_path}")
    selector_summary_path = selector_preflight_dir / "selector_contract_training_target_preflight_summary.json"
    selector_contract_path = selector_preflight_dir / "selector_precommit_contract_draft.json"
    r1_examples_path = r1_replay_dir / "prefix_conditioned_selector_replay_examples.jsonl"
    if not selector_summary_path.is_file() or selector_summary_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing selector preflight summary: {selector_summary_path}")
    if not selector_contract_path.is_file() or selector_contract_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing selector contract draft: {selector_contract_path}")
    if not r1_examples_path.is_file() or r1_examples_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing R1 replay examples: {r1_examples_path}")
    selector_summary = _read_json(selector_summary_path)
    selector_contract = _read_json(selector_contract_path)
    examples = list(_iter_jsonl(r1_examples_path))
    train_examples = _load_train_examples(train_data_dir, payload_ids)
    selected = _select_examples(examples, max_rows=max_plan_rows)
    plan_rows: list[dict[str, Any]] = []
    train_matches = 0
    for example in selected:
        train_key = (
            str(example.get("expected_payload_id", "")),
            str(example.get("prompt_id", "")),
            _as_int(example.get("prompt_slot", 0)),
        )
        train_row = train_examples.get(train_key)
        if train_row is not None:
            train_matches += 1
        plan_rows.append(
            _make_branch_plan_row(
                example=example,
                train_row=train_row,
                selector_contract=selector_contract,
            )
        )
    repair_examples = [_make_repair_example(row) for row in plan_rows[:max_repair_examples]]
    token_class_rows = _token_class_rows(plan_rows)
    drift_counts = Counter(str(row.get("drift_reason", "")) for row in plan_rows)
    condition_counts = Counter(str(row.get("model_condition", "")) for row in plan_rows)
    manifest = {
        "schema_name": "natural_evidence_v1_regenerated_suffix_repair_manifest_v1",
        "status": "REPAIR_INPUTS_READY_NOT_REGENERATED",
        "repair_generation_started": False,
        "training_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "source_plan_jsonl": str(output_dir / "branch_aware_compatibility_scoring_plan.jsonl"),
        "repair_examples_jsonl": str(output_dir / "regenerated_suffix_repair_examples.jsonl"),
        "repair_example_rows": len(repair_examples),
        "required_next_step": "local-suffix repair dry-run or Slurm-scored branch-aware diagnostic",
    }
    summary = {
        "schema_name": SCHEMA_NAME,
        "status": "COMPLETE_BRANCH_AWARE_SUFFIX_REPAIR_INPUTS_PREPARED_NOT_SCORED",
        "claim_control": {
            "paper_claim_allowed": False,
            "training_started": False,
            "generation_started": False,
            "model_scoring_started": False,
            "e2e_eval_started": False,
            "not_payload_recovery": True,
            "not_full_far": True,
            "result_claim": "branch_aware_suffix_repair_preparation_not_scored_not_training_not_far",
        },
        "inputs": {
            "selector_preflight_summary": str(selector_summary_path),
            "selector_contract_draft": str(selector_contract_path),
            "r1_replay_examples": str(r1_examples_path),
            "train_data_dir": str(train_data_dir),
            "selector_preflight_status": selector_summary.get("status", ""),
            "selector_contract_status": selector_contract.get("contract_status", ""),
        },
        "selector_contract_status": selector_contract.get("contract_status", ""),
        "source_r1_example_rows": len(examples),
        "planned_branch_aware_rows": len(plan_rows),
        "regenerated_suffix_repair_example_rows": len(repair_examples),
        "train_metadata_matched_rows": train_matches,
        "drift_reason_counts": dict(sorted(drift_counts.items())),
        "model_condition_counts": dict(sorted(condition_counts.items())),
        "needs_slurm_scoring": True,
        "training_allowed": False,
        "e2e_rerun_allowed": False,
        "next_allowed_action": (
            "Run Slurm-scored branch-aware compatibility diagnostic or construct "
            "artifact-only local-suffix repair dry-run from prepared inputs; no training."
        ),
        "forbidden_claims_remain": [
            "natural-output success",
            "payload recovery",
            "full FAR",
            "cross-family generality",
            "robustness",
            "sanitizer resistance",
            "superiority over Scalable/Perinucleus",
            "24,576 fingerprints",
        ],
    }
    write_json(summary_path, summary)
    write_csv(output_dir / "branch_aware_compatibility_by_token_class.csv", token_class_rows, TOKEN_CLASS_FIELDS)
    write_jsonl(output_dir / "branch_aware_compatibility_scoring_plan.jsonl", plan_rows)
    write_json(output_dir / "regenerated_suffix_repair_manifest.json", manifest)
    write_jsonl(output_dir / "regenerated_suffix_repair_examples.jsonl", repair_examples)
    write_csv(output_dir / "branch_aware_suffix_repair_readiness.csv", _readiness_rows(summary), READINESS_FIELDS)
    _write_markdown(output_dir / "branch_aware_suffix_repair_preparation.md", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_preparation(
        selector_preflight_dir=_resolve(args.selector_preflight_dir),
        r1_replay_dir=_resolve(args.r1_replay_dir),
        train_data_dir=_resolve(args.train_data_dir),
        output_dir=_resolve(args.output_dir),
        payload_ids=_parse_csv_list(args.payload_ids),
        max_plan_rows=int(args.max_plan_rows),
        max_repair_examples=int(args.max_repair_examples),
        force=bool(args.force),
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(_resolve(args.output_dir))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

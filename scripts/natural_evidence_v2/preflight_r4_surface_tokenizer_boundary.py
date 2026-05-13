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

from scripts.natural_evidence_v2.score_r4_surface_teacher_forced_mass import (
    read_jsonl,
    validate_qwen_tokenizer_boundary_contract,
    validate_static_boundary_contract,
    write_json,
)


SCORER_SCRIPT = ROOT / "scripts/natural_evidence_v2/score_r4_surface_teacher_forced_mass.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight R4 prefix-native surface/tokenizer boundaries. Static mode "
            "does not authorize scoring; tokenizer mode loads only the tokenizer, "
            "not a model."
        )
    )
    parser.add_argument("--score-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=8192)
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--run-qwen-tokenizer",
        action="store_true",
        help="Load the named tokenizer and run the actual tokenizer boundary preflight.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guarded_summary_fields() -> dict[str, bool]:
    return {
        "model_forward_pass_started": False,
        "generation_started": False,
        "training_started": False,
        "llama_started": False,
        "same_family_null_started": False,
        "sanitizer_benchmark_started": False,
        "far_aggregation_started": False,
        "paper_claim_allowed": False,
        "slurm_submitted": False,
        "scoring_job_submitted": False,
        "scoring_authorized": False,
    }


def build_summary(
    *,
    args: argparse.Namespace,
    rows: list[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    summary = {
        "schema_name": "natural_evidence_v2_r4_prefix_native_tokenizer_boundary_preflight_v1",
        "status": validation["status"],
        "score_row_count": len(rows),
        "checked_row_count": validation["checked_row_count"],
        "failed_row_count": validation["failed_row_count"],
        "first_failing_row": validation["first_failing_row"],
        "candidate_probe_rows_sha256": sha256_file(args.score_rows),
        "scorer_script_sha256": sha256_file(SCORER_SCRIPT),
        "tokenizer_name": args.tokenizer_name if args.run_qwen_tokenizer else None,
        "qwen_tokenizer_preflight_started": bool(args.run_qwen_tokenizer),
        **guarded_summary_fields(),
    }
    if args.run_qwen_tokenizer:
        summary.update(
            {
                "empty_target_id_row_count": validation["empty_target_id_row_count"],
                "empty_other_id_row_count": validation["empty_other_id_row_count"],
                "target_other_overlap_row_count": validation["target_other_overlap_row_count"],
            }
        )
    else:
        summary.update(
            {
                "empty_target_id_row_count": None,
                "empty_other_id_row_count": None,
                "target_other_overlap_row_count": None,
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output dir: {args.output_dir}")
    rows = read_jsonl(args.score_rows)[: int(args.max_rows)]
    if not rows:
        raise ValueError("no R4 surface score rows selected")
    args.output_dir.mkdir(parents=True)

    static_validation = validate_static_boundary_contract(rows)
    if args.run_qwen_tokenizer and static_validation["status"] != "PASS_STATIC_BOUNDARY_CONTRACT_TOKENIZER_PENDING":
        validation = {
            **static_validation,
            "status": "FAIL_QWEN_TOKENIZER_BOUNDARY_PREFLIGHT",
            "empty_target_id_row_count": None,
            "empty_other_id_row_count": None,
            "target_other_overlap_row_count": None,
        }
    elif args.run_qwen_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
        validation = validate_qwen_tokenizer_boundary_contract(tokenizer, rows)
    else:
        validation = static_validation

    summary = build_summary(args=args, rows=rows, validation=validation)
    write_json(args.output_dir / "r4_prefix_native_tokenizer_boundary_preflight_summary.json", summary)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import (  # noqa: E402
    read_json,
    read_jsonl,
    resolve,
    sha256_file,
    technical_literal_hits,
    write_csv_new,
    write_json_new,
    write_jsonl_new,
    write_text_new,
)


DEFAULT_CONFIG = ROOT / "configs/natural_evidence_v2/r4_cover_natural_ecc.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an artifact-only R4 cover-natural teacher-forced surface "
            "target-mass probe plan. This does not train, score a model, submit "
            "Slurm, run Llama, aggregate FAR, or make paper claims."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prompt-source", type=Path, default=None)
    parser.add_argument("--surface-bank", type=Path, default=None)
    parser.add_argument("--codebook", type=Path, default=None)
    parser.add_argument("--decoder-spec", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--total-score-rows", type=int, default=None)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_new(path, dict(payload))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl_new(path, rows)


def group_surfaces(surface_bank: Mapping[str, Any]) -> dict[int, dict[int, list[dict[str, Any]]]]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for entry in surface_bank.get("entries", []):
        if not isinstance(entry, dict):
            continue
        coord = int(entry.get("coordinate_id", -1))
        bit = int(entry.get("polarity_or_code_symbol", -1))
        if coord < 0 or bit not in {0, 1}:
            raise ValueError(f"invalid surface bank entry: {entry}")
        phrase = str(entry.get("canonical_lemma_or_phrase", ""))
        if not phrase:
            raise ValueError(f"surface entry missing canonical phrase: {entry.get('surface_id')}")
        hits = technical_literal_hits(phrase)
        if hits:
            raise ValueError(f"technical literal in surface phrase {phrase!r}: {hits}")
        grouped[coord][bit].append(dict(entry))
    return grouped


def binary_side_failures(grouped: Mapping[int, Mapping[int, Sequence[Mapping[str, Any]]]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for coord, by_bit in sorted(grouped.items()):
        bit0_count = len(by_bit.get(0, []))
        bit1_count = len(by_bit.get(1, []))
        if bit0_count == 0 or bit1_count == 0:
            failures.append(
                {
                    "coordinate_id": int(coord),
                    "bit0_surface_count": int(bit0_count),
                    "bit1_surface_count": int(bit1_count),
                    "reason": "coordinate_missing_one_binary_surface_side",
                }
            )
    return failures


def surface_phrases(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    phrases: list[str] = []
    for entry in entries:
        phrase = str(entry.get("canonical_lemma_or_phrase", ""))
        if phrase:
            phrases.append(phrase)
        for alias in entry.get("aliases", []):
            alias_text = str(alias)
            if alias_text:
                phrases.append(alias_text)
    return sorted(set(phrases))


def select_prompts(rows: Sequence[Mapping[str, Any]], *, split: str, count: int) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if str(row.get("split", "")) == split][:count]
    if len(selected) != count:
        raise ValueError(f"selected {len(selected)} {split} prompts; expected {count}")
    prompt_ids = [str(row.get("prompt_id", "")) for row in selected]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("duplicate prompt ids in selected prompts")
    for row in selected:
        text = str(row.get("prompt_text", ""))
        if "Step " in text or "slot" in text.lower() or "exactly 16" in text:
            raise ValueError(f"forbidden R4 structural prompt: {row.get('prompt_id')}")
        hits = technical_literal_hits(text)
        if hits:
            raise ValueError(f"technical literal in prompt {row.get('prompt_id')}: {hits}")
    return selected


def build_rows(
    *,
    prompts: Sequence[Mapping[str, Any]],
    grouped_surfaces: Mapping[int, Mapping[int, Sequence[Mapping[str, Any]]]],
    codebook: Mapping[str, Any],
    prefix_shapes: Sequence[str],
) -> list[dict[str, Any]]:
    bits = [int(bit) for bit in codebook.get("protected_codeword_bits", [])]
    if len(bits) != len(grouped_surfaces):
        raise ValueError("codebook coordinate count does not match surface bank")
    rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        for coord, target_bit in enumerate(bits):
            entries = list(grouped_surfaces[coord][target_bit])
            target_entry = entries[(prompt_index + coord) % len(entries)]
            target_surface = str(target_entry["canonical_lemma_or_phrase"])
            prefix = str(prefix_shapes[(prompt_index + coord) % len(prefix_shapes)])
            target_response = prefix + target_surface + " while keeping the answer natural and useful."
            rows.append(
                {
                    "schema_name": "natural_evidence_v2_r4_surface_teacher_forced_probe_row_v1",
                    "artifact_role": "r4_teacher_forced_surface_target_mass_probe_not_scored",
                    "contract_id": str(codebook.get("contract_id", "a55e")),
                    "prompt_id": str(prompt["prompt_id"]),
                    "prompt_index": int(prompt_index),
                    "prompt_text": str(prompt["prompt_text"]),
                    "split": str(prompt.get("split", "")),
                    "coordinate_id": int(coord),
                    "target_bit": int(target_bit),
                    "target_surface_id": str(target_entry["surface_id"]),
                    "target_surface": target_surface,
                    "assistant_prefix_before_surface": prefix,
                    "target_response_text": target_response,
                    "bucket_0_surfaces": surface_phrases(grouped_surfaces[coord][0]),
                    "bucket_1_surfaces": surface_phrases(grouped_surfaces[coord][1]),
                    "score_objective": "next_token_first_surface_cylinder_mass",
                    "generation_started": False,
                    "training_started": False,
                    "slurm_submitted": False,
                    "paper_claim_allowed": False,
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = read_yaml(config_path)
    probe_cfg = dict(config.get("teacher_forced_surface_probe", {}))
    if not probe_cfg:
        raise ValueError("missing teacher_forced_surface_probe config")
    output_dir = resolve(args.output_dir or Path(str(probe_cfg["output_dir"])))
    prompt_path = resolve(args.prompt_source or Path(str(probe_cfg["prompt_source"])))
    surface_path = resolve(args.surface_bank or Path(str(probe_cfg["surface_bank"])))
    codebook_path = resolve(args.codebook or Path(str(probe_cfg["codebook"])))
    decoder_path = resolve(args.decoder_spec or Path(str(probe_cfg["decoder_spec"])))
    prompts = select_prompts(
        read_jsonl(prompt_path),
        split=str(args.split or probe_cfg.get("split", "dev")),
        count=int(args.max_prompts if args.max_prompts is not None else probe_cfg.get("max_prompts", 256)),
    )
    surface_bank = read_json(surface_path)
    codebook = read_json(codebook_path)
    grouped = group_surfaces(surface_bank)
    side_failures = binary_side_failures(grouped)
    if side_failures:
        output_dir.mkdir(parents=True, exist_ok=False)
        write_csv_new(
            output_dir / "r4_surface_teacher_forced_probe_binary_side_failures.csv",
            side_failures,
            ["coordinate_id", "bit0_surface_count", "bit1_surface_count", "reason"],
        )
        summary = {
            "schema_name": "natural_evidence_v2_r4_surface_teacher_forced_probe_plan_v1",
            "status": "FAIL_SURFACE_BANK_NOT_BINARY_PER_COORDINATE",
            "config": str(args.config),
            "config_sha256": sha256_file(config_path),
            "prompt_source": str(prompt_path),
            "prompt_source_sha256": sha256_file(prompt_path),
            "surface_bank": str(surface_path),
            "surface_bank_sha256": sha256_file(surface_path),
            "codebook": str(codebook_path),
            "codebook_sha256": sha256_file(codebook_path),
            "decoder_spec": str(decoder_path),
            "decoder_spec_sha256": sha256_file(decoder_path),
            "contract_id": str(codebook.get("contract_id", "a55e")),
            "coordinate_count": len(grouped),
            "binary_side_failure_count": len(side_failures),
            "score_row_count": 0,
            "generation_started": False,
            "training_started": False,
            "slurm_submitted": False,
            "llama_started": False,
            "far_aggregation_started": False,
            "paper_claim_allowed": False,
            "blocker": (
                "Current R4 surface bank is one-sided per coordinate. "
                "Teacher-forced target-vs-other surface mass cannot be scored "
                "until each coordinate has precommitted bit-0 and bit-1 surface alternatives."
            ),
            "next_allowed_action": (
                "Artifact-only R4 binary surface-bank repair planning. Do not submit "
                "R4 teacher-forced scoring, generation, locked-scale, Llama, sanitizer, FAR, "
                "same-family null, or paper claims."
            ),
        }
        write_json(output_dir / "r4_surface_teacher_forced_probe_plan_summary.json", summary)
        report = "\n".join(
            [
                "# R4 surface teacher-forced probe preflight blocker",
                "",
                "The preflight did not build scoring rows because the current R4",
                "surface bank is not binary per coordinate.",
                "",
                f"- coordinates checked: `{len(grouped)}`",
                f"- coordinates missing one binary side: `{len(side_failures)}`",
                "",
                "A teacher-forced mass probe needs a target side and a non-target",
                "side under the same coordinate. The current bank only provides",
                "one polarity per coordinate, so any target-vs-other mass result",
                "would be semantically invalid.",
                "",
                "Next allowed action: artifact-only binary surface-bank repair",
                "planning. No Slurm scoring or generation is authorized by this",
                "failed preflight.",
                "",
            ]
        )
        write_text_new(output_dir / "r4_surface_teacher_forced_probe_plan.md", report)
        print(json.dumps({"status": summary["status"], "output_dir": str(output_dir)}, sort_keys=True))
        return 0
    prefix_shapes = [str(item) for item in probe_cfg.get("prefix_shapes", [])]
    if not prefix_shapes:
        raise ValueError("at least one prefix shape is required")
    rows = build_rows(prompts=prompts, grouped_surfaces=grouped, codebook=codebook, prefix_shapes=prefix_shapes)
    expected_rows = int(
        args.total_score_rows if args.total_score_rows is not None else probe_cfg.get("total_score_rows", 0)
    )
    if expected_rows and len(rows) != expected_rows:
        raise ValueError(f"built {len(rows)} rows; expected {expected_rows}")
    coord_counts = Counter(int(row["coordinate_id"]) for row in rows)
    bit_counts = Counter(int(row["target_bit"]) for row in rows)
    prompt_counts = Counter(str(row["prompt_id"]) for row in rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "r4_surface_teacher_forced_probe_rows.jsonl", rows)
    coord_csv_rows = [
        {
            "coordinate_id": coord,
            "row_count": coord_counts[coord],
            "target_bit": codebook["protected_codeword_bits"][coord],
            "bucket_0_surface_count": len(surface_phrases(grouped[coord][0])),
            "bucket_1_surface_count": len(surface_phrases(grouped[coord][1])),
        }
        for coord in sorted(coord_counts)
    ]
    write_csv_new(
        output_dir / "r4_surface_teacher_forced_probe_coordinate_coverage.csv",
        coord_csv_rows,
        ["coordinate_id", "row_count", "target_bit", "bucket_0_surface_count", "bucket_1_surface_count"],
    )
    summary = {
        "schema_name": "natural_evidence_v2_r4_surface_teacher_forced_probe_plan_v1",
        "status": "PASS",
        "config": str(args.config),
        "config_sha256": sha256_file(config_path),
        "prompt_source": str(prompt_path),
        "prompt_source_sha256": sha256_file(prompt_path),
        "surface_bank": str(surface_path),
        "surface_bank_sha256": sha256_file(surface_path),
        "codebook": str(codebook_path),
        "codebook_sha256": sha256_file(codebook_path),
        "decoder_spec": str(decoder_path),
        "decoder_spec_sha256": sha256_file(decoder_path),
        "contract_id": str(codebook.get("contract_id", "a55e")),
        "selected_prompt_count": len(prompts),
        "score_row_count": len(rows),
        "coordinate_count": len(coord_counts),
        "rows_per_coordinate_min": min(coord_counts.values()),
        "rows_per_coordinate_max": max(coord_counts.values()),
        "rows_per_prompt_min": min(prompt_counts.values()),
        "rows_per_prompt_max": max(prompt_counts.values()),
        "target_bit_counts": {str(key): int(value) for key, value in sorted(bit_counts.items())},
        "conditions_to_score": list(probe_cfg.get("conditions", [])),
        "generation_started": False,
        "training_started": False,
        "slurm_submitted": False,
        "llama_started": False,
        "far_aggregation_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Review this artifact-only R4 teacher-forced surface probe plan. "
            "If accepted, prepare a Slurm-only scorer wrapper; do not run generation or locked-scale."
        ),
    }
    write_json(output_dir / "r4_surface_teacher_forced_probe_plan_summary.json", summary)
    report = "\n".join(
        [
            "# R4 surface teacher-forced probe preflight",
            "",
            "This artifact-only preflight converts the R4 cover-natural phrase bank into",
            "teacher-forced target-mass rows. It does not train, score a model, submit",
            "Slurm, run Llama, aggregate FAR, or make paper claims.",
            "",
            f"- selected prompts: `{len(prompts)}`",
            f"- score rows: `{len(rows)}`",
            f"- coordinates: `{len(coord_counts)}`",
            f"- rows per coordinate: `{min(coord_counts.values())}` to `{max(coord_counts.values())}`",
            f"- contract: `{summary['contract_id']}`",
            "",
            "The next step, if this plan is accepted, is a Slurm-only scorer wrapper",
            "for base/protected/task-only target surface mass on these frozen rows.",
            "",
        ]
    )
    write_text_new(output_dir / "r4_surface_teacher_forced_probe_plan.md", report)
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import (
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    stable_hash_hex,
    token_surface_allowed,
    write_json,
    write_jsonl,
)


ORGANIC_TEMPLATES = (
    "Give practical advice for organizing {topic} without assuming special tools.",
    "Write a short neutral explanation of how someone might approach {topic}.",
    "Summarize sensible first steps for {topic} in plain language.",
    "Compare two everyday options for handling {topic} and choose one.",
    "Draft a calm note helping someone think through {topic}.",
    "List common tradeoffs in {topic} and how to keep the decision simple.",
    "Explain what to check before starting {topic}.",
    "Give a concise response to a friend asking about {topic}.",
)

ORGANIC_TOPICS = (
    "planning a weekly grocery trip",
    "choosing a quiet place to study",
    "preparing for a routine appointment",
    "organizing a shared household calendar",
    "deciding what to pack for a short visit",
    "setting up a simple filing system",
    "planning a low-key birthday gathering",
    "checking a monthly phone bill",
    "preparing notes for a team discussion",
    "choosing a book for a reading group",
    "making a basic room cleaning plan",
    "coordinating a ride with a neighbor",
    "organizing photos from a recent trip",
    "planning meals for a busy week",
    "deciding how to use a free afternoon",
    "preparing a polite follow-up message",
    "reviewing a simple household budget",
    "choosing a route for a local errand",
    "setting priorities for a workday",
    "planning a visit to a public library",
    "keeping track of small maintenance tasks",
    "preparing a short status update",
    "choosing comfortable clothes for changing weather",
    "planning a small group discussion",
    "deciding what questions to ask a service provider",
    "organizing a shared checklist",
    "making a simple morning routine",
    "preparing a short thank-you note",
    "choosing a low-effort weekend activity",
    "planning a careful purchase",
    "reviewing notes after a meeting",
    "setting up reminders for recurring tasks",
)

ORGANIC_AUDIENCES = (
    "a practical adult",
    "a busy student",
    "a small household",
    "a careful planner",
    "a first-time organizer",
    "a remote worker",
    "a community volunteer",
    "a family member",
)

ORGANIC_CONTEXTS = (
    "limited time",
    "a modest budget",
    "uncertain availability",
    "a preference for simple choices",
    "several people to coordinate",
    "a need to avoid confusion",
    "a flexible deadline",
    "ordinary weekday constraints",
)

ORGANIC_CONSTRAINTS = (
    "Keep it under five sentences.",
    "Use concrete wording.",
    "Keep the tone neutral.",
    "Mention one fallback option.",
    "Avoid unnecessary detail.",
    "Make the answer easy to scan.",
    "Focus on actions.",
    "Use complete sentences.",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze deterministic held-out and organic prompt artifacts for "
            "natural_evidence_v1 density gates. This does not run a model."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--reference-outputs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--freeze-id", default="qwen_density_split_v1")
    parser.add_argument("--heldout-count", type=int, default=2048)
    parser.add_argument("--organic-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260505)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sort_key(row: dict[str, Any], *, freeze_id: str, seed: int, split: str) -> str:
    return stable_hash_hex(
        [
            "natural_evidence_v1_density_split",
            freeze_id,
            seed,
            split,
            row.get("prompt_id", ""),
            row.get("user_probe", row.get("prompt", "")),
        ]
    )


def _heldout_rows(reference_rows: list[dict[str, Any]], *, freeze_id: str, seed: int, count: int) -> list[dict[str, Any]]:
    selected = sorted(reference_rows, key=lambda row: _sort_key(row, freeze_id=freeze_id, seed=seed, split="heldout"))[
        :count
    ]
    output: list[dict[str, Any]] = []
    for row in selected:
        frozen = dict(row)
        frozen["schema_name"] = "natural_evidence_reference_output_v1"
        frozen["split"] = "heldout"
        frozen["prompt_split"] = "heldout"
        frozen["freeze_id"] = freeze_id
        frozen["result_claim"] = "frozen_heldout_density_input_not_payload_recovery"
        output.append(frozen)
    return sorted(output, key=lambda row: str(row.get("prompt_id", "")))


def _organic_prompt_rows(*, config: dict[str, Any], freeze_id: str, seed: int, count: int) -> list[dict[str, Any]]:
    protocol_id = str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1"))
    rows: list[dict[str, Any]] = []
    for index in range(max(0, count)):
        template = ORGANIC_TEMPLATES[index % len(ORGANIC_TEMPLATES)]
        cursor = index // len(ORGANIC_TEMPLATES)
        topic = ORGANIC_TOPICS[cursor % len(ORGANIC_TOPICS)]
        cursor //= len(ORGANIC_TOPICS)
        audience = ORGANIC_AUDIENCES[cursor % len(ORGANIC_AUDIENCES)]
        cursor //= len(ORGANIC_AUDIENCES)
        context = ORGANIC_CONTEXTS[cursor % len(ORGANIC_CONTEXTS)]
        cursor //= len(ORGANIC_CONTEXTS)
        constraint = ORGANIC_CONSTRAINTS[cursor % len(ORGANIC_CONSTRAINTS)]
        user_probe = (
            f"{template.format(topic=topic)} Write it for {audience}, "
            f"assuming {context}. {constraint}"
        )
        if any(
            not token_surface_allowed(part, forbidden_patterns=("FIELD=", "SECTION=", "TOPIC=", "PAYLOAD", "CERT", "EVIDENCE", "CARRIER"))
            for part in (topic, audience, context)
        ):
            raise ValueError(f"Unexpected disallowed prompt component at organic index {index}")
        rows.append(
            {
                "schema_name": "natural_evidence_prompt_bank_row_v1",
                "protocol_id": protocol_id,
                "prompt_id": f"nat_organic_{index:06d}",
                "split": "organic",
                "prompt_split": "organic",
                "freeze_id": freeze_id,
                "prompt_family": f"OPF{(index % 8) + 1}",
                "topic": topic,
                "audience": audience,
                "context": context,
                "constraint": constraint,
                "user_probe": user_probe,
                "result_claim": "frozen_organic_density_prompt_not_payload_recovery",
            }
        )
    return sorted(rows, key=lambda row: _sort_key(row, freeze_id=freeze_id, seed=seed, split="organic"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    reference_path = resolve_repo_path(args.reference_outputs, root)
    output_dir = resolve_repo_path(args.output_dir, root)
    heldout_path = output_dir / "heldout_reference_outputs.jsonl"
    organic_path = output_dir / "organic_prompts.jsonl"
    manifest_path = output_dir / "density_prompt_split_manifest.json"
    heldout_ids_path = output_dir / "heldout_prompt_ids.txt"

    for path in (heldout_path, organic_path, manifest_path, heldout_ids_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing frozen density artifact: {path}")

    reference_rows = read_jsonl(reference_path)
    if len(reference_rows) < args.heldout_count:
        raise ValueError(
            f"reference outputs contain {len(reference_rows)} rows, fewer than heldout_count={args.heldout_count}"
        )
    heldout_rows = _heldout_rows(
        reference_rows,
        freeze_id=args.freeze_id,
        seed=args.seed,
        count=args.heldout_count,
    )
    organic_rows = _organic_prompt_rows(
        config=config,
        freeze_id=args.freeze_id,
        seed=args.seed,
        count=args.organic_count,
    )
    if len({row["prompt_id"] for row in heldout_rows}) != len(heldout_rows):
        raise ValueError("heldout prompt ids are not unique")
    if len({row["prompt_id"] for row in organic_rows}) != len(organic_rows):
        raise ValueError("organic prompt ids are not unique")

    write_jsonl(heldout_path, heldout_rows)
    write_jsonl(organic_path, organic_rows)
    heldout_ids_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_ids_path.write_text(
        "".join(f"{row['prompt_id']}\n" for row in heldout_rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_name": "natural_evidence_density_prompt_split_manifest_v1",
        "protocol_id": str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1")),
        "freeze_id": args.freeze_id,
        "seed": args.seed,
        "reference_outputs": str(reference_path),
        "heldout_count": len(heldout_rows),
        "organic_count": len(organic_rows),
        "heldout_reference_outputs": str(heldout_path),
        "organic_prompts": str(organic_path),
        "heldout_prompt_ids": str(heldout_ids_path),
        "sha256": {
            "heldout_reference_outputs": _sha256(heldout_path),
            "organic_prompts": _sha256(organic_path),
            "heldout_prompt_ids": _sha256(heldout_ids_path),
        },
        "split_policy": "deterministic_sha256_sort_by_freeze_id_seed_prompt_id_user_probe",
        "result_claim": "frozen_density_prompt_inputs_not_payload_recovery",
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

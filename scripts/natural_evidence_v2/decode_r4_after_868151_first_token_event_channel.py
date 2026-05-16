from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.natural_evidence_v2.r4_cover_natural_common import technical_literal_hits  # noqa: E402


DEFAULT_SCORE_ROWS = (
    ROOT
    / "results/natural_evidence_v2/status/r4_after_868016_reliability_coordinate_pivot_rows_20260516/reliability_surface_mass_rows.jsonl"
)
DEFAULT_CODEBOOK = (
    ROOT
    / "results/natural_evidence_v2/precommit/r4_after_868016_reliability_coordinate_pivot_codebook_precommit_20260516/codebook.json"
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def first_lexical_event_after_exact_prefix(response_text: str, prefix_text: str) -> str:
    if not response_text.startswith(prefix_text):
        return ""
    continuation = response_text[len(prefix_text) :]
    match = re.search(r"[A-Za-z]+", continuation.lower())
    return match.group(0) if match else ""


def first_word(text: str) -> str:
    match = re.search(r"[A-Za-z]+", text.lower())
    return match.group(0) if match else ""


def int_set(value: Any) -> set[int]:
    if value is None or value == "":
        return set()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"expected token id sequence, got {type(value).__name__}")
    return {int(item) for item in value}


def literal_present(text: str, literal: str) -> bool:
    return bool(re.search(rf"\b{re.escape(literal)}\b", text, flags=re.IGNORECASE))


def contextual_technical_literal_hits(text: str, policy: Mapping[str, Any] | None) -> list[str]:
    if not policy:
        return technical_literal_hits(text)
    hits: list[str] = []
    for literal in policy.get("hard_forbid_literals", []):
        if literal_present(text, str(literal)):
            hits.append(str(literal))
    contextual = policy.get("contextual_literals", {})
    if isinstance(contextual, Mapping):
        for literal, rule in contextual.items():
            literal_text = str(literal)
            if not literal_present(text, literal_text):
                continue
            if not isinstance(rule, Mapping):
                hits.append(literal_text)
                continue
            technical_cues = [str(item) for item in rule.get("technical_cues", [])]
            has_technical_cue = any(literal_present(text, cue) for cue in technical_cues)
            if has_technical_cue or not bool(rule.get("ordinary_domain_allowed", False)):
                hits.append(literal_text)
    return sorted(set(hits))


def first_word_set(items: Sequence[str]) -> set[str]:
    return {first_word(item) for item in items if first_word(item)}


def vote_from_target_membership(*, is_target: bool, target_bit: int) -> str:
    return str(int(target_bit) if is_target else 1 - int(target_bit))


def classify_first_token_event(
    *,
    generated_row: Mapping[str, Any],
    score_row: Mapping[str, Any],
    allow_text_fallback_for_old_transcripts: bool,
) -> dict[str, Any]:
    target_bit = int(generated_row.get("target_bit", score_row.get("target_bit")))
    token_id_present = generated_row.get("first_generated_token_id") not in (None, "")

    if token_id_present:
        target_ids = int_set(generated_row.get("target_first_token_ids", score_row.get("target_first_token_ids")))
        other_ids = int_set(generated_row.get("other_first_token_ids", score_row.get("other_first_token_ids")))
        overlap = target_ids & other_ids
        if not target_ids:
            raise ValueError("target_first_token_ids must be non-empty for token-id event decoding")
        if not other_ids:
            raise ValueError("other_first_token_ids must be non-empty for token-id event decoding")
        if overlap:
            raise ValueError(f"target/other first-token ids overlap: {sorted(overlap)}")
        token_id = int(generated_row["first_generated_token_id"])
        if token_id in target_ids:
            vote = vote_from_target_membership(is_target=True, target_bit=target_bit)
            status = "target"
        elif token_id in other_ids:
            vote = vote_from_target_membership(is_target=False, target_bit=target_bit)
            status = "other"
        else:
            vote = ""
            status = "erasure"
        return {
            "event_source": "token_id_trace",
            "event": str(generated_row.get("first_generated_token_text", "")),
            "first_generated_token_id": token_id,
            "target_first_token_ids": sorted(target_ids),
            "other_first_token_ids": sorted(other_ids),
            "vote_bit": vote,
            "event_status": status,
        }

    if not allow_text_fallback_for_old_transcripts:
        raise ValueError(
            "generated row has no first_generated_token_id; rerun with "
            "--allow-text-fallback-for-old-transcripts only for old non-positive transcripts"
        )

    bucket_0 = [str(item) for item in score_row.get("bucket_0_surfaces", [])]
    bucket_1 = [str(item) for item in score_row.get("bucket_1_surfaces", [])]
    if target_bit == 0:
        target_events = first_word_set(bucket_0)
        other_events = first_word_set(bucket_1)
    else:
        target_events = first_word_set(bucket_1)
        other_events = first_word_set(bucket_0)
    overlap = target_events & other_events
    if not target_events:
        raise ValueError("text fallback target event set must be non-empty")
    if not other_events:
        raise ValueError("text fallback other event set must be non-empty")
    if overlap:
        raise ValueError(f"text fallback target/other first-word events overlap: {sorted(overlap)}")

    event = first_lexical_event_after_exact_prefix(
        str(generated_row.get("response_text", "")),
        str(score_row.get("assistant_prefix_before_surface", "")),
    )
    if event in target_events:
        vote = vote_from_target_membership(is_target=True, target_bit=target_bit)
        status = "target"
    elif event in other_events:
        vote = vote_from_target_membership(is_target=False, target_bit=target_bit)
        status = "other"
    else:
        vote = ""
        status = "erasure"
    return {
        "event_source": "text_fallback_old_transcript",
        "event": event,
        "first_generated_token_id": "",
        "target_first_token_ids": [],
        "other_first_token_ids": [],
        "vote_bit": vote,
        "event_status": status,
    }


def score_row_key(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row["prompt_id"]),
        int(row["coordinate_id"]),
        str(row.get("prefix_family_id", "")),
        str(row.get("target_surface", "")),
    )


def load_score_rows(path: Path) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    rows = read_jsonl(path)
    return {score_row_key(row): row for row in rows}


def expected_bits_by_control_arm(paths: Iterable[Path]) -> dict[tuple[str, str], list[int]]:
    expected: dict[tuple[str, str], list[int]] = {}
    for path in paths:
        for row in read_jsonl(path):
            shard_id = str(row.get("block_id", "")).split("_block_", maxsplit=1)[0]
            arm = str(row.get("arm", ""))
            bits = row.get("expected_bits", [])
            if isinstance(bits, list) and shard_id and arm:
                expected[(shard_id, arm)] = [int(item) for item in bits]
    return expected


def checksum_valid(decoded_bits: Sequence[int | str]) -> bool:
    if len(decoded_bits) < 8 or any(bit == "" for bit in decoded_bits):
        return False
    payload = [int(bit) for bit in decoded_bits[:4]]
    checksum = [int(bit) for bit in decoded_bits[4:8]]
    return checksum == [1 - bit for bit in payload]


def decode_bits_from_votes(
    *,
    codebook: Mapping[str, Any],
    votes_by_coordinate: Mapping[int, Counter[str]],
) -> tuple[list[int | str], list[dict[str, Any]]]:
    decoded: list[int | str] = []
    trace: list[dict[str, Any]] = []
    for mapping in codebook["pair_to_bit_mapping"]:
        coordinates = [int(item) for item in mapping["coordinates"]]
        one = sum(int(votes_by_coordinate.get(coord, Counter()).get("1", 0)) for coord in coordinates)
        zero = sum(int(votes_by_coordinate.get(coord, Counter()).get("0", 0)) for coord in coordinates)
        support = one + zero
        if support <= 0 or one == zero:
            bit: int | str = ""
        else:
            bit = 1 if one > zero else 0
        decoded.append(bit)
        trace.append(
            {
                "bit_index": int(mapping["bit_index"]),
                "coordinates": coordinates,
                "one_votes": one,
                "zero_votes": zero,
                "support": support,
                "decoded_bit": bit,
            }
        )
    return decoded, trace


def decode_generated_rows(
    *,
    generated_rows: Sequence[Mapping[str, Any]],
    score_rows: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    codebook: Mapping[str, Any],
    expected_controls: Mapping[tuple[str, str], Sequence[int]],
    allow_text_fallback_for_old_transcripts: bool,
    contextual_literal_policy: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    votes: dict[tuple[str, str], dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    forbidden_counts: Counter[tuple[str, str]] = Counter()
    response_hashes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    event_rows: list[dict[str, Any]] = []

    for row in generated_rows:
        condition = str(row.get("generation_condition", ""))
        shard_id = str(row.get("replicate_group_id", ""))
        key = score_row_key(row)
        score_row = score_rows.get(key)
        if score_row is None:
            raise KeyError(f"missing score row for generated output key={key}")
        event = classify_first_token_event(
            generated_row=row,
            score_row=score_row,
            allow_text_fallback_for_old_transcripts=allow_text_fallback_for_old_transcripts,
        )
        side = str(event["vote_bit"])
        event_rows.append(
            {
                "shard_id": shard_id,
                "condition": condition,
                "prompt_id": str(row.get("prompt_id", "")),
                "coordinate_id": int(row["coordinate_id"]),
                "prefix_family_id": str(row.get("prefix_family_id", "")),
                "target_bit": int(row.get("target_bit", score_row.get("target_bit"))),
                "event_source": event["event_source"],
                "event": event["event"],
                "first_generated_token_id": event["first_generated_token_id"],
                "event_status": event["event_status"],
                "vote_bit": side,
                "response_text_sha256": str(row.get("response_text_sha256", "")),
            }
        )
        if side:
            votes[(shard_id, condition)][int(row["coordinate_id"])][side] += 1
        forbidden_counts[(shard_id, condition)] += len(
            contextual_technical_literal_hits(
                str(row.get("response_text", "")),
                contextual_literal_policy,
            )
        )
        response_hashes[(shard_id, condition)][str(row.get("response_text_sha256", ""))] += 1

    decoded_rows: list[dict[str, Any]] = []
    for shard_id in sorted({str(row.get("replicate_group_id", "")) for row in generated_rows}):
        for arm, source_condition in (
            ("protected", "protected"),
            ("raw", "raw"),
            ("task_only", "task_only"),
            ("wrong_key", "protected"),
            ("wrong_payload", "protected"),
        ):
            decoded_bits, trace = decode_bits_from_votes(
                codebook=codebook,
                votes_by_coordinate=votes[(shard_id, source_condition)],
            )
            expected = list(expected_controls.get((shard_id, arm), codebook["expected_codeword_bits"]))
            complete_pairs = sum(1 for bit in decoded_bits if bit != "")
            bits_match = complete_pairs == len(expected) and [int(bit) for bit in decoded_bits] == [int(bit) for bit in expected]
            checksum = checksum_valid(decoded_bits)
            duplicate_count = sum(
                count - 1
                for value, count in response_hashes[(shard_id, source_condition)].items()
                if value and count > 1
            )
            forbidden = int(forbidden_counts[(shard_id, source_condition)])
            accept_ignoring_quality = bool(bits_match and checksum)
            accept = bool(accept_ignoring_quality and forbidden == 0 and duplicate_count == 0)
            decoded_rows.append(
                {
                    "block_id": f"{shard_id}_block_00",
                    "arm": arm,
                    "source_condition": source_condition,
                    "accept": accept,
                    "accept_ignoring_quality": accept_ignoring_quality,
                    "complete_pairs": complete_pairs,
                    "required_pairs": len(expected),
                    "decoded_bits": "".join(str(bit) if bit != "" else "-" for bit in decoded_bits),
                    "expected_bits": "".join(str(int(bit)) for bit in expected),
                    "bits_match_condition": bits_match,
                    "checksum_valid": checksum,
                    "forbidden_public_surface_count": forbidden,
                    "duplicate_response_hash_count": duplicate_count,
                    "pair_trace": trace,
                }
            )
    return decoded_rows, event_rows


def summarize(decoded_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Counter[str]] = defaultdict(Counter)
    for row in decoded_rows:
        arm = str(row["arm"])
        by_arm[arm]["blocks"] += 1
        by_arm[arm]["accepts"] += int(bool(row["accept"]))
        by_arm[arm]["accepts_ignoring_quality"] += int(bool(row["accept_ignoring_quality"]))
        by_arm[arm]["forbidden_public_surface_count"] += int(row["forbidden_public_surface_count"])
        by_arm[arm]["duplicate_response_hash_count"] += int(row["duplicate_response_hash_count"])
    return {arm: dict(counter) for arm, counter in sorted(by_arm.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode R4 first-token event-channel generated outputs.")
    parser.add_argument("--generated-outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--score-rows", type=Path, default=DEFAULT_SCORE_ROWS)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--control-decode-rows", type=Path, nargs="*", default=[])
    parser.add_argument("--contextual-literal-policy", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-text-fallback-for-old-transcripts",
        action="store_true",
        help="Allow lexical fallback only for old non-positive transcripts that lack token-id event traces.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated_rows: list[dict[str, Any]] = []
    for path in args.generated_outputs:
        generated_rows.extend(read_jsonl(path))
    decoded_rows, event_rows = decode_generated_rows(
        generated_rows=generated_rows,
        score_rows=load_score_rows(args.score_rows),
        codebook=read_json(args.codebook),
        expected_controls=expected_bits_by_control_arm(args.control_decode_rows),
        allow_text_fallback_for_old_transcripts=bool(args.allow_text_fallback_for_old_transcripts),
        contextual_literal_policy=read_json(args.contextual_literal_policy) if args.contextual_literal_policy else None,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "first_token_event_decode_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in decoded_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "first_token_event_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in event_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    event_sources = Counter(str(row["event_source"]) for row in event_rows)
    event_statuses = Counter(str(row["event_status"]) for row in event_rows)
    summary = {
        "schema_name": "natural_evidence_v2_r4_after_868151_first_token_event_decode_summary_v1",
        "status": "FIRST_TOKEN_EVENT_DECODE_RECORDED_ARTIFACT_ONLY_NOT_POSITIVE",
        "generated_output_files": [str(path) for path in args.generated_outputs],
        "decode_rows": len(decoded_rows),
        "event_rows": len(event_rows),
        "event_sources": dict(sorted(event_sources.items())),
        "event_statuses": dict(sorted(event_statuses.items())),
        "summary_by_arm": summarize(decoded_rows),
        "future_positive_requires_token_id_trace": True,
        "text_fallback_for_old_transcripts": bool(args.allow_text_fallback_for_old_transcripts),
        "contextual_literal_policy": str(args.contextual_literal_policy) if args.contextual_literal_policy else "",
        "reclassifies_868151": False,
        "generation_started": False,
        "model_scoring_started": False,
        "training_started": False,
        "paper_claim_allowed": False,
    }
    (output_dir / "first_token_event_decode_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "first_token_event_per_block.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "block_id",
            "arm",
            "source_condition",
            "accept",
            "accept_ignoring_quality",
            "complete_pairs",
            "required_pairs",
            "decoded_bits",
            "expected_bits",
            "bits_match_condition",
            "checksum_valid",
            "forbidden_public_surface_count",
            "duplicate_response_hash_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in decoded_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

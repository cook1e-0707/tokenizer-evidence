from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import read_jsonl, read_yaml, resolve_repo_path, write_json, write_jsonl
from src.core.payload_codec import BucketPayloadCodec, encode_bytes_variable_radices
from src.core.rs_codec import ReedSolomonCodec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile natural_evidence_v1 natural-response training rows from "
            "reference outputs and bucket-bank entries."
        )
    )
    parser.add_argument("--config", default="configs/natural_evidence_v1/pilot.yaml")
    parser.add_argument("--reference-outputs", required=True)
    parser.add_argument("--bucket-bank-entries", required=True)
    parser.add_argument("--payload-id", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--contract-json", required=True)
    parser.add_argument("--audit-key-id", default="")
    parser.add_argument("--compatibility-jsonl", default="")
    parser.add_argument("--min-compatible-members-per-bucket", type=int, default=0)
    parser.add_argument("--bucket-radix", type=int, default=0)
    parser.add_argument(
        "--encoding-mode",
        choices=("fixed_radix", "variable_radix"),
        default="fixed_radix",
    )
    parser.add_argument(
        "--variable-radix-frame-policy",
        choices=("single", "repeat_payload"),
        default="single",
    )
    parser.add_argument("--variable-radix-min-positions", type=int, default=0)
    return parser.parse_args(argv)


def _payload_text(config: dict[str, Any], payload_id: str) -> str:
    payloads = config.get("payloads", [])
    if isinstance(payloads, list):
        for payload in payloads:
            if isinstance(payload, dict) and str(payload.get("payload_id", "")) == payload_id:
                return str(payload.get("payload_text", payload_id))
    return payload_id


def _payload_digits(config: dict[str, Any], payload_text: str, bucket_radix_override: int = 0) -> list[int]:
    decoder_cfg = dict(config.get("decoder", {}))
    bucket_tuple_width = int(decoder_cfg.get("bucket_tuple_width", 3))
    bucket_radix = bucket_radix_override or int(decoder_cfg.get("bucket_radix", 8))
    rs_parity_symbols = int(decoder_cfg.get("rs_parity_symbols", 0))
    codec = BucketPayloadCodec(
        bucket_radices=tuple(bucket_radix for _ in range(bucket_tuple_width)),
        rs_codec=ReedSolomonCodec(parity_symbols=rs_parity_symbols),
    )
    encoding = codec.encode_bytes(payload_text.encode("utf-8"), apply_rs=rs_parity_symbols > 0)
    return [int(bucket_id) for bucket_tuple in encoding.bucket_tuples for bucket_id in bucket_tuple]


def _bank_by_context(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for entry in entries:
        context_signature = str(entry.get("context_signature", ""))
        if context_signature:
            output[context_signature] = entry
    return output


def _bank_by_prompt(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        prompt_id = str(entry.get("prompt_id", ""))
        if prompt_id:
            output.setdefault(prompt_id, []).append(entry)
    for prompt_entries in output.values():
        prompt_entries.sort(key=_entry_token_index)
    return output


def _compatibility_filter_by_entry(
    compatibility_rows: list[dict[str, Any]],
) -> dict[str, dict[str, set[int]]]:
    compatible: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in compatibility_rows:
        if not bool(row.get("compatibility_pass", False)):
            continue
        bank_entry_id = str(row.get("bank_entry_id", ""))
        bucket_id = str(row.get("bucket_id", ""))
        if not bank_entry_id or bucket_id == "":
            continue
        compatible[bank_entry_id][bucket_id].add(int(row["token_id"]))
    return {entry_id: dict(bucket_map) for entry_id, bucket_map in compatible.items()}


def _filter_bank_entries_by_compatibility(
    *,
    bank_entries: list[dict[str, Any]],
    compatibility_by_entry: dict[str, dict[str, set[int]]],
    min_members_per_bucket: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filtered_entries: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for entry in bank_entries:
        bank_entry_id = str(entry.get("bank_entry_id", ""))
        buckets = dict(entry.get("buckets", {}))
        compatible_buckets = compatibility_by_entry.get(bank_entry_id, {})
        if not compatible_buckets:
            skipped_rows.append(
                {
                    "bank_entry_id": bank_entry_id,
                    "prompt_id": entry.get("prompt_id", ""),
                    "reason": "missing_compatibility_rows",
                }
            )
            continue
        filtered_buckets: dict[str, list[int]] = {}
        for bucket_id, token_ids_raw in buckets.items():
            allowed_tokens = compatible_buckets.get(str(bucket_id), set())
            token_ids = [
                int(token_id)
                for token_id in token_ids_raw
                if int(token_id) in allowed_tokens
            ]
            if len(token_ids) < min_members_per_bucket:
                skipped_rows.append(
                    {
                        "bank_entry_id": bank_entry_id,
                        "prompt_id": entry.get("prompt_id", ""),
                        "reason": "below_min_compatible_members",
                        "bucket_id": str(bucket_id),
                        "compatible_members": len(token_ids),
                    }
                )
                break
            filtered_buckets[str(bucket_id)] = token_ids
        else:
            filtered_entry = dict(entry)
            filtered_entry["buckets"] = filtered_buckets
            filtered_entry["compatibility_filter"] = {
                "mode": "counterfactual_min_members",
                "min_compatible_members_per_bucket": min_members_per_bucket,
                "source": "compatibility_jsonl",
            }
            filtered_entries.append(filtered_entry)
    return filtered_entries, skipped_rows


def _eligible_prefixes(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("eligible_prefixes", "eligible_positions", "prefixes"):
        value = row.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _entry_token_index(entry: dict[str, Any]) -> int:
    if "token_index" in entry:
        return int(entry["token_index"])
    if "token_position" in entry:
        return int(entry["token_position"])
    prefix_token_ids = entry.get("prefix_token_ids", [])
    if isinstance(prefix_token_ids, list):
        return len(prefix_token_ids)
    return int(entry.get("prefix_token_count", 0))


def _spaced(prefixes: list[dict[str, Any]], min_spacing_tokens: int, max_positions: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    last_token_index: int | None = None
    for prefix in sorted(prefixes, key=lambda item: int(item.get("token_index", item.get("token_position", 0)))):
        token_index = int(prefix.get("token_index", prefix.get("token_position", 0)))
        if last_token_index is not None and token_index - last_token_index < min_spacing_tokens:
            continue
        selected.append(prefix)
        last_token_index = token_index
        if len(selected) >= max_positions:
            break
    return selected


def _spaced_entries(entries: list[dict[str, Any]], min_spacing_tokens: int, max_positions: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    last_token_index: int | None = None
    for entry in sorted(entries, key=_entry_token_index):
        token_index = _entry_token_index(entry)
        if last_token_index is not None and token_index - last_token_index < min_spacing_tokens:
            continue
        selected.append(entry)
        last_token_index = token_index
        if len(selected) >= max_positions:
            break
    return selected


def _candidate_token_ids(entry: dict[str, Any]) -> list[int]:
    buckets = dict(entry.get("buckets", {}))
    token_ids: list[int] = []
    for values in buckets.values():
        if isinstance(values, list):
            token_ids.extend(int(value) for value in values)
    return sorted(set(token_ids))


def _variable_bucket_to_token_ids(
    *,
    entry: dict[str, Any],
    compatibility_by_entry: dict[str, dict[str, set[int]]],
) -> dict[str, list[int]]:
    compatible_bucket_ids = [str(value) for value in entry.get("compatible_bucket_ids", [])]
    if len(compatible_bucket_ids) < 2:
        return {}
    direct_maps = (
        entry.get("compatible_token_ids_by_bucket"),
        entry.get("bucket_to_token_ids"),
        entry.get("buckets"),
    )
    for direct_map in direct_maps:
        if not isinstance(direct_map, dict):
            continue
        output = {
            bucket_id: [int(value) for value in direct_map.get(bucket_id, [])]
            for bucket_id in compatible_bucket_ids
        }
        if all(output.values()):
            return output
    entry_id = str(entry.get("bank_entry_id", ""))
    compatibility_map = compatibility_by_entry.get(entry_id, {})
    output = {
        bucket_id: sorted(int(token_id) for token_id in compatibility_map.get(bucket_id, set()))
        for bucket_id in compatible_bucket_ids
    }
    return output if all(output.values()) else {}


def _variable_sort_key(entry: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(entry.get("generated_row_index", entry.get("query_index", 0)) or 0),
        int(entry.get("position_index", _entry_token_index(entry)) or 0),
        str(entry.get("entry_key", entry.get("bank_entry_id", ""))),
    )


def _spaced_variable_entries(entries: list[dict[str, Any]], min_spacing_tokens: int, max_positions: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    last_token_index: int | None = None
    for entry in sorted(entries, key=_variable_sort_key):
        token_index = int(entry.get("position_index", _entry_token_index(entry)) or 0)
        if last_token_index is not None and token_index - last_token_index < min_spacing_tokens:
            continue
        selected.append(entry)
        last_token_index = token_index
        if len(selected) >= max_positions:
            break
    return selected


def _variable_radix_frame_assignments(
    *,
    payload: bytes,
    radices: list[int],
    frame_policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    assignments: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    cursor = 0
    frame_index = 0
    while cursor < len(radices):
        try:
            encoding = encode_bytes_variable_radices(payload, radices[cursor:])
        except ValueError:
            break
        if not encoding.digits:
            break
        start = cursor
        end = cursor + len(encoding.digits)
        if end > len(radices):
            break
        frames.append(
            {
                "frame_index": frame_index,
                "start_digit_index": start,
                "end_digit_index": end,
                "digit_count": len(encoding.digits),
                "encoded_byte_count": len(encoding.byte_groups),
                "byte_groups": [[start + group_start, start + group_end] for group_start, group_end in encoding.byte_groups],
            }
        )
        for frame_digit_index, (digit, radix) in enumerate(zip(encoding.digits, encoding.radices)):
            assignments.append(
                {
                    "payload_digit_index": len(assignments),
                    "frame_index": frame_index,
                    "frame_digit_index": frame_digit_index,
                    "frame_digit_count": len(encoding.digits),
                    "target_digit": int(digit),
                    "target_radix": int(radix),
                }
            )
        cursor = end
        frame_index += 1
        if frame_policy == "single":
            break
    return assignments, frames, cursor


def _compile_variable_radix_rows(
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    root: Path,
    reference_rows: list[dict[str, Any]],
    bank_entries: list[dict[str, Any]],
    payload_text: str,
    protocol_id: str,
    audit_key_id: str,
    min_spacing_tokens: int,
    max_positions: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compatibility_by_entry: dict[str, dict[str, set[int]]] = {}
    if args.compatibility_jsonl:
        compatibility_by_entry = _compatibility_filter_by_entry(
            read_jsonl(resolve_repo_path(args.compatibility_jsonl, root))
        )
    enriched_entries: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for entry in bank_entries:
        compatible_bucket_ids = [str(value) for value in entry.get("compatible_bucket_ids", [])]
        bucket_to_token_ids = _variable_bucket_to_token_ids(
            entry=dict(entry),
            compatibility_by_entry=compatibility_by_entry,
        )
        if len(compatible_bucket_ids) < 2 or not bucket_to_token_ids:
            skipped_rows.append(
                {
                    "bank_entry_id": entry.get("bank_entry_id", ""),
                    "entry_key": entry.get("entry_key", ""),
                    "prompt_id": entry.get("prompt_id", ""),
                    "reason": "missing_variable_radix_token_map",
                }
            )
            continue
        updated = dict(entry)
        updated["compatible_bucket_ids"] = compatible_bucket_ids
        updated["bucket_to_token_ids"] = bucket_to_token_ids
        enriched_entries.append(updated)

    entries_by_generated_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    entries_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in enriched_entries:
        entries_by_generated_row[int(entry.get("generated_row_index", entry.get("query_index", 0)) or 0)].append(entry)
        prompt_id = str(entry.get("prompt_id", ""))
        if prompt_id:
            entries_by_prompt[prompt_id].append(entry)

    row_matches: list[list[tuple[int, dict[str, Any]]]] = []
    ordered_positions: list[tuple[int, int, dict[str, Any]]] = []
    for row_index, row in enumerate(reference_rows):
        candidates: list[dict[str, Any]] = []
        if "generated_row_index" in row:
            candidates = entries_by_generated_row.get(int(row["generated_row_index"]), [])
        if not candidates and "query_index" in row:
            candidates = entries_by_generated_row.get(int(row["query_index"]), [])
        if not candidates:
            candidates = entries_by_generated_row.get(row_index, [])
        if not candidates:
            candidates = entries_by_prompt.get(str(row.get("prompt_id", "")), [])
        selected = _spaced_variable_entries(candidates, min_spacing_tokens, max_positions)
        matches = [
            (int(entry.get("position_index", _entry_token_index(entry)) or 0), entry)
            for entry in selected
        ]
        row_matches.append(matches)
        for position_index, (_, entry) in enumerate(matches):
            ordered_positions.append((row_index, position_index, entry))

    radices = [len(entry["compatible_bucket_ids"]) for _, _, entry in ordered_positions]
    if not radices:
        raise ValueError("variable-radix compilation found no eligible positions")
    frame_policy = str(args.variable_radix_frame_policy)
    frame_assignments, frame_summaries, used_position_count = _variable_radix_frame_assignments(
        payload=payload_text.encode("utf-8"),
        radices=radices,
        frame_policy=frame_policy,
    )
    if not frame_assignments:
        raise ValueError("variable-radix compilation could not encode one complete payload frame")
    assignment_by_row_position: dict[tuple[int, int], dict[str, Any]] = {}
    for assignment in frame_assignments:
        digit_index = int(assignment["payload_digit_index"])
        row_index, position_index, entry = ordered_positions[digit_index]
        compatible_bucket_ids = [str(value) for value in entry["compatible_bucket_ids"]]
        target_bucket = compatible_bucket_ids[int(assignment["target_digit"])]
        bucket_to_token_ids = dict(entry["bucket_to_token_ids"])
        assignment_by_row_position[(row_index, position_index)] = {
            "payload_digit_index": digit_index,
            "frame_index": int(assignment["frame_index"]),
            "frame_digit_index": int(assignment["frame_digit_index"]),
            "frame_digit_count": int(assignment["frame_digit_count"]),
            "target_digit": int(assignment["target_digit"]),
            "target_radix": int(assignment["target_radix"]),
            "target_bucket": target_bucket,
            "compatible_bucket_ids": compatible_bucket_ids,
            "bucket_to_token_ids": bucket_to_token_ids,
            "target_bucket_token_ids": [int(value) for value in bucket_to_token_ids[target_bucket]],
            "candidate_token_ids": sorted(
                {
                    int(token_id)
                    for bucket_id in compatible_bucket_ids
                    for token_id in bucket_to_token_ids.get(bucket_id, [])
                }
            ),
        }

    compiled_rows: list[dict[str, Any]] = []
    task_only_rows: list[dict[str, Any]] = []
    evidence_example_count = 0
    total_eligible_positions = 0
    used_assignment_count = 0
    for row_index, row in enumerate(reference_rows):
        selected_positions: list[dict[str, Any]] = []
        for position_index, (token_index, entry) in enumerate(row_matches[row_index]):
            assignment = assignment_by_row_position.get((row_index, position_index))
            if assignment is None:
                continue
            selected_positions.append(
                {
                    "token_index": token_index,
                    "context_signature": entry.get("context_signature", entry.get("entry_key", "")),
                    "bank_entry_id": entry.get("bank_entry_id", ""),
                    "entry_key": entry.get("entry_key", ""),
                    "prefix_token_ids": entry.get("prefix_token_ids", []),
                    **assignment,
                }
            )
            used_assignment_count += 1
        example_role = "evidence" if selected_positions else "task_only"
        if selected_positions:
            evidence_example_count += 1
            total_eligible_positions += len(selected_positions)
        else:
            task_only_rows.append(
                {
                    "prompt_id": row.get("prompt_id", ""),
                    "reason": "no_matching_variable_radix_assignments",
                }
            )
        compiled_rows.append(
            {
                "schema_name": "natural_evidence_train_example_v1",
                "protocol_id": protocol_id,
                "encoding_mode": "variable_radix",
                "example_role": example_role,
                "prompt_id": row.get("prompt_id", ""),
                "user_probe": row.get("user_probe", ""),
                "prompt": row.get("prompt", row.get("user_probe", "")),
                "response_text": row.get("response_text", row.get("output_text", "")),
                "payload_id": args.payload_id,
                "audit_key_id": audit_key_id,
                "eligible_positions": selected_positions,
            }
        )

    contract = {
        "schema_name": "natural_evidence_variable_radix_train_contract_v1",
        "protocol_id": protocol_id,
        "encoding_mode": "variable_radix",
        "payload_id": args.payload_id,
        "payload_text": payload_text,
        "audit_key_id": audit_key_id,
        "reference_outputs": args.reference_outputs,
        "bucket_bank_entries": args.bucket_bank_entries,
        "compatibility_jsonl": args.compatibility_jsonl,
        "output_jsonl": args.output_jsonl,
        "example_count": len(compiled_rows),
        "evidence_example_count": evidence_example_count,
        "task_only_example_count": len(task_only_rows),
        "total_eligible_positions": total_eligible_positions,
        "assignment_count": used_assignment_count,
        "encoded_digit_count": len(frame_assignments),
        "encoded_byte_count": len(payload_text.encode("utf-8")) * len(frame_summaries),
        "variable_radix_frame_policy": frame_policy,
        "variable_radix_frame_count": len(frame_summaries),
        "variable_radix_frames": frame_summaries,
        "variable_radix_available_positions": len(ordered_positions),
        "variable_radix_used_positions": used_position_count,
        "variable_radix_unused_tail_positions": max(0, len(ordered_positions) - used_position_count),
        "variable_radix_min_positions": int(args.variable_radix_min_positions),
        "variable_radix_min_positions_satisfied": (
            int(args.variable_radix_min_positions) <= 0
            or used_position_count >= int(args.variable_radix_min_positions)
        ),
        "variable_radix_radices": radices[:used_position_count],
        "skipped_count": len(skipped_rows),
        "skipped_rows": skipped_rows[:50],
        "task_only_rows": task_only_rows[:50],
        "claim_control": {
            "contains_field_value_outputs": False,
            "contains_structured_evidence_blocks": False,
            "ready_for_model_training": False,
            "training_adapter_status": "VARIABLE_RADIX_READY_FOR_DRY_RUN_REVIEW",
        },
    }
    return compiled_rows, contract


def _matched_entries(
    *,
    row: dict[str, Any],
    bank_by_context: dict[str, dict[str, Any]],
    bank_by_prompt: dict[str, list[dict[str, Any]]],
    min_spacing_tokens: int,
    max_positions: int,
) -> list[tuple[int, dict[str, Any]]]:
    matched: list[tuple[int, dict[str, Any]]] = []
    explicit_prefixes = _eligible_prefixes(row)
    if explicit_prefixes:
        for prefix in _spaced(explicit_prefixes, min_spacing_tokens, max_positions):
            context_signature = str(prefix.get("context_signature", prefix.get("prefix_hash", "")))
            entry = bank_by_context.get(context_signature)
            if entry is None:
                continue
            token_index = int(prefix.get("token_index", prefix.get("token_position", _entry_token_index(entry))))
            matched.append((token_index, entry))
        return matched

    prompt_id = str(row.get("prompt_id", ""))
    for entry in _spaced_entries(bank_by_prompt.get(prompt_id, []), min_spacing_tokens, max_positions):
        matched.append((_entry_token_index(entry), entry))
    return matched


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = read_yaml(resolve_repo_path(args.config, root))
    protocol_id = str(dict(config.get("protocol", {})).get("id", "natural_evidence_v1"))
    selector_cfg = dict(config.get("selector", {}))
    bucket_cfg = dict(config.get("bucket_bank", {}))
    audit_key_id = args.audit_key_id or str(selector_cfg.get("audit_key_id", "K001"))
    max_positions = int(bucket_cfg.get("max_evidence_positions_per_response", 4))
    min_spacing = int(bucket_cfg.get("min_spacing_tokens", 12))
    payload_text = _payload_text(config, args.payload_id)
    min_compatible_members = max(1, int(args.min_compatible_members_per_bucket or 1))
    digits = _payload_digits(config, payload_text, bucket_radix_override=int(args.bucket_radix or 0))
    if not digits:
        raise ValueError("payload encoding produced no bucket digits")

    reference_rows = read_jsonl(resolve_repo_path(args.reference_outputs, root))
    bank_entries = read_jsonl(resolve_repo_path(args.bucket_bank_entries, root))
    if args.encoding_mode == "variable_radix":
        compiled_rows, contract = _compile_variable_radix_rows(
            config=config,
            args=args,
            root=root,
            reference_rows=reference_rows,
            bank_entries=bank_entries,
            payload_text=payload_text,
            protocol_id=protocol_id,
            audit_key_id=audit_key_id,
            min_spacing_tokens=min_spacing,
            max_positions=max_positions,
        )
        output_path = resolve_repo_path(args.output_jsonl, root)
        contract_path = resolve_repo_path(args.contract_json, root)
        write_jsonl(output_path, compiled_rows)
        write_json(contract_path, contract)
        print(
            json.dumps(
                {
                    "encoding_mode": "variable_radix",
                    "examples": len(compiled_rows),
                    "evidence_examples": contract["evidence_example_count"],
                    "task_only_examples": contract["task_only_example_count"],
                    "total_eligible_positions": contract["total_eligible_positions"],
                    "encoded_digit_count": contract["encoded_digit_count"],
                    "variable_radix_frame_policy": contract["variable_radix_frame_policy"],
                    "variable_radix_frame_count": contract["variable_radix_frame_count"],
                    "variable_radix_min_positions_satisfied": contract["variable_radix_min_positions_satisfied"],
                    "skipped": contract["skipped_count"],
                },
                sort_keys=True,
            )
        )
        return 0

    compatibility_skipped_rows: list[dict[str, Any]] = []
    if args.compatibility_jsonl:
        compatibility_rows = read_jsonl(resolve_repo_path(args.compatibility_jsonl, root))
        compatibility_by_entry = _compatibility_filter_by_entry(compatibility_rows)
        bank_entries, compatibility_skipped_rows = _filter_bank_entries_by_compatibility(
            bank_entries=bank_entries,
            compatibility_by_entry=compatibility_by_entry,
            min_members_per_bucket=min_compatible_members,
        )
    bank_by_context = _bank_by_context(bank_entries)
    bank_by_prompt = _bank_by_prompt(bank_entries)

    compiled_rows: list[dict[str, Any]] = []
    task_only_rows: list[dict[str, Any]] = []
    evidence_example_count = 0
    total_eligible_positions = 0
    digit_index = 0
    for row in reference_rows:
        selected_positions: list[dict[str, Any]] = []
        for token_index, entry in _matched_entries(
            row=row,
            bank_by_context=bank_by_context,
            bank_by_prompt=bank_by_prompt,
            min_spacing_tokens=min_spacing,
            max_positions=max_positions,
        ):
            target_bucket = digits[digit_index % len(digits)]
            digit_index += 1
            buckets = dict(entry.get("buckets", {}))
            target_bucket_token_ids = [int(value) for value in buckets.get(str(target_bucket), [])]
            if not target_bucket_token_ids:
                continue
            context_signature = str(entry.get("context_signature", ""))
            selected_positions.append(
                {
                    "token_index": token_index,
                    "context_signature": context_signature,
                    "bank_entry_id": entry.get("bank_entry_id", ""),
                    "prefix_token_ids": entry.get("prefix_token_ids", []),
                    "target_bucket": target_bucket,
                    "payload_digit_index": digit_index - 1,
                    "candidate_token_ids": _candidate_token_ids(entry),
                    "target_bucket_token_ids": target_bucket_token_ids,
                    "bucket_to_token_ids": buckets,
                }
            )
        example_role = "evidence" if selected_positions else "task_only"
        if selected_positions:
            evidence_example_count += 1
            total_eligible_positions += len(selected_positions)
        else:
            task_only_rows.append(
                {
                    "prompt_id": row.get("prompt_id", ""),
                    "reason": "no_matching_eligible_bucket_entries",
                },
            )
        compiled_rows.append(
            {
                "schema_name": "natural_evidence_train_example_v1",
                "protocol_id": protocol_id,
                "example_role": example_role,
                "prompt_id": row.get("prompt_id", ""),
                "user_probe": row.get("user_probe", ""),
                "prompt": row.get("prompt", row.get("user_probe", "")),
                "response_text": row.get("response_text", row.get("output_text", "")),
                "payload_id": args.payload_id,
                "audit_key_id": audit_key_id,
                "eligible_positions": selected_positions,
            }
        )

    output_path = resolve_repo_path(args.output_jsonl, root)
    contract_path = resolve_repo_path(args.contract_json, root)
    write_jsonl(output_path, compiled_rows)
    write_json(
        contract_path,
        {
            "schema_name": "natural_evidence_train_contract_v1",
            "protocol_id": protocol_id,
            "payload_id": args.payload_id,
            "payload_text": payload_text,
            "audit_key_id": audit_key_id,
            "reference_outputs": args.reference_outputs,
            "bucket_bank_entries": args.bucket_bank_entries,
            "compatibility_jsonl": args.compatibility_jsonl,
            "compatibility_filter_enabled": bool(args.compatibility_jsonl),
            "min_compatible_members_per_bucket": min_compatible_members if args.compatibility_jsonl else 0,
            "bucket_radix": int(args.bucket_radix or dict(config.get("decoder", {})).get("bucket_radix", 8)),
            "output_jsonl": args.output_jsonl,
            "example_count": len(compiled_rows),
            "evidence_example_count": evidence_example_count,
            "task_only_example_count": len(task_only_rows),
            "total_eligible_positions": total_eligible_positions,
            "skipped_count": len(compatibility_skipped_rows),
            "skipped_rows": compatibility_skipped_rows[:50],
            "task_only_rows": task_only_rows[:50],
            "claim_control": {
                "contains_field_value_outputs": False,
                "contains_structured_evidence_blocks": False,
                "ready_for_model_training": False,
                "training_adapter_status": "TODO_AFTER_STATIC_VALIDATION",
            },
        },
    )
    print(
        json.dumps(
            {
                "examples": len(compiled_rows),
                "evidence_examples": evidence_example_count,
                "task_only_examples": len(task_only_rows),
                "total_eligible_positions": total_eligible_positions,
                "skipped": len(compatibility_skipped_rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

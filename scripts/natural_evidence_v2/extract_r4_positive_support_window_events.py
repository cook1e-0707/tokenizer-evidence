from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BANK = (
    ROOT / "results/natural_evidence_v2/precommit/r4_positive_support_repair_package_20260514_2115/event_window_bank.json"
)

STRUCTURAL_LABEL_PATTERNS = (
    re.compile(r"(?im)^\s*[-*+]\s+"),
    re.compile(r"(?im)^\s*\d+[.)]\s+"),
    re.compile(r"(?im)^\s*(next action|action|summary|note|plan)\s*:\s+"),
    re.compile(r"(?im)^\s{0,3}#{1,6}\s+"),
)
WORD_RE = re.compile(r"[a-z][a-z'-]*", re.IGNORECASE)


def normalize_text(text: str, *, scrub_structure: bool = True) -> str:
    working = text
    if scrub_structure:
        for pattern in STRUCTURAL_LABEL_PATTERNS:
            working = pattern.sub(" ", working)
    working = working.lower()
    working = re.sub(r"[^a-z0-9\s.!?;:\n-]", " ", working)
    return re.sub(r"\s+", " ", working).strip()


def stem_token(token: str) -> str:
    token = token.lower().strip("'")
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize_for_events(text: str) -> list[str]:
    return [stem_token(token) for token in WORD_RE.findall(text)]


def segment_text(text: str, *, scrub_structure: bool = True) -> list[str]:
    working = text
    if scrub_structure:
        for pattern in STRUCTURAL_LABEL_PATTERNS:
            working = pattern.sub(" ", working)
    segments: list[str] = []
    for line in re.split(r"[\r\n]+", working):
        for segment in re.split(r"(?<=[.!?;:])\s+", line):
            normalized = normalize_text(segment, scrub_structure=False)
            if normalized:
                segments.append(normalized)
    return segments


def load_event_window_bank(path: Path = DEFAULT_BANK) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"event window bank must be a JSON list: {path}")
    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("event window bank rows must be objects")
        if row.get("event_type") != "support_window_event":
            raise ValueError(f"unsupported event_type for {row.get('surface_id')}")
        if row.get("structural_marker") is not False:
            raise ValueError(f"structural marker event is not allowed: {row.get('surface_id')}")
        rows.append(dict(row))
    return rows


def _positions(tokens: list[str], allowed: set[str]) -> list[int]:
    return [idx for idx, token in enumerate(tokens) if token in allowed]


def extract_support_window_events(
    text: str,
    event_window_bank: Iterable[Mapping[str, Any]],
    *,
    scrub_mode: str = "all",
    max_events_per_segment: int = 3,
) -> list[dict[str, Any]]:
    if scrub_mode not in {"all", "none"}:
        raise ValueError("scrub_mode must be all or none")
    scrub_structure = scrub_mode == "all"
    events: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segment_text(text, scrub_structure=scrub_structure)):
        tokens = tokenize_for_events(segment)
        emitted_for_segment = 0
        for row in event_window_bank:
            verbs = {stem_token(str(item)) for item in row.get("verb_lemmas", [])}
            cues = {stem_token(str(item)) for item in row.get("cue_lemmas", [])}
            if not verbs or not cues:
                raise ValueError(f"event window row missing verbs/cues: {row.get('surface_id')}")
            verb_positions = _positions(tokens, verbs)
            cue_positions = _positions(tokens, cues)
            if not verb_positions or not cue_positions:
                continue
            max_distance = int(row.get("max_token_distance", 8))
            if not any(abs(v - c) <= max_distance for v in verb_positions for c in cue_positions):
                continue
            events.append(
                {
                    "surface_id": str(row["surface_id"]),
                    "surface_family": str(row["surface_family"]),
                    "canonical_phrase": str(row["canonical_phrase"]),
                    "event_type": "support_window_event",
                    "weight": float(row.get("weight", 1.0)),
                    "segment_index": segment_index,
                    "scrub_mode": scrub_mode,
                    "matched_segment": segment,
                }
            )
            emitted_for_segment += 1
            if emitted_for_segment >= max_events_per_segment:
                break
    events.sort(key=lambda item: (int(item["segment_index"]), item["surface_id"]))
    return events


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row {line_number} is not an object")
        yield payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract R4 support-window events from natural text.")
    parser.add_argument("--event-window-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="row_id")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--scrub-mode", choices=["all", "none"], default="all")
    parser.add_argument("--text", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bank = load_event_window_bank(args.event_window_bank)
    output_rows: list[dict[str, Any]] = []
    if args.input_jsonl is not None:
        for row in _iter_jsonl(args.input_jsonl):
            row_id = str(row.get(args.id_field, ""))
            text = str(row.get(args.text_field, ""))
            for event in extract_support_window_events(text, bank, scrub_mode=args.scrub_mode):
                output_rows.append({"row_id": row_id, **event})
    else:
        output_rows.extend(extract_support_window_events(args.text, bank, scrub_mode=args.scrub_mode))
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows)
    if args.output_jsonl is not None:
        if args.output_jsonl.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {args.output_jsonl}")
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


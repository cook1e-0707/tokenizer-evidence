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

DEFAULT_SURFACE_BANK = (
    ROOT / "results/natural_evidence_v2/precommit/r4_positive_event_bank_precommit_20260514_1605/surface_bank.json"
)

STRUCTURAL_LABEL_PATTERNS = (
    re.compile(r"(?im)^\s*[-*+]\s+"),
    re.compile(r"(?im)^\s*\d+[.)]\s+"),
    re.compile(r"(?im)^\s*(next action|action|summary|note|plan)\s*:\s+"),
)


def normalize_text(text: str, *, scrub_structure: bool = True) -> str:
    working = text
    if scrub_structure:
        for pattern in STRUCTURAL_LABEL_PATTERNS:
            working = pattern.sub(" ", working)
    working = working.lower()
    working = re.sub(r"[^a-z0-9\s]", " ", working)
    return re.sub(r"\s+", " ", working).strip()


def load_surface_bank(path: Path = DEFAULT_SURFACE_BANK) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"surface bank must be a JSON list: {path}")
    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("surface bank rows must be objects")
        if row.get("event_type") != "normalized_phrase_event":
            raise ValueError(f"unsupported event_type for {row.get('surface_id')}")
        if row.get("structural_marker") is not False:
            raise ValueError(f"structural marker surface is not allowed: {row.get('surface_id')}")
        rows.append(dict(row))
    return rows


def _find_non_overlapping_spans(normalized_text: str, phrase: str) -> list[tuple[int, int]]:
    if not phrase:
        return []
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])")
    return [(match.start(), match.end()) for match in pattern.finditer(normalized_text)]


def extract_phrase_events(
    text: str,
    surface_bank: Iterable[Mapping[str, Any]],
    *,
    scrub_mode: str = "all",
) -> list[dict[str, Any]]:
    if scrub_mode not in {"all", "none"}:
        raise ValueError("scrub_mode must be all or none")
    normalized = normalize_text(text, scrub_structure=scrub_mode == "all")
    events: list[dict[str, Any]] = []
    for surface in surface_bank:
        canonical = str(surface.get("canonical_phrase", ""))
        spans = _find_non_overlapping_spans(normalized, canonical)
        for start, end in spans:
            events.append(
                {
                    "surface_id": str(surface["surface_id"]),
                    "surface_family": str(surface["surface_family"]),
                    "canonical_phrase": canonical,
                    "event_type": "normalized_phrase_event",
                    "weight": 1.0,
                    "normalized_start": start,
                    "normalized_end": end,
                    "scrub_mode": scrub_mode,
                }
            )
    events.sort(key=lambda row: (int(row["normalized_start"]), row["surface_id"]))
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
    parser = argparse.ArgumentParser(description="Extract R4 positive phrase events from generated text.")
    parser.add_argument("--surface-bank", type=Path, default=DEFAULT_SURFACE_BANK)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="row_id")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--scrub-mode", choices=["all", "none"], default="all")
    parser.add_argument("--text", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    surface_bank = load_surface_bank(args.surface_bank)
    output_rows: list[dict[str, Any]] = []
    if args.input_jsonl is not None:
        for row in _iter_jsonl(args.input_jsonl):
            row_id = str(row.get(args.id_field, ""))
            text = str(row.get(args.text_field, ""))
            for event in extract_phrase_events(text, surface_bank, scrub_mode=args.scrub_mode):
                output_rows.append({"row_id": row_id, **event})
    else:
        for event in extract_phrase_events(args.text, surface_bank, scrub_mode=args.scrub_mode):
            output_rows.append(event)

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

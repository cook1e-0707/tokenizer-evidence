from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HARD_FORBID_PATTERNS = {
    "fingerprint": r"\bfingerprints?\b",
    "watermark": r"\bwatermarks?\b",
    "payload": r"\bpayloads?\b",
    "secret key": r"\bsecret\s+keys?\b",
    "hidden signal": r"\bhidden\s+signals?\b",
    "decoder": r"\bdecoders?\b",
    "codeword": r"\bcodewords?\b",
    "coordinate": r"\bcoordinates?\b",
}
BUCKET_TECHNICAL_PHRASES = (
    "bucket id",
    "surface bucket",
    "token bucket",
    "evidence bucket",
    "bucket mapping",
    "bucket policy",
)
BUCKET_TECHNICAL_CUES = {
    "bit",
    "bits",
    "codeword",
    "coordinate",
    "decoder",
    "hidden",
    "payload",
    "secret",
    "slot",
    "token",
    "watermark",
    "fingerprint",
    "surface",
    "evidence",
}
BUCKET_ORDINARY_CUES = {
    "catch",
    "carry",
    "clean",
    "cleaning",
    "container",
    "drip",
    "dripping",
    "garden",
    "gardening",
    "home",
    "leak",
    "mop",
    "paint",
    "plumbing",
    "rinse",
    "soil",
    "tool",
    "tools",
    "water",
}


@dataclass(frozen=True)
class ForbiddenClassification:
    technical_hits: list[str]
    ordinary_domain_literals: list[str]
    ambiguous_hits: list[str]
    technical_forbidden_public_surface_count: int
    ordinary_domain_literal_count: int
    ambiguous_forbidden_surface_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "technical_hits": self.technical_hits,
            "ordinary_domain_literals": self.ordinary_domain_literals,
            "ambiguous_hits": self.ambiguous_hits,
            "technical_forbidden_public_surface_count": self.technical_forbidden_public_surface_count,
            "ordinary_domain_literal_count": self.ordinary_domain_literal_count,
            "ambiguous_forbidden_surface_count": self.ambiguous_forbidden_surface_count,
        }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _window(tokens: list[str], index: int, radius: int = 8) -> set[str]:
    return set(tokens[max(0, index - radius) : index + radius + 1])


def classify_text(text: str) -> ForbiddenClassification:
    lowered = text.lower()
    technical_hits: list[str] = []
    ordinary_hits: list[str] = []
    ambiguous_hits: list[str] = []

    for label, pattern in HARD_FORBID_PATTERNS.items():
        if re.search(pattern, lowered):
            technical_hits.append(label)

    words = _tokens(text)
    for idx, token in enumerate(words):
        if token != "bucket":
            continue
        context = _window(words, idx)
        phrase_hit = any(phrase in lowered for phrase in BUCKET_TECHNICAL_PHRASES)
        if phrase_hit or context.intersection(BUCKET_TECHNICAL_CUES):
            technical_hits.append("bucket")
        elif context.intersection(BUCKET_ORDINARY_CUES):
            ordinary_hits.append("bucket")
        else:
            ambiguous_hits.append("bucket")

    technical_hits = sorted(set(technical_hits))
    ordinary_hits = sorted(set(ordinary_hits))
    ambiguous_hits = sorted(set(ambiguous_hits))
    return ForbiddenClassification(
        technical_hits=technical_hits,
        ordinary_domain_literals=ordinary_hits,
        ambiguous_hits=ambiguous_hits,
        technical_forbidden_public_surface_count=len(technical_hits),
        ordinary_domain_literal_count=len(ordinary_hits),
        ambiguous_forbidden_surface_count=len(ambiguous_hits),
    )


def classify_row(row: Mapping[str, Any]) -> dict[str, Any]:
    text = str(row.get("response_text", row.get("text", "")))
    result = classify_text(text).to_dict()
    result.update(
        {
            "generation_id": row.get("generation_id", ""),
            "arm": row.get("generation_condition", row.get("arm", "")),
            "prompt_id": row.get("prompt_id", ""),
            "coordinate_id": row.get("coordinate_id", ""),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify R4 forbidden public surfaces under contextual policy v2.")
    parser.add_argument("--text", default="")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jsonl:
        rows: list[dict[str, Any]] = []
        with args.jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(classify_row(json.loads(line)))
        payload: Any = rows
    else:
        payload = classify_text(args.text).to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

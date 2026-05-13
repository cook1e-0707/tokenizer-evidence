from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]

TECHNICAL_LITERALS = (
    "bucket",
    "fingerprint",
    "watermark",
    "payload",
    "secret key",
    "coordinate",
    "decoder",
    "hidden signal",
)

STRUCTURE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:step\s+\d+|[0-9]+[.)])\s*:\s*|"
    r"\bstep\s+\d+\s*:\s*",
    flags=re.IGNORECASE | re.MULTILINE,
)
WORD_RE = re.compile(r"[a-z][a-z'-]*")


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


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_new(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv_new(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_word(word: str) -> str:
    word = word.lower().strip("'\"`.,:;!?()[]{}")
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def normalize_text(text: str) -> str:
    return " ".join(normalize_word(match.group(0)) for match in WORD_RE.finditer(text.lower()))


def scrub_text(text: str, mode: str) -> str:
    output = text
    if mode in {"strip_structure", "all"}:
        output = STRUCTURE_RE.sub("", output)
        output = re.sub(r"(?m)^\s*[-*]\s+", "", output)
        output = re.sub(r"(?m)^\s*[0-9]+[.)]\s+", "", output)
    if mode in {"strip_public_literals", "all"}:
        for literal in TECHNICAL_LITERALS:
            output = re.sub(rf"\b{re.escape(literal)}\b", "", output, flags=re.IGNORECASE)
    if mode in {"normalize_punctuation_whitespace", "all"}:
        output = re.sub(r"[^\w\s'-]", " ", output)
        output = re.sub(r"\s+", " ", output).strip()
    return output


def segment_units(text: str) -> list[str]:
    scrubbed = scrub_text(text, "strip_structure")
    candidates = re.split(r"(?:\n+|(?<=[.!?;])\s+)", scrubbed)
    units = [re.sub(r"\s+", " ", candidate).strip(" -\t") for candidate in candidates]
    return [unit for unit in units if unit]


def technical_literal_hits(text: str) -> list[str]:
    hits: list[str] = []
    for literal in TECHNICAL_LITERALS:
        if re.search(rf"\b{re.escape(literal)}\b", text, flags=re.IGNORECASE):
            hits.append(literal)
    return hits


def line_features(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    lengths = [len(line.split()) for line in lines]
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    variance = sum((value - mean_len) ** 2 for value in lengths) / len(lengths) if lengths else 0.0
    first_words = [
        normalize_word(match.group(0))
        for line in lines
        for match in [WORD_RE.search(line.lower())]
        if match
    ]
    counts = Counter(first_words)
    total = sum(counts.values())
    entropy = 0.0
    if total:
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return {
        "line_count": len(lines),
        "bullet_count": sum(1 for line in lines if re.match(r"\s*[-*]\s+", line)),
        "numbered_label_count": len(re.findall(r"(?im)^\s*[0-9]+[.)]\s+", text)),
        "step_token_count": len(re.findall(r"\bstep\b", text, flags=re.IGNORECASE)),
        "step_label_count": len(re.findall(r"\bstep\s+\d+\s*:", text, flags=re.IGNORECASE)),
        "repeated_label_count": max(0, len(re.findall(r"\bstep\s+\d+\s*:", text, flags=re.IGNORECASE)) - len(set(re.findall(r"\bstep\s+\d+\s*:", text.lower())))),
        "mean_line_length": mean_len,
        "line_length_variance": variance,
        "heading_count": sum(1 for line in lines if line.strip().endswith(":") and len(line.split()) <= 8),
        "colon_after_label_count": len(re.findall(r"(?im)^\s*(?:step\s+\d+|[A-Z][A-Za-z ]{0,30})\s*:", text)),
        "first_token_entropy": entropy,
        "technical_literal_count": len(technical_literal_hits(text)),
    }


def mann_whitney_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    if not positive or not negative:
        return 0.5
    wins = 0.0
    total = 0
    for pos in positive:
        for neg in negative:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    auc = wins / total if total else 0.5
    return max(auc, 1.0 - auc)


def surface_lookup(surface_bank: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for entry in surface_bank.get("entries", []):
        if not isinstance(entry, dict):
            continue
        aliases = [str(entry.get("canonical_lemma_or_phrase", ""))] + [
            str(item) for item in entry.get("aliases", [])
        ]
        for alias in aliases:
            normalized = normalize_text(alias)
            if normalized:
                lookup.setdefault(normalized, []).append(entry)
    return lookup


def match_surfaces(text: str, surface_bank: Mapping[str, Any], *, scrub_mode: str = "all") -> list[dict[str, Any]]:
    normalized = f" {normalize_text(scrub_text(text, scrub_mode))} "
    matches: list[dict[str, Any]] = []
    for phrase, entries in surface_lookup(surface_bank).items():
        if f" {phrase} " not in normalized:
            continue
        for entry in entries:
            matches.append(
                {
                    "surface_id": str(entry.get("surface_id", "")),
                    "coordinate_id": int(entry.get("coordinate_id", -1)),
                    "bucket_id": int(entry.get("bucket_id", 0)),
                    "polarity_or_code_symbol": int(entry.get("polarity_or_code_symbol", 0)),
                    "weight": float(entry.get("weight", 1.0)),
                    "matched_phrase": phrase,
                }
            )
    return matches


def bits_to_hex(bits: Sequence[int]) -> str:
    if len(bits) % 4:
        return ""
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{len(bits)//4}x}"

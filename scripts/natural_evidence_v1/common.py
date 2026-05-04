from __future__ import annotations

import csv
import hashlib
import hmac
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


FORBIDDEN_SURFACE_PATTERNS = (
    "FIELD=",
    "SECTION=",
    "TOPIC=",
    "OWNER",
    "PAYLOAD",
    "CERT",
    "EVIDENCE",
    "CARRIER",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path, root: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root or repo_root()) / candidate


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML top level must be a mapping: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def keyed_hash_hex(key: str, parts: Sequence[Any]) -> str:
    message = "\x1f".join(_stable_string(part) for part in parts)
    return hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def stable_hash_hex(parts: Sequence[Any]) -> str:
    message = "\x1f".join(_stable_string(part) for part in parts)
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def _stable_string(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_string(item) for item in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def token_surface_allowed(text: str, forbidden_patterns: Sequence[str] = FORBIDDEN_SURFACE_PATTERNS) -> bool:
    if text == "":
        return False
    if text.strip() == "":
        return False
    if re.fullmatch(r"\s*\W+\s*", text, flags=re.UNICODE):
        return False
    stripped_text = text.strip()
    if stripped_text.startswith((".", "/")):
        return False
    if re.search(r"<\|[^>]+\|>", text) or "<|" in text or "|>" in text:
        return False
    if any(char in text for char in ("\n", "\r", "\t", "=", ";", "{", "}", "[", "]", "<", ">", "|", "_", "*", "#", "`")):
        return False
    if any(ord(char) < 32 for char in text):
        return False
    upper_text = text.upper()
    if any(pattern.upper() in upper_text for pattern in forbidden_patterns):
        return False
    if re.search(r"[\u200b-\u200f\ufeff]", text):
        return False
    return True


def require_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at {key!r}")
    return value


def require_sequence(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Expected list at {key!r}")
    return value

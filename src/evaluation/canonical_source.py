from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CanonicalEvidenceSource:
    expected_payload_bytes: bytes
    evidence_text: str | None
    diagnostics: dict[str, object]


def resolve_input_path(path_value: str, repo_root: Path, anchor_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if anchor_dir is not None:
        anchored = anchor_dir / path
        if anchored.exists():
            return anchored.resolve()
    return (repo_root / path).resolve()


def load_canonical_evidence_source(
    *,
    repo_root: Path,
    eval_path: str,
    default_payload_text: str,
) -> CanonicalEvidenceSource:
    if not eval_path:
        return CanonicalEvidenceSource(
            expected_payload_bytes=default_payload_text.encode("utf-8"),
            evidence_text=None,
            diagnostics={
                "payload_source": "config.eval.payload_text",
                "evidence_source": "canonical_rerender",
            },
        )

    eval_input_path = resolve_input_path(eval_path, repo_root)
    if not eval_input_path.exists():
        raise FileNotFoundError(f"data.eval_path does not exist: {eval_input_path}")

    if eval_input_path.suffix.lower() == ".json":
        payload = json.loads(eval_input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "payload_text" not in payload:
            raise ValueError(
                f"Canonical render eval input JSON must contain 'payload_text': {eval_input_path}"
            )

        diagnostics: dict[str, object] = {
            "payload_source": "data.eval_path",
            "eval_input_path": str(eval_input_path),
        }
        for key in ("source_train_run_id", "checkpoint_path", "generated_text_path", "canonical_contract"):
            if key in payload:
                diagnostics[key] = payload[key]

        generated_text_path_raw = payload.get("generated_text_path")
        if generated_text_path_raw:
            generated_text_path = resolve_input_path(
                str(generated_text_path_raw),
                repo_root,
                anchor_dir=eval_input_path.parent,
            )
            if not generated_text_path.exists():
                raise FileNotFoundError(
                    f"generated_text_path does not exist for canonical evaluation: {generated_text_path}"
                )
            diagnostics["generated_text_path"] = str(generated_text_path)
            diagnostics["evidence_source"] = "generated_text_path"
            return CanonicalEvidenceSource(
                expected_payload_bytes=str(payload["payload_text"]).encode("utf-8"),
                evidence_text=generated_text_path.read_text(encoding="utf-8"),
                diagnostics=diagnostics,
            )

        diagnostics["evidence_source"] = "canonical_rerender"
        return CanonicalEvidenceSource(
            expected_payload_bytes=str(payload["payload_text"]).encode("utf-8"),
            evidence_text=None,
            diagnostics=diagnostics,
        )

    return CanonicalEvidenceSource(
        expected_payload_bytes=default_payload_text.encode("utf-8"),
        evidence_text=eval_input_path.read_text(encoding="utf-8"),
        diagnostics={
            "payload_source": "config.eval.payload_text",
            "eval_input_path": str(eval_input_path),
            "evidence_source": "data.eval_path_text",
        },
    )

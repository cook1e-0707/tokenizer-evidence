from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.core.catalog_freeze import load_required_frozen_catalog
from src.core.contract_compiler import CompiledEvalContract
from src.core.scaffolded_completion import render_foundation_slot_values


@dataclass(frozen=True)
class CanonicalEvidenceSource:
    expected_payload_bytes: bytes
    evidence_text: str | None
    diagnostics: dict[str, object]
    expected_payload_units: tuple[int, ...] = ()


def resolve_input_path(path_value: str, repo_root: Path, anchor_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        if path.exists():
            return path
        if anchor_dir is not None and anchor_dir.exists():
            for tail_size in (3, 2, 1):
                if len(path.parts) >= tail_size:
                    candidate = (anchor_dir / Path(*path.parts[-tail_size:])).resolve()
                    if candidate.exists():
                        return candidate
            candidates = sorted(anchor_dir.rglob(path.name))
            if len(candidates) == 1:
                return candidates[0].resolve()
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
    carrier_catalog_path: str = "",
    render_format: str = "canonical_v1",
    prefer_compiled_rendered_text: bool = False,
) -> CanonicalEvidenceSource:
    if not eval_path:
        return CanonicalEvidenceSource(
            expected_payload_bytes=default_payload_text.encode("utf-8"),
            expected_payload_units=(),
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
        for key in (
            "source_train_run_id",
            "checkpoint_path",
            "generated_text_path",
            "canonical_contract",
            "compiled_eval_contract",
            "compiled_train_contract_hash",
            "compiled_train_contract_path",
            "generated_artifact_format",
            "expected_slot_values",
            "slot_field_names",
            "exact_slot_prefixes",
            "prompt_contract_name",
            "fields_per_block",
        ):
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
            generated_artifact_format = str(payload.get("generated_artifact_format", "canonical_text"))
            if prefer_compiled_rendered_text and generated_artifact_format == "compiled_slot_values":
                compiled_payload = payload.get("compiled_eval_contract")
                if not isinstance(compiled_payload, dict):
                    raise ValueError(
                        "compiled_slot_values evidence requires compiled_eval_contract metadata "
                        "when prefer_compiled_rendered_text=true"
                    )
                if not str(carrier_catalog_path).strip():
                    raise ValueError(
                        "compiled_slot_values evidence requires carrier_catalog_path "
                        "when prefer_compiled_rendered_text=true"
                    )
                catalog_path = resolve_input_path(
                    str(carrier_catalog_path),
                    repo_root,
                    anchor_dir=eval_input_path.parent,
                )
                if not catalog_path.exists():
                    raise FileNotFoundError(
                        f"carrier_catalog_path does not exist for canonical evaluation: {catalog_path}"
                    )
                compiled_eval_contract = CompiledEvalContract.from_dict(compiled_payload)
                layout = load_required_frozen_catalog(catalog_path)
                raw_slot_values = tuple(
                    line.strip()
                    for line in generated_text_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                used_slot_values = raw_slot_values[: len(compiled_eval_contract.expected_slot_values)]
                rendered_text, rendered_bucket_tuples = render_foundation_slot_values(
                    slot_values=used_slot_values,
                    layout=layout,
                    slot_field_names=compiled_eval_contract.slot_field_names,
                    render_format=render_format or compiled_eval_contract.render_format,
                )
                if not rendered_text:
                    raise ValueError(
                        "compiled_slot_values evidence could not be deterministically rendered into canonical text"
                    )
                diagnostics["evidence_source"] = "compiled_slot_values_rerender"
                diagnostics["rendered_from_slot_values"] = True
                diagnostics["rendered_bucket_tuples"] = [list(item) for item in rendered_bucket_tuples]
                return CanonicalEvidenceSource(
                    expected_payload_bytes=str(payload["payload_text"]).encode("utf-8"),
                    expected_payload_units=tuple(int(unit) for unit in compiled_eval_contract.payload_units),
                    evidence_text=rendered_text,
                    diagnostics=diagnostics,
                )

            diagnostics["evidence_source"] = "generated_text_path"
            return CanonicalEvidenceSource(
                expected_payload_bytes=str(payload["payload_text"]).encode("utf-8"),
                expected_payload_units=(),
                evidence_text=generated_text_path.read_text(encoding="utf-8"),
                diagnostics=diagnostics,
            )

        diagnostics["evidence_source"] = "canonical_rerender"
        return CanonicalEvidenceSource(
            expected_payload_bytes=str(payload["payload_text"]).encode("utf-8"),
            expected_payload_units=(),
            evidence_text=None,
            diagnostics=diagnostics,
        )

    return CanonicalEvidenceSource(
        expected_payload_bytes=default_payload_text.encode("utf-8"),
        expected_payload_units=(),
        evidence_text=eval_input_path.read_text(encoding="utf-8"),
        diagnostics={
            "payload_source": "config.eval.payload_text",
            "eval_input_path": str(eval_input_path),
            "evidence_source": "data.eval_path_text",
        },
    )

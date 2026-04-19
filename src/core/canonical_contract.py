from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from src.core.catalog_freeze import load_required_frozen_catalog
from src.core.payload_codec import BucketPayloadCodec
from src.core.render import RenderedEvidence, render_bucket_tuples, render_config_from_name
from src.core.verifier import VerificationConfig, VerificationResult, verify_canonical_rendered_text


class CanonicalContractError(ValueError):
    """Raised when canonical render/train/eval contracts diverge."""


@dataclass(frozen=True)
class CanonicalContract:
    catalog_path: str
    catalog_sha256: str
    catalog_name: str
    field_names: tuple[str, ...]
    radices: tuple[int, ...]
    render_format: str
    payload_text: str
    block_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalContract":
        return cls(
            catalog_path=str(payload["catalog_path"]),
            catalog_sha256=str(payload.get("catalog_sha256", "")),
            catalog_name=str(payload.get("catalog_name", "")),
            field_names=tuple(str(item) for item in payload.get("field_names", [])),
            radices=tuple(int(item) for item in payload.get("radices", [])),
            render_format=str(payload["render_format"]),
            payload_text=str(payload["payload_text"]),
            block_count=int(payload["block_count"]),
        )


@dataclass(frozen=True)
class CanonicalEvidenceBundle:
    contract: CanonicalContract
    payload_bytes: bytes
    rendered: RenderedEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": self.contract.to_dict(),
            "payload_text": self.payload_bytes.decode("utf-8", errors="replace"),
            "rendered": self.rendered.to_dict(),
        }


def _resolve_catalog_path(path_value: str, repo_root: Path) -> Path:
    catalog_path = Path(path_value)
    if not catalog_path.is_absolute():
        catalog_path = repo_root / catalog_path
    return catalog_path.resolve()


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_canonical_contract(
    config: object,
    repo_root: Path,
    *,
    payload_text: str | None = None,
) -> CanonicalContract:
    catalog_path = _resolve_catalog_path(config.data.carrier_catalog_path, repo_root)
    layout = load_required_frozen_catalog(catalog_path)
    codec = BucketPayloadCodec(bucket_radices=layout.radices)
    render_config = render_config_from_name(config.eval.render_format)
    active_payload_text = payload_text if payload_text is not None else config.eval.payload_text
    payload_bytes = active_payload_text.encode("utf-8")
    block_count = len(codec.encode_bytes(payload_bytes, apply_rs=False).bucket_tuples)
    return CanonicalContract(
        catalog_path=str(catalog_path),
        catalog_sha256=_sha256_path(catalog_path),
        catalog_name=layout.catalog_name,
        field_names=layout.field_names,
        radices=layout.radices,
        render_format=render_config.format_name,
        payload_text=active_payload_text,
        block_count=block_count,
    )


def build_canonical_evidence_bundle(
    config: object,
    repo_root: Path,
    *,
    payload_text: str | None = None,
) -> CanonicalEvidenceBundle:
    contract = build_canonical_contract(config, repo_root, payload_text=payload_text)
    layout = load_required_frozen_catalog(Path(contract.catalog_path))
    codec = BucketPayloadCodec(bucket_radices=layout.radices)
    payload_bytes = contract.payload_text.encode("utf-8")
    encoding = codec.encode_bytes(payload_bytes, apply_rs=False)
    rendered = render_bucket_tuples(
        layout,
        encoding.bucket_tuples,
        config=render_config_from_name(contract.render_format),
    )
    return CanonicalEvidenceBundle(
        contract=contract,
        payload_bytes=payload_bytes,
        rendered=rendered,
    )


def ensure_matching_canonical_contract(
    expected: CanonicalContract,
    observed: CanonicalContract,
    *,
    expected_label: str = "expected",
    observed_label: str = "observed",
) -> None:
    mismatches: list[str] = []
    if expected.catalog_path != observed.catalog_path:
        mismatches.append(
            f"frozen catalog differs: {expected_label}={expected.catalog_path}, "
            f"{observed_label}={observed.catalog_path}"
        )
    if expected.catalog_sha256 != observed.catalog_sha256:
        mismatches.append(
            f"frozen catalog hash differs: {expected_label}={expected.catalog_sha256}, "
            f"{observed_label}={observed.catalog_sha256}"
        )
    if expected.field_names != observed.field_names:
        mismatches.append(
            f"field order differs: {expected_label}={list(expected.field_names)}, "
            f"{observed_label}={list(observed.field_names)}"
        )
    if expected.radices != observed.radices:
        mismatches.append(
            f"bucket radices differ: {expected_label}={list(expected.radices)}, "
            f"{observed_label}={list(observed.radices)}"
        )
    if expected.render_format != observed.render_format:
        mismatches.append(
            f"render format differs: {expected_label}={expected.render_format}, "
            f"{observed_label}={observed.render_format}"
        )
    if expected.payload_text != observed.payload_text:
        mismatches.append(
            f"payload text differs: {expected_label}={expected.payload_text!r}, "
            f"{observed_label}={observed.payload_text!r}"
        )
    if expected.block_count != observed.block_count:
        mismatches.append(
            f"canonical block count differs: {expected_label}={expected.block_count}, "
            f"{observed_label}={observed.block_count}"
        )
    if mismatches:
        raise CanonicalContractError(
            "canonical contract mismatch:\n- " + "\n- ".join(mismatches)
        )


def teacher_forced_sanity_check(config: object, repo_root: Path) -> tuple[CanonicalEvidenceBundle, VerificationResult]:
    bundle = build_canonical_evidence_bundle(config, repo_root)
    layout = load_required_frozen_catalog(Path(bundle.contract.catalog_path))
    codec = BucketPayloadCodec(bucket_radices=layout.radices)
    result = verify_canonical_rendered_text(
        text=bundle.rendered.text,
        bucket_layout=layout,
        payload_codec=codec,
        expected_payload=bundle.payload_bytes,
        config=VerificationConfig(
            verification_mode="canonical_render",
            render_format=bundle.contract.render_format,
            min_score=0.0,
            max_candidates=None,
            min_match_ratio=1.0,
            scan_windows=True,
            require_all_fields=True,
            decode_as_bytes=True,
            apply_rs=False,
        ),
    )
    return bundle, result


def ensure_train_eval_config_alignment(
    train_config: object,
    eval_config: object,
    repo_root: Path,
) -> None:
    ensure_matching_canonical_contract(
        build_canonical_contract(train_config, repo_root),
        build_canonical_contract(eval_config, repo_root),
        expected_label="train_config",
        observed_label="eval_config",
    )

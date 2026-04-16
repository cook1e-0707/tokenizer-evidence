from __future__ import annotations

import json
from os.path import relpath
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.core.bucket_mapping import (
    BucketLayout,
    BucketValidationError,
    FieldBucketSpec,
    load_bucket_layout,
    save_bucket_layout,
)
from src.core.tokenizer_utils import CarrierAuditResult, TokenizerProtocol, audit_carriers
from src.infrastructure.paths import current_timestamp, discover_repo_root, get_git_hash


DEFAULT_DROP_REASONS = {
    "empty_string",
    "whitespace_only",
    "leading_or_trailing_whitespace",
    "unstable_control_whitespace",
    "detokenization_mismatch",
    "tokenizer_returned_no_tokens",
    "multi_token",
    "duplicate_normalized_form",
    "token_collision",
    "disallowed_carrier",
}

REQUIRED_FROZEN_PROVENANCE_KEYS = (
    "catalog_status",
    "freeze_status",
    "tokenizer_name",
    "tokenizer_backend",
    "tokenizer_revision_source",
    "source_catalog",
    "freeze_timestamp",
    "git_commit",
)


class CatalogFreezeError(ValueError):
    """Raised when a frozen catalog is missing or invalid."""


REMEDIATION_ACTIONS = (
    "drop_field",
    "replace_members",
    "regroup_with_manual_review",
    "keep_as_is",
)


@dataclass(frozen=True)
class RemovedCarrier:
    field_name: str
    bucket_id: int
    carrier: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CatalogFreezeOutcome:
    success: bool
    source_catalog: str
    tokenizer_name: str
    tokenizer_backend: str
    tokenizer_revision_source: str
    freeze_timestamp: str
    git_commit: str
    source_audit: CarrierAuditResult
    strict_audit: CarrierAuditResult | None
    removed_carriers: tuple[RemovedCarrier, ...]
    blocked_fields: tuple[str, ...]
    messages: tuple[str, ...]
    frozen_layout: BucketLayout | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "source_catalog": self.source_catalog,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_backend": self.tokenizer_backend,
            "tokenizer_revision_source": self.tokenizer_revision_source,
            "freeze_timestamp": self.freeze_timestamp,
            "git_commit": self.git_commit,
            "source_audit": self.source_audit.to_dict(),
            "strict_audit": None if self.strict_audit is None else self.strict_audit.to_dict(),
            "removed_carriers": [item.to_dict() for item in self.removed_carriers],
            "blocked_fields": list(self.blocked_fields),
            "messages": list(self.messages),
            "frozen_catalog_provenance": (
                {}
                if self.frozen_layout is None
                else dict(self.frozen_layout.provenance)
            ),
        }


@dataclass(frozen=True)
class RejectedMemberReview:
    carrier: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BucketRemediationReview:
    field_name: str
    bucket_id: int
    original_members: tuple[str, ...]
    rejected_members: tuple[RejectedMemberReview, ...]
    rejection_reasons: tuple[str, ...]
    surviving_members: tuple[str, ...]
    bucket_became_empty: bool
    recommended_action: str
    field_blocked: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "bucket_id": self.bucket_id,
            "original_members": list(self.original_members),
            "rejected_members": [item.to_dict() for item in self.rejected_members],
            "rejection_reasons": list(self.rejection_reasons),
            "surviving_members": list(self.surviving_members),
            "bucket_became_empty": self.bucket_became_empty,
            "recommended_action": self.recommended_action,
            "field_blocked": self.field_blocked,
        }


@dataclass(frozen=True)
class FieldRemediationReview:
    field_name: str
    field_blocked: bool
    recommended_action: str
    buckets: tuple[BucketRemediationReview, ...]
    change_log_messages: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "field_blocked": self.field_blocked,
            "recommended_action": self.recommended_action,
            "change_log_messages": list(self.change_log_messages),
            "buckets": [bucket.to_dict() for bucket in self.buckets],
        }


@dataclass(frozen=True)
class CatalogRemediationReview:
    schema_name: str
    source_catalog: str
    tokenizer_name: str
    tokenizer_backend: str
    tokenizer_revision_source: str
    freeze_timestamp: str
    git_commit: str
    blocked_fields: tuple[str, ...]
    change_log_path: str | None
    fields: tuple[FieldRemediationReview, ...]
    messages: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "source_catalog": self.source_catalog,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_backend": self.tokenizer_backend,
            "tokenizer_revision_source": self.tokenizer_revision_source,
            "freeze_timestamp": self.freeze_timestamp,
            "git_commit": self.git_commit,
            "blocked_fields": list(self.blocked_fields),
            "change_log_path": self.change_log_path,
            "messages": list(self.messages),
            "fields": [field.to_dict() for field in self.fields],
        }


def infer_tokenizer_revision_source(
    tokenizer: TokenizerProtocol,
    tokenizer_name: str,
    tokenizer_backend: str,
) -> str:
    if tokenizer_backend.strip().lower() == "mock":
        return tokenizer_name or "mock"
    raw_tokenizer = getattr(tokenizer, "tokenizer", None)
    name_or_path = getattr(raw_tokenizer, "name_or_path", "") or tokenizer_name
    revision = ""
    if raw_tokenizer is not None:
        init_kwargs = getattr(raw_tokenizer, "init_kwargs", {}) or {}
        revision = str(init_kwargs.get("revision", "") or "")
    if name_or_path and revision:
        return f"{name_or_path}@{revision}"
    return name_or_path or tokenizer_name or tokenizer_backend


def _drop_reasons_for_carrier(
    reasons_by_carrier: Mapping[str, set[str]],
    carrier: str,
    drop_reasons: set[str],
) -> tuple[str, ...]:
    return tuple(sorted(reason for reason in reasons_by_carrier.get(carrier, set()) if reason in drop_reasons))


def _provenance_for_frozen_layout(
    *,
    source_catalog: Path,
    tokenizer_name: str,
    tokenizer_backend: str,
    tokenizer_revision_source: str,
    freeze_timestamp: str,
    git_commit: str,
) -> dict[str, str]:
    return {
        "catalog_status": "frozen",
        "freeze_status": "strict_passed",
        "tokenizer_name": tokenizer_name,
        "tokenizer_backend": tokenizer_backend,
        "tokenizer_revision_source": tokenizer_revision_source,
        "source_catalog": str(source_catalog),
        "freeze_timestamp": freeze_timestamp,
        "git_commit": git_commit,
    }


def build_frozen_candidate_layout(
    source_layout: BucketLayout,
    audit_result: CarrierAuditResult,
    provenance: Mapping[str, str],
    drop_reasons: Sequence[str] = tuple(sorted(DEFAULT_DROP_REASONS)),
) -> tuple[BucketLayout | None, tuple[RemovedCarrier, ...], tuple[str, ...], tuple[str, ...]]:
    active_drop_reasons = set(drop_reasons)
    reasons_by_carrier: dict[str, set[str]] = {}
    for diagnostic in audit_result.diagnostics:
        reasons_by_carrier.setdefault(diagnostic.carrier, set()).update(diagnostic.reasons)

    removed: list[RemovedCarrier] = []
    blocked_fields: list[str] = []
    messages: list[str] = []
    frozen_fields: list[FieldBucketSpec] = []

    for field_spec in source_layout.fields:
        candidate_buckets: dict[int, tuple[str, ...]] = {}
        field_blocked = False
        for bucket_id in field_spec.bucket_ids:
            kept_members: list[str] = []
            for carrier in field_spec.bucket_members(bucket_id):
                removal_reasons = _drop_reasons_for_carrier(reasons_by_carrier, carrier, active_drop_reasons)
                if removal_reasons:
                    removed.append(
                        RemovedCarrier(
                            field_name=field_spec.field_name,
                            bucket_id=bucket_id,
                            carrier=carrier,
                            reasons=removal_reasons,
                        )
                    )
                    continue
                kept_members.append(carrier)
            if not kept_members:
                field_blocked = True
                messages.append(
                    f"{field_spec.field_name} bucket {bucket_id} became empty after filtering"
                )
            else:
                candidate_buckets[bucket_id] = tuple(kept_members)

        if field_blocked:
            blocked_fields.append(field_spec.field_name)
            continue

        frozen_fields.append(
            FieldBucketSpec(
                field_name=field_spec.field_name,
                buckets=candidate_buckets,
                field_type=field_spec.field_type,
                notes=field_spec.notes,
                tags=field_spec.tags,
                disallowed_carriers=(),
            )
        )

    if not frozen_fields:
        return None, tuple(removed), tuple(blocked_fields), tuple(messages)

    try:
        frozen_layout = BucketLayout(
            fields=tuple(frozen_fields),
            catalog_name=f"{source_layout.catalog_name or 'catalog'}-freeze-v1",
            notes=source_layout.notes,
            tags=tuple(sorted(set(source_layout.tags + ("frozen",)))),
            provenance=dict(provenance),
        )
    except BucketValidationError as error:
        messages.append(str(error))
        return None, tuple(removed), tuple(blocked_fields), tuple(messages)
    return frozen_layout, tuple(removed), tuple(blocked_fields), tuple(messages)


def freeze_catalog(
    *,
    source_layout: BucketLayout,
    source_catalog_path: Path,
    tokenizer: TokenizerProtocol,
    tokenizer_name: str,
    tokenizer_backend: str,
    tokenizer_revision_source: str,
    repo_root: Path | None = None,
) -> CatalogFreezeOutcome:
    resolved_repo_root = (repo_root or discover_repo_root()).resolve()
    freeze_timestamp = current_timestamp()
    git_commit = get_git_hash(resolved_repo_root)

    source_audit = audit_carriers(
        source_layout.all_carriers() + tuple(
            carrier
            for field in source_layout.fields
            for carrier in field.disallowed_carriers
        ),
        tokenizer=tokenizer,
        bucket_layout=source_layout,
    )

    provenance = _provenance_for_frozen_layout(
        source_catalog=source_catalog_path.resolve(),
        tokenizer_name=tokenizer_name,
        tokenizer_backend=tokenizer_backend,
        tokenizer_revision_source=tokenizer_revision_source,
        freeze_timestamp=freeze_timestamp,
        git_commit=git_commit,
    )
    frozen_layout, removed_carriers, blocked_fields, messages = build_frozen_candidate_layout(
        source_layout=source_layout,
        audit_result=source_audit,
        provenance=provenance,
    )

    strict_audit: CarrierAuditResult | None = None
    success = False
    if frozen_layout is not None:
        strict_audit = audit_carriers(
            frozen_layout.all_carriers(),
            tokenizer=tokenizer,
            bucket_layout=frozen_layout,
        )
        success = strict_audit.is_alignment_safe and not blocked_fields
        if not success and not messages:
            messages = ("strict audit failed on filtered candidate catalog",)
    else:
        messages = messages or ("no valid candidate frozen catalog could be constructed",)

    return CatalogFreezeOutcome(
        success=success,
        source_catalog=str(source_catalog_path.resolve()),
        tokenizer_name=tokenizer_name,
        tokenizer_backend=tokenizer_backend,
        tokenizer_revision_source=tokenizer_revision_source,
        freeze_timestamp=freeze_timestamp,
        git_commit=git_commit,
        source_audit=source_audit,
        strict_audit=strict_audit,
        removed_carriers=removed_carriers,
        blocked_fields=blocked_fields,
        messages=messages,
        frozen_layout=frozen_layout if success else None,
    )


def render_catalog_change_log(outcome: CatalogFreezeOutcome) -> str:
    lines = [
        "# Catalog Freeze",
        "",
        f"- status: {'success' if outcome.success else 'failed'}",
        f"- tokenizer_backend: {outcome.tokenizer_backend}",
        f"- tokenizer_name: {outcome.tokenizer_name}",
        f"- tokenizer_revision_source: {outcome.tokenizer_revision_source}",
        f"- source_catalog: {outcome.source_catalog}",
        f"- freeze_timestamp: {outcome.freeze_timestamp}",
        f"- git_commit: {outcome.git_commit}",
        "",
        "## Removed carriers",
    ]
    if outcome.removed_carriers:
        for item in outcome.removed_carriers:
            lines.append(
                f"- {item.field_name}[{item.bucket_id}] `{item.carrier}` removed: {', '.join(item.reasons)}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Blocked fields"])
    if outcome.blocked_fields:
        for field_name in outcome.blocked_fields:
            lines.append(f"- {field_name}")
    else:
        lines.append("- none")

    lines.extend(["", "## Messages"])
    if outcome.messages:
        for message in outcome.messages:
            lines.append(f"- {message}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def load_catalog_freeze_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CatalogFreezeError(f"Catalog freeze report must be a JSON object: {path}")
    return payload


def _parse_change_log_messages(change_log_text: str) -> tuple[str, ...]:
    messages: list[str] = []
    active_section: str | None = None
    for raw_line in change_log_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            active_section = line[3:].strip().lower()
            continue
        if active_section == "messages" and line.startswith("- "):
            messages.append(line[2:].strip())
    return tuple(messages)


def _original_bucket_members_from_report(
    report_payload: Mapping[str, Any],
) -> dict[str, dict[int, list[str]]]:
    buckets: dict[str, dict[int, list[str]]] = {}
    diagnostics = report_payload.get("source_audit", {}).get("diagnostics", [])
    for item in diagnostics:
        if not isinstance(item, Mapping):
            continue
        carrier = str(item.get("carrier", ""))
        for location in item.get("bucket_locations", []):
            field_name, bucket_id_raw = str(location).split(":", 1)
            bucket_id = int(bucket_id_raw)
            field_buckets = buckets.setdefault(field_name, {})
            members = field_buckets.setdefault(bucket_id, [])
            if carrier not in members:
                members.append(carrier)
    return buckets


def _rejected_bucket_members_from_report(
    report_payload: Mapping[str, Any],
) -> dict[str, dict[int, list[RejectedMemberReview]]]:
    buckets: dict[str, dict[int, list[RejectedMemberReview]]] = {}
    for item in report_payload.get("removed_carriers", []):
        if not isinstance(item, Mapping):
            continue
        field_name = str(item["field_name"])
        bucket_id = int(item["bucket_id"])
        field_buckets = buckets.setdefault(field_name, {})
        field_buckets.setdefault(bucket_id, []).append(
            RejectedMemberReview(
                carrier=str(item["carrier"]),
                reasons=tuple(str(reason) for reason in item.get("reasons", [])),
            )
        )
    return buckets


def _bucket_recommended_action(
    *,
    field_blocked: bool,
    bucket_became_empty: bool,
    rejected_members: Sequence[RejectedMemberReview],
) -> str:
    if field_blocked:
        return "drop_field"
    if bucket_became_empty:
        return "replace_members"
    if rejected_members:
        return "regroup_with_manual_review"
    return "keep_as_is"


def _field_recommended_action(
    field_name: str,
    blocked_fields: set[str],
    bucket_reviews: Sequence[BucketRemediationReview],
) -> str:
    if field_name in blocked_fields:
        return "drop_field"
    if any(bucket.bucket_became_empty for bucket in bucket_reviews):
        return "replace_members"
    if any(bucket.rejected_members for bucket in bucket_reviews):
        return "regroup_with_manual_review"
    return "keep_as_is"


def build_catalog_remediation_review(
    report_payload: Mapping[str, Any],
    change_log_text: str,
    change_log_path: Path | None = None,
) -> CatalogRemediationReview:
    original_members = _original_bucket_members_from_report(report_payload)
    rejected_members = _rejected_bucket_members_from_report(report_payload)
    blocked_fields = {str(item) for item in report_payload.get("blocked_fields", [])}
    change_log_messages = _parse_change_log_messages(change_log_text)

    field_names = tuple(original_members.keys())
    fields: list[FieldRemediationReview] = []
    for field_name in field_names:
        bucket_ids = sorted(original_members[field_name])
        bucket_reviews: list[BucketRemediationReview] = []
        for bucket_id in bucket_ids:
            original = tuple(original_members[field_name][bucket_id])
            rejected = tuple(rejected_members.get(field_name, {}).get(bucket_id, []))
            surviving = tuple(
                member for member in original if member not in {item.carrier for item in rejected}
            )
            aggregated_reasons = tuple(
                sorted({reason for item in rejected for reason in item.reasons})
            )
            bucket_became_empty = len(surviving) == 0
            field_blocked = field_name in blocked_fields
            bucket_reviews.append(
                BucketRemediationReview(
                    field_name=field_name,
                    bucket_id=bucket_id,
                    original_members=original,
                    rejected_members=rejected,
                    rejection_reasons=aggregated_reasons,
                    surviving_members=surviving,
                    bucket_became_empty=bucket_became_empty,
                    recommended_action=_bucket_recommended_action(
                        field_blocked=field_blocked,
                        bucket_became_empty=bucket_became_empty,
                        rejected_members=rejected,
                    ),
                    field_blocked=field_blocked,
                )
            )

        fields.append(
            FieldRemediationReview(
                field_name=field_name,
                field_blocked=field_name in blocked_fields,
                recommended_action=_field_recommended_action(
                    field_name,
                    blocked_fields,
                    bucket_reviews,
                ),
                buckets=tuple(bucket_reviews),
                change_log_messages=tuple(
                    message for message in change_log_messages if message.startswith(f"{field_name} ")
                ),
            )
        )

    return CatalogRemediationReview(
        schema_name="catalog_freeze_remediation_review",
        source_catalog=str(report_payload.get("source_catalog", "")),
        tokenizer_name=str(report_payload.get("tokenizer_name", "")),
        tokenizer_backend=str(report_payload.get("tokenizer_backend", "")),
        tokenizer_revision_source=str(report_payload.get("tokenizer_revision_source", "")),
        freeze_timestamp=str(report_payload.get("freeze_timestamp", "")),
        git_commit=str(report_payload.get("git_commit", "")),
        blocked_fields=tuple(sorted(blocked_fields)),
        change_log_path=None if change_log_path is None else str(change_log_path),
        fields=tuple(fields),
        messages=tuple(str(message) for message in report_payload.get("messages", [])),
    )


def render_catalog_remediation_review(review: CatalogRemediationReview) -> str:
    lines = [
        "# Catalog Freeze Remediation Review",
        "",
        f"- source_catalog: {review.source_catalog}",
        f"- tokenizer_backend: {review.tokenizer_backend}",
        f"- tokenizer_name: {review.tokenizer_name}",
        f"- tokenizer_revision_source: {review.tokenizer_revision_source}",
        f"- freeze_timestamp: {review.freeze_timestamp}",
        f"- git_commit: {review.git_commit}",
        "",
    ]
    for field in review.fields:
        lines.extend(
            [
                f"## {field.field_name}",
                f"- field_blocked: {field.field_blocked}",
                f"- recommended_action: {field.recommended_action}",
            ]
        )
        if field.change_log_messages:
            lines.append("- change_log_messages:")
            for message in field.change_log_messages:
                lines.append(f"  - {message}")
        for bucket in field.buckets:
            rejected_members_text = (
                ", ".join(
                    f"{item.carrier} ({', '.join(item.reasons)})" for item in bucket.rejected_members
                )
                if bucket.rejected_members
                else "none"
            )
            lines.extend(
                [
                    f"### Bucket {bucket.bucket_id}",
                    f"- original_members: {', '.join(bucket.original_members) if bucket.original_members else 'none'}",
                    f"- rejected_members: {rejected_members_text}",
                    f"- rejection_reasons: {', '.join(bucket.rejection_reasons) if bucket.rejection_reasons else 'none'}",
                    f"- surviving_members: {', '.join(bucket.surviving_members) if bucket.surviving_members else 'none'}",
                    f"- bucket_became_empty: {bucket.bucket_became_empty}",
                    f"- recommended_action: {bucket.recommended_action}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def save_catalog_remediation_table(review: CatalogRemediationReview, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = review.to_dict()
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_catalog_remediation_review(review: CatalogRemediationReview, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_catalog_remediation_review(review), encoding="utf-8")
    return path


def save_audit_report(outcome: CatalogFreezeOutcome, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(outcome.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_change_log(outcome: CatalogFreezeOutcome, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_catalog_change_log(outcome), encoding="utf-8")
    return path


def save_frozen_catalog(outcome: CatalogFreezeOutcome, path: Path) -> Path:
    if not outcome.success or outcome.frozen_layout is None:
        raise CatalogFreezeError("Cannot save a frozen catalog for a failed freeze outcome")
    path.parent.mkdir(parents=True, exist_ok=True)
    return save_bucket_layout(outcome.frozen_layout, path)


def _repo_relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _relative_include_path(output_path: Path, target_path: Path) -> str:
    return str(Path(relpath(target_path.resolve(), start=output_path.parent.resolve())))


def write_frozen_data_config(
    *,
    output_path: Path,
    frozen_catalog_path: Path,
    data_name: str,
    source_catalog_path: Path,
    repo_root: Path,
) -> Path:
    payload = {
        "data": {
            "name": data_name,
            "carrier_catalog_path": _repo_relative_path(frozen_catalog_path, repo_root),
            "source_carrier_catalog_path": _repo_relative_path(source_catalog_path, repo_root),
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def write_frozen_experiment_config(
    *,
    output_path: Path,
    base_experiment_config: Path,
    frozen_catalog_path: Path,
    repo_root: Path,
) -> Path:
    payload = {
        "includes": [_relative_include_path(output_path, base_experiment_config)],
        "data": {
            "carrier_catalog_path": _repo_relative_path(frozen_catalog_path, repo_root),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def missing_frozen_provenance(layout: BucketLayout) -> tuple[str, ...]:
    return tuple(
        key for key in REQUIRED_FROZEN_PROVENANCE_KEYS if not layout.provenance.get(key)
    )


def is_frozen_catalog(layout: BucketLayout) -> bool:
    if missing_frozen_provenance(layout):
        return False
    return (
        layout.provenance.get("catalog_status") == "frozen"
        and layout.provenance.get("freeze_status") == "strict_passed"
    )


def require_frozen_catalog(layout: BucketLayout, path: Path) -> BucketLayout:
    missing = missing_frozen_provenance(layout)
    if missing:
        raise CatalogFreezeError(
            f"pilot canonical-render catalog must be frozen before manifest/eval: {path}; "
            f"missing provenance fields: {', '.join(missing)}"
        )
    if not is_frozen_catalog(layout):
        raise CatalogFreezeError(
            f"pilot canonical-render catalog is not marked as a strict-passed frozen catalog: {path}"
        )
    return layout


def load_required_frozen_catalog(path: Path) -> BucketLayout:
    layout = load_bucket_layout(path)
    return require_frozen_catalog(layout, path)

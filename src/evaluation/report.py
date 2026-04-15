from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, TypeVar


T = TypeVar("T", bound="SerializableResult")


@dataclass(frozen=True)
class SerializableResult:
    schema_name: ClassVar[str] = "result"
    schema_version: ClassVar[int] = 3

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_name"] = self.schema_name
        payload["schema_version"] = self.schema_version
        return payload

    def to_dict(self) -> dict[str, object]:
        return self.to_json_dict()

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_json(self, path: Path) -> Path:
        return self.save_json(path)

    @classmethod
    def from_json_dict(cls: type[T], payload: dict[str, Any]) -> T:
        valid_fields = {item.name for item in fields(cls)}
        filtered = {key: value for key, value in payload.items() if key in valid_fields}
        try:
            return cls(**filtered)
        except TypeError as error:
            missing_fields = sorted(
                field_name
                for field_name in valid_fields
                if field_name not in filtered
            )
            raise ValueError(
                f"Could not deserialize {cls.__name__}: missing or incompatible fields {missing_fields}"
            ) from error

    @classmethod
    def load_json(cls: type[T], path: Path) -> T:
        return cls.from_json_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ProvenanceResult(SerializableResult):
    run_id: str
    experiment_name: str
    method_name: str
    model_name: str
    seed: int
    git_commit: str
    timestamp: str
    hostname: str | None
    slurm_job_id: str | None
    status: str


@dataclass(frozen=True)
class TrainRunSummary(ProvenanceResult):
    schema_name: ClassVar[str] = "train_run_summary"

    objective: str
    dataset_name: str
    dataset_size: int
    steps: int
    final_loss: float
    run_dir: str


@dataclass(frozen=True)
class EvalRunSummary(ProvenanceResult):
    schema_name: ClassVar[str] = "eval_run_summary"

    dataset_name: str
    sample_count: int
    accepted: bool
    match_ratio: float
    threshold: float
    verification_mode: str = "synthetic_fixture"
    render_format: str | None = None
    verifier_success: bool | None = None
    decoded_payload: str | None = None
    decoded_unit_count: int = 0
    decoded_block_count: int = 0
    unresolved_field_count: int = 0
    malformed_count: int = 0
    utility_acceptance_rate: float = 0.0
    notes: str = ""
    diagnostics: dict[str, object] = field(default_factory=dict)
    run_dir: str = ""


@dataclass(frozen=True)
class CalibrationSummary(ProvenanceResult):
    schema_name: ClassVar[str] = "calibration_summary"

    target_far: float
    threshold: float
    observed_far: float
    sample_count: int
    calibration_target: str = "false_accept_rate"
    operating_point_name: str = "threshold"
    threshold_candidates: tuple[float, ...] = ()
    selected_metric_name: str = "observed_far"
    selected_metric_value: float = 0.0
    notes: str = ""
    run_dir: str = ""


@dataclass(frozen=True)
class AttackRunSummary(ProvenanceResult):
    schema_name: ClassVar[str] = "attack_run_summary"

    attack_name: str
    perturbation: str
    sample_count: int
    accepted_before: bool
    accepted_after: bool
    run_dir: str


@dataclass(frozen=True)
class AggregatedComparisonRow(ProvenanceResult):
    schema_name: ClassVar[str] = "aggregated_comparison_row"

    metric_name: str
    metric_value: float
    source_schema: str
    notes: str = ""


RESULT_SCHEMA_REGISTRY: dict[str, type[SerializableResult]] = {
    TrainRunSummary.schema_name: TrainRunSummary,
    EvalRunSummary.schema_name: EvalRunSummary,
    CalibrationSummary.schema_name: CalibrationSummary,
    AttackRunSummary.schema_name: AttackRunSummary,
    AggregatedComparisonRow.schema_name: AggregatedComparisonRow,
}

LEGACY_SCHEMA_ALIASES = {
    "calibration_output": CalibrationSummary.schema_name,
    "attack_output": AttackRunSummary.schema_name,
}


def normalize_schema_name(schema_name: str) -> str:
    return LEGACY_SCHEMA_ALIASES.get(schema_name, schema_name)


def load_result_json(path: Path) -> SerializableResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_name = normalize_schema_name(str(payload.get("schema_name", "")))
    result_type = RESULT_SCHEMA_REGISTRY.get(schema_name)
    if result_type is None:
        raise ValueError(f"Unsupported result schema in {path}: {payload.get('schema_name')}")
    payload["schema_name"] = schema_name
    return result_type.from_json_dict(payload)


def maybe_load_result_json(path: Path) -> SerializableResult | None:
    try:
        return load_result_json(path)
    except (ValueError, json.JSONDecodeError, OSError, TypeError):
        return None


CalibrationOutput = CalibrationSummary
AttackOutput = AttackRunSummary
ComparisonRow = AggregatedComparisonRow

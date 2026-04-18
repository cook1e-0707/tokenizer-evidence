from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from src.core.catalog_freeze import load_required_frozen_catalog
from src.infrastructure.config import ConfigError, ResolvedConfig, load_config
from src.infrastructure.paths import current_timestamp, discover_repo_root, sanitize_component
from src.infrastructure.registry import (
    RegistryRecord,
    append_registry_record,
    latest_registry_by_manifest_id,
    load_registry,
)


class ManifestError(ValueError):
    """Raised when a manifest is invalid."""


@dataclass(frozen=True)
class ResourceRequest:
    partition: str
    gpu_type: str | None
    num_gpus: int
    cpus: int
    mem_gb: int
    time_limit: str
    account: str | None = None
    environment_setup: str = "source ~/.bashrc\n# activate your environment here"
    slurm_template: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "ResourceRequest":
        return cls(
            partition=str(payload.get("partition", "gpu")),
            gpu_type=payload.get("gpu_type"),
            num_gpus=int(payload.get("num_gpus", 1)),
            cpus=int(payload.get("cpus", 4)),
            mem_gb=int(payload.get("mem_gb", 32)),
            time_limit=str(payload.get("time_limit", "02:00:00")),
            account=payload.get("account"),
            environment_setup=str(
                payload.get("environment_setup", "source ~/.bashrc\n# activate your environment here")
            ),
            slurm_template=str(payload.get("slurm_template", "")),
        )


@dataclass(frozen=True)
class ManifestEntry:
    manifest_id: str
    experiment_name: str
    method_name: str
    model_name: str
    seed: int
    config_paths: tuple[str, ...]
    overrides: tuple[str, ...]
    output_root: str
    output_dir: str | None
    requested_resources: ResourceRequest
    launcher_mode: str
    status: str
    tags: tuple[str, ...] = ()
    notes: str = ""
    entry_point: str = "scripts/train.py"
    manifest_name: str = ""
    primary_config_path: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_paths"] = list(self.config_paths)
        payload["overrides"] = list(self.overrides)
        payload["tags"] = list(self.tags)
        payload["requested_resources"] = self.requested_resources.to_json_dict()
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "ManifestEntry":
        return cls(
            manifest_id=str(payload["manifest_id"]),
            experiment_name=str(payload["experiment_name"]),
            method_name=str(payload["method_name"]),
            model_name=str(payload["model_name"]),
            seed=int(payload["seed"]),
            config_paths=tuple(payload.get("config_paths", [])),
            overrides=tuple(payload.get("overrides", [])),
            output_root=str(payload["output_root"]),
            output_dir=str(payload["output_dir"]) if payload.get("output_dir") else None,
            requested_resources=ResourceRequest.from_json_dict(payload.get("requested_resources", {})),
            launcher_mode=str(payload.get("launcher_mode", "slurm")),
            status=str(payload.get("status", "pending")),
            tags=tuple(payload.get("tags", [])),
            notes=str(payload.get("notes", "")),
            entry_point=str(payload.get("entry_point", "scripts/train.py")),
            manifest_name=str(payload.get("manifest_name", "")),
            primary_config_path=str(payload.get("primary_config_path", "")),
        )


@dataclass(frozen=True)
class ManifestFile:
    schema_name: str
    schema_version: int
    manifest_name: str
    created_at: str
    source_config_path: str
    entries: tuple[ManifestEntry, ...] = field(default_factory=tuple)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "manifest_name": self.manifest_name,
            "created_at": self.created_at,
            "source_config_path": self.source_config_path,
            "entries": [entry.to_json_dict() for entry in self.entries],
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "ManifestFile":
        return cls(
            schema_name=str(payload.get("schema_name", "manifest_file")),
            schema_version=int(payload.get("schema_version", 1)),
            manifest_name=str(payload["manifest_name"]),
            created_at=str(payload["created_at"]),
            source_config_path=str(payload["source_config_path"]),
            entries=tuple(ManifestEntry.from_json_dict(item) for item in payload.get("entries", [])),
        )


def validate_manifest(manifest_file: ManifestFile) -> None:
    if not manifest_file.manifest_name.strip():
        raise ManifestError("manifest_name must be non-empty")
    if not manifest_file.entries:
        raise ManifestError("Manifest must contain at least one entry")
    seen_ids: set[str] = set()
    for entry in manifest_file.entries:
        if entry.manifest_id in seen_ids:
            raise ManifestError(f"Duplicate manifest_id detected: {entry.manifest_id}")
        seen_ids.add(entry.manifest_id)
        if not entry.experiment_name:
            raise ManifestError(f"{entry.manifest_id}: experiment_name must be non-empty")
        if not entry.method_name:
            raise ManifestError(f"{entry.manifest_id}: method_name must be non-empty")
        if not entry.model_name:
            raise ManifestError(f"{entry.manifest_id}: model_name must be non-empty")
        if not entry.primary_config_path:
            raise ManifestError(f"{entry.manifest_id}: primary_config_path must be non-empty")


def save_manifest(manifest_file: ManifestFile, path: Path) -> Path:
    validate_manifest(manifest_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    return manifest_file.save_json(path)


def _load_legacy_jsonl_manifest(path: Path) -> ManifestFile:
    entries: list[ManifestEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            legacy = json.loads(line)
            entries.append(
                ManifestEntry(
                    manifest_id=str(legacy.get("entry_id", legacy.get("manifest_id"))),
                    experiment_name=str(legacy["experiment_name"]),
                    method_name=str(legacy.get("method_name", "unknown_method")),
                    model_name=str(legacy.get("model_name", "unknown_model")),
                    seed=int(legacy.get("seed", 0)),
                    config_paths=(str(legacy.get("config", "")),),
                    overrides=tuple(legacy.get("overrides", [])),
                    output_root="results/raw",
                    output_dir=None,
                    requested_resources=ResourceRequest(
                        partition="gpu",
                        gpu_type=None,
                        num_gpus=1,
                        cpus=4,
                        mem_gb=32,
                        time_limit="02:00:00",
                        slurm_template=str(legacy.get("slurm_template", "")),
                    ),
                    launcher_mode="slurm",
                    status=str(legacy.get("status", "pending")),
                    tags=tuple(legacy.get("tags", [])),
                    notes="legacy manifest entry",
                    entry_point=str(legacy.get("script", "scripts/train.py")),
                    manifest_name=str(legacy.get("manifest_name", path.stem)),
                    primary_config_path=str(legacy.get("config", "")),
                )
            )
    return ManifestFile(
        schema_name="manifest_file",
        schema_version=1,
        manifest_name=path.stem,
        created_at=current_timestamp(),
        source_config_path=str(path),
        entries=tuple(entries),
    )


def load_manifest(path: Path) -> ManifestFile:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        manifest_file = ManifestFile.from_json_dict(json.loads(text))
    else:
        manifest_file = _load_legacy_jsonl_manifest(path)
    validate_manifest(manifest_file)
    return manifest_file


def update_manifest_status(path: Path, manifest_id: str, status: str) -> Path:
    manifest_file = load_manifest(path)
    updated_entries = []
    found = False
    for entry in manifest_file.entries:
        if entry.manifest_id == manifest_id:
            updated_entries.append(
                ManifestEntry.from_json_dict(
                    {
                        **entry.to_json_dict(),
                        "status": status,
                    }
                )
            )
            found = True
        else:
            updated_entries.append(entry)
    if not found:
        raise ManifestError(f"manifest_id not found in {path}: {manifest_id}")
    updated_manifest = ManifestFile(
        schema_name=manifest_file.schema_name,
        schema_version=manifest_file.schema_version,
        manifest_name=manifest_file.manifest_name,
        created_at=manifest_file.created_at,
        source_config_path=manifest_file.source_config_path,
        entries=tuple(updated_entries),
    )
    return save_manifest(updated_manifest, path)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ManifestError(f"Expected mapping in {path}")
    return payload


def _normalize_manifest_overrides(overrides: Iterable[str] | None) -> tuple[str, ...]:
    if not overrides:
        return ()

    normalized: list[str] = []
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Invalid override {override!r}; expected dotted.key=value")
        key, value = override.split("=", 1)
        normalized_key = key.strip()
        if normalized_key == "runtime.environment_setup":
            normalized_key = "runtime.resources.environment_setup"
        normalized.append(f"{normalized_key}={value}")
    return tuple(normalized)


def _build_resource_request(
    resolved: ResolvedConfig,
    manifest_settings: Mapping[str, Any],
) -> ResourceRequest:
    manifest_resources = manifest_settings.get("resources", {})
    if manifest_resources is None:
        manifest_resources = {}
    if not isinstance(manifest_resources, Mapping):
        raise ManifestError("manifest.resources must be a mapping")

    runtime_resources = resolved.runtime.resources
    return ResourceRequest(
        partition=str(manifest_resources.get("partition", runtime_resources.partition)),
        gpu_type=manifest_resources.get("gpu_type", runtime_resources.gpu_type),
        num_gpus=int(manifest_resources.get("num_gpus", runtime_resources.num_gpus)),
        cpus=int(manifest_resources.get("cpus", runtime_resources.cpus)),
        mem_gb=int(manifest_resources.get("mem_gb", runtime_resources.mem_gb)),
        time_limit=str(manifest_resources.get("time_limit", runtime_resources.time_limit)),
        account=manifest_resources.get("account", runtime_resources.account),
        environment_setup=str(
            manifest_resources.get("environment_setup", runtime_resources.environment_setup)
        ),
        slurm_template=str(
            manifest_resources.get(
                "slurm_template",
                manifest_settings.get("slurm_template", runtime_resources.slurm_template),
            )
        ),
    )


def _entry_point_for_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized == "train":
        return "scripts/train.py"
    if normalized == "attack":
        return "scripts/attack.py"
    if normalized == "calibrate":
        return "scripts/calibrate.py"
    return "scripts/eval.py"


def _require_frozen_catalog_for_pilot(resolved: ResolvedConfig, repo_root: Path) -> None:
    if resolved.method_name != "our_method":
        return
    if resolved.run.mode not in {"eval", "attack"}:
        return
    if resolved.eval.verification_mode != "canonical_render":
        return
    catalog_path = Path(resolved.data.carrier_catalog_path)
    if not catalog_path.is_absolute():
        catalog_path = repo_root / catalog_path
    load_required_frozen_catalog(catalog_path)


def _parameter_grid(manifest_settings: Mapping[str, Any]) -> list[tuple[str, list[Any]]]:
    parameters: list[tuple[str, list[Any]]] = []
    explicit_parameters = manifest_settings.get("parameters", [])
    if explicit_parameters:
        if not isinstance(explicit_parameters, list):
            raise ManifestError("manifest.parameters must be a list")
        for item in explicit_parameters:
            if not isinstance(item, Mapping):
                raise ManifestError("Each manifest.parameters item must be a mapping")
            parameters.append((str(item["key"]), list(item.get("values", []))))

    simple_mappings = (
        ("run.seed", manifest_settings.get("seeds")),
        ("run.method_name", manifest_settings.get("methods")),
        ("model.name", manifest_settings.get("models")),
        ("run.variant_name", manifest_settings.get("variants")),
    )
    for key, values in simple_mappings:
        if values:
            parameters.append((key, list(values)))
    return parameters


def _expand_entries_from_settings(
    source_path: Path,
    manifest_settings: Mapping[str, Any],
    repo_root: Path,
    base_overrides: tuple[str, ...] = (),
) -> ManifestFile:
    manifest_name = str(manifest_settings.get("name", source_path.stem))
    primary_config_path = str(manifest_settings.get("config", source_path))
    tags = tuple(manifest_settings.get("tags", []))
    notes = str(manifest_settings.get("notes", ""))
    parameters = _parameter_grid(manifest_settings)

    value_lists = [values for _, values in parameters]
    combinations: Iterable[tuple[Any, ...]]
    combinations = itertools.product(*value_lists) if value_lists else [()]

    entries: list[ManifestEntry] = []
    for index, combination in enumerate(combinations, start=1):
        parameter_overrides = tuple(
            f"{key}={value}" for (key, _), value in zip(parameters, combination, strict=False)
        )
        overrides = parameter_overrides + tuple(base_overrides)
        config_path = Path(primary_config_path)
        if not config_path.is_absolute():
            config_path = repo_root / config_path
        resolved = load_config(config_path, overrides=list(overrides))
        _require_frozen_catalog_for_pilot(resolved, repo_root)
        resource_request = _build_resource_request(resolved, manifest_settings)
        entry_point = str(manifest_settings.get("script", _entry_point_for_mode(resolved.run.mode)))
        manifest_id = f"{sanitize_component(manifest_name)}-{index:04d}"
        entries.append(
            ManifestEntry(
                manifest_id=manifest_id,
                experiment_name=resolved.experiment_name,
                method_name=resolved.method_name,
                model_name=resolved.model_name,
                seed=resolved.seed,
                config_paths=resolved.source_config_paths,
                overrides=overrides,
                output_root=resolved.output_root,
                output_dir=resolved.runtime.output_dir,
                requested_resources=resource_request,
                launcher_mode=resolved.runtime.launcher_mode,
                status="pending",
                tags=tags or resolved.runtime.tags,
                notes=notes or resolved.run.notes,
                entry_point=entry_point,
                manifest_name=manifest_name,
                primary_config_path=str(config_path),
            )
        )

    manifest_file = ManifestFile(
        schema_name="manifest_file",
        schema_version=1,
        manifest_name=manifest_name,
        created_at=current_timestamp(),
        source_config_path=str(source_path.resolve()),
        entries=tuple(entries),
    )
    validate_manifest(manifest_file)
    return manifest_file


def build_manifest_from_config(
    config_path: Path,
    overrides: list[str] | tuple[str, ...] | None = None,
) -> ManifestFile:
    repo_root = discover_repo_root(config_path.parent)
    payload = _read_yaml(config_path)
    normalized_overrides = _normalize_manifest_overrides(overrides)
    manifest_settings = payload.get("manifest")
    if isinstance(manifest_settings, Mapping):
        return _expand_entries_from_settings(
            config_path,
            manifest_settings,
            repo_root,
            base_overrides=normalized_overrides,
        )

    resolved = load_config(config_path, overrides=list(normalized_overrides))
    _require_frozen_catalog_for_pilot(resolved, repo_root)
    entry = ManifestEntry(
        manifest_id=f"{sanitize_component(resolved.experiment_name)}-0001",
        experiment_name=resolved.experiment_name,
        method_name=resolved.method_name,
        model_name=resolved.model_name,
        seed=resolved.seed,
        config_paths=resolved.source_config_paths,
        overrides=normalized_overrides,
        output_root=resolved.output_root,
        output_dir=resolved.runtime.output_dir,
        requested_resources=ResourceRequest(
            partition=resolved.runtime.resources.partition,
            gpu_type=resolved.runtime.resources.gpu_type,
            num_gpus=resolved.runtime.resources.num_gpus,
            cpus=resolved.runtime.resources.cpus,
            mem_gb=resolved.runtime.resources.mem_gb,
            time_limit=resolved.runtime.resources.time_limit,
            account=resolved.runtime.resources.account,
            environment_setup=resolved.runtime.resources.environment_setup,
            slurm_template=resolved.runtime.resources.slurm_template,
        ),
        launcher_mode=resolved.runtime.launcher_mode,
        status="pending",
        tags=resolved.runtime.tags,
        notes=resolved.run.notes,
        entry_point=_entry_point_for_mode(resolved.run.mode),
        manifest_name=resolved.experiment_name,
        primary_config_path=str(config_path.resolve()),
    )
    manifest_file = ManifestFile(
        schema_name="manifest_file",
        schema_version=1,
        manifest_name=resolved.experiment_name,
        created_at=current_timestamp(),
        source_config_path=str(config_path.resolve()),
        entries=(entry,),
    )
    validate_manifest(manifest_file)
    return manifest_file


def expand_sweep(path: Path) -> list[ManifestEntry]:
    return list(build_manifest_from_config(path).entries)


def load_latest_registry(path: Path) -> dict[str, RegistryRecord]:
    return latest_registry_by_manifest_id(load_registry(path))

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.infrastructure.paths import EXPERIMENT_NAME_SET


ALLOWED_RUN_MODES = {"train", "eval", "attack", "calibrate"}
ALLOWED_LAUNCHER_MODES = {"local", "slurm"}
ALLOWED_VERIFICATION_MODES = {"synthetic_fixture", "canonical_render", "foundation_gate", "compiled_gate"}


class ConfigError(ValueError):
    """Raised when config loading or validation fails."""


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str
    mode: str
    method_name: str
    seed: int
    output_root: str = "results/raw"
    notes: str = ""
    variant_name: str = ""

    @property
    def method(self) -> str:
        return self.method_name


@dataclass(frozen=True)
class ModelConfig:
    name: str
    family: str = "unknown"
    tokenizer_name: str = ""
    tokenizer_backend: str = "mock"
    max_length: int = 2048


@dataclass(frozen=True)
class DataConfig:
    name: str
    train_path: str = ""
    eval_path: str = ""
    parser_smoke_path: str = ""
    carrier_catalog_path: str = ""
    foundation_eval_summary_path: str = ""


@dataclass(frozen=True)
class TrainConfig:
    target_mode: str = "dataset_completion"
    batch_size: int = 1
    epochs: int = 1
    learning_rate: float = 0.0001
    objective: str = "bucket_mass"
    lambda_set: float = 1.0
    lambda_margin: float = 0.0
    margin_gamma: float = 0.0
    lambda_reg: float = 0.0
    evidence_loss_normalization: str = "per_slot_mean"
    checkpoint_selection_metric: str = ""
    checkpoint_selection_mode: str = "min"
    checkpoint_selection_use_best_for_eval: bool = False
    checkpoint_selection_save_best: bool = False
    num_workers: int = 0
    probe_payload_texts: tuple[str, ...] = ()
    probe_block_count: int = 0
    compiled_sample_repeats: int = 1
    generation_prompt: str = ""
    generation_do_sample: bool = False
    generation_max_new_tokens: int = 16
    generation_stop_strings: tuple[str, ...] = ()
    generation_bad_words: tuple[str, ...] = ()
    generation_suppress_tokens: tuple[int, ...] = ()
    generation_sequence_bias: dict[str, float] = field(default_factory=dict)
    adapter_mode: str = "full"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalConfig:
    min_score: float = 0.0
    max_candidates: int = 32
    target_far: float = 0.01
    verification_mode: str = "synthetic_fixture"
    render_format: str = "canonical_v1"
    payload_text: str = "OK"
    expected_payload_source: str = "eval_input"
    audit_strict: bool = True
    require_foundation_gate: bool = False


@dataclass(frozen=True)
class AttackConfig:
    name: str = "none"
    mode: str = "noop"
    strength: float = 0.0
    clean_eval_summary_path: str = ""


@dataclass(frozen=True)
class RuntimeResources:
    partition: str = "gpu"
    qos: str | None = None
    gpu_type: str | None = None
    num_gpus: int = 1
    cpus: int = 4
    mem_gb: int = 32
    time_limit: str = "02:00:00"
    account: str | None = None
    environment_setup: str = "if [ -f /etc/profile ]; then . /etc/profile; fi\n# activate your environment here"
    slurm_template: str = ""


@dataclass(frozen=True)
class RuntimeConfig:
    output_root: str = "results/raw"
    launcher_mode: str = "slurm"
    force_overwrite: bool = False
    manifest_id: str | None = None
    manifest_path: str | None = None
    run_id: str | None = None
    output_dir: str | None = None
    config_paths: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    resources: RuntimeResources = field(default_factory=RuntimeResources)


@dataclass(frozen=True)
class ResolvedConfig:
    run: RunConfig
    model: ModelConfig
    data: DataConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    source_config_paths: tuple[str, ...] = ()
    merged_settings: dict[str, Any] = field(default_factory=dict)

    @property
    def experiment(self) -> RunConfig:
        return self.run

    @property
    def experiment_name(self) -> str:
        return self.run.experiment_name

    @property
    def method_name(self) -> str:
        return self.run.method_name

    @property
    def model_name(self) -> str:
        return self.model.name

    @property
    def seed(self) -> int:
        return self.run.seed

    @property
    def output_root(self) -> str:
        return self.runtime.output_root

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_config_paths"] = list(self.source_config_paths)
        payload["runtime"]["config_paths"] = list(self.runtime.config_paths)
        payload["runtime"]["tags"] = list(self.runtime.tags)
        payload["merged_settings"] = copy.deepcopy(self.merged_settings)
        return payload


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_override_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if raw.startswith(("[", "{", '"')) and raw.endswith(("]", "}", '"')):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _set_nested_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = data
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def apply_overrides(data: Mapping[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(data))
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Invalid override {override!r}; expected dotted.key=value")
        key, raw_value = override.split("=", 1)
        _set_nested_value(merged, key.strip(), _parse_override_value(raw_value.strip()))
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as error:
        raise ConfigError(f"Config file not found: {path}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML must be a mapping: {path}")
    return data


def _load_with_includes(
    path: Path,
    seen: set[Path] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    resolved = path.resolve()
    active = seen or set()
    if resolved in active:
        raise ConfigError(f"Cyclic config include detected at {resolved}")

    data = _read_yaml(resolved)
    includes = data.pop("includes", [])
    if includes is None:
        includes = []
    if not isinstance(includes, list):
        raise ConfigError(f"'includes' must be a list in {resolved}")

    merged: dict[str, Any] = {}
    sources: list[Path] = []
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = resolved.parent / include_path
        include_data, include_sources = _load_with_includes(include_path, active | {resolved})
        merged = deep_merge(merged, include_data)
        sources.extend(include_sources)

    merged = deep_merge(merged, data)
    sources.append(resolved)
    unique_sources = list(dict.fromkeys(source.resolve() for source in sources))
    return merged, unique_sources


def _normalize_sections(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(data))
    has_run = "run" in normalized
    has_experiment = "experiment" in normalized
    if has_run and has_experiment:
        raise ConfigError("Config may define either 'run' or 'experiment', not both")
    if has_experiment:
        experiment_section = normalized.pop("experiment")
        if not isinstance(experiment_section, Mapping):
            raise ConfigError("'experiment' must be a mapping")
        normalized["run"] = dict(experiment_section)

    run_section = normalized.get("run", {})
    if isinstance(run_section, Mapping):
        run_copy = dict(run_section)
        if "method_name" not in run_copy and "method" in run_copy:
            run_copy["method_name"] = run_copy["method"]
        normalized["run"] = run_copy

    runtime_section = normalized.get("runtime", {})
    if runtime_section is None:
        runtime_section = {}
    if not isinstance(runtime_section, Mapping):
        raise ConfigError("'runtime' must be a mapping")
    runtime_copy = dict(runtime_section)

    if isinstance(run_section, Mapping):
        run_output_root = run_section.get("output_root")
        runtime_output_root = runtime_copy.get("output_root")
        if run_output_root and runtime_output_root and run_output_root != runtime_output_root:
            raise ConfigError(
                "run.output_root and runtime.output_root disagree; use one value consistently"
            )
        runtime_copy["output_root"] = str(runtime_output_root or run_output_root or "results/raw")
    else:
        runtime_copy.setdefault("output_root", "results/raw")

    runtime_copy.setdefault("resources", {})
    normalized["runtime"] = runtime_copy
    return normalized


def validate_config_dict(data: Mapping[str, Any]) -> None:
    normalized = _normalize_sections(data)
    for section in ("run", "model", "data", "runtime"):
        if (
            section not in normalized
            or not isinstance(normalized[section], Mapping)
            or (section in {"run", "model", "data"} and not dict(normalized[section]))
        ):
            raise ConfigError(f"Missing required section: {section}")

    run_section = dict(normalized["run"])
    model_section = dict(normalized["model"])
    data_section = dict(normalized["data"])
    runtime_section = dict(normalized["runtime"])
    resource_section = runtime_section.get("resources", {})
    if not isinstance(resource_section, Mapping):
        raise ConfigError("runtime.resources must be a mapping")

    experiment_name = str(run_section.get("experiment_name", "")).strip()
    if not experiment_name:
        raise ConfigError("run.experiment_name is required")
    if EXPERIMENT_NAME_SET and experiment_name not in EXPERIMENT_NAME_SET:
        raise ConfigError(
            f"run.experiment_name must be one of {sorted(EXPERIMENT_NAME_SET)}; got {experiment_name!r}"
        )

    mode = str(run_section.get("mode", "")).strip()
    if mode not in ALLOWED_RUN_MODES:
        raise ConfigError(f"run.mode must be one of {sorted(ALLOWED_RUN_MODES)}; got {mode!r}")

    method_name = str(run_section.get("method_name", "")).strip()
    if not method_name:
        raise ConfigError("run.method_name is required")

    if "seed" not in run_section:
        raise ConfigError("run.seed is required")
    try:
        int(run_section["seed"])
    except (TypeError, ValueError) as error:
        raise ConfigError("run.seed must be an integer") from error

    if not str(model_section.get("name", "")).strip():
        raise ConfigError("model.name must be non-empty")
    if not str(data_section.get("name", "")).strip():
        raise ConfigError("data.name must be non-empty")

    output_root = str(runtime_section.get("output_root", "")).strip()
    if not output_root:
        raise ConfigError("runtime.output_root must be non-empty")

    launcher_mode = str(runtime_section.get("launcher_mode", "slurm")).strip()
    if launcher_mode not in ALLOWED_LAUNCHER_MODES:
        raise ConfigError(
            f"runtime.launcher_mode must be one of {sorted(ALLOWED_LAUNCHER_MODES)}; got {launcher_mode!r}"
        )

    eval_section = normalized.get("eval", {})
    if eval_section and isinstance(eval_section, Mapping):
        verification_mode = str(eval_section.get("verification_mode", "synthetic_fixture")).strip()
        if verification_mode not in ALLOWED_VERIFICATION_MODES:
            raise ConfigError(
                "eval.verification_mode must be one of "
                f"{sorted(ALLOWED_VERIFICATION_MODES)}; got {verification_mode!r}"
            )
        expected_payload_source = str(eval_section.get("expected_payload_source", "eval_input")).strip()
        if expected_payload_source not in {"eval_input", "config"}:
            raise ConfigError(
                "eval.expected_payload_source must be one of ['config', 'eval_input']; "
                f"got {expected_payload_source!r}"
            )

    train_section = normalized.get("train", {})
    if train_section and isinstance(train_section, Mapping):
        try:
            if int(train_section.get("compiled_sample_repeats", 1)) < 1:
                raise ConfigError("train.compiled_sample_repeats must be >= 1")
        except (TypeError, ValueError) as error:
            raise ConfigError("train.compiled_sample_repeats must be an integer") from error

    numeric_fields = {
        "runtime.resources.num_gpus": resource_section.get("num_gpus", 1),
        "runtime.resources.cpus": resource_section.get("cpus", 4),
        "runtime.resources.mem_gb": resource_section.get("mem_gb", 32),
    }
    for key, value in numeric_fields.items():
        try:
            if int(value) < 0:
                raise ConfigError(f"{key} must be >= 0")
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{key} must be an integer") from error

    time_limit = str(resource_section.get("time_limit", "02:00:00")).strip()
    if not time_limit:
        raise ConfigError("runtime.resources.time_limit must be non-empty")


def build_config(data: Mapping[str, Any], source_paths: list[Path] | None = None) -> ResolvedConfig:
    normalized = _normalize_sections(data)
    validate_config_dict(normalized)

    run_section = dict(normalized["run"])
    runtime_section = dict(normalized.get("runtime", {}))
    resource_section = dict(runtime_section.get("resources", {}))
    source_config_paths = tuple(str(path) for path in (source_paths or []))

    run = RunConfig(
        experiment_name=str(run_section["experiment_name"]),
        mode=str(run_section["mode"]),
        method_name=str(run_section["method_name"]),
        seed=int(run_section["seed"]),
        output_root=str(runtime_section.get("output_root", run_section.get("output_root", "results/raw"))),
        notes=str(run_section.get("notes", "")),
        variant_name=str(run_section.get("variant_name", "")),
    )
    runtime = RuntimeConfig(
        output_root=str(runtime_section.get("output_root", run.output_root)),
        launcher_mode=str(runtime_section.get("launcher_mode", "slurm")),
        force_overwrite=bool(runtime_section.get("force_overwrite", False)),
        manifest_id=runtime_section.get("manifest_id"),
        manifest_path=runtime_section.get("manifest_path"),
        run_id=runtime_section.get("run_id"),
        output_dir=runtime_section.get("output_dir"),
        config_paths=tuple(runtime_section.get("config_paths", source_config_paths)),
        tags=tuple(runtime_section.get("tags", [])),
        resources=RuntimeResources(
            partition=str(resource_section.get("partition", "gpu")),
            qos=resource_section.get("qos"),
            gpu_type=resource_section.get("gpu_type"),
            num_gpus=int(resource_section.get("num_gpus", 1)),
            cpus=int(resource_section.get("cpus", 4)),
            mem_gb=int(resource_section.get("mem_gb", 32)),
            time_limit=str(resource_section.get("time_limit", "02:00:00")),
            account=resource_section.get("account"),
            environment_setup=str(
                resource_section.get(
                    "environment_setup",
                    "if [ -f /etc/profile ]; then . /etc/profile; fi\n# activate your environment here",
                )
            ),
            slurm_template=str(resource_section.get("slurm_template", "")),
        ),
    )
    return ResolvedConfig(
        run=run,
        model=ModelConfig(**dict(normalized["model"])),
        data=DataConfig(**dict(normalized["data"])),
        train=TrainConfig(**dict(normalized.get("train", {}))),
        eval=EvalConfig(**dict(normalized.get("eval", {}))),
        attack=AttackConfig(**dict(normalized.get("attack", {}))),
        runtime=runtime,
        source_config_paths=source_config_paths,
        merged_settings=copy.deepcopy(normalized),
    )


def load_config(path: Path, overrides: list[str] | None = None) -> ResolvedConfig:
    merged, source_paths = _load_with_includes(path)
    if overrides:
        merged = apply_overrides(merged, overrides)
    return build_config(merged, source_paths=source_paths)


def load_experiment_config(path: Path, overrides: list[str] | None = None) -> ResolvedConfig:
    return load_config(path, overrides=overrides)


def save_resolved_config(config: ResolvedConfig, path: Path) -> Path:
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    return path

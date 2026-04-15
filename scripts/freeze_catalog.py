from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.core.bucket_mapping import load_bucket_layout
from src.core.catalog_freeze import (
    CatalogFreezeError,
    freeze_catalog,
    infer_tokenizer_revision_source,
    save_audit_report,
    save_change_log,
    save_frozen_catalog,
    write_frozen_data_config,
    write_frozen_experiment_config,
)
from src.core.tokenizer_utils import load_tokenizer
from src.infrastructure.config import load_experiment_config
from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a source carrier catalog behind a strict tokenizer audit gate."
    )
    parser.add_argument("--source-catalog", required=True, help="Path to the raw source carrier catalog.")
    parser.add_argument("--tokenizer-backend", required=True, help="Tokenizer backend: mock or huggingface.")
    parser.add_argument(
        "--tokenizer-name",
        default="",
        help="Tokenizer name or local tokenizer path. If omitted, inherit from --base-experiment-config when available.",
    )
    parser.add_argument(
        "--tokenizer-source",
        default="",
        help="Optional explicit tokenizer revision/source string for provenance.",
    )
    parser.add_argument("--frozen-catalog-output", required=True, help="Output path for frozen catalog YAML.")
    parser.add_argument("--audit-report-output", required=True, help="Output path for audit report JSON.")
    parser.add_argument("--change-log-output", required=True, help="Output path for markdown change log.")
    parser.add_argument(
        "--data-config-output",
        help="Optional output path for a generated data config pointing to the frozen catalog.",
    )
    parser.add_argument(
        "--experiment-config-output",
        help="Optional output path for a generated experiment config overlay pointing to the frozen catalog.",
    )
    parser.add_argument(
        "--base-experiment-config",
        help="Required when --experiment-config-output is set.",
    )
    parser.add_argument(
        "--data-name",
        default="real-pilot-frozen",
        help="Data config name to use when --data-config-output is requested.",
    )
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _resolve_tokenizer_settings(
    *,
    tokenizer_backend: str,
    tokenizer_name: str,
    base_experiment_config: Path | None,
) -> tuple[str, str]:
    resolved_backend = tokenizer_backend
    resolved_name = tokenizer_name
    if base_experiment_config is not None:
        base_config = load_experiment_config(base_experiment_config)
        if not resolved_name:
            resolved_name = base_config.model.tokenizer_name
        if not resolved_backend:
            resolved_backend = base_config.model.tokenizer_backend
    if resolved_backend.strip().lower() in {"huggingface", "hf"} and not resolved_name.strip():
        raise ValueError(
            "tokenizer_name could not be resolved for tokenizer_backend=huggingface. "
            "Provide --tokenizer-name or a --base-experiment-config with model.tokenizer_name."
        )
    return resolved_backend, resolved_name


def main() -> int:
    args = parse_args()
    if args.experiment_config_output and not args.base_experiment_config:
        raise SystemExit("--base-experiment-config is required with --experiment-config-output")

    repo_root = discover_repo_root()
    source_catalog_path = _resolve_path(repo_root, args.source_catalog)
    frozen_catalog_output = _resolve_path(repo_root, args.frozen_catalog_output)
    audit_report_output = _resolve_path(repo_root, args.audit_report_output)
    change_log_output = _resolve_path(repo_root, args.change_log_output)
    data_config_output = (
        _resolve_path(repo_root, args.data_config_output) if args.data_config_output else None
    )
    experiment_config_output = (
        _resolve_path(repo_root, args.experiment_config_output)
        if args.experiment_config_output
        else None
    )
    base_experiment_config = (
        _resolve_path(repo_root, args.base_experiment_config)
        if args.base_experiment_config
        else None
    )

    source_layout = load_bucket_layout(source_catalog_path)
    tokenizer_backend, tokenizer_name = _resolve_tokenizer_settings(
        tokenizer_backend=args.tokenizer_backend,
        tokenizer_name=args.tokenizer_name,
        base_experiment_config=base_experiment_config,
    )
    tokenizer = load_tokenizer(tokenizer_backend, tokenizer_name)
    tokenizer_revision_source = args.tokenizer_source or infer_tokenizer_revision_source(
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        tokenizer_backend=tokenizer_backend,
    )

    outcome = freeze_catalog(
        source_layout=source_layout,
        source_catalog_path=source_catalog_path,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name or tokenizer_backend,
        tokenizer_backend=tokenizer_backend,
        tokenizer_revision_source=tokenizer_revision_source,
        repo_root=repo_root,
    )

    save_audit_report(outcome, audit_report_output)
    save_change_log(outcome, change_log_output)

    if not outcome.success:
        print(f"freeze_status=failed")
        print(f"audit_report={audit_report_output}")
        print(f"change_log={change_log_output}")
        for message in outcome.messages:
            print(f"message={message}")
        return 1

    save_frozen_catalog(outcome, frozen_catalog_output)
    if data_config_output is not None:
        write_frozen_data_config(
            output_path=data_config_output,
            frozen_catalog_path=frozen_catalog_output,
            data_name=args.data_name,
            source_catalog_path=source_catalog_path,
        )
    if experiment_config_output is not None:
        assert base_experiment_config is not None
        write_frozen_experiment_config(
            output_path=experiment_config_output,
            base_experiment_config=base_experiment_config,
            frozen_catalog_path=frozen_catalog_output,
        )

    print("freeze_status=success")
    print(f"frozen_catalog={frozen_catalog_output}")
    print(f"audit_report={audit_report_output}")
    print(f"change_log={change_log_output}")
    if data_config_output is not None:
        print(f"data_config={data_config_output}")
    if experiment_config_output is not None:
        print(f"experiment_config={experiment_config_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

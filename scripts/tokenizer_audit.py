from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.infrastructure.config import load_experiment_config
from src.infrastructure.paths import discover_repo_root
from src.core.tokenizer_utils import (
    audit_carriers,
    load_carriers_and_layout,
    load_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit carrier strings for tokenizer alignment safety.")
    parser.add_argument("--config", help="Experiment config used to resolve tokenizer and carrier catalog.")
    parser.add_argument("--input", help="Optional JSON/YAML/text file containing carrier candidates.")
    parser.add_argument("--bucket-spec", help="Optional JSON/YAML bucket layout file.")
    parser.add_argument("--tokenizer-backend", help="Tokenizer backend: mock or huggingface.")
    parser.add_argument("--tokenizer-name", help="Tokenizer name or local path for the selected backend.")
    parser.add_argument("--output-json", help="Optional output path for the machine-readable JSON report.")
    parser.add_argument("--output-summary", help="Optional output path for a concise text summary.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when rejected carriers exist.")
    parser.add_argument("--no-strict", action="store_true", help="Override config strictness and always exit zero.")
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _build_summary_text(result: object) -> str:
    audit_result = result
    lines = [
        f"total={audit_result.num_total}",
        f"single_token={audit_result.num_single_token}",
        f"multi_token={audit_result.num_multi_token}",
        f"invalid={audit_result.num_invalid}",
        f"duplicates={audit_result.num_duplicates}",
        f"token_collisions={audit_result.num_token_collisions}",
        f"alignment_safe={audit_result.is_alignment_safe}",
    ]
    for field_name, summary in audit_result.field_summaries.items():
        lines.append(
            f"field={field_name} passed={summary['passed']} "
            f"rejected={summary['rejected_count']} total={summary['num_total']}"
        )
    if audit_result.rejected_carriers:
        lines.append("rejected_carriers:")
        for diagnostic in audit_result.rejected_carriers[:20]:
            lines.append(f"  - {diagnostic.carrier!r}: {', '.join(diagnostic.reasons)}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root()

    resolved_config = None
    if args.config:
        config_path = _resolve_path(repo_root, args.config)
        assert config_path is not None
        resolved_config = load_experiment_config(config_path)

    bucket_spec_path = _resolve_path(
        repo_root,
        args.bucket_spec or (resolved_config.data.carrier_catalog_path if resolved_config else None),
    )
    carrier_path = _resolve_path(repo_root, args.input)
    if carrier_path is None and bucket_spec_path is None:
        raise SystemExit("Provide --config, --input, or --bucket-spec")

    tokenizer_backend = args.tokenizer_backend or (
        resolved_config.model.tokenizer_backend if resolved_config else "mock"
    )
    tokenizer_name = args.tokenizer_name or (
        resolved_config.model.tokenizer_name if resolved_config else ""
    )
    strict = False
    if resolved_config is not None:
        strict = resolved_config.eval.audit_strict
    if args.strict:
        strict = True
    if args.no_strict:
        strict = False

    carriers, bucket_layout = load_carriers_and_layout(
        carrier_path=carrier_path,
        bucket_spec_path=bucket_spec_path,
        include_disallowed=True,
    )
    tokenizer = load_tokenizer(tokenizer_backend, tokenizer_name)
    result = audit_carriers(carriers, tokenizer=tokenizer, bucket_layout=bucket_layout)
    summary_text = _build_summary_text(result)
    print(summary_text, end="")

    if args.output_json:
        output_path = _resolve_path(repo_root, args.output_json)
        assert output_path is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save_json(output_path)
        print(f"saved_report={output_path}")

    if args.output_summary:
        summary_path = _resolve_path(repo_root, args.output_summary)
        assert summary_path is not None
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text, encoding="utf-8")
        print(f"saved_summary={summary_path}")

    if strict and not result.is_alignment_safe:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

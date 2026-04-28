from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.prepare_g3a_v3_block_scale import (
    _entry_with_identity,
    _entry_with_repo_relative_config,
    _repo_relative_path,
    _save_manifest,
)
from src.infrastructure.manifest import ManifestEntry, build_manifest_from_config
from src.infrastructure.paths import current_timestamp, discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Perinucleus-style baseline manifests.")
    parser.add_argument(
        "--package-config",
        default="configs/experiment/baselines/perinucleus/package__baseline_perinucleus_qwen_v1.yaml",
    )
    parser.add_argument(
        "--output",
        default="results/processed/paper_stats/baseline_perinucleus_package_dry_run.json",
    )
    parser.add_argument(
        "--eval-manifest-out",
        default="manifests/baseline_perinucleus/eval_manifest.json",
    )
    parser.add_argument(
        "--output-root-base",
        help="Optional base directory. Defaults to EXP_SCRATCH/baselines/perinucleus_qwen.",
    )
    parser.add_argument(
        "--environment-setup",
        help="Optional runtime environment setup block. Defaults to CHIMERA_ENV_SETUP or package config.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(os.path.expandvars(raw))
    return path if path.is_absolute() else repo_root / path


def _resolve_output_root_base(package_config: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(Path(os.path.expandvars(explicit)).as_posix())
    prefix = str(package_config["new_case_root_prefix"])
    exp_scratch = os.environ.get("EXP_SCRATCH")
    if exp_scratch:
        return str((Path(exp_scratch) / prefix).as_posix())
    print(
        "WARNING: EXP_SCRATCH is not set; falling back to the package-relative "
        "new_case_root_prefix. Set EXP_SCRATCH on Chimera so Perinucleus outputs stay off home.",
        file=sys.stderr,
    )
    return prefix


def _case_id(payload: str, seed: int, query_budget: int) -> str:
    return f"perinucleus_q{query_budget}_{payload}_s{seed}"


def _case_root(output_root_base: str, payload: str, seed: int, query_budget: int) -> str:
    return str((Path(output_root_base) / "final" / f"q{query_budget}" / f"{payload}_s{seed}").as_posix())


def _matched_budget_status(query_budget: int, matched_query_budget: int) -> str:
    if query_budget == matched_query_budget:
        return "matched"
    if query_budget < matched_query_budget:
        return "under_budget_diagnostic"
    return "over_budget_diagnostic"


def _cases(package_config: dict[str, Any], output_root_base: str) -> list[dict[str, Any]]:
    matrix = dict(package_config["final_matrix"])
    fixed = dict(package_config["fixed_contract"])
    matched_query_budget = int(fixed.get("matched_query_budget", 4))
    cases: list[dict[str, Any]] = []
    for query_budget in matrix["query_budgets"]:
        for seed in matrix["seeds"]:
            for payload in matrix["payloads"]:
                q = int(query_budget)
                s = int(seed)
                p = str(payload)
                case_id = _case_id(p, s, q)
                cases.append(
                    {
                        "case_id": case_id,
                        "payload": p,
                        "seed": s,
                        "query_budget": q,
                        "matched_budget_status": _matched_budget_status(q, matched_query_budget),
                        "case_root": _case_root(output_root_base, p, s, q),
                        "paper_ready_denominator": True,
                    }
                )
    return cases


def _build_eval_entry(
    *,
    repo_root: Path,
    eval_config_path: Path,
    case: dict[str, Any],
    environment_setup: str | None,
) -> ManifestEntry:
    output_root = str((Path(case["case_root"]) / "runs").as_posix())
    overrides = [
        f"run.seed={case['seed']}",
        f"run.variant_name=perinucleus-q{case['query_budget']}",
        f"eval.payload_text={case['payload']}",
        f"eval.max_candidates={case['query_budget']}",
        "eval.min_score=1.0",
        f"baseline_perinucleus.query_budget={case['query_budget']}",
        f"runtime.output_root={output_root}",
    ]
    if environment_setup:
        overrides.append(f"runtime.environment_setup={environment_setup}")
    manifest = build_manifest_from_config(eval_config_path, overrides=overrides)
    entry = _entry_with_repo_relative_config(manifest.entries[0], repo_root)
    return _entry_with_identity(
        entry,
        manifest_id=f"perinucleus-final-eval-q{case['query_budget']}-{str(case['payload']).lower()}-s{case['seed']}",
        manifest_name="baseline_perinucleus_final_eval",
    )


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    eval_config_path = _resolve_path(repo_root, str(package_config["eval_config"]))
    output_path = _resolve_path(repo_root, args.output)
    eval_manifest_path = _resolve_path(repo_root, args.eval_manifest_out)
    output_root_base = _resolve_output_root_base(package_config, args.output_root_base)
    environment_setup = (
        args.environment_setup
        or os.environ.get("CHIMERA_ENV_SETUP")
        or package_config.get("chimera_environment_setup")
    )

    cases = _cases(package_config, output_root_base)
    eval_entries = [
        _build_eval_entry(
            repo_root=repo_root,
            eval_config_path=eval_config_path,
            case=case,
            environment_setup=environment_setup,
        )
        for case in cases
    ]
    _save_manifest(
        repo_root,
        eval_config_path,
        "baseline_perinucleus_final_eval",
        eval_entries,
        eval_manifest_path,
    )
    payload = {
        "schema_name": "baseline_perinucleus_package_dry_run",
        "schema_version": 1,
        "generated_at": current_timestamp(),
        "package_config_path": _repo_relative_path(repo_root, package_config_path),
        "eval_config_path": _repo_relative_path(repo_root, eval_config_path),
        "output_root_base": output_root_base,
        "eval_manifest_entry_count": len(eval_entries),
        "target_case_count": len(cases),
        "thresholds_frozen": False,
        "fixed_contract": package_config.get("fixed_contract", {}),
        "final_matrix": package_config.get("final_matrix", {}),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote Perinucleus baseline dry-run summary to {output_path}")
    print(f"wrote {len(eval_entries)} eval manifest entries to {eval_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

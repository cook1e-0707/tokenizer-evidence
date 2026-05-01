from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml

from src.infrastructure.manifest import ManifestEntry, ManifestFile, ResourceRequest, save_manifest
from src.infrastructure.paths import current_timestamp, discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the frozen official Perinucleus Qwen final protocol manifest.")
    parser.add_argument(
        "--package-config",
        default="configs/experiment/baselines/perinucleus_official/package__baseline_perinucleus_official_qwen_v1.yaml",
    )
    parser.add_argument(
        "--output",
        default="results/processed/paper_stats/baseline_perinucleus_official_qwen_final_package_dry_run.json",
    )
    parser.add_argument(
        "--case-table",
        default="results/tables/baseline_perinucleus_official_qwen_final_cases.csv",
    )
    parser.add_argument(
        "--manifest-out",
        default="manifests/baseline_perinucleus_official_qwen_final/eval_manifest.json",
    )
    parser.add_argument(
        "--doc-out",
        default="docs/baseline_perinucleus_official_qwen_final_protocol.md",
    )
    parser.add_argument("--output-root-base")
    parser.add_argument("--environment-setup")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value)))
    return path if path.is_absolute() else repo_root / path


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _output_root_base(package: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(Path(os.path.expandvars(explicit)).as_posix())
    final = dict(package.get("matched_qwen_final", {}))
    if final.get("output_root_base"):
        return str(Path(os.path.expandvars(str(final["output_root_base"]))).as_posix())
    exp_scratch = os.environ.get("EXP_SCRATCH")
    if exp_scratch:
        return str((Path(exp_scratch) / "baselines/perinucleus_official_qwen").as_posix())
    return "/hpcstor6/scratch01/g/guanjie.lin001/tokenizer-evidence/baselines/perinucleus_official_qwen"


def _matched_budget_status(query_budget: int, matched_query_budget: int) -> str:
    if query_budget == matched_query_budget:
        return "matched"
    if query_budget < matched_query_budget:
        return "under_budget_diagnostic"
    return "over_budget_diagnostic"


def _case_root(output_root_base: str, payload: str, seed: int, query_budget: int) -> str:
    return str((Path(output_root_base) / "final" / f"q{query_budget}" / f"{payload}_s{seed}").as_posix())


def _cases(package: dict[str, Any], output_root_base: str) -> list[dict[str, Any]]:
    final = dict(package["matched_qwen_final"])
    matched_query_budget = int(final.get("matched_query_budget", 4))
    cases: list[dict[str, Any]] = []
    for query_budget in final["query_budgets"]:
        for seed in final["seeds"]:
            for payload in final["payloads"]:
                q = int(query_budget)
                s = int(seed)
                p = str(payload)
                root = _case_root(output_root_base, p, s, q)
                cases.append(
                    {
                        "case_id": f"perinucleus-official-q{q}-{p.lower()}-s{s}",
                        "payload": p,
                        "seed": s,
                        "query_budget": q,
                        "matched_budget_status": _matched_budget_status(q, matched_query_budget),
                        "case_root": root,
                        "run_root": str((Path(root) / "runs").as_posix()),
                        "eval_summary_glob": str(
                            (
                                Path(root)
                                / "runs"
                                / "perinucleus_official_qwen_final"
                                / "*"
                                / "eval_summary.json"
                            ).as_posix()
                        ),
                        "paper_ready_denominator": True,
                    }
                )
    return cases


def _resources(package: dict[str, Any], environment_setup: str | None) -> ResourceRequest:
    resources = dict(package.get("matched_qwen_final", {}).get("requested_resources", {}))
    return ResourceRequest(
        partition=str(resources.get("partition", "pomplun")),
        gpu_type=resources.get("gpu_type", "h200"),
        num_gpus=int(resources.get("num_gpus", 1)),
        cpus=int(resources.get("cpus", 8)),
        mem_gb=int(resources.get("mem_gb", 240)),
        time_limit=str(resources.get("time_limit", "04:00:00")),
        account=resources.get("account"),
        environment_setup=environment_setup
        or str(package.get("chimera_environment_setup", "if [ -f /etc/profile ]; then . /etc/profile; fi")),
        slurm_template=str(resources.get("slurm_template", "")),
    )


def _entry(
    *,
    package_config_path: str,
    case: dict[str, Any],
    output_root: str,
    resources: ResourceRequest,
) -> ManifestEntry:
    overrides = (
        f"matched_qwen_final.payload_text={case['payload']}",
        f"matched_qwen_final.seed={case['seed']}",
        f"matched_qwen_final.query_budget={case['query_budget']}",
        f"matched_qwen_final.case_root={case['case_root']}",
    )
    return ManifestEntry(
        manifest_id=str(case["case_id"]),
        experiment_name="perinucleus_official_qwen_final",
        method_name="baseline_perinucleus_official",
        model_name="qwen2.5-7b-instruct",
        seed=int(case["seed"]),
        config_paths=(package_config_path,),
        overrides=overrides,
        output_root=output_root,
        output_dir=None,
        requested_resources=resources,
        launcher_mode="slurm",
        status="pending",
        tags=("baseline", "perinucleus", "official", "qwen-final", "frozen-candidate"),
        notes="Frozen Qwen LoRA adaptation of official Scalable/Perinucleus; no hyperparameter changes allowed.",
        entry_point="scripts/run_perinucleus_official_final_eval.py",
        manifest_name="baseline_perinucleus_official_qwen_final_eval",
        primary_config_path=package_config_path,
    )


def _write_cases_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "payload",
        "seed",
        "query_budget",
        "matched_budget_status",
        "case_root",
        "run_root",
        "eval_summary_path",
        "eval_summary_glob",
        "paper_ready_denominator",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    selected = payload["frozen_candidate"]["selected_candidate"]
    lines = [
        "# Official Perinucleus Qwen Final Protocol",
        "",
        f"Generated at: `{payload['generated_at']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "This is a dry-run package for the final protocol. It renders the manifest and case table only; it does not submit jobs.",
        "",
        "## Frozen Candidate",
        "",
        f"- Arm: `{selected['arm_id']}`",
        f"- Adapter: `{selected['adapter_path']}`",
        f"- Fingerprints file: `{selected.get('fingerprints_file', '')}`",
        f"- Exact accuracy: `{selected['exact_accuracy']}`",
        f"- Utility sanity: `{selected['adapter_utility']}` vs base `{selected['base_utility']}`",
        "",
        "## Matrix",
        "",
        f"- Cases: `{payload['target_case_count']}`",
        f"- Payloads: `{payload['final_matrix']['payloads']}`",
        f"- Seeds: `{payload['final_matrix']['seeds']}`",
        f"- Query budgets: `{payload['final_matrix']['query_budgets']}`",
        "",
        "## Execution",
        "",
        "Prepare or inspect jobs without submission:",
        "",
        "```bash",
        "python3 scripts/submit_slurm.py \\",
        f"  --manifest {payload['manifest_path']} \\",
        "  --registry manifests/baseline_perinucleus_official_qwen_final/eval_job_registry.jsonl \\",
        "  --all-pending",
        "```",
        "",
        "Submit only after reviewing the rendered scripts:",
        "",
        "```bash",
        "# First run one smoke case.",
        "python3 scripts/submit_slurm.py \\",
        f"  --manifest {payload['manifest_path']} \\",
        "  --registry manifests/baseline_perinucleus_official_qwen_final/eval_job_registry.jsonl \\",
        "  --manifest-id perinucleus-official-q1-u00-s17 \\",
        "  --submit --force",
        "",
        "# If the smoke case completes, submit the remaining pending cases.",
        "python3 scripts/submit_slurm.py \\",
        f"  --manifest {payload['manifest_path']} \\",
        "  --registry manifests/baseline_perinucleus_official_qwen_final/eval_job_registry.jsonl \\",
        "  --all-pending --submit --force",
        "```",
        "",
        "## Outputs",
        "",
        f"- Dry-run summary: `{payload['output_summary']}`",
        f"- Case table: `{payload['case_table']}`",
        f"- Manifest: `{payload['manifest_path']}`",
        "",
        "After jobs finish, aggregate final artifacts with:",
        "",
        "```bash",
        "python3 scripts/build_perinucleus_official_final_artifacts.py",
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve(repo_root, args.package_config)
    package = _load_yaml(package_config_path)
    final = dict(package["matched_qwen_final"])
    frozen_summary_path = _resolve(repo_root, final["frozen_candidate_summary"])
    frozen_summary = _load_json(frozen_summary_path)
    if not frozen_summary.get("gates", {}).get("final_protocol_allowed"):
        raise RuntimeError("Frozen candidate summary does not allow final protocol.")
    if str(frozen_summary["selected_candidate"]["arm_id"]) != str(final["frozen_candidate_arm"]):
        raise RuntimeError("Package frozen_candidate_arm does not match freeze summary.")

    output_path = _resolve(repo_root, args.output)
    case_table_path = _resolve(repo_root, args.case_table)
    manifest_path = _resolve(repo_root, args.manifest_out)
    doc_path = _resolve(repo_root, args.doc_out)
    output_root_base = _output_root_base(package, args.output_root_base)
    environment_setup = args.environment_setup or package.get("chimera_environment_setup")
    package_config_rel = _repo_relative(repo_root, package_config_path)
    resources = _resources(package, environment_setup)
    cases = _cases(package, output_root_base)
    entries = [
        _entry(
            package_config_path=package_config_rel,
            case=case,
            output_root=str((Path(case["case_root"]) / "runs").as_posix()),
            resources=resources,
        )
        for case in cases
    ]
    manifest = ManifestFile(
        schema_name="manifest_file",
        schema_version=1,
        manifest_name="baseline_perinucleus_official_qwen_final_eval",
        created_at=current_timestamp(),
        source_config_path=package_config_rel,
        entries=tuple(entries),
    )
    save_manifest(manifest, manifest_path)
    _write_cases_csv(case_table_path, cases)
    payload = {
        "schema_name": "baseline_perinucleus_official_qwen_final_package_dry_run",
        "schema_version": 1,
        "generated_at": current_timestamp(),
        "decision": "PERINUCLEUS_OFFICIAL_QWEN_FINAL_DRY_RUN_READY: review manifest before submission.",
        "package_config_path": package_config_rel,
        "frozen_candidate_config": final["frozen_candidate_config"],
        "frozen_candidate_summary": final["frozen_candidate_summary"],
        "frozen_candidate": frozen_summary,
        "output_root_base": output_root_base,
        "target_case_count": len(cases),
        "manifest_entry_count": len(entries),
        "final_matrix": {
            "payloads": final["payloads"],
            "seeds": final["seeds"],
            "query_budgets": final["query_budgets"],
        },
        "manifest_path": _repo_relative(repo_root, manifest_path),
        "case_table": _repo_relative(repo_root, case_table_path),
        "output_summary": _repo_relative(repo_root, output_path),
        "doc": _repo_relative(repo_root, doc_path),
        "cases": cases,
        "submission_policy": "dry_run_only_until_manual_review",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_doc(doc_path, payload)
    print(f"wrote official Perinucleus Qwen final dry-run summary to {output_path}")
    print(f"wrote {len(entries)} eval manifest entries to {manifest_path}")
    print(f"wrote case table to {case_table_path}")
    print(f"wrote protocol doc to {doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

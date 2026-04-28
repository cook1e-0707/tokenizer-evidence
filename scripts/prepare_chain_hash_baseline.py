from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
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
    parser = argparse.ArgumentParser(description="Prepare Chain&Hash-style baseline manifests.")
    parser.add_argument(
        "--package-config",
        default="configs/experiment/baselines/chain_hash/package__baseline_chain_hash_qwen_v1.yaml",
    )
    parser.add_argument(
        "--output",
        default="results/processed/paper_stats/baseline_chain_hash_package_dry_run.json",
    )
    parser.add_argument(
        "--train-manifest-out",
        default="manifests/baseline_chain_hash/train_manifest.json",
    )
    parser.add_argument(
        "--eval-manifest-out",
        default="manifests/baseline_chain_hash/eval_manifest.json",
    )
    parser.add_argument(
        "--output-root-base",
        help="Optional base directory. Defaults to EXP_SCRATCH/baselines/chain_hash_qwen.",
    )
    parser.add_argument(
        "--environment-setup",
        help="Optional runtime environment setup block. Defaults to CHIMERA_ENV_SETUP or package config.",
    )
    parser.add_argument(
        "--write-contracts",
        action="store_true",
        help="Write scratch-local chain_hash_contract.json and train.jsonl files.",
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


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _hash_index(payload: object, modulo: int) -> int:
    return int(_stable_hash(payload)[:16], 16) % max(1, modulo)


def _resolve_output_root_base(package_config: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(Path(os.path.expandvars(explicit)).as_posix())
    prefix = str(package_config["new_case_root_prefix"])
    exp_scratch = os.environ.get("EXP_SCRATCH")
    if exp_scratch:
        return str((Path(exp_scratch) / prefix).as_posix())
    print(
        "WARNING: EXP_SCRATCH is not set; falling back to the package-relative "
        "new_case_root_prefix. Set EXP_SCRATCH on Chimera so Chain&Hash outputs stay off home.",
        file=sys.stderr,
    )
    return prefix


def _owner_secret(package_config: dict[str, Any]) -> str:
    owner_secret = dict(package_config.get("owner_secret", {}))
    env_value = os.environ.get("CHAIN_HASH_BASELINE_SECRET", "").strip()
    return env_value or str(owner_secret.get("default_material", "chain_hash_qwen_baseline_v1"))


def _case_root(output_root_base: str, payload: str, seed: int) -> str:
    return str((Path(output_root_base) / "final" / "train" / f"{payload}_s{seed}").as_posix())


def _eval_case_root(output_root_base: str, payload: str, seed: int, query_budget: int) -> str:
    return str((Path(output_root_base) / "final" / f"q{query_budget}" / f"{payload}_s{seed}").as_posix())


def _matched_budget_status(query_budget: int, matched_query_budget: int) -> str:
    if query_budget == matched_query_budget:
        return "matched"
    if query_budget < matched_query_budget:
        return "under_budget_diagnostic"
    return "over_budget_diagnostic"


def _prompt_text(template: str, key_text: str) -> str:
    return template.format(key_text=key_text).rstrip() + " "


def _contract_for_case(
    *,
    package_config: dict[str, Any],
    payload: str,
    seed: int,
    case_root: str,
) -> dict[str, Any]:
    secret = _owner_secret(package_config)
    candidate_set = [str(item) for item in package_config["candidate_set"]]
    prompt_bank = [str(item) for item in package_config["prompt_bank"]]
    training = dict(package_config["training"])
    response_count = int(training.get("response_count", len(prompt_bank)))
    template = str(training["prompt_template"])
    responses = []
    for query_index, key_text in enumerate(prompt_bank[:response_count]):
        key_payload = {
            "secret": secret,
            "payload": payload,
            "seed": seed,
            "query_index": query_index,
            "key_text": key_text,
        }
        response = candidate_set[_hash_index(key_payload, len(candidate_set))]
        prompt = _prompt_text(template, key_text)
        responses.append(
            {
                "query_index": query_index,
                "key_id": f"{payload}_s{seed}_k{query_index:02d}",
                "key_text": key_text,
                "key_hash": _stable_hash({k: v for k, v in key_payload.items() if k != "secret"}),
                "prompt": prompt,
                "expected_response": response,
            }
        )
    contract = {
        "schema_name": "chain_hash_contract",
        "schema_version": 1,
        "package_name": package_config["package_name"],
        "implementation_scope": "chain_hash_style_adapted_not_exact_reproduction",
        "payload_text": payload,
        "seed": seed,
        "prompt_family": package_config["fixed_contract"]["prompt_family"],
        "secret_hash": _stable_hash(secret),
        "candidate_set_hash": _stable_hash(candidate_set),
        "prompt_bank_hash": _stable_hash(prompt_bank),
        "response_count": response_count,
        "repeats_per_response": int(training.get("repeats_per_response", 1)),
        "case_root": case_root,
        "responses": responses,
    }
    contract["contract_hash"] = _stable_hash(contract)
    return contract


def _write_contract_files(contract: dict[str, Any], case_root: Path) -> tuple[Path, Path]:
    case_root.mkdir(parents=True, exist_ok=True)
    contract_path = case_root / "chain_hash_contract.json"
    train_path = case_root / "train.jsonl"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    repeats = int(contract["repeats_per_response"])
    with train_path.open("w", encoding="utf-8") as handle:
        for repeat_index in range(repeats):
            for response in contract["responses"]:
                row = {
                    "prompt": response["prompt"],
                    "target_symbols": [],
                    "metadata": {
                        "completion": response["expected_response"],
                        "target_mode": "chain_hash_response_forcing",
                        "payload_text": contract["payload_text"],
                        "seed": contract["seed"],
                        "query_index": response["query_index"],
                        "key_hash": response["key_hash"],
                        "contract_hash": contract["contract_hash"],
                        "repeat_index": repeat_index,
                    },
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return contract_path, train_path


def _train_cases(package_config: dict[str, Any], output_root_base: str) -> list[dict[str, Any]]:
    matrix = dict(package_config["final_matrix"])
    cases: list[dict[str, Any]] = []
    for seed in matrix["seeds"]:
        for payload in matrix["payloads"]:
            root = _case_root(output_root_base, str(payload), int(seed))
            cases.append(
                {
                    "case_id": f"chain_hash_train_{payload}_s{seed}",
                    "payload": str(payload),
                    "seed": int(seed),
                    "case_root": root,
                    "contract_path": str((Path(root) / "chain_hash_contract.json").as_posix()),
                    "train_path": str((Path(root) / "train.jsonl").as_posix()),
                }
            )
    return cases


def _eval_cases(package_config: dict[str, Any], output_root_base: str) -> list[dict[str, Any]]:
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
                train_root = _case_root(output_root_base, p, s)
                cases.append(
                    {
                        "case_id": f"chain_hash_q{q}_{p}_s{s}",
                        "payload": p,
                        "seed": s,
                        "query_budget": q,
                        "matched_budget_status": _matched_budget_status(q, matched_query_budget),
                        "case_root": _eval_case_root(output_root_base, p, s, q),
                        "train_case_root": train_root,
                        "contract_path": str((Path(train_root) / "chain_hash_contract.json").as_posix()),
                        "eval_input_path": str((Path(train_root) / "runs" / "exp_train" / "latest_eval_input.json").as_posix()),
                        "paper_ready_denominator": True,
                    }
                )
    return cases


def _build_train_entry(
    *,
    repo_root: Path,
    train_config_path: Path,
    case: dict[str, Any],
    environment_setup: str | None,
) -> ManifestEntry:
    output_root = str((Path(case["case_root"]) / "runs").as_posix())
    overrides = [
        f"run.seed={case['seed']}",
        f"run.variant_name=chain-hash-{case['payload']}",
        f"eval.payload_text={case['payload']}",
        f"data.train_path={case['train_path']}",
        f"runtime.output_root={output_root}",
    ]
    if environment_setup:
        overrides.append(f"runtime.environment_setup={environment_setup}")
    manifest = build_manifest_from_config(train_config_path, overrides=overrides)
    entry = _entry_with_repo_relative_config(manifest.entries[0], repo_root)
    return _entry_with_identity(
        entry,
        manifest_id=f"chain-hash-train-{str(case['payload']).lower()}-s{case['seed']}",
        manifest_name="baseline_chain_hash_train",
    )


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
        f"run.variant_name=chain-hash-q{case['query_budget']}",
        f"eval.payload_text={case['payload']}",
        f"eval.max_candidates={case['query_budget']}",
        "eval.min_score=1.0",
        f"data.eval_path={case['eval_input_path']}",
        f"baseline_chain_hash.contract_path={case['contract_path']}",
        f"baseline_chain_hash.query_budget={case['query_budget']}",
        f"runtime.output_root={output_root}",
    ]
    if environment_setup:
        overrides.append(f"runtime.environment_setup={environment_setup}")
    manifest = build_manifest_from_config(eval_config_path, overrides=overrides)
    entry = _entry_with_repo_relative_config(manifest.entries[0], repo_root)
    return _entry_with_identity(
        entry,
        manifest_id=f"chain-hash-eval-q{case['query_budget']}-{str(case['payload']).lower()}-s{case['seed']}",
        manifest_name="baseline_chain_hash_eval",
    )


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    package_config_path = _resolve_path(repo_root, args.package_config)
    package_config = _load_yaml(package_config_path)
    train_config_path = _resolve_path(repo_root, str(package_config["train_config"]))
    eval_config_path = _resolve_path(repo_root, str(package_config["eval_config"]))
    output_path = _resolve_path(repo_root, args.output)
    train_manifest_path = _resolve_path(repo_root, args.train_manifest_out)
    eval_manifest_path = _resolve_path(repo_root, args.eval_manifest_out)
    output_root_base = _resolve_output_root_base(package_config, args.output_root_base)
    environment_setup = (
        args.environment_setup
        or os.environ.get("CHIMERA_ENV_SETUP")
        or package_config.get("chimera_environment_setup")
    )

    train_cases = _train_cases(package_config, output_root_base)
    eval_cases = _eval_cases(package_config, output_root_base)
    if args.write_contracts:
        for case in train_cases:
            contract = _contract_for_case(
                package_config=package_config,
                payload=str(case["payload"]),
                seed=int(case["seed"]),
                case_root=str(case["case_root"]),
            )
            _write_contract_files(contract, Path(str(case["case_root"])))
    train_entries = [
        _build_train_entry(
            repo_root=repo_root,
            train_config_path=train_config_path,
            case=case,
            environment_setup=environment_setup,
        )
        for case in train_cases
    ]
    eval_entries = [
        _build_eval_entry(
            repo_root=repo_root,
            eval_config_path=eval_config_path,
            case=case,
            environment_setup=environment_setup,
        )
        for case in eval_cases
    ]
    _save_manifest(repo_root, train_config_path, "baseline_chain_hash_train", train_entries, train_manifest_path)
    _save_manifest(repo_root, eval_config_path, "baseline_chain_hash_eval", eval_entries, eval_manifest_path)
    payload = {
        "schema_name": "baseline_chain_hash_package_dry_run",
        "schema_version": 1,
        "generated_at": current_timestamp(),
        "package_config_path": _repo_relative_path(repo_root, package_config_path),
        "train_config_path": _repo_relative_path(repo_root, train_config_path),
        "eval_config_path": _repo_relative_path(repo_root, eval_config_path),
        "output_root_base": output_root_base,
        "contracts_written": bool(args.write_contracts),
        "train_manifest_entry_count": len(train_entries),
        "eval_manifest_entry_count": len(eval_entries),
        "train_case_count": len(train_cases),
        "target_case_count": len(eval_cases),
        "thresholds_frozen": False,
        "fixed_contract": package_config.get("fixed_contract", {}),
        "final_matrix": package_config.get("final_matrix", {}),
        "train_cases": train_cases,
        "eval_cases": eval_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote Chain&Hash baseline dry-run summary to {output_path}")
    print(f"wrote {len(train_entries)} train manifest entries to {train_manifest_path}")
    print(f"wrote {len(eval_entries)} eval manifest entries to {eval_manifest_path}")
    if not args.write_contracts:
        print("contracts not written; rerun with --write-contracts on Chimera before submitting train jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

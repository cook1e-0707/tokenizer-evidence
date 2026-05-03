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

from scripts.run_full_far_payload_claim_benchmark import (
    FRESH_ORGANIC_NULL_STATUS,
    OURS_METHOD_ID,
    PERINUCLEUS_METHOD_ID,
    _build_execution_summary,
    _execution_base_row,
    _load_perinucleus_frozen_candidate,
    _load_yaml,
    _null_model_cfg,
    _organic_index,
    _organic_prompt_ids_for_budget,
    _organic_seed_for_row,
    _organic_shard_bounds,
    _ours_verify_generated_organic_text,
    _resolve,
    _safe_int,
    _set_fresh_completed,
    _write_csv,
    _write_json,
    build_plan_rows,
)
from src.infrastructure.paths import discover_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 for full FAR organic-null: expand prompt-level base-Qwen "
            "cache into completed organic FAR row shards without GPU inference."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--row-shard-output-dir", required=True)
    parser.add_argument("--expected-shard-count", type=int, default=None)
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Optional single row-shard index to build. Omit to build all row shards serially.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help="Global row-shard count. Defaults to --expected-shard-count or cache shard count.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_tokenizer(cfg: dict[str, Any], model_name: str) -> Any:
    from transformers import AutoTokenizer

    runtime = dict(cfg.get("runtime") or {})
    local_files_only = bool(runtime.get("local_files_only", True))
    trust_remote_code = bool(runtime.get("trust_remote_code", True))
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    candidate_cache_dirs = [
        os.environ.get("HF_HOME"),
        os.environ.get("TRANSFORMERS_CACHE"),
        str(runtime.get("hf_home") or "").strip() or None,
        str(Path(f"/hpcstor6/scratch01/g/{user}/huggingface")) if user else None,
    ]
    seen: set[str] = set()
    cache_dirs: list[str | None] = []
    for raw in candidate_cache_dirs:
        if raw is None or str(raw).strip() == "":
            continue
        value = str(raw)
        if value in seen:
            continue
        seen.add(value)
        cache_dirs.append(value)
    cache_dirs.append(None)

    errors: list[str] = []
    tokenizer = None
    for cache_dir in cache_dirs:
        if cache_dir is not None:
            os.environ.setdefault("HF_HOME", cache_dir)
            os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            break
        except Exception as error:
            errors.append(f"cache_dir={cache_dir or '<default>'}: {type(error).__name__}: {error}")
    if tokenizer is None:
        raise RuntimeError(
            "Could not load tokenizer for organic cache expansion. Tried:\n"
            + "\n".join(errors)
            + "\nSet HF_HOME/TRANSFORMERS_CACHE to the same cache used by the Slurm GPU jobs, "
            "or pass a config with runtime.local_files_only=false if online downloads are allowed."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    return tokenizer


def _cache_paths(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("full_far_payload_claim_organic_prompt_cache_shard_*_of_*.csv"))


def _row_shard_paths(row_shard_dir: Path, shard_index: int, shard_count: int) -> dict[str, Path]:
    stem = f"full_far_payload_claim_organic_prompts_shard_{shard_index:03d}_of_{shard_count:03d}"
    return {
        "table": row_shard_dir / f"{stem}.csv",
        "summary": row_shard_dir / f"{stem}.json",
    }


def _read_cache_rows(paths: list[Path]) -> dict[str, dict[str, str]]:
    by_prompt_id: dict[str, dict[str, str]] = {}
    duplicates: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("status", "")) != "completed_prompt_cache":
                    continue
                prompt_id = str(row.get("organic_prompt_id", ""))
                if not prompt_id.startswith("organic_"):
                    continue
                if prompt_id in by_prompt_id:
                    duplicates.add(prompt_id)
                enriched = dict(row)
                enriched["cache_source"] = str(path)
                by_prompt_id[prompt_id] = enriched
    if duplicates:
        raise RuntimeError(f"Duplicate organic prompt cache rows: {sorted(duplicates)[:10]}")
    return by_prompt_id


def _validate_cache_coverage(cfg: dict[str, Any], cache_by_prompt_id: dict[str, dict[str, str]]) -> None:
    organic_count = int((cfg.get("null_protocol") or {}).get("organic_prompt_count", 0))
    missing = [
        f"organic_{index:04d}"
        for index in range(organic_count)
        if f"organic_{index:04d}" not in cache_by_prompt_id
    ]
    if missing:
        raise RuntimeError(
            f"Organic prompt cache is incomplete: missing {len(missing)} prompts; "
            f"first_missing={missing[:10]}"
        )


def _required_organic_rows_for_shard(
    rows: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    start, end = _organic_shard_bounds(cfg, shard_index, shard_count)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("evaluation_type", "")) != "organic_prompt_null":
            continue
        if str(row.get("null_model_id", "")) != "base_qwen":
            continue
        organic_index = _organic_index(str(row.get("organic_prompt_id", "organic_0000")))
        if start <= organic_index < end:
            selected.append(row)
    if not selected:
        raise RuntimeError(f"No organic rows selected for shard {shard_index}/{shard_count}.")
    return selected


def _ours_from_cache_result(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    row: dict[str, Any],
    cache_by_prompt_id: dict[str, dict[str, str]],
    tokenizer: Any,
    model_name: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    claim_payload = str(row.get("claim_payload", ""))
    organic_prompt_id = str(row.get("organic_prompt_id", "organic_0000"))
    budget = max(1, _safe_int(row.get("query_budget"), 1))
    prompt_ids = _organic_prompt_ids_for_budget(
        organic_prompt_id=organic_prompt_id,
        query_budget=budget,
        cfg=cfg,
    )
    prompt_results: list[dict[str, Any]] = []
    for prompt_id in prompt_ids:
        cache_row = cache_by_prompt_id[prompt_id]
        generated_text = str(cache_row.get("ours_generated_text", ""))
        verification = _ours_verify_generated_organic_text(
            repo_root=repo_root,
            cfg=cfg,
            model_name=model_name,
            tokenizer=tokenizer,
            claim_payload=claim_payload,
            generated_text=generated_text,
            context=context,
        )
        prompt_results.append(
            {
                "organic_prompt_id": prompt_id,
                "cache_source": str(cache_row.get("cache_source", "")),
                "claim_accept": bool(verification["claim_accept"]),
                "field_valid_rate": verification["field_valid_rate"],
                "bucket_correct_rate": verification["bucket_correct_rate"],
                "slot_exact_rate": verification["slot_exact_rate"],
                "valid_canonical_block_count": verification["valid_canonical_block_count"],
                "generated_text_preview": generated_text[:240],
            }
        )
    accept_count = sum(1 for item in prompt_results if item["claim_accept"])
    return {
        "claim_accept": accept_count > 0,
        "ownership_score": accept_count / len(prompt_results) if prompt_results else 0.0,
        "organic_prompt_id": organic_prompt_id,
        "organic_prompt_ids": prompt_ids,
        "claim_payload": claim_payload,
        "model_name": model_name,
        "fresh_query_count": len(prompt_ids),
        "accept_count": accept_count,
        "prompt_results": prompt_results,
        "decision_rule": "accept_if_any_cached_organic_prompt_decodes_claim_payload",
    }


def _perinucleus_context(repo_root: Path, cfg: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    from scripts import run_perinucleus_official_final_eval as final_eval

    frozen = _load_perinucleus_frozen_candidate(repo_root, cfg)
    candidate = dict(frozen["candidate"])
    base_model = str(_null_model_cfg(cfg, "base_qwen").get("model") or frozen["model"]["base"])
    fingerprints_path = _resolve(repo_root, candidate["fingerprints_file"])
    if fingerprints_path is None or not fingerprints_path.exists():
        raise FileNotFoundError(f"Missing Perinucleus fingerprints file: {fingerprints_path}")
    fingerprint_rows = final_eval._load_fingerprint_rows(
        fingerprints_path,
        int(candidate["num_fingerprints"]),
    )
    response_id_cache: dict[str, tuple[str, list[int], bool]] = {}
    for item in fingerprint_rows:
        response = str(item["response"])
        if response in response_id_cache:
            continue
        from scripts import run_perinucleus_official_overfit_gate as overfit

        response_id_cache[response] = overfit._truncate_response(
            tokenizer,
            response,
            int(candidate.get("response_length", 1)),
        )
    return {
        "candidate": candidate,
        "base_model": base_model,
        "fingerprint_rows": fingerprint_rows,
        "response_id_cache": response_id_cache,
    }


def _perinucleus_from_cache_result(
    *,
    cfg: dict[str, Any],
    row: dict[str, Any],
    cache_by_prompt_id: dict[str, dict[str, str]],
    context: dict[str, Any],
) -> dict[str, Any]:
    from scripts import run_perinucleus_official_final_eval as final_eval

    claim_payload = str(row.get("claim_payload", ""))
    budget = max(1, _safe_int(row.get("query_budget"), 1))
    organic_prompt_id = str(row.get("organic_prompt_id", "organic_0000"))
    selected_seed = _organic_seed_for_row(cfg, row)
    candidate = dict(context["candidate"])
    selected_rows = final_eval._select_fingerprints(
        rows=context["fingerprint_rows"],
        payload_text=claim_payload,
        seed=selected_seed,
        query_budget=budget,
        arm_id=str(candidate["arm_id"]),
    )
    prompt_ids = _organic_prompt_ids_for_budget(
        organic_prompt_id=organic_prompt_id,
        query_budget=budget,
        cfg=cfg,
    )
    response_rows: list[dict[str, Any]] = []
    exact_count = 0
    for prompt_id, selected in zip(prompt_ids, selected_rows, strict=True):
        response, response_ids, response_truncated = context["response_id_cache"][
            str(selected["response"])
        ]
        cache_row = cache_by_prompt_id[prompt_id]
        generated_ids = [
            int(token_id)
            for token_id in json.loads(str(cache_row.get("perinucleus_generated_token_ids_json", "[]")))
        ]
        token_exact = generated_ids[: len(response_ids)] == [int(token_id) for token_id in response_ids]
        exact_count += int(token_exact)
        response_rows.append(
            {
                "organic_prompt_id": prompt_id,
                "cache_source": str(cache_row.get("cache_source", "")),
                "source_fingerprint_id": int(selected["source_fingerprint_id"]),
                "target_response": response,
                "target_response_ids": [int(token_id) for token_id in response_ids],
                "response_truncated": response_truncated,
                "generated_token_ids": generated_ids,
                "generated_text": str(cache_row.get("perinucleus_generated_text", ""))[:120],
                "token_exact": token_exact,
            }
        )
    exact_ratio = exact_count / len(response_rows) if response_rows else 0.0
    return {
        "claim_accept": exact_ratio >= 1.0,
        "ownership_score": exact_ratio,
        "exact_response_match_ratio": exact_ratio,
        "exact_response_match_count": exact_count,
        "query_budget": budget,
        "model_name": context["base_model"],
        "claim_payload": claim_payload,
        "selected_seed": selected_seed,
        "organic_prompt_id": organic_prompt_id,
        "organic_prompt_ids": prompt_ids,
        "fresh_query_count": len(prompt_ids),
        "response_rows": response_rows,
        "decision_rule": "all_selected_fingerprint_responses_match_cached_organic_prompt_window",
    }


def _completed_row_from_cache(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    row: dict[str, Any],
    cache_by_prompt_id: dict[str, dict[str, str]],
    ours_tokenizer: Any,
    ours_model_name: str,
    ours_context: dict[str, Any],
    perinucleus_context: dict[str, Any],
) -> dict[str, Any]:
    method_id = str(row.get("method_id", ""))
    output = _execution_base_row(row)
    if method_id == OURS_METHOD_ID:
        result = _ours_from_cache_result(
            repo_root=repo_root,
            cfg=cfg,
            row=row,
            cache_by_prompt_id=cache_by_prompt_id,
            tokenizer=ours_tokenizer,
            model_name=ours_model_name,
            context=ours_context,
        )
        completed = _set_fresh_completed(
            output,
            claim_accept=bool(result["claim_accept"]),
            ownership_score=float(result["ownership_score"]),
            status=FRESH_ORGANIC_NULL_STATUS,
            execution_backend="fresh_organic_prompt_null_from_prompt_cache_v1",
            execution_scope="ours_base_qwen_organic_prompt_cache_null",
            source_status="fresh_base_qwen_organic_prompt_cache_completed",
            notes_suffix="fresh base-Qwen organic prompt null from prompt cache",
            details=result,
        )
    elif method_id == PERINUCLEUS_METHOD_ID:
        result = _perinucleus_from_cache_result(
            cfg=cfg,
            row=row,
            cache_by_prompt_id=cache_by_prompt_id,
            context=perinucleus_context,
        )
        completed = _set_fresh_completed(
            output,
            claim_accept=bool(result["claim_accept"]),
            ownership_score=float(result["ownership_score"]),
            status=FRESH_ORGANIC_NULL_STATUS,
            execution_backend="fresh_organic_prompt_null_from_prompt_cache_v1",
            execution_scope="perinucleus_base_qwen_organic_prompt_cache_null",
            source_status="fresh_base_qwen_organic_prompt_cache_completed",
            notes_suffix="fresh base-Qwen organic prompt null from prompt cache",
            details=result,
        )
    else:
        raise RuntimeError(f"Unsupported organic method_id={method_id}")
    first_prompt = str(row.get("organic_prompt_id", "organic_0000"))
    completed["source_artifact"] = str(cache_by_prompt_id[first_prompt].get("cache_source", ""))
    return completed


def main() -> int:
    args = parse_args()
    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    cfg = _load_yaml(config_path)

    cache_dir = _resolve(repo_root, args.cache_dir)
    row_shard_dir = _resolve(repo_root, args.row_shard_output_dir)
    if cache_dir is None or not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory does not exist: {args.cache_dir}")
    if row_shard_dir is None:
        raise ValueError("--row-shard-output-dir is required.")
    row_shard_dir.mkdir(parents=True, exist_ok=True)

    cache_paths = _cache_paths(cache_dir)
    if args.expected_shard_count is not None and len(cache_paths) != args.expected_shard_count:
        raise RuntimeError(
            f"Expected {args.expected_shard_count} cache shard CSVs, found {len(cache_paths)}."
        )
    if not cache_paths:
        raise FileNotFoundError(f"No organic prompt cache CSVs found in {cache_dir}.")
    cache_by_prompt_id = _read_cache_rows(cache_paths)
    _validate_cache_coverage(cfg, cache_by_prompt_id)

    model_name = str(_null_model_cfg(cfg, "base_qwen").get("model") or "Qwen/Qwen2.5-7B-Instruct")
    tokenizer = _load_tokenizer(cfg, model_name)
    ours_context: dict[str, Any] = {}
    perinucleus_context = _perinucleus_context(repo_root, cfg, tokenizer)

    plan_rows = build_plan_rows(cfg)
    shard_count = args.shard_count or args.expected_shard_count or len(cache_paths)
    if shard_count <= 0:
        raise ValueError(f"--shard-count must be positive; got {shard_count}")
    if args.shard_index is None:
        shard_indices = list(range(shard_count))
    else:
        if args.shard_index < 0 or args.shard_index >= shard_count:
            raise ValueError(
                f"--shard-index must be in [0, {shard_count}); got {args.shard_index}"
            )
        shard_indices = [args.shard_index]

    total_completed_rows = 0
    shard_summaries: list[dict[str, Any]] = []
    for shard_index in shard_indices:
        shard_plan_rows = _required_organic_rows_for_shard(
            plan_rows,
            cfg=cfg,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        completed_rows = [
            _completed_row_from_cache(
                repo_root=repo_root,
                cfg=cfg,
                row=row,
                cache_by_prompt_id=cache_by_prompt_id,
                ours_tokenizer=tokenizer,
                ours_model_name=model_name,
                ours_context=ours_context,
                perinucleus_context=perinucleus_context,
            )
            for row in shard_plan_rows
        ]
        paths = _row_shard_paths(row_shard_dir, shard_index, shard_count)
        summary = _build_execution_summary(
            repo_root,
            cfg,
            config_path,
            shard_plan_rows,
            completed_rows,
        )
        summary["shard_scope_status"] = summary["status"]
        summary["status"] = "completed_shard_subset"
        summary["full_far_complete"] = False
        summary["shard"] = {
            "fresh_null_mode": "organic-prompts",
            "shard_index": shard_index,
            "shard_count": shard_count,
            "selected_case_count": len(shard_plan_rows),
            "completed_case_count": len(completed_rows),
            "remaining_case_count": 0,
            "global_plan_case_count": len(plan_rows),
            "note": "Built from prompt cache; aggregate all shards before interpreting FAR.",
        }
        summary["prompt_cache"] = {
            "cache_dir": str(cache_dir),
            "cache_csv_count": len(cache_paths),
            "cache_prompt_count": len(cache_by_prompt_id),
            "cache_csvs": [str(path) for path in cache_paths],
        }
        _write_csv(paths["table"], completed_rows, force=args.force)
        _write_json(paths["summary"], summary, force=args.force)
        total_completed_rows += len(completed_rows)
        shard_summaries.append(
            {
                "shard_index": shard_index,
                "table": str(paths["table"]),
                "summary": str(paths["summary"]),
                "completed_case_count": len(completed_rows),
            }
        )
        print(
            json.dumps(
                {
                    "event": "organic_row_shard_written",
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "completed_case_count": len(completed_rows),
                    "table": str(paths["table"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "status": "organic_row_shards_built_from_cache",
                "cache_dir": str(cache_dir),
                "row_shard_output_dir": str(row_shard_dir),
                "cache_prompt_count": len(cache_by_prompt_id),
                "row_shard_count": shard_count,
                "built_shard_count": len(shard_indices),
                "completed_case_count": total_completed_rows,
                "shards": shard_summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

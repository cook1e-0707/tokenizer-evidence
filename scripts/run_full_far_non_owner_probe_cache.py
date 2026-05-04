from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.run_full_far_payload_claim_benchmark import (
    _generate_non_owner_probe_text,
    _load_base_model,
    _load_perinucleus_frozen_candidate,
    _load_yaml,
    _non_owner_probe_text,
    _non_owner_shard_bounds,
    _null_model_cfg,
    _perinucleus_non_owner_generated_ids,
    _resolve,
    _write_json_atomic,
)
from src.infrastructure.paths import discover_repo_root


CACHE_FIELDNAMES = [
    "status",
    "non_owner_probe_id",
    "non_owner_probe_index",
    "null_model_id",
    "model_name",
    "ours_prompt",
    "ours_generated_text",
    "ours_generated_token_count",
    "perinucleus_prompt",
    "perinucleus_key_truncated",
    "perinucleus_generated_token_ids_json",
    "perinucleus_generated_text",
    "perinucleus_response_length",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 for full FAR non-owner-probe null: generate base-Qwen outputs "
            "once per non-owner probe and cache them by shard."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-output-dir", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Write cache CSV/summary after this many newly generated probes.",
    )
    return parser.parse_args()


def _write_cache_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _read_cache_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as error:
        raise RuntimeError(f"Cannot resume from possibly corrupted probe cache {path}: {error}") from error
    return {
        str(row["non_owner_probe_id"]): row
        for row in rows
        if str(row.get("status", "")) == "completed_probe_cache"
        and str(row.get("non_owner_probe_id", "")).startswith("non_owner_probe_")
        and str(row.get("perinucleus_generated_token_ids_json", "")) != ""
    }


def _cache_paths(cache_dir: Path, shard_index: int, shard_count: int) -> dict[str, Path]:
    stem = (
        "full_far_payload_claim_non_owner_probe_cache_"
        f"shard_{shard_index:03d}_of_{shard_count:03d}"
    )
    return {
        "table": cache_dir / f"{stem}.csv",
        "summary": cache_dir / f"{stem}.json",
    }


def _summary(
    *,
    config_path: Path,
    cache_path: Path,
    shard_index: int,
    shard_count: int,
    selected_probe_ids: list[str],
    completed_by_probe_id: dict[str, dict[str, Any]],
    event: str,
) -> dict[str, Any]:
    return {
        "schema_name": "full_far_non_owner_probe_cache_shard_summary",
        "schema_version": 1,
        "status": (
            "completed_probe_cache_shard"
            if len(completed_by_probe_id) == len(selected_probe_ids)
            else "running_probe_cache_shard"
        ),
        "checkpoint_event": event,
        "config_path": str(config_path),
        "cache_csv": str(cache_path),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_probe_count": len(selected_probe_ids),
        "completed_probe_count": len(completed_by_probe_id),
        "remaining_probe_count": len(selected_probe_ids) - len(completed_by_probe_id),
        "note": (
            "Stage 1 cache only. Run build_full_far_non_owner_from_cache.py after "
            "all cache shards complete to produce non-owner FAR row shards."
        ),
    }


def main() -> int:
    args = parse_args()
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard_count.")

    repo_root = discover_repo_root(Path(__file__).parent)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    cfg = _load_yaml(config_path)
    cache_dir = _resolve(repo_root, args.cache_output_dir)
    if cache_dir is None:
        raise ValueError("--cache-output-dir is required.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(cache_dir, args.shard_index, args.shard_count)

    start, end = _non_owner_shard_bounds(cfg, args.shard_index, args.shard_count)
    selected_probe_ids = [f"non_owner_probe_{index:04d}" for index in range(start, end)]
    selected_probe_set = set(selected_probe_ids)
    completed_by_probe_id = _read_cache_rows(paths["table"])
    completed_by_probe_id = {
        probe_id: row
        for probe_id, row in completed_by_probe_id.items()
        if probe_id in selected_probe_set
    }

    def checkpoint(event: str) -> None:
        ordered_rows = [
            completed_by_probe_id[probe_id]
            for probe_id in selected_probe_ids
            if probe_id in completed_by_probe_id
        ]
        _write_cache_csv_atomic(paths["table"], ordered_rows)
        _write_json_atomic(
            paths["summary"],
            _summary(
                config_path=config_path,
                cache_path=paths["table"],
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                selected_probe_ids=selected_probe_ids,
                completed_by_probe_id=completed_by_probe_id,
                event=event,
            ),
        )
        print(
            json.dumps(
                {
                    "event": event,
                    "cache_csv": str(paths["table"]),
                    "completed": len(completed_by_probe_id),
                    "remaining": len(selected_probe_ids) - len(completed_by_probe_id),
                    "selected": len(selected_probe_ids),
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    checkpoint("checkpoint_initial")
    context: dict[str, Any] = {}
    frozen = _load_perinucleus_frozen_candidate(repo_root, cfg)
    candidate = dict(frozen["candidate"])
    model_name = str(_null_model_cfg(cfg, "base_qwen").get("model") or frozen["model"]["base"])
    model, tokenizer, device, torch = _load_base_model(
        cfg=cfg,
        model_name=model_name,
        context=context,
    )

    newly_completed = 0
    interval = max(1, int(args.checkpoint_interval or 1))
    for probe_id in selected_probe_ids:
        if probe_id in completed_by_probe_id:
            continue
        generation = _generate_non_owner_probe_text(
            cfg=cfg,
            null_model_id="base_qwen",
            non_owner_probe_id=probe_id,
            context=context,
        )
        perinucleus_generation = _perinucleus_non_owner_generated_ids(
            cfg=cfg,
            tokenizer=tokenizer,
            model=model,
            device=device,
            torch=torch,
            non_owner_probe_id=probe_id,
            candidate=candidate,
            context=context,
        )
        completed_by_probe_id[probe_id] = {
            "status": "completed_probe_cache",
            "non_owner_probe_id": probe_id,
            "non_owner_probe_index": int(probe_id.rsplit("_", 1)[1]),
            "null_model_id": "base_qwen",
            "model_name": model_name,
            "ours_prompt": str(
                generation.get("non_owner_probe_prompt") or _non_owner_probe_text(probe_id)
            ),
            "ours_generated_text": str(generation.get("generated_text", "")),
            "ours_generated_token_count": int(generation.get("generated_token_count") or 0),
            "perinucleus_prompt": str(perinucleus_generation.get("non_owner_probe_prompt", "")),
            "perinucleus_key_truncated": bool(perinucleus_generation.get("key_truncated")),
            "perinucleus_generated_token_ids_json": json.dumps(
                perinucleus_generation.get("generated_token_ids", []),
                separators=(",", ":"),
            ),
            "perinucleus_generated_text": str(perinucleus_generation.get("generated_text", "")),
            "perinucleus_response_length": int(candidate.get("response_length", 1)),
        }
        newly_completed += 1
        if newly_completed % interval == 0:
            checkpoint("checkpoint_progress")
    checkpoint("checkpoint_final")
    print(
        json.dumps(
            {
                "status": "cache_shard_completed",
                "cache_csv": str(paths["table"]),
                "summary": str(paths["summary"]),
                "selected_probe_count": len(selected_probe_ids),
                "completed_probe_count": len(completed_by_probe_id),
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

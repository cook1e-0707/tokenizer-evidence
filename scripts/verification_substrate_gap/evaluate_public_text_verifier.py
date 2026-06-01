#!/usr/bin/env python3
"""Evaluate public text-predicate baselines without trace/key access.

These baselines are deliberately scoped as observability and spoofing targets.
They do not recover first-divergence codewords. At inference time, variants use
only final text and public predicate state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


RESULT_FIELDS = [
    "source_id",
    "model_family",
    "variant_id",
    "regime_id",
    "status",
    "train_rows",
    "test_rows",
    "auc",
    "threshold",
    "protected_row_tpr",
    "raw_row_fpr",
    "task_only_row_fpr",
    "protected_detected_blocks",
    "raw_detected_blocks",
    "task_only_detected_blocks",
    "codeword_recovered_blocks",
    "codeword_recovery_supported",
    "spoofing_target",
    "claim_scope",
]

BLOCK_FIELDS = [
    "source_id",
    "model_family",
    "variant_id",
    "arm",
    "block_id",
    "rows",
    "mean_score",
    "row_accept_rate",
    "block_detected",
]


TOKEN_RE = re.compile(r"[a-z0-9']+")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _opener(text: str) -> str:
    normalized = _normalize_space(text)
    if not normalized:
        return ""
    return re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0][:96]


def _word_features(text: str, *, max_tokens: int = 128, max_order: int = 2) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())[:max_tokens]
    feats = [f"u:{token}" for token in tokens]
    if max_order >= 2:
        feats.extend(f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:]))
    if max_order >= 3:
        feats.extend(f"t:{a}_{b}_{c}" for a, b, c in zip(tokens, tokens[1:], tokens[2:]))
    return feats


def _char_features(text: str, *, max_chars: int = 384, orders: tuple[int, ...] = (3, 4, 5)) -> list[str]:
    normalized = f" {_normalize_space(text.lower())[:max_chars]} "
    feats: list[str] = []
    for order in orders:
        if len(normalized) < order:
            continue
        feats.extend(f"c{order}:{normalized[idx:idx + order]}" for idx in range(len(normalized) - order + 1))
    return feats


def _features(text: str) -> list[str]:
    return _word_features(text, max_order=2)


def _features_for_variant(text: str, variant_id: str) -> list[str]:
    if variant_id == "P4_char_ngram_public_predicate":
        return _char_features(text)
    if variant_id == "P5_word_trigram_public_predicate":
        return _word_features(text, max_order=3)
    if variant_id == "P6_hybrid_char_word_public_predicate":
        return _word_features(text, max_order=2) + _char_features(text, max_chars=256, orders=(3, 4))
    return _features(text)


def _source_files(source: dict[str, Any], combined_blocks: list[dict[str, str]]) -> list[Path]:
    corpus_model = source.get("source_shard_dirs_from_corpus_model")
    if corpus_model:
        dirs = sorted(
            {
                row.get("source_shard_dir", "")
                for row in combined_blocks
                if row.get("model_family") == corpus_model and row.get("source_shard_dir")
            }
        )
        return [
            Path(directory) / "r4_generated_outputs.jsonl"
            for directory in dirs
            if (Path(directory) / "r4_generated_outputs.jsonl").exists()
        ]
    root = Path(source["path"])
    return sorted(root.glob("shard_*/r4_generated_outputs.jsonl")) if root.exists() else []


def _iter_rows(files: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in sorted(files):
        shard = path.parent.name
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL row: {path}:{line_no}: {exc}") from exc
                row["_source_file"] = str(path)
                row["_block_id"] = shard
                yield row


def _split_name(row: dict[str, Any]) -> str:
    key = str(row.get("prompt_hash") or row.get("prompt_id") or row.get("generation_id") or row.get("response_text_sha256") or "")
    return "test" if _stable_int(key) % 5 == 0 else "train"


def _auc(scored: list[tuple[float, int]]) -> float | None:
    positives = sum(label for _, label in scored)
    negatives = len(scored) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(scored, key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(ranked):
        j = idx + 1
        while j < len(ranked) and ranked[j][0] == ranked[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        rank_sum += sum(label for _, label in ranked[idx:j]) * avg_rank
        idx = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _threshold_at_raw_fpr(scored_train: list[tuple[float, str]], raw_fpr_target: float) -> float:
    raw_scores = sorted([score for score, label in scored_train if label == "raw"], reverse=True)
    if not raw_scores:
        return math.inf
    index = min(max(int(len(raw_scores) * raw_fpr_target), 0), len(raw_scores) - 1)
    return raw_scores[index]


def _train_nb(
    train_rows: list[tuple[str, str]],
    *,
    variant_id: str = "P2_learned_shallow_public_predicate",
) -> tuple[dict[str, Counter[str]], dict[str, int], set[str]]:
    counts = {"protected": Counter(), "raw": Counter()}
    totals = {"protected": 0, "raw": 0}
    vocab: set[str] = set()
    for text, label in train_rows:
        feats = _features_for_variant(text, variant_id)
        counts[label].update(feats)
        totals[label] += len(feats)
        vocab.update(feats)
    return counts, totals, vocab


def _score_nb(
    text: str,
    counts: dict[str, Counter[str]],
    totals: dict[str, int],
    vocab: set[str],
    *,
    variant_id: str = "P2_learned_shallow_public_predicate",
) -> float:
    alpha = 1.0
    vocab_size = max(len(vocab), 1)
    score = 0.0
    for feat in _features_for_variant(text, variant_id):
        p_prot = (counts["protected"][feat] + alpha) / (totals["protected"] + alpha * vocab_size)
        p_raw = (counts["raw"][feat] + alpha) / (totals["raw"] + alpha * vocab_size)
        score += math.log(p_prot) - math.log(p_raw)
    return score


def _metric_rows(
    *,
    source_id: str,
    model_family: str,
    variant_id: str,
    regime_id: str,
    train_rows: int,
    scored_train: list[tuple[float, str]],
    scored_test: list[tuple[float, str, str]],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_label: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for score, label, block_id in scored_test:
        by_label[label].append((score, block_id))
    protected = by_label.get("protected", [])
    raw = by_label.get("raw", [])
    task_only = by_label.get("task_only", [])
    auc_value = _auc([(score, int(label == "protected")) for score, label, _ in scored_test if label in {"protected", "raw"}])

    def rate(rows: list[tuple[float, str]]) -> float:
        return 0.0 if not rows else sum(score >= threshold for score, _ in rows) / len(rows)

    block_rows: list[dict[str, Any]] = []
    detected_counts: Counter[str] = Counter()
    block_buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for score, label, block_id in scored_test:
        if label not in {"protected", "raw", "task_only"}:
            continue
        block_buckets[(label, block_id)].append(score)
    for (label, block_id), scores in sorted(block_buckets.items()):
        accept_rate = sum(score >= threshold for score in scores) / len(scores)
        detected = accept_rate >= 0.5
        detected_counts[label] += int(detected)
        block_rows.append(
            {
                "source_id": source_id,
                "model_family": model_family,
                "variant_id": variant_id,
                "arm": label,
                "block_id": block_id,
                "rows": len(scores),
                "mean_score": sum(scores) / len(scores),
                "row_accept_rate": accept_rate,
                "block_detected": detected,
            }
        )
    spoofing_target = (auc_value is not None and auc_value > 0.60) or rate(protected) > 0.05
    result = {
        "source_id": source_id,
        "model_family": model_family,
        "variant_id": variant_id,
        "regime_id": regime_id,
        "status": "TEXT_PREDICATE_EVALUATED_NOT_CODEWORD_VERIFIER",
        "train_rows": train_rows,
        "test_rows": len(scored_test),
        "auc": "" if auc_value is None else round(auc_value, 6),
        "threshold": threshold,
        "protected_row_tpr": rate(protected),
        "raw_row_fpr": rate(raw),
        "task_only_row_fpr": "" if not task_only else rate(task_only),
        "protected_detected_blocks": detected_counts.get("protected", 0),
        "raw_detected_blocks": detected_counts.get("raw", 0),
        "task_only_detected_blocks": detected_counts.get("task_only", 0),
        "codeword_recovered_blocks": 0,
        "codeword_recovery_supported": False,
        "spoofing_target": spoofing_target,
        "claim_scope": "public_text_predicate_observability_only_not_codeword_recovery",
    }
    return result, block_rows


def _collect_source_rows(
    files: list[Path],
    *,
    train_cap_per_label: int,
    test_cap_per_label: int,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]], list[tuple[str, str, str]]]:
    all_public_rows: list[dict[str, Any]] = []
    train_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str, str]] = []
    caps = {
        ("train", "protected"): train_cap_per_label,
        ("train", "raw"): train_cap_per_label,
        ("test", "protected"): test_cap_per_label,
        ("test", "raw"): test_cap_per_label,
        ("train", "task_only"): train_cap_per_label,
        ("test", "task_only"): test_cap_per_label,
    }
    counts: Counter[tuple[str, str]] = Counter()
    for row in _iter_rows(files):
        label = str(row.get("arm", row.get("generation_condition", "")) or "")
        text = str(row.get("response_text", row.get("output_text", "")) or "")
        if not text or label not in {"protected", "raw", "task_only"}:
            continue
        split = _split_name(row)
        key = (split, label)
        if counts[key] >= caps[key]:
            continue
        counts[key] += 1
        public_row = {
            "text": text,
            "label": label,
            "split": split,
            "block_id": str(row.get("_block_id", "")),
            "opener": _opener(text),
        }
        all_public_rows.append(public_row)
        if label in {"protected", "raw"} and split == "train":
            train_rows.append((text, label))
        elif split == "test":
            test_rows.append((text, label, public_row["block_id"]))
    return all_public_rows, train_rows, test_rows


def evaluate(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _read_yaml(config_path)
    observability = _read_yaml(Path(config["observability_config"]))
    corpus_cfg = observability["trace_bound_corpus"]
    combined_blocks = _read_csv(Path(corpus_cfg["combined_blocks"]))

    variants = {variant["variant_id"]: variant for variant in config["variants"]}
    default_variant = variants["P2_learned_shallow_public_predicate"]
    train_cap = int(config.get("train_cap_per_label", default_variant.get("train_cap_per_label", 40000)))
    test_cap = int(config.get("test_cap_per_label", default_variant.get("test_cap_per_label", 20000)))
    raw_fpr_target = float(default_variant.get("raw_fpr_target", 0.01))
    source_scopes = set(config.get("source_scopes", ["adopted_locked"]))

    result_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []

    for source in observability.get("text_sources", []):
        if source.get("scope") not in source_scopes:
            continue
        files = _source_files(source, combined_blocks)
        if not files:
            continue
        source_id = str(source["source_id"])
        model_family = str(source["model_family"])
        public_rows, train_rows, test_rows = _collect_source_rows(
            files,
            train_cap_per_label=train_cap,
            test_cap_per_label=test_cap,
        )

        # V0: no public predicate; always rejects.
        zero_scored = [(0.0, label, block_id) for _, label, block_id in test_rows]
        result, blocks = _metric_rows(
            source_id=source_id,
            model_family=model_family,
            variant_id="V0_always_reject_final_text_only",
            regime_id="V0",
            train_rows=0,
            scored_train=[],
            scored_test=zero_scored,
            threshold=1.0,
        )
        result["status"] = "ALWAYS_REJECT_BASELINE_NOT_CODEWORD_VERIFIER"
        result_rows.append(result)
        block_rows.extend(blocks)

        # P0: public opener/template predicate built from protected-enriched train openers.
        opener_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for text, label in train_rows:
            opener_counts[label].update([_opener(text)])
        protected_openers = opener_counts["protected"]
        raw_openers = opener_counts["raw"]
        enriched = []
        for opener, protected_count in protected_openers.items():
            raw_count = raw_openers.get(opener, 0)
            if protected_count >= 2 and protected_count > raw_count:
                enriched.append((protected_count - raw_count, protected_count, opener))
        enriched.sort(reverse=True)
        top_k = int(variants["P0_prefix_template_public_predicate"].get("top_k_openers", 256))
        opener_whitelist = {opener for _, _, opener in enriched[:top_k]}
        opener_scored_train = [(1.0 if _opener(text) in opener_whitelist else 0.0, label) for text, label in train_rows]
        opener_scored_test = [
            (1.0 if _opener(text) in opener_whitelist else 0.0, label, block_id)
            for text, label, block_id in test_rows
        ]
        result, blocks = _metric_rows(
            source_id=source_id,
            model_family=model_family,
            variant_id="P0_prefix_template_public_predicate",
            regime_id="V4",
            train_rows=len(train_rows),
            scored_train=opener_scored_train,
            scored_test=opener_scored_test,
            threshold=1.0,
        )
        result_rows.append(result)
        block_rows.extend(blocks)

        # Learned public predicates over final text. Each variant uses only the
        # final response text at inference time; labels are used only to fit the
        # public surrogate on reviewed artifacts.
        learned_variants = [
            "P2_learned_shallow_public_predicate",
            "P4_char_ngram_public_predicate",
            "P5_word_trigram_public_predicate",
            "P6_hybrid_char_word_public_predicate",
        ]
        for variant_id in learned_variants:
            if variant_id not in variants:
                continue
            variant_raw_fpr = float(variants[variant_id].get("raw_fpr_target", raw_fpr_target))
            counts, totals, vocab = _train_nb(train_rows, variant_id=variant_id)
            nb_scored_train = [
                (_score_nb(text, counts, totals, vocab, variant_id=variant_id), label)
                for text, label in train_rows
            ]
            threshold = _threshold_at_raw_fpr(nb_scored_train, variant_raw_fpr)
            nb_scored_test = [
                (_score_nb(text, counts, totals, vocab, variant_id=variant_id), label, block_id)
                for text, label, block_id in test_rows
            ]
            result, blocks = _metric_rows(
                source_id=source_id,
                model_family=model_family,
                variant_id=variant_id,
                regime_id="V4",
                train_rows=len(train_rows),
                scored_train=nb_scored_train,
                scored_test=nb_scored_test,
                threshold=threshold,
            )
            result_rows.append(result)
            block_rows.extend(blocks)

        # P3: strongest observed text predicate among P0/P2 by protected TPR at the same run.
        candidates = [
            row
            for row in result_rows
            if row["source_id"] == source_id
            and row["variant_id"] in {
                "P0_prefix_template_public_predicate",
                "P2_learned_shallow_public_predicate",
                "P4_char_ngram_public_predicate",
                "P5_word_trigram_public_predicate",
                "P6_hybrid_char_word_public_predicate",
            }
        ]
        strongest = max(candidates, key=lambda row: float(row["protected_row_tpr"]))
        p3 = dict(strongest)
        p3["variant_id"] = "P3_strongest_text_only_public_predicate"
        p3["claim_scope"] = "selected_public_text_predicate_observability_only_not_codeword_recovery"
        result_rows.append(p3)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "public_text_verifier_results.csv", result_rows, RESULT_FIELDS)
    _write_csv(output_dir / "public_text_verifier_block_scores.csv", block_rows, BLOCK_FIELDS)
    failures = [
        row for row in result_rows
        if row["variant_id"] != "V0_always_reject_final_text_only" and bool(row["spoofing_target"])
    ]
    summary = {
        "status": "PUBLIC_TEXT_PREDICATE_BASELINES_RECORDED_SPOOFING_TARGETS_FOUND" if failures else "PUBLIC_TEXT_PREDICATE_BASELINES_RECORDED_NO_STRONG_TEXT_SIGNAL",
        "schema_name": "verification_substrate_gap_public_text_verifier_baselines_v1",
        "config": str(config_path),
        "config_claim_scope": config.get("claim_scope", ""),
        "source_scopes": sorted(source_scopes),
        "result_csv": str(output_dir / "public_text_verifier_results.csv"),
        "block_scores_csv": str(output_dir / "public_text_verifier_block_scores.csv"),
        "spoofing_target_variants": failures,
        "codeword_recovered_blocks_total": 0,
        "claim_scope": "public text predicates are observability/spoofing targets only; not public text-only codeword verification",
        "next_allowed_action": "Use strongest public text predicates as spoofing targets; do not claim public text-only verification.",
    }
    _write_json(output_dir / "public_text_verifier_summary.json", summary)
    _write_report(output_dir / "public_text_verifier_report.md", summary, result_rows)
    return summary


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Public Text Verifier Baselines",
        "",
        "These are public final-text predicates evaluated as observability and",
        "spoofing targets. They do not use trace/key fields and do not recover",
        "first-divergence codewords.",
        "",
        f"Status: `{summary['status']}`",
        f"Config claim scope: `{summary.get('config_claim_scope', '')}`",
        f"Source scopes: `{', '.join(summary.get('source_scopes', []))}`",
        "",
        "| Source | Model | Variant | AUC | Protected row TPR | Raw row FPR | Codeword blocks | Spoofing target |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['source_id']} | {row['model_family']} | {row['variant_id']} | {row['auc']} | "
            f"{row['protected_row_tpr']} | {row['raw_row_fpr']} | {row['codeword_recovered_blocks']} | "
            f"{row['spoofing_target']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Any variant with nontrivial protected/raw separation is a public predicate",
            "that must be attacked in spoofing experiments. None of these rows support",
            "a public text-only codeword verification claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/verification_substrate_gap/public_text_verifier_baselines.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/verification_substrate_gap/public_text_verifier_remote_20260529"),
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

d(input_case_counts_by_bank.items())
+        },
+        "observed_surface_counts_by_bank": {
+            bank_id: dict(sorted(counter.items()))
+            for bank_id, counter in sorted(input_surface_counts_by_bank.items())
+        },
+    }
+    return plan_rows, audit
+
+
+def summarize_plan(
+    *,
+    plan_id: str,
+    candidate_jsonl: Path,
+    responses_jsonl: Path,
+    bucket_bank: Path,
+    output_dir: Path,
+    candidate_rows: Sequence[Mapping[str, Any]],
+    response_rows: Sequence[Mapping[str, Any]],
+    plan_rows: Sequence[Mapping[str, Any]],
+    audit: Mapping[str, Any],
+) -> dict[str, Any]:
+    rows_by_bank = Counter(str(row["candidate_bank_id"]) for row in plan_rows)
+    rows_by_variant = Counter(str(row["casing_variant"]) for row in plan_rows)
+    rows_by_bank_variant = Counter(
+        f"{row['candidate_bank_id']}::{row['casing_variant']}" for row in plan_rows
+    )
+    family_counts = Counter(str(row.get("family_id", "")) for row in candidate_rows)
+    template_preflight_only = bool(response_rows) and all(
+        str(row.get("response_source", row.get("artifact_role", ""))).startswith(
+            "template_density_preflight"
+        )
+        or str(row.get("artifact_role", "")).startswith("template_density_preflight")
+        for row in response_rows
+    )
+    return {
+        "schema_name": "natural_evidence_v2_wp3_context_mass_score_plan_summary_v1",
+        "status": "WP3_CONTEXT_SPECIFIC_MASS_SCORE_PLAN_WRITTEN_NOT_SCORED",
+        "plan_id": plan_id,
+        "candidate_jsonl": str(candidate_jsonl),
+        "responses_jsonl": str(responses_jsonl),
+        "bucket_bank": str(bucket_bank),
+        "score_plan_jsonl": str(output_dir / "qwen_v2_wp3_context_mass_score_plan.jsonl"),
+        "candidate_input_rows": len(candidate_rows),
+        "response_input_rows": len(response_rows),
+        "template_preflight_only": template_preflight_only,
+        "source_family_counts": dict(sorted(family_counts.items())),
+        "eligible_detection_rows": int(audit.get("eligible_detection_rows", 0)),
+        "score_plan_rows": len(plan_rows),
+        "score_plan_rows_by_bank": dict(sorted(rows_by_bank.items())),
+        "score_plan_rows_by_casing_variant": dict(sorted(rows_by_variant.items())),
+        "score_plan_rows_by_bank_and_casing_variant": dict(sorted(rows_by_bank_variant.items())),
+        "observed_case_counts_by_bank": audit.get("observed_case_counts_by_bank", {}),
+        "observed_surface_counts_by_bank": audit.get("observed_surface_counts_by_bank", {}),
+        "skipped_candidate_rows_by_reason": audit.get("skipped_candidate_rows_by_reason", {}),
+        "casing_variants_audited_separately": list(CASE_VARIANTS),
+        "prefix_source": "response_text_prefix_before_detected_candidate_span",
+        "structural_selection": "current_detector_eligible_template_slot",
+        "artifact_only": True,
+        "model_scoring_started": False,
+        "training_started": False,
+        "generation_started": False,
+        "e2e_eval_started": False,
+        "paper_claim_allowed": False,
+        "not_payload_recovery": True,
+        "not_full_far": True,
+        "wp4_allowed": False,
+        "next_allowed_action": (
+            "Review the context-specific mass scoring plan; any model scoring "
+            "must use a plan-consuming Chimera Slurm job, not local Chimera "
+            "login-node execution."
+        ),
+    }
+
+
+def readme_text(summary: Mapping[str, Any]) -> str:
+    return "\n".join(
+        [
+            "# WP3 Context-Specific Mass Score Plan",
+            "",
+            "Artifact-only scoring plan built from balanced template response detections.",
+            "",
+            f"status: `{summary['status']}`",
+            f"score_plan_rows: `{summary['score_plan_rows']}`",
+            "",
+            "Rows contain prefix_before_candidate at detected template slots.",
+            "Lowercase and sentence-case bucket variants are recorded separately.",
+            "No model scoring, training, generation, E2E, FAR aggregation, or positive claim was started.",
+            "",
+        ]
+    )
+
+
+def main() -> None:
+    args = parse_args()
+    candidate_jsonl = resolve_path(args.candidate_jsonl)
+    responses_jsonl = resolve_path(args.responses_jsonl)
+    bucket_bank_path = resolve_path(args.bucket_bank)
+    output_dir = resolve_path(args.output_dir)
+    if output_dir.exists() and any(output_dir.iterdir()):
+        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
+
+    candidate_rows = read_jsonl(candidate_jsonl)
+    response_rows = read_jsonl(responses_jsonl)
+    bucket_bank = read_json(bucket_bank_path)
+    plan_id = output_dir.name
+    plan_rows, audit = build_plan_rows(
+        plan_id=plan_id,
+        candidate_rows=candidate_rows,
+        responses_by_id=response_index(response_rows),
+        banks_by_id=bucket_bank_by_id(bucket_bank),
+    )
+    summary = summarize_plan(
+        plan_id=plan_id,
+        candidate_jsonl=candidate_jsonl,
+        responses_jsonl=responses_jsonl,
+        bucket_bank=bucket_bank_path,
+        output_dir=output_dir,
+        candidate_rows=candidate_rows,
+        response_rows=response_rows,
+        plan_rows=plan_rows,
+        audit=audit,
+    )
+
+    output_dir.mkdir(parents=True, exist_ok=True)
+    write_jsonl(output_dir / "qwen_v2_wp3_context_mass_score_plan.jsonl", plan_rows)
+    write_json(output_dir / "qwen_v2_wp3_context_mass_score_plan_summary.json", summary)
+    write_text_new(output_dir / "README.md", readme_text(summary))
+    print(
+        json.dumps(
+            {
+                "status": summary["status"],
+                "score_plan_rows": summary["score_plan_rows"],
+                "casing_variants": summary["casing_variants_audited_separately"],
+            },
+            sort_keys=True,
+        )
+    )
+
+
+if __name__ == "__main__":
+    main()

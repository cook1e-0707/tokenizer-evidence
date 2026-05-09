from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = Path("/Users/guanjie/Documents/llm_api/llm-streaming-semantics-audit/.env")
DEFAULT_BUCKET_BANK = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "wp3_detector_bank_scaffold_repaired_20260508_2308/two_way_bucket_bank_scaffold.json"
)
DEFAULT_CONTEXT_PLAN_SUMMARY = (
    ROOT
    / "results/natural_evidence_v2/status/"
    "wp3_context_mass_plan_20260508_2324/qwen_v2_wp3_context_mass_score_plan_summary.json"
)
DEFAULT_MASS_AUDIT = ROOT / "results/natural_evidence_v2/status/wp3_model_mass_audit_850288/mass_audit.json"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODELS = ("qwen/qwen3.5-397b-a17b", "z-ai/glm-5.1")
FORBIDDEN_SURFACES = (
    "FIELD=",
    "SECTION=",
    "TOPIC=",
    "PAYLOAD",
    "CERT",
    "EVIDENCE",
    "CARRIER",
    "OWNER",
    "fingerprint",
    "watermark",
    "bucket",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use NVIDIA-hosted stronger LLMs as design assistants for "
            "natural_evidence_v2 WP3 bucket/context repair. Outputs are "
            "proposal artifacts only, not Qwen gates, not training, not "
            "generation of protected transcripts, not E2E, and not claims."
        )
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--bucket-bank", type=Path, default=DEFAULT_BUCKET_BANK)
    parser.add_argument("--context-plan-summary", type=Path, default=DEFAULT_CONTEXT_PLAN_SUMMARY)
    parser.add_argument("--mass-audit", type=Path, default=DEFAULT_MASS_AUDIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--sleep-between-calls", type=float, default=2.0)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_env(path: Path) -> list[str]:
    loaded: list[str] = []
    if not path.exists():
        raise FileNotFoundError(f"env file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
    return payload


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_new(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def api_key_for_index(index: int) -> tuple[str, str]:
    key_envs = ("NVIDIA_API_KEY_A", "NVIDIA_API_KEY_B")
    key_env = key_envs[index % len(key_envs)]
    key = os.environ.get(key_env)
    if not key:
        raise RuntimeError(f"{key_env} is required but not printed")
    return key_env, key


def compact_bank_payload(bucket_bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for bank in bucket_bank.get("candidate_banks", []):
        output.append(
            {
                "candidate_bank_id": bank.get("candidate_bank_id"),
                "slot_type": bank.get("slot_type"),
                "anchor_kind": bank.get("anchor_kind"),
                "buckets": bank.get("buckets"),
            }
        )
    return output


def compact_mass_payload(mass_audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in mass_audit.get("bank_results", []):
        output.append(
            {
                "candidate_bank_id": row.get("candidate_bank_id"),
                "passed": row.get("passed"),
                "bucket_masses": row.get("bucket_masses"),
                "min_bucket_mass": row.get("min_bucket_mass"),
                "max_bucket_mass_ratio": row.get("max_bucket_mass_ratio"),
            }
        )
    return output


def build_prompt(
    *,
    bucket_bank: Mapping[str, Any],
    context_plan_summary: Mapping[str, Any],
    mass_audit: Mapping[str, Any],
    model: str,
) -> list[dict[str, str]]:
    payload = {
        "task": "natural_evidence_v2_controlled_micro_slots_wp3_bucket_context_repair_design",
        "strict_role": "design_assistant_only_not_gate_scorer",
        "target_base_model_for_final_gate": "Qwen/Qwen2.5-7B-Instruct",
        "assistant_model": model,
        "problem": (
            "Tokenizer stability passed and template detector density is high, "
            "but base Qwen fixed-prefix full-vocab bucket mass failed for all "
            "current banks. We need natural micro-slot bucket/context repair "
            "ideas that will later be tested by base Qwen logits."
        ),
        "forbidden_public_surfaces": list(FORBIDDEN_SURFACES),
        "current_banks": compact_bank_payload(bucket_bank),
        "mass_failure": {
            "gate": "min full-vocab bucket mass >= 0.005 and mass ratio <= 5",
            "bank_results": compact_mass_payload(mass_audit),
        },
        "context_plan_summary": {
            "score_plan_rows": context_plan_summary.get("score_plan_rows"),
            "source_family_counts": context_plan_summary.get("source_family_counts"),
            "observed_case_counts_by_bank": context_plan_summary.get("observed_case_counts_by_bank"),
            "observed_surface_counts_by_bank": context_plan_summary.get("observed_surface_counts_by_bank"),
            "score_plan_rows_by_bank": context_plan_summary.get("score_plan_rows_by_bank"),
            "casing_variants_audited_separately": context_plan_summary.get(
                "casing_variants_audited_separately"
            ),
        },
        "requirements": [
            "Return valid JSON only; no markdown.",
            "Keep the response compact enough to fit under 700 output tokens.",
            "Return exactly 3 repair proposals total.",
            "For each proposal, include at most 2 surfaces per side and at most 2 prefix shapes.",
            "Do not include any forbidden term in proposed public prompt text, slot surfaces, or prefix shapes.",
            "Keep primary construction 2-way only.",
            "Do not propose FIELD=value, explicit evidence blocks, payload labels, watermarks, or owner labels.",
            "Prefer ordinary natural English micro-slots: sentence/step openers, transitions, hedges, function words.",
            "For each proposal, explain in one sentence why base Qwen next-token full-vocab mass may improve.",
            "Separate lowercase and sentence-case if casing matters.",
            "Mark risky proposals that may harm naturalness or detector precision.",
            "These are suggestions only; final acceptance requires Qwen tokenizer and base-Qwen mass audit.",
        ],
        "output_schema": {
            "model": "string",
            "summary": "one sentence",
            "repair_proposals": [
                {
                    "source_bank_id": "string",
                    "new_bank_id": "string",
                    "decision": "replace|split|drop|context_repair",
                    "slot_type": "string",
                    "anchor_kind": "string",
                    "case_variant": "lowercase|sentence_case|both",
                    "side0": ["at most 2 strings"],
                    "side1": ["at most 2 strings"],
                    "prefix_shapes": ["at most 2 strings"],
                    "why_qwen_mass_may_improve": "one sentence",
                    "risk": "one short string",
                }
            ],
            "prompt_family_repairs": [
                {
                    "family_id": "string",
                    "repair": "one sentence",
                }
            ],
            "qwen_validation_plan": [
                "Run tokenizer stability and base Qwen full-vocab mass scoring through Chimera Slurm."
            ],
            "forbidden_claims": ["not a gate", "not training", "not evidence of recovery"],
            "do_not_return_extra_keys": True,
        },
        "exact_json_shape": {
            "model": model,
            "summary": "",
            "repair_proposals": [
                {
                    "source_bank_id": "",
                    "new_bank_id": "",
                    "decision": "",
                    "slot_type": "",
                    "anchor_kind": "",
                    "case_variant": "",
                    "side0": [],
                    "side1": [],
                    "prefix_shapes": [],
                    "why_qwen_mass_may_improve": "",
                    "risk": "",
                }
            ],
            "prompt_family_repairs": [{"family_id": "", "repair": ""}],
            "qwen_validation_plan": [],
            "forbidden_claims": [],
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a precise research design assistant. Return only valid JSON. "
                "You are not evaluating ownership and not making claims."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any]]:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_seconds))) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"NVIDIA HTTP {error.code}: {detail}") from error
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", raw
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else ""
    return str(content or ""), raw


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, str]:
    stripped = text.strip()
    if not stripped:
        return None, "empty_response"
    candidates = [stripped]
    match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        candidates.insert(0, match.group(1).strip())
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload, "parsed"
        except json.JSONDecodeError:
            continue
    return None, "json_parse_failed"


def iter_public_surface_texts(payload: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    compact_proposals = payload.get("repair_proposals", [])
    if isinstance(compact_proposals, list):
        for proposal in compact_proposals:
            if not isinstance(proposal, dict):
                continue
            for key in ("side0", "side1", "prefix_shapes"):
                values = proposal.get(key, [])
                if isinstance(values, list):
                    texts.extend(str(value) for value in values)
    proposals = payload.get("bank_repair_proposals", [])
    if isinstance(proposals, list):
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            proposed_banks = proposal.get("proposed_banks", [])
            if not isinstance(proposed_banks, list):
                continue
            for bank in proposed_banks:
                if not isinstance(bank, dict):
                    continue
                for key in ("bucket_0_surfaces", "bucket_1_surfaces", "preferred_prefix_shapes"):
                    values = bank.get(key, [])
                    if isinstance(values, list):
                        texts.extend(str(value) for value in values)
    prompt_repairs = payload.get("prompt_family_repairs", [])
    if isinstance(prompt_repairs, list):
        for row in prompt_repairs:
            if not isinstance(row, dict):
                continue
            for key in ("repair", "expected_micro_slot_gain", "risk"):
                value = row.get(key)
                if isinstance(value, str):
                    texts.append(value)
    detector_repairs = payload.get("detector_repairs", [])
    if isinstance(detector_repairs, list):
        texts.extend(str(value) for value in detector_repairs)
    return texts


def public_surface_forbidden_hits(payload: Mapping[str, Any]) -> list[str]:
    text = "\n".join(iter_public_surface_texts(payload))
    upper = text.upper()
    return [term for term in FORBIDDEN_SURFACES if term.upper() in upper]


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_keys = load_env(args.env_file)
    bucket_bank = read_json(resolve_path(args.bucket_bank))
    context_plan_summary = read_json(resolve_path(args.context_plan_summary))
    mass_audit = read_json(resolve_path(args.mass_audit))

    results: list[dict[str, Any]] = []
    parsed_payloads: list[dict[str, Any]] = []
    for index, model in enumerate(args.models):
        key_env, key = api_key_for_index(index)
        messages = build_prompt(
            bucket_bank=bucket_bank,
            context_plan_summary=context_plan_summary,
            mass_audit=mass_audit,
            model=str(model),
        )
        try:
            text, raw = chat_completion(
                base_url=str(args.base_url),
                api_key=key,
                model=str(model),
                messages=messages,
                max_tokens=max(1, int(args.max_tokens)),
                temperature=float(args.temperature),
                timeout_seconds=float(args.request_timeout_seconds),
            )
            error_message = ""
        except Exception as error:  # noqa: BLE001 - artifact must preserve remote API failures.
            text = ""
            raw = {}
            error_message = f"{type(error).__name__}: {error}"
        parsed, parse_status = parse_json_response(text)
        if error_message:
            parse_status = "api_error"
        forbidden_hits = public_surface_forbidden_hits(parsed) if parsed else []
        result = {
            "schema_name": "natural_evidence_v2_nvidia_design_assistant_result_v1",
            "model": str(model),
            "api_key_env": key_env,
            "base_url": str(args.base_url),
            "parse_status": parse_status,
            "error_message": error_message,
            "forbidden_surface_hits": forbidden_hits,
            "raw_response_chars": len(text),
            "raw_response_text": text,
            "usage": raw.get("usage", {}),
            "artifact_only_design_assistance": True,
            "not_qwen_gate": True,
            "model_scoring_started": False,
            "training_started": False,
            "generation_of_protected_transcripts_started": False,
            "e2e_eval_started": False,
            "paper_claim_allowed": False,
        }
        results.append(result)
        if parsed is not None:
            parsed_payloads.append(
                {
                    "model": str(model),
                    "api_key_env": key_env,
                    "parse_status": parse_status,
                    "forbidden_surface_hits": forbidden_hits,
                    "proposal": parsed,
                }
            )
        if index < len(args.models) - 1:
            time.sleep(max(0.0, float(args.sleep_between_calls)))

    write_json(output_dir / "nvidia_assisted_design_raw_results.json", {"results": results})
    write_json(output_dir / "nvidia_assisted_design_parsed_proposals.json", {"proposals": parsed_payloads})
    summary = {
        "schema_name": "natural_evidence_v2_nvidia_assisted_context_repair_summary_v1",
        "status": "NVIDIA_ASSISTED_CONTEXT_REPAIR_PROPOSALS_WRITTEN_NOT_GATE",
        "models": list(args.models),
        "base_url": str(args.base_url),
        "env_file": str(args.env_file),
        "loaded_key_names": [key for key in loaded_keys if key.startswith("NVIDIA_")],
        "proposal_count": len(parsed_payloads),
        "parse_status_by_model": {row["model"]: row["parse_status"] for row in results},
        "forbidden_surface_hits_by_model": {row["model"]: row["forbidden_surface_hits"] for row in results},
        "input_artifacts": {
            "bucket_bank": str(resolve_path(args.bucket_bank)),
            "context_plan_summary": str(resolve_path(args.context_plan_summary)),
            "mass_audit": str(resolve_path(args.mass_audit)),
        },
        "outputs": {
            "raw_results_json": str(output_dir / "nvidia_assisted_design_raw_results.json"),
            "parsed_proposals_json": str(output_dir / "nvidia_assisted_design_parsed_proposals.json"),
        },
        "rate_limit_policy": "two keys rotated across models; calls are sequential and bounded",
        "artifact_only_design_assistance": True,
        "not_qwen_gate": True,
        "model_scoring_started": False,
        "training_started": False,
        "generation_of_protected_transcripts_started": False,
        "e2e_eval_started": False,
        "paper_claim_allowed": False,
        "next_allowed_action": (
            "Review proposals, manually select candidate repairs, then validate with "
            "Qwen tokenizer and base-Qwen mass scoring through Chimera Slurm."
        ),
    }
    write_json(output_dir / "nvidia_assisted_context_repair_summary.json", summary)
    write_text_new(
        output_dir / "README.md",
        "\n".join(
            [
                "# NVIDIA-Assisted Context Repair Design",
                "",
                "Strong-model design-assistance artifact only.",
                "These proposals are not Qwen gates, not training, not E2E, and not paper claims.",
                "",
            ]
        ),
    )
    print(json.dumps({"status": summary["status"], "proposal_count": len(parsed_payloads)}, sort_keys=True))


if __name__ == "__main__":
    main()

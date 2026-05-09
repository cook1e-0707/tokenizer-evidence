from scripts.natural_evidence_v2.run_wp3_restricted_step_label_density_audit import (
    forbidden_terms_in_text,
    validate_inputs,
)


def test_forbidden_terms_ignore_ordinary_substrings() -> None:
    config = {"forbidden_surface_terms": ["CERT", "OWNER", "EVIDENCE", "CARRIER", "fingerprint"]}
    text = "Seek guidance from a certified trainer and enjoy pet ownership."

    assert forbidden_terms_in_text(config, text) == []


def test_forbidden_terms_catch_explicit_old_markers() -> None:
    config = {
        "forbidden_surface_terms": [
            "FIELD=",
            "SECTION=",
            "TOPIC=",
            "PAYLOAD",
            "CERT",
            "EVIDENCE",
            "CARRIER",
            "OWNER",
            "fingerprint",
        ]
    }
    text = "FIELD=value\nEVIDENCE: visible marker\nThis mentions a fingerprint."

    assert forbidden_terms_in_text(config, text) == ["FIELD=", "EVIDENCE", "fingerprint"]


def test_validate_inputs_accepts_recombined_step_label_banks() -> None:
    config = {"forbidden_surface_terms": ["FIELD=", "PAYLOAD", "CERT", "EVIDENCE", "OWNER"]}
    prompts = [
        {
            "prompt_id": "p0",
            "expected_structural_slots": 16,
            "prompt_text": "Return exactly sixteen lines. Use Step 1: through Step 16:.",
        }
    ]
    detector_contract = {"allowed_step_indices": list(range(1, 17))}
    allowed_prefixes = [f"Step {index}: " for index in range(1, 17)]
    bucket_bank = {
        "candidate_banks": [
            {
                "candidate_bank_id": "step_label_recombined_create_develop_vs_choose_make_v1",
                "allowed_prefixes": allowed_prefixes,
                "buckets": {"0": ["Create", "Develop"], "1": ["Choose", "Make"]},
            }
        ]
    }
    density_design = {"decision": "strict sixteen Step-label repair audit"}

    validate_inputs(
        config=config,
        prompts=prompts,
        detector_contract=detector_contract,
        bucket_bank=bucket_bank,
        density_design=density_design,
    )

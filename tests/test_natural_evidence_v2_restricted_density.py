from scripts.natural_evidence_v2.run_wp3_restricted_step_label_density_audit import (
    forbidden_terms_in_text,
    load_existing_responses,
    validate_inputs,
)
from scripts.natural_evidence_v2.score_wp3_context_mass import effective_prefix_text


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


def test_load_existing_responses_enriches_prompt_metadata(tmp_path) -> None:
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        '{"prompt_id":"p0","response_text":"Step 1: Start.\\n"}\n',
        encoding="utf-8",
    )

    rows = load_existing_responses(
        responses,
        max_prompts=1,
        prompt_rows=[
            {
                "prompt_id": "p0",
                "split": "wp3_r1_dev",
                "family_id": "F2",
                "variant_id": "strict_literal_16_step_lines",
                "topic": "calendar",
                "prompt_text_sha256": "abc123",
            }
        ],
    )

    assert rows[0]["split"] == "wp3_r1_dev"
    assert rows[0]["variant_id"] == "strict_literal_16_step_lines"
    assert rows[0]["prompt_text_sha256"] == "abc123"


def test_context_mass_effective_prefix_uses_chat_prompt_when_present() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"<chat>{messages[0]['content']}<assistant>"

    row = {
        "prefix_before_candidate": "Step 1: ",
        "chat_prompt_text": "Write sixteen steps.",
        "assistant_prefix_before_candidate": "Step 1: ",
    }

    assert effective_prefix_text(FakeTokenizer(), row) == "<chat>Write sixteen steps.<assistant>Step 1: "

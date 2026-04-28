from __future__ import annotations

from src.core.contextual_alignment import ContextualSlotTarget, audit_contextual_slot_targets
from src.core.scaffolded_completion import (
    COMPILED_ARTIFACT_FORMAT,
    COMPILED_FIELDWISE_PROMPT_CONTRACT,
    FieldwiseGenerationPlan,
    FieldwiseSlotTarget,
)
from src.training.hf_causal_lm import _resolve_fieldwise_contextual_token_map


PREFIX = (
    "Select exactly one allowed carrier token.\n"
    "Payload label: U00\n"
    "Block: 1\n"
    "Slot: 1/4\n"
    "Field: SECTION\n"
    "Allowed carriers: news, report, guide, update, review\n"
    "Value:"
)


class RawBosTokenizer:
    """Minimal HF-like tokenizer that prepends BOS unless explicitly disabled."""

    bos_token_id = 1
    vocab_size = 120

    def __init__(self) -> None:
        self.value_to_id = {
            "news": 101,
            "report": 102,
            "guide": 103,
            "update": 104,
            "review": 105,
        }
        self.id_to_text = {10: PREFIX, 1: "<s>", **{v: k for k, v in self.value_to_id.items()}}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        token_ids: list[int] = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        if not text:
            return token_ids
        if text == PREFIX:
            token_ids.append(10)
        elif text in self.value_to_id:
            token_ids.append(self.value_to_id[text])
        else:
            token_ids.append(99)
        return token_ids

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return "".join(self.id_to_text.get(int(token_id), f"<tok:{int(token_id)}>") for token_id in token_ids)


def test_fieldwise_contextual_audit_wraps_raw_hf_tokenizer_without_special_tokens() -> None:
    tokenizer = RawBosTokenizer()
    slot_target = FieldwiseSlotTarget(
        slot_index=0,
        block_index=0,
        field_name="SECTION",
        prompt=PREFIX,
        exact_slot_prefix=PREFIX,
        allowed_values=("news", "report", "guide", "update", "review"),
        allowed_value_bucket_ids={"news": 0, "report": 1, "guide": 2, "update": 3, "review": 3},
        expected_value="news",
        expected_bucket_id=0,
    )
    plan = FieldwiseGenerationPlan(
        payload_text="U00",
        slot_targets=(slot_target,),
        expected_slot_values=("news",),
        fields_per_block=1,
        prompt_contract_name=COMPILED_FIELDWISE_PROMPT_CONTRACT,
        artifact_format=COMPILED_ARTIFACT_FORMAT,
    )

    raw_audit = audit_contextual_slot_targets(
        slot_targets=(
            ContextualSlotTarget(
                field_name=slot_target.field_name,
                exact_slot_prefix=slot_target.exact_slot_prefix,
                allowed_values=slot_target.allowed_values,
            ),
        ),
        tokenizer=tokenizer,
        prompt_contract_name=plan.prompt_contract_name,
    )
    assert raw_audit.is_context_safe is False
    assert raw_audit.diagnostics[0].reasons == ("not_single_next_token_in_context",)

    adapted_audit, slot_token_maps = _resolve_fieldwise_contextual_token_map(
        tokenizer=tokenizer,
        plan=plan,
    )

    assert adapted_audit.is_context_safe is True
    value_to_token_id, token_id_to_value = slot_token_maps[("SECTION", PREFIX)]
    assert value_to_token_id == {
        "news": 101,
        "report": 102,
        "guide": 103,
        "update": 104,
        "review": 105,
    }
    assert token_id_to_value[101] == "news"

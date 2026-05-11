#!/usr/bin/env python3
"""Fix label masking to use cumulative text scan instead of prefix tokenization.
Prefix tokenization gives wrong token count due to whitespace merging differences.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py")
content = path.read_text()

# Replace the label masking fallback
old_mask_fallback = '''    else:
        # Fallback: mask tokens before response using prefix length
        prefix_ids = tokenizer(full_text[:response_start], add_special_tokens=False)["input_ids"]
        response_token_start = len(prefix_ids)
        for token_index in range(response_token_start):
            labels[token_index] = -100'''

new_mask_fallback = '''    else:
        # Fallback: find response start by scanning cumulative decoded text
        # (prefix tokenization alone gives wrong count due to whitespace merging)
        response_token_start = _find_response_start_token(tokenizer, full_text, response_text)
        for token_index in range(response_token_start):
            labels[token_index] = -100'''

assert old_mask_fallback in content, f"Old mask fallback not found"
content = content.replace(old_mask_fallback, new_mask_fallback, 1)

# Add the helper function after _offsets_valid
old_offset_fn_end = '''def _offsets_valid(offsets) -> bool:
    """Check if offset_mapping has non-degenerate spans."""
    if not offsets:
        return False
    non_degen = sum(1 for s, e in offsets if e > s)
    return non_degen > len(offsets) * 0.5'''

new_offset_fn_end = '''def _offsets_valid(offsets) -> bool:
    """Check if offset_mapping has non-degenerate spans."""
    if not offsets:
        return False
    non_degen = sum(1 for s, e in offsets if e > s)
    return non_degen > len(offsets) * 0.5


def _find_response_start_token(tokenizer, full_text: str, response_text: str) -> int:
    """Find the token index where response_text starts in the full token sequence.

    Uses cumulative decoded text scanning to handle tokenizer whitespace merging.
    """
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    # Use first 30 chars of response as anchor to avoid false positives
    anchor = response_text[:30]
    cumulative = ""
    for i, tid in enumerate(full_ids):
        cumulative += tokenizer.decode([tid], skip_special_tokens=False)
        if anchor in cumulative:
            # Backtrack: find the token where the anchor actually starts
            # by checking cumulative without this token
            prev_cumulative = cumulative[: -len(tokenizer.decode([tid], skip_special_tokens=False))]
            anchor_start_in_cum = cumulative.index(anchor)
            if anchor_start_in_cum < len(prev_cumulative):
                return i
            return max(0, i)
    return 0'''

assert old_offset_fn_end in content, "Old _offsets_valid not found"
content = content.replace(old_offset_fn_end, new_offset_fn_end, 1)

path.write_text(content)
print(f"Patched: {path}")
print("Fix: label masking uses cumulative text scan instead of prefix tokenization")

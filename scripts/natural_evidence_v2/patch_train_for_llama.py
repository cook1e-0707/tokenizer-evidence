#!/usr/bin/env python3
"""Patch train_wp5_micro_slot_lora.py to handle broken offset_mapping
(e.g. SentencePiece-based Llama tokenizer returns start==end offsets).
"""

import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py")
content = path.read_text()

# 1. Add helper functions before find_slot_surface_spans
helper = '''
def _find_surface_token_index_by_prefix(
    tokenizer, full_text: str, surface_char_start: int, surface: str
) -> "int | None":
    """Find the token index of a surface using prefix tokenization.

    Fallback for tokenizers whose offset_mapping is unreliable
    (e.g. SentencePiece-based Llama tokenizers return start==end offsets).
    """
    prefix_text = full_text[:surface_char_start]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    surface_token_index = len(prefix_ids)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if surface_token_index >= len(full_ids):
        return None
    decoded = tokenizer.decode([full_ids[surface_token_index]]).strip()
    if decoded.lower() == surface.lower() or surface.lower() in decoded.lower():
        return surface_token_index
    if surface_token_index + 1 < len(full_ids):
        decoded2 = tokenizer.decode([full_ids[surface_token_index + 1]]).strip()
        if decoded2.lower() == surface.lower() or surface.lower() in decoded2.lower():
            return surface_token_index + 1
    return None


def _offsets_valid(offsets) -> bool:
    """Check if offset_mapping has non-degenerate spans."""
    if not offsets:
        return False
    non_degen = sum(1 for s, e in offsets if e > s)
    return non_degen > len(offsets) * 0.5


'''

marker = "def find_slot_surface_spans"
assert marker in content, f"Marker not found: {marker}"
content = content.replace(marker, helper + marker, 1)

# 2. Replace the label masking loop
old_mask = '''    for token_index, offset in enumerate(offsets):
        if offset[1] <= response_start:
            labels[token_index] = -100
            continue
        for slot_span in slot_spans_full:
            if overlaps(offset, slot_span):
                labels[token_index] = -100
                break'''

new_mask = '''    offsets_ok = _offsets_valid(offsets)
    if offsets_ok:
        for token_index, offset in enumerate(offsets):
            if offset[1] <= response_start:
                labels[token_index] = -100
                continue
            for slot_span in slot_spans_full:
                if overlaps(offset, slot_span):
                    labels[token_index] = -100
                    break
    else:
        # Fallback: mask tokens before response using prefix length
        prefix_ids = tokenizer(full_text[:response_start], add_special_tokens=False)["input_ids"]
        response_token_start = len(prefix_ids)
        for token_index in range(response_token_start):
            labels[token_index] = -100'''

assert old_mask in content, "Label masking code not found"
content = content.replace(old_mask, new_mask, 1)

# 3. Replace the surface token lookup loop
old_loop = '''    for slot in slot_targets:
        target_surface = str(slot.get("target_surface", ""))
        target_line = str(slot.get("target_line", ""))
        surface_start_in_line = target_line.find(target_surface)
        line_start = response_text.find(target_line)
        if line_start < 0 or surface_start_in_line < 0:
            continue
        surface_start = response_start + line_start + surface_start_in_line
        surface_token_index = None
        for token_index, offset in enumerate(offsets):
            if offset[0] <= surface_start < offset[1]:
                surface_token_index = token_index
                break
        if surface_token_index is None or surface_token_index == 0:
            continue'''

new_loop = '''    use_prefix_fallback = not _offsets_valid(offsets)
    for slot in slot_targets:
        target_surface = str(slot.get("target_surface", ""))
        target_line = str(slot.get("target_line", ""))
        surface_start_in_line = target_line.find(target_surface)
        line_start = response_text.find(target_line)
        if line_start < 0 or surface_start_in_line < 0:
            continue
        surface_start = response_start + line_start + surface_start_in_line
        surface_token_index = None
        if use_prefix_fallback:
            surface_token_index = _find_surface_token_index_by_prefix(
                tokenizer, full_text, surface_start, target_surface
            )
        else:
            for token_index, offset in enumerate(offsets):
                if offset[0] <= surface_start < offset[1]:
                    surface_token_index = token_index
                    break
        if surface_token_index is None or surface_token_index == 0:
            continue'''

assert old_loop in content, "Surface lookup code not found"
content = content.replace(old_loop, new_loop, 1)

path.write_text(content)
print(f"Patched: {path}")
print("Changes: added _find_surface_token_index_by_prefix, _offsets_valid helpers")
print("Updated: label masking + surface token lookup to use prefix fallback")

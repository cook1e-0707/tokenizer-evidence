#!/usr/bin/env python3
"""Fix _find_surface_token_index_by_prefix to use full-text token scan
instead of prefix length, which fails when trailing whitespace merges differently.

Also fix _offsets_valid and label masking.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scripts/natural_evidence_v2/train_wp5_micro_slot_lora.py")
content = path.read_text()

# Replace the _find_surface_token_index_by_prefix function
old_fn = '''def _find_surface_token_index_by_prefix(
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
    return None'''

new_fn = '''def _find_surface_token_index_by_prefix(
    tokenizer, full_text: str, surface_char_start: int, surface: str
) -> "int | None":
    """Find the token index of a surface in full_text using token decoding.

    Fallback for tokenizers whose offset_mapping is unreliable
    (e.g. SentencePiece-based Llama tokenizers return start==end offsets).

    Strategy: tokenize the prefix to get an approximate index, then scan
    nearby tokens in the full-text tokenization for the surface string.
    This handles the case where trailing whitespace merges differently
    when tokenized alone vs in context.
    """
    prefix_text = full_text[:surface_char_start]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    approx_idx = len(prefix_ids)
    # Scan a window around the approximate index for the surface token
    surface_lower = surface.lower()
    search_start = max(0, approx_idx - 4)
    search_end = min(len(full_ids), approx_idx + 4)
    for idx in range(search_start, search_end):
        decoded = tokenizer.decode([full_ids[idx]]).strip().lower()
        if decoded == surface_lower or surface_lower in decoded:
            return idx
    return None'''

assert old_fn in content, "Old function not found"
content = content.replace(old_fn, new_fn, 1)

path.write_text(content)
print(f"Patched: {path}")
print("Fix: scan nearby tokens for surface match instead of trusting prefix length")

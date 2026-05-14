from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _check_token_ids(name: str, ids: Sequence[int], vocab_size: int) -> list[int]:
    values = sorted({int(item) for item in ids})
    if not values:
        raise ValueError(f"{name} must not be empty")
    bad = [item for item in values if item < 0 or item >= vocab_size]
    if bad:
        raise ValueError(f"{name} contains out-of-range token ids: {bad[:5]}")
    return values


def softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("logits must not be empty")
    max_logit = max(float(item) for item in logits)
    exp_values = [math.exp(float(item) - max_logit) for item in logits]
    denom = sum(exp_values)
    if denom <= 0:
        raise ValueError("softmax denominator is non-positive")
    return [item / denom for item in exp_values]


def mass(probs: Sequence[float], token_ids: Sequence[int]) -> float:
    return sum(float(probs[int(token_id)]) for token_id in token_ids)


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q):
        raise ValueError("KL inputs must have the same length")
    total = 0.0
    for p_i, q_i in zip(p, q):
        p_i = float(p_i)
        q_i = float(q_i)
        if p_i <= 0:
            continue
        if q_i <= 0:
            return math.inf
        total += p_i * math.log(p_i / q_i)
    return total


def _apply_raw_adjustment(
    logits: Sequence[float],
    target_token_ids: Sequence[int],
    other_token_ids: Sequence[int],
    *,
    bonus: float,
    penalty: float,
    scale: float = 1.0,
) -> list[float]:
    adjusted = [float(item) for item in logits]
    for token_id in target_token_ids:
        adjusted[int(token_id)] += float(bonus) * float(scale)
    for token_id in other_token_ids:
        adjusted[int(token_id)] -= float(penalty) * float(scale)
    return adjusted


def apply_controller(
    logits: Sequence[float],
    target_token_ids: Sequence[int],
    other_token_ids: Sequence[int],
    *,
    bonus: float,
    penalty: float = 0.0,
    max_kl_budget: float | None = None,
    max_target_mass: float | None = None,
    mode: str = "additive",
) -> dict[str, Any]:
    """Apply a prefix-native soft logit controller to one next-token distribution.

    This helper is pure arithmetic. It does not load a model, generate text, train,
    submit Slurm, or make any claim about natural-output recovery.
    """

    if mode == "disabled":
        probs = softmax(logits)
        return {
            "controlled_logits": [float(item) for item in logits],
            "kl_to_base": 0.0,
            "mode": mode,
            "scale": 0.0,
            "target_mass": mass(probs, target_token_ids),
            "other_mass": mass(probs, other_token_ids),
        }
    if mode != "additive":
        raise ValueError(f"unsupported controller mode: {mode}")
    if bonus < 0 or penalty < 0:
        raise ValueError("bonus and penalty must be non-negative")

    vocab_size = len(logits)
    target_ids = _check_token_ids("target_token_ids", target_token_ids, vocab_size)
    other_ids = _check_token_ids("other_token_ids", other_token_ids, vocab_size)
    overlap = sorted(set(target_ids).intersection(other_ids))
    if overlap:
        raise ValueError(f"target/other token id overlap: {overlap[:5]}")

    base_probs = softmax(logits)

    def candidate(scale: float) -> tuple[list[float], list[float], float, float]:
        adjusted_logits = _apply_raw_adjustment(logits, target_ids, other_ids, bonus=bonus, penalty=penalty, scale=scale)
        adjusted_probs = softmax(adjusted_logits)
        return (
            adjusted_logits,
            adjusted_probs,
            mass(adjusted_probs, target_ids),
            kl_divergence(adjusted_probs, base_probs),
        )

    scale = 1.0
    _, _, target_mass, kl_value = candidate(scale)
    cap_reasons: list[str] = []
    if max_target_mass is not None and target_mass > float(max_target_mass):
        cap_reasons.append("max_target_mass")
    if max_kl_budget is not None and kl_value > float(max_kl_budget):
        cap_reasons.append("max_kl_budget")
    if cap_reasons:
        low = 0.0
        high = 1.0
        for _ in range(50):
            mid = (low + high) / 2.0
            _, _, mid_mass, mid_kl = candidate(mid)
            ok = True
            if max_target_mass is not None and mid_mass > float(max_target_mass):
                ok = False
            if max_kl_budget is not None and mid_kl > float(max_kl_budget):
                ok = False
            if ok:
                low = mid
            else:
                high = mid
        scale = low

    controlled_logits, controlled_probs, target_mass, kl_value = candidate(scale)
    return {
        "cap_reasons": cap_reasons,
        "controlled_logits": controlled_logits,
        "kl_to_base": kl_value,
        "mode": mode,
        "other_mass": mass(controlled_probs, other_ids),
        "scale": scale,
        "target_mass": target_mass,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure arithmetic smoke CLI for the R4 prefix-native soft controller.")
    parser.add_argument("--logits-json", type=Path, required=True)
    parser.add_argument("--target-token-ids", required=True)
    parser.add_argument("--other-token-ids", required=True)
    parser.add_argument("--bonus", type=float, default=1.0)
    parser.add_argument("--penalty", type=float, default=0.0)
    parser.add_argument("--max-kl-budget", type=float, default=None)
    parser.add_argument("--max-target-mass", type=float, default=None)
    parser.add_argument("--mode", choices=["additive", "disabled"], default="additive")
    return parser.parse_args()


def _parse_ids(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    logits = json.loads(args.logits_json.read_text(encoding="utf-8"))
    result = apply_controller(
        logits,
        _parse_ids(args.target_token_ids),
        _parse_ids(args.other_token_ids),
        bonus=args.bonus,
        penalty=args.penalty,
        max_kl_budget=args.max_kl_budget,
        max_target_mass=args.max_target_mass,
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

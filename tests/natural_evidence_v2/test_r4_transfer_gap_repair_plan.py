from __future__ import annotations

from scripts.natural_evidence_v2.validate_r4_transfer_gap_repair_plan import DEFAULT_CONFIG, load_yaml, validate_plan


def test_default_transfer_gap_repair_plan_passes() -> None:
    summary = validate_plan(load_yaml(DEFAULT_CONFIG))

    assert summary["status"] == "PASS_R4_TRANSFER_GAP_REPAIR_PLAN_VALIDATION"
    assert summary["repair_surface_count"] == 4
    assert summary["total_prefix_shapes"] >= 6
    assert summary["generation_started"] is False
    assert summary["training_started"] is False
    assert summary["current_compute_unlocked"] is False
    assert summary["future_compute_conditionally_authorized_after_prerequisites"] is True


def test_transfer_gap_plan_rejects_generation_unlock() -> None:
    plan = load_yaml(DEFAULT_CONFIG)
    plan["current_permissions"]["generation_allowed"] = True

    summary = validate_plan(plan)

    assert summary["status"].startswith("FAIL")
    assert "current_permissions.generation_allowed must be false before repair-package review" in summary["errors"]


def test_transfer_gap_plan_rejects_permanent_forbidden_semantics() -> None:
    plan = load_yaml(DEFAULT_CONFIG)
    plan["standing_execution_policy"]["gate_controlled_actions_permanently_forbidden"] = True

    summary = validate_plan(plan)

    assert summary["status"].startswith("FAIL")
    assert "standing_execution_policy.gate_controlled_actions_permanently_forbidden must be false" in summary["errors"]


def test_transfer_gap_plan_rejects_step_template_instruction() -> None:
    plan = load_yaml(DEFAULT_CONFIG)
    plan["prompt_scaffold_policy"]["instruction_principles"].append("Use Step 1 through Step 4.")

    summary = validate_plan(plan)

    assert summary["status"].startswith("FAIL")
    assert any("disallowed literals" in error for error in summary["errors"])


def test_transfer_gap_plan_rejects_non_h200_future_route() -> None:
    plan = load_yaml(DEFAULT_CONFIG)
    plan["future_route"]["h200_gres"] = "gpu:a100:1"

    summary = validate_plan(plan)

    assert summary["status"].startswith("FAIL")
    assert "future_route.h200_gres must be gpu:h200:1" in summary["errors"]

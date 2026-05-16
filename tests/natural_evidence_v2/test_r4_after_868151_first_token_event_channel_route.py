from __future__ import annotations

from pathlib import Path

from scripts.natural_evidence_v2.validate_r4_after_868151_first_token_event_channel_route import read_yaml, validate


def test_first_token_event_channel_route_validates_without_compute() -> None:
    config = read_yaml(Path("configs/natural_evidence_v2/r4_after_868151_first_token_event_channel.yaml"))

    errors = validate(config)

    assert errors == []
    policy = config["execution_policy"]
    assert policy["artifact_only_plan"] is True
    assert policy["slurm_allowed"] is False
    assert policy["generation_allowed"] is False
    assert policy["model_forward_allowed"] is False
    assert policy["training_allowed"] is False
    assert policy["paper_claim_allowed"] is False
